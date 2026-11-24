"""zh-800m joint pinyin pretraining: config, plan extension and the microbatched joint update."""
import copy
from pathlib import Path
import sys
import tomllib
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.training import joint_zh13, joint_zh200m
from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2

IDENTITY = {'output_dir', 'log_dir', 'tracking', 'pinyin'}
STORE = ROOT / 'artifacts/training-data/zh-pinyin-pt-12k'


def load(name):
    return tomllib.loads((ROOT / f'configs/zh/{name}.toml').read_text('utf-8'))


def batch(rng, rows, length, vocab):
    ids = torch.randint(0, vocab, (rows, length), generator=rng)
    labels = ids.clone()
    labels[:, : length // 3] = -100
    return {'input_ids': ids, 'labels': labels, 'lengths': torch.full((rows,), length)}


class Zh800mPinyinConfigTests(unittest.TestCase):
    def test_config_is_the_plain_arm_plus_pinyin(self):
        plain, joint = load('train-zh800m-12k-c512'), load('train-zh800m-12k-c512-pinyin')
        self.assertEqual({k: v for k, v in joint.items() if k not in IDENTITY},
                         {k: v for k, v in plain.items() if k not in IDENTITY})
        self.assertNotEqual(joint['output_dir'], plain['output_dir'])
        self.assertEqual(joint['pinyin']['mix'], {'full': 96, 'mixed': 128, 'initials': 112, 'error': 48})

    @unittest.skipUnless((STORE / 'manifest.json').exists(), 'pinyin store not present')
    def test_add_pinyin_extends_the_plan(self):
        joint = load('train-zh800m-12k-c512-pinyin')
        manifest = joint_zh200m.read(STORE / 'manifest.json')
        plan = {'tokenizer_identity': {'tokenizer_model_sha256': manifest['tokenizer_sha256']},
                'settings': {'lm_batch': 2048, 'max_steps': 1000}, 'signatures': {}, 'required_files': {},
                'task_contract': {'objective': 'lm'}}
        joint_zh200m.add_pinyin(plan, joint['pinyin'], ROOT, verify_files=False)
        p = plan['pinyin']
        self.assertEqual(p['batch'], 2048 + 384)
        self.assertAlmostEqual(sum(p['weights'].values()), 1.0)
        self.assertEqual(plan['task_contract']['objective'], 'joint_lm_and_pinyin_per_kind')
        self.assertEqual(plan['signatures']['pinyin'], joint['pinyin']['manifest_sha256'])
        for bad, message in (({**joint['pinyin'], 'manifest_sha256': '0' * 64}, 'changed'),
                             ({**joint['pinyin'], 'mix': {'full': 1, 'plain': 1}}, 'mix'),
                             ({**joint['pinyin'], 'plain_arm': 'x'}, 'Unknown')):
            with self.assertRaises(ValueError) as caught:
                joint_zh200m.add_pinyin(copy.deepcopy(plan), bad, ROOT, verify_files=False)
            self.assertIn(message, str(caught.exception))
        mismatch = {**plan, 'tokenizer_identity': {'tokenizer_model_sha256': '0' * 64}}
        with self.assertRaises(ValueError):
            joint_zh200m.add_pinyin(mismatch, joint['pinyin'], ROOT, verify_files=False)


class Zh800mJointUpdateTests(unittest.TestCase):
    def grads(self, update, microbatch):
        torch.manual_seed(0)
        model = TinyGPTV2(GPTV2Config(vocab_size=64, context_length=16, d_model=32, n_heads=2,
                                              n_layers=2, d_ff=64, dropout=0.0))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        rng = torch.Generator().manual_seed(1)
        batches = {'lm': batch(rng, 8, 16, 64), 'full': batch(rng, 3, 10, 64), 'error': batch(rng, 5, 12, 64)}
        weights = {'lm': 0.5, 'full': 0.2, 'error': 0.3}
        s = {'lm_microbatch': microbatch, 'grad_clip': 1e9}
        result = update(model, optimizer, batches, weights, s)
        return result, torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    def test_matches_the_unsplit_joint_update(self):
        def reference(model, optimizer, batches, weights, s):
            extra = {k: v for k, v in batches.items() if k != 'lm'}
            return joint_zh13.update_joint(model, optimizer, batches['lm'], extra, weights, s, torch.device('cpu'), 'fp32')

        def ported(model, optimizer, batches, weights, s):
            return joint_zh200m.update_joint(model, optimizer, batches, weights, s, torch.device('cpu'), 'fp32')

        want, want_grad = self.grads(reference, 8)
        # Every task is packed: 8 LM rows of 16 plus 3 rows of 10 and 5 of 12, all of width 16.
        # Microbatch 3 (48 positions) cuts the 16 rows 3 at a time and pads the last single row
        # with an empty one; microbatch 1 (16 positions) has no room for that.
        for microbatch, rows in ((8, 16), (3, 17), (1, 16)):
            got, grad = self.grads(ported, microbatch)
            torch.testing.assert_close(grad, want_grad, rtol=1e-5, atol=1e-6)
            self.assertAlmostEqual(got['loss'], want['loss'], places=5)
            self.assertEqual(set(got['pinyin']), {'full', 'error'})
            for kind in got['pinyin']:
                self.assertAlmostEqual(got['pinyin'][kind]['loss'], want['pinyin'][kind]['loss'], places=5)
            self.assertEqual(got['padded_positions'], rows * 16)

    def test_packed_rows(self):
        rng = torch.Generator().manual_seed(3)
        a, b = batch(rng, 5, 16, 64), batch(rng, 4, 10, 64)
        a['lengths'] = torch.tensor([16, 3, 9, 12, 1])
        b['lengths'] = torch.tensor([10, 2, 7, 10])
        for x in (a, b):
            for r, n in enumerate(x['lengths'].tolist()):
                x['input_ids'][r, n:] = 0
                x['labels'][r, n:] = -100
        groups = joint_zh200m.packed_rows([(0, a), (1, b)], 32)
        rows = [(int((ids[j] != 0).sum()), int(kind[j])) for ids, _, kind in groups for j in range(len(kind))]
        for ids, labels, kind in groups:
            self.assertLessEqual(ids.numel(), 32)
            self.assertEqual(ids.shape, labels.shape)
            self.assertEqual(ids.shape[1] % 8, 0)
        self.assertEqual(sorted(k for _, k in rows), [0] * 5 + [1] * 4)
        # Every supervised (row, token) appears once with its label.
        want = sorted((k, tuple(x['labels'][r][x['labels'][r] != -100].tolist()))
                      for k, x in ((0, a), (1, b)) for r in range(len(x['lengths'])))
        got = sorted((int(kind[j]), tuple(labels[j][labels[j] != -100].tolist()))
                     for _, labels, kind in groups for j in range(len(kind)))
        self.assertEqual(got, want)
        with self.assertRaises(ValueError):
            joint_zh200m.packed_rows([(0, a)], 8)

    def test_segments_trade_padding_against_passes(self):
        # 4 rows of width 64 and 40 of width 16: merged 44 * 64 = 2816 positions in one pass,
        # split 256 + 640 = 896 in two. One pass costs 1000: split; costs 3000: merge.
        widths = [64] * 4 + [16] * 40
        self.assertEqual(joint_zh200m.segments(widths, 4096, pass_cost=1000), [0, 4])
        self.assertEqual(joint_zh200m.segments(widths, 4096, pass_cost=3000), [0])
        # Without a pass cost every width level is its own segment.
        self.assertEqual(joint_zh200m.segments([24, 24, 16, 8, 8], 4096, pass_cost=0), [0, 2, 3])
        self.assertEqual(joint_zh200m.segments([], 64), [])

    def test_packed_rows_keep_long_pinyin_rows_away_from_short_lm_rows(self):
        rng = torch.Generator().manual_seed(4)
        lm, pinyin = batch(rng, 64, 40, 64), batch(rng, 6, 104, 64)
        lm['lengths'] = torch.full((64,), 38)
        pinyin['lengths'] = torch.tensor([104, 90, 70, 30, 20, 12])
        groups = joint_zh200m.packed_rows([(0, lm), (1, pinyin)], 64 * 512, pass_cost=1024)
        self.assertEqual([tuple(ids.shape) for ids, _, _ in groups], [(3, 104), (67, 40)])
        merged = joint_zh200m.packed_rows([(0, lm), (1, pinyin)], 64 * 512, pass_cost=10 ** 9)
        self.assertEqual([tuple(ids.shape) for ids, _, _ in merged], [(70, 104)])


class FakeLoader:
    def __init__(self, data, batches):
        self.dataset, self.batch_sampler = data, batches

    @staticmethod
    def collate_fn(rows):
        width = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
        labels = torch.where(ids > 0, ids, torch.full_like(ids, -100))
        return {'input_ids': ids, 'labels': labels, 'lengths': torch.tensor([len(r) for r in rows])}


class SumModel(torch.nn.Module):
    def forward(self, ids, labels):
        valid = labels != -100
        return {'loss_sum': (ids.float() * 0.37)[valid].sum(), 'token_count': valid.sum()}


class SharedValidationTests(unittest.TestCase):
    def test_rank_shares_add_up_to_one_device(self):
        from vimeml.posttraining.train import validate
        from vimeml.training.distributed import Context

        class Own(Context):     # each rank's partial sums, added up here instead of all-reduced
            active = False
        rng = random_lists()
        loader = FakeLoader(rng, [[0, 1, 2], [3, 4], [5, 6, 7, 8], [9], [10, 11]])
        model = SumModel()
        want = validate(model, loader_iter(loader), 'cpu', characters=100)
        self.assertEqual(joint_zh200m.shared_validate(model, loader, 'cpu', Context(), characters=100), want)
        for world in (2, 3):
            parts = [joint_zh200m.shared_validate(model, loader, 'cpu', Own(world, r)) for r in range(world)]
            for key in ('nll_sum', 'supervised_tokens', 'input_tokens', 'examples'):
                self.assertAlmostEqual(sum(p[key] for p in parts), want[key], places=9)


def random_lists():
    g = torch.Generator().manual_seed(5)
    return [torch.randint(1, 50, (int(n),), generator=g).tolist() for n in torch.randint(2, 9, (12,), generator=g)]


def loader_iter(loader):
    return [loader.collate_fn([loader.dataset[j] for j in b]) for b in loader.batch_sampler]


class BackgroundSaverTests(unittest.TestCase):
    def test_writes_a_snapshot_and_reports_errors(self):
        import tempfile
        weight = torch.ones(4)
        saver = joint_zh200m.BackgroundSaver()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pt'
            saver.submit(path, {'model': {'w': weight}, 'step': 3, 'rng': [torch.zeros(2)]})
            weight += 1                     # training goes on; the snapshot was taken at submit
            saver.wait()
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved['step'], 3)
            torch.testing.assert_close(saved['model']['w'], torch.ones(4))
            self.assertFalse(path.with_suffix('.pt.tmp').exists())
            saver.submit(Path(folder) / 'missing' / 'last.pt', {'step': 4})
            with self.assertRaises(RuntimeError):     # torch.save: parent directory missing
                saver.wait()
            saver.wait()                    # the error is reported once


if __name__ == '__main__':
    unittest.main()
