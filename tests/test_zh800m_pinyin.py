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
    return tomllib.loads((ROOT / f'configs/{name}.toml').read_text('utf-8'))


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
        for microbatch in (8, 3, 1):
            got, grad = self.grads(ported, microbatch)
            torch.testing.assert_close(grad, want_grad, rtol=1e-5, atol=1e-6)
            self.assertAlmostEqual(got['loss'], want['loss'], places=5)
            self.assertEqual(set(got['pinyin']), {'full', 'error'})
            for kind in got['pinyin']:
                self.assertAlmostEqual(got['pinyin'][kind]['loss'], want['pinyin'][kind]['loss'], places=5)
            self.assertEqual(got['padded_positions'], 8 * 16)


if __name__ == '__main__':
    unittest.main()
