"""V4.5 student table, twin configs, plan guards and the distillation loss, without training."""
import copy
import math
from pathlib import Path
import sys
import tomllib
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.training import joint_v45 as v45
from vimeml.training.model_v4 import TinyGPTV4, GPTV4Config


def load(arm):
    return tomllib.loads((ROOT / f'configs/train-v45-{arm}.toml').read_text('utf-8'))


class V45ConfigTests(unittest.TestCase):
    def test_eight_students_twin_the_four_teachers(self):
        self.assertEqual(sorted(v45.ARMS), sorted(f'{p}-{s}' for p in ('geo', 'mix') for s in ('v42', 's1', 's2', 's3')))
        self.assertEqual(set(v45.TWINS), set(v45.TEACHERS))
        for name, spec in v45.TEACHERS.items():
            self.assertRegex(spec['sha256'], '^[0-9a-f]{64}$')

    def test_configs_equal_their_twin_outside_the_student_fields(self):
        for arm, (pool, seed) in v45.ARMS.items():
            config, twin = load(arm), v45.twin_config(seed)
            self.assertEqual(v45.strip_student(config), v45.strip_student(twin), arm)
            self.assertEqual(config['training']['seed'], twin['training']['seed'])
            self.assertEqual(config['dataset_sha256'], twin['dataset_sha256'])
            self.assertEqual(config['distillation'], {'pool': pool, 'twin': seed, 'weight': 0.5,
                                                      'teachers': sorted(v45.TEACHERS)})
            self.assertEqual(config['tracking']['group'], 'v45-distillation')
            self.assertIn(f'v45-{arm}-', config['output_dir'])

    def test_plan_rejects_drift_from_the_twin_or_arm(self):
        config = load('geo-s1')
        for path, value in ((('training', 'seed'), 1), (('training', 'learning_rate'), 1e-3),
                            (('training', 'prefix_crop_probability'), .6)):
            broken = copy.deepcopy(config)
            broken[path[0]][path[1]] = value
            with self.assertRaisesRegex(ValueError, 'twin'):
                v45.build_plan(broken, ROOT, verify_files=False)
        broken = copy.deepcopy(config)
        broken['distillation']['pool'] = 'mix'
        with self.assertRaisesRegex(ValueError, 'registered arm'):
            v45.build_plan(broken, ROOT, verify_files=False)
        broken = copy.deepcopy(config)
        broken['arm'] = 'geo-s4'
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            v45.build_plan(broken, ROOT, verify_files=False)


class V45LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.models = [TinyGPTV4(GPTV4Config()).eval() for _ in range(2)]
        self.ids = torch.randint(4, 8192, (2, 12))
        self.labels = torch.randint(4, 8192, (2, 12))
        self.labels[1, :3] = -100
        self.valid = self.labels != -100
        self.cpu = torch.device('cpu')

    def logp(self, model):
        with torch.no_grad():
            return model(self.ids)[self.valid].float().log_softmax(-1)

    def test_pools_are_normalized_and_match_their_definitions(self):
        a, b = (self.logp(m) for m in self.models)
        geo = v45.pooled_teacher(self.models, self.ids, self.valid, 'geo', self.cpu, 'fp32')
        mix = v45.pooled_teacher(self.models, self.ids, self.valid, 'mix', self.cpu, 'fp32')
        for pooled in (geo, mix):
            torch.testing.assert_close(pooled.exp().sum(-1), torch.ones(len(pooled)))
        torch.testing.assert_close(geo, ((a + b) / 2).log_softmax(-1))
        torch.testing.assert_close(mix, torch.log((a.exp() + b.exp()) / 2))
        same = v45.pooled_teacher([self.models[0]] * 3, self.ids, self.valid, 'geo', self.cpu, 'fp32')
        torch.testing.assert_close(same, a)

    def test_update_reports_ce_kl_and_trains(self):
        student = TinyGPTV4(GPTV4Config())
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        batch = {'input_ids': self.ids, 'labels': self.labels, 'lengths': torch.tensor([12, 12])}
        settings = {'lm_microbatch': 1, 'grad_clip': 1.0}
        result = v45.update_lm(student, optimizer, [batch], settings, self.cpu, 'fp32', self.models, 'geo')
        self.assertEqual(result['supervised_tokens'], int(self.valid.sum()))
        w = v45.DISTILLATION_WEIGHT
        self.assertAlmostEqual(result['loss'], (1 - w) * result['ce'] + w * result['soft'], places=4)
        self.assertGreater(result['kl'], 0)
        self.assertTrue(all(math.isfinite(v) for v in result.values()))
        # A student equal to the pooled single teacher has zero KL.
        twin = copy.deepcopy(self.models[0]).train()
        optimizer = torch.optim.AdamW(twin.parameters(), lr=0.0)
        result = v45.update_lm(twin, optimizer, [batch], settings, self.cpu, 'fp32', [self.models[0]], 'mix')
        self.assertAlmostEqual(result['kl'], 0.0, places=4)
        self.assertAlmostEqual(result['teacher_ce'], result['ce'], places=4)


if __name__ == '__main__':
    unittest.main()
