"""zh-200m arm table, configs and plan guards, without model/corpus training."""
from pathlib import Path
import sys
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.training.joint_zh200m import TOKENS_PER_UPDATE, build_plan
from vimeml.training.model_zh200m import SHAPES
from vimeml.training.zh200m_data import ARMS, TAGS

# Keys an arm config may differ in from the default 24k-c512-s1 arm.
IDENTITY = {'stage', 'arm', 'output_dir', 'log_dir', 'dataset_sha256'}


def load(name):
    return tomllib.loads((ROOT / f'configs/{name}.toml').read_text('utf-8'))


def plan_error(config):
    try:
        build_plan(config)
    except ValueError as error:
        return str(error)
    raise AssertionError('build_plan accepted an invalid configuration')


class Zh200mArmTests(unittest.TestCase):
    def test_arms_match_registered_shapes_and_batch_policy(self):
        for arm, (vocab, context, pool, epochs, seed) in ARMS.items():
            self.assertIn(vocab, TAGS)
            self.assertEqual((pool, epochs), ('P1', 1), arm)  # every arm is fresh single-epoch P1
            self.assertEqual(context, 512 if '-c512' in arm else 128, arm)
        self.assertEqual(sorted(ARMS), ['16k-c512', '24k-c128', '24k-c512-s1', '24k-c512-s2',
                                         '24k-c512-s3', '32k-c512'])
        seeds = [ARMS[a][4] for a in ARMS]
        for a in set(seeds):
            for b in set(seeds) - {a}:
                self.assertGreater(abs(a - b), 20)  # chat streams must never overlap
        for (vocab, context), (layers, parameters) in SHAPES.items():
            self.assertEqual(parameters, vocab * 1024 + context * 1024
                             + layers * 12_060_672 + 1024)

    def test_configs_are_the_default_arm_plus_identity(self):
        default = load('train-zh200m-24k-c512-s1')
        for arm in ARMS:
            config = load(f'train-zh200m-{arm}')
            vocab, context, _, _, seed = ARMS[arm]
            self.assertEqual(config['stage'], 'zh-200m')
            self.assertEqual(config['architecture'], 'tiny_gpt_zh200m')
            self.assertEqual(config['arm'], arm)
            self.assertEqual(config['model']['vocab_size'], vocab, arm)
            self.assertEqual(config['model']['context_length'], context, arm)
            self.assertEqual(config['model']['n_layers'], SHAPES[(vocab, context)][0], arm)
            self.assertEqual(config['training']['seed'], seed, arm)
            self.assertEqual(config['training']['lm_batch'] * context, TOKENS_PER_UPDATE, arm)
            self.assertEqual(config['data_mixture']['directory'], f'artifacts/training-data/zh200m-{arm}', arm)
            self.assertRegex(config['dataset_sha256'], r'^(|[0-9a-f]{64})$',
                             f'{arm}: pin the mixture SHA before training')
            trimmed = {k: v for k, v in config.items() if k not in IDENTITY | {'data_mixture'}}
            trimmed['model'] = {k: v for k, v in config['model'].items()
                                if k not in ('vocab_size', 'context_length', 'n_layers')}
            trimmed['training'] = {k: v for k, v in config['training'].items()
                                   if k not in ('seed', 'lm_batch', 'lm_microbatch', 'validation_batch')}
            reference = {k: v for k, v in default.items() if k not in IDENTITY | {'data_mixture'}}
            reference['model'] = {k: v for k, v in default['model'].items()
                                  if k not in ('vocab_size', 'context_length', 'n_layers')}
            reference['training'] = {k: v for k, v in default['training'].items()
                                     if k not in ('seed', 'lm_batch', 'lm_microbatch', 'validation_batch')}
            self.assertEqual(trimmed, reference, arm)

    def test_plan_rejects_drift_before_touching_data(self):
        config = load('train-zh200m-24k-c512-s1')
        self.assertIn('Unknown zh-200m arm', plan_error({**config, 'arm': '24k-c256'}))
        wrong_layers = {**config, 'model': {**config['model'], 'n_layers': 15}}
        self.assertIn('zh-200m shapes', plan_error(wrong_layers))
        wrong_batch = {**config, 'training': {**config['training'], 'lm_batch': 512}}
        self.assertIn('tokens per update', plan_error(wrong_batch))
        wrong_crop = {**config, 'training': {**config['training'], 'prefix_crop_probability': 0.6}}
        self.assertIn('contract mismatch', plan_error(wrong_crop))
        foreign_stage = {**config, 'stage': 'zh-v1.3'}
        self.assertIn('explicit random initialization', plan_error(foreign_stage))

    def test_collator_accepts_the_arm_context(self):
        import torch
        from vimeml.posttraining.streams import TaskCollator
        sample = [{"input_ids": list(range(500)), "labels": list(range(500))}]
        self.assertEqual(TaskCollator(512)(sample)["input_ids"].shape, (1, 504))
        with self.assertRaises(ValueError):
            TaskCollator()(sample)  # the pre-200m default still caps at 128


if __name__ == '__main__':
    unittest.main()
