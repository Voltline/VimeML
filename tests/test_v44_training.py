"""V4.4 arm table, configs and plan guards, without model/corpus training."""
import copy
from pathlib import Path
import sys
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.training.joint_v44 import ARMS, build_plan

TRAINED = sorted(set(ARMS) - {'control'})
# Keys an arm config may differ in from V4.2, besides its single arm variable.
IDENTITY = {'stage', 'arm', 'output_dir', 'log_dir', 'dataset_sha256'}


def load(name):
    return tomllib.loads((ROOT / f'configs/{name}.toml').read_text('utf-8'))


def changed(arm):
    """Arm variables that differ from V4.2, as (seed, chat, crop) positions."""
    return [i for i, (a, b) in enumerate(zip(ARMS[arm], ARMS['control'])) if a != b]


class V44ArmTests(unittest.TestCase):
    def test_every_arm_changes_exactly_one_variable(self):
        self.assertEqual(ARMS['control'], (1219827974, .05, .30))
        for arm in TRAINED:
            self.assertEqual(len(changed(arm)), 1, arm)
        seeds = [ARMS[a][0] for a in ARMS]
        # Chat sampling uses seed+epoch; replicate streams must never overlap.
        for a in set(seeds):
            for b in set(seeds) - {a}:
                self.assertGreater(abs(a - b), 20)

    def test_configs_are_v42_plus_the_arm(self):
        base = load('train-v42')
        for arm in TRAINED:
            config = load(f'train-v44-{arm}')
            seed, chat, crop = ARMS[arm]
            self.assertEqual(config['arm'], arm)
            self.assertEqual((config['training']['seed'], config['data_mixture']['chat_fraction'],
                              config['training']['prefix_crop_probability']), (seed, chat, crop))
            self.assertEqual(config['data_mixture']['directory'],
                             f'artifacts/training-data/corpus-v44-{arm}-c128')
            self.assertEqual(config['tracking'], {'enabled': True, 'project': 'vimeml',
                                                  'group': 'v44-ablation', 'mode': 'online'})
            self.assertRegex(config['dataset_sha256'], '^[0-9a-f]{64}$')
            for key in set(base) | set(config):
                if key in IDENTITY or key in ('tracking', 'data_mixture', 'training'):
                    continue
                self.assertEqual(config.get(key), base.get(key), (arm, key))
            training = dict(config['training'])
            training.update(seed=base['training']['seed'],
                            prefix_crop_probability=base['training']['prefix_crop_probability'])
            self.assertEqual(training, base['training'], arm)

    def test_plan_rejects_a_config_that_drifts_from_its_arm(self):
        config = load('train-v44-chat10')
        for path, value in ((('training', 'seed'), 1), (('data_mixture', 'chat_fraction'), .05),
                            (('training', 'prefix_crop_probability'), .6)):
            broken = copy.deepcopy(config)
            broken[path[0]][path[1]] = value
            with self.assertRaisesRegex(ValueError, 'registered arm'):
                build_plan(broken, ROOT, verify_files=False)
        broken = copy.deepcopy(config)
        broken['arm'] = 'control'
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            build_plan(broken, ROOT, verify_files=False)


if __name__ == '__main__':
    unittest.main()
