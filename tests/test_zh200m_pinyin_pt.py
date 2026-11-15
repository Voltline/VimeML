"""The 12K joint-pinyin 200M: arm 3's config with only the model swapped, and its plan guards."""
from pathlib import Path
import sys
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.training.joint_zh13 import FAMILIES, build_plan
from vimeml.training.model_zh200m import SHAPES, GPTZh200mConfig, TinyGPTZh200m

IDENTITY = {'arm', 'architecture', 'output_dir', 'log_dir', 'model', 'tracking'}


def load(name):
    return tomllib.loads((ROOT / f'configs/{name}.toml').read_text('utf-8'))


class Zh200mPinyinTests(unittest.TestCase):
    def test_config_is_arm3_with_the_200m_model(self):
        arm3, large = load('train-zh-pinyin-pt'), load('train-zh200m-12k-pinyin-pt')
        self.assertEqual({k: v for k, v in large.items() if k not in IDENTITY},
                         {k: v for k, v in arm3.items() if k not in IDENTITY})
        self.assertEqual(large['architecture'], 'tiny_gpt_zh200m')
        self.assertIn(large['architecture'], FAMILIES)
        model = large['model']
        self.assertEqual((model['vocab_size'], model['context_length'], model['n_layers']), (12288, 128, 15))
        self.assertEqual({k: v for k, v in model.items() if k not in ('d_model', 'n_heads', 'n_layers', 'd_ff')},
                         {k: v for k, v in arm3['model'].items() if k not in ('d_model', 'n_heads', 'n_layers', 'd_ff')})
        GPTZh200mConfig(**model)

    def test_shape_budget(self):
        self.assertEqual(SHAPES[(12288, 128)], (15, 193_625_088))
        model = TinyGPTZh200m(GPTZh200mConfig(vocab_size=12288, context_length=128, n_layers=15))
        self.assertEqual(model.parameter_count(), 193_625_088)

    def test_plan_rejects_a_shape_off_the_plain_arm(self):
        config = load('train-zh200m-12k-pinyin-pt')
        for model, message in (({**config['model'], 'vocab_size': 24576, 'n_layers': 14}, 'vocabulary and context'),
                               ({**config['model'], 'n_layers': 14}, 'zh-200m shapes')):
            with self.assertRaises(ValueError) as caught:
                build_plan({**config, 'model': model}, verify_files=False)
            self.assertIn(message, str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            build_plan({**config, 'architecture': 'tiny_gpt_v2'}, verify_files=False)
        self.assertIn('architecture', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
