"""Desktop checkpoint headers map to their model classes (zh-200m, zh-800m); anything else is not a desktop model."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.model_zh200m import GPTZh200mConfig, TinyGPTZh200m  # noqa: E402
from vimeml.training.model_zh800m import GPTZh800mConfig, TinyGPTZh800m  # noqa: E402
from vimeml.training.zh_desktop import desktop_family  # noqa: E402


class DesktopFamilyTest(unittest.TestCase):
    def test_headers(self):
        self.assertEqual(desktop_family({"format": "vimeml_tiny_gpt_zh200m", "architecture": "tiny_gpt_zh200m"}),
                         (GPTZh200mConfig, TinyGPTZh200m))
        self.assertEqual(desktop_family({"format": "vimeml_tiny_gpt_zh800m", "architecture": "tiny_gpt_zh800m"}),
                         (GPTZh800mConfig, TinyGPTZh800m))
        self.assertIsNone(desktop_family({"format": "vimeml_tiny_gpt_zh800m", "architecture": "tiny_gpt_zh200m"}))
        self.assertIsNone(desktop_family({"format": "vimeml_tiny_gpt_zh13", "architecture": "tiny_gpt_zh13"}))
        self.assertIsNone(desktop_family({}))

    def test_800m_config_is_the_registered_shape(self):
        self.assertEqual(GPTZh800mConfig().n_layers, 27)
        with self.assertRaises(ValueError):
            GPTZh800mConfig(n_layers=26)


if __name__ == "__main__":
    unittest.main()
