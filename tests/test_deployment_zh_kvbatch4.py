import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vimeml.deployment import zh_v1_batch3 as z3, zh_v1_batch4 as z4  # noqa: E402
from vimeml.training.model_zh13 import GPTZh13Config, TinyGPTZh13  # noqa: E402


def model():
    torch.manual_seed(1120)
    return TinyGPTZh13(GPTZh13Config()).float().eval()


class KVBatchV4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = model()
        cls.reference = z3.torch_reference(cls.model)

    def test_all_functions_match_full_forward(self):
        checks = z4.run_checks(self.reference, z4.TorchRunner(self.model))
        self.assertEqual({c["kind"] for c in checks}, {"logits", "scored", "score", "prefix"})
        self.assertEqual([c for c in checks if c["rows_outside_tolerance"]], [])
        self.assertEqual({c.get("prefill") for c in checks if c["kind"] == "score"}, {"scored", "prefix"})

    def test_fp16_kv_within_its_tolerance(self):
        checks = z4.run_checks(self.reference, z4.TorchRunner(self.model, kv="fp16"), variant="fp16kv")
        self.assertEqual([c for c in checks if c["rows_outside_tolerance"]], [])
        self.assertGreater(max(c["max_abs"] for c in checks), z3.TOLERANCE / 100)   # the rounding is really applied

    def test_prefix_extension_equals_one_call(self):
        runner = z4.TorchRunner(self.model)
        ids = [2, 50, 60, 70, 80, 90]
        one = z4.prefix_call(runner, ids)
        two = z4.prefix_call(runner, ids[3:], z4.prefix_call(runner, ids[:3]))
        self.assertEqual(int(two["prefix_length"][0]), len(ids))
        np.testing.assert_allclose(one["prefix_key"], two["prefix_key"], atol=1e-5)
        np.testing.assert_allclose(one["prefix_value"], two["prefix_value"], atol=1e-5)
        _, via_scored = z3.prompt_call(runner, "scored", ids, [5])
        np.testing.assert_allclose(one["prefix_key"], via_scored["prefix_key"], atol=1e-5)

    def test_prefix_inputs_not_modified_and_shapes_flexible(self):
        runner = z4.TorchRunner(graphs={f: z4.trace(self.model, f) for f in z4.FUNCTIONS})
        shared = z4.prefix_call(runner, [2, 11, 12])
        before = {k: v.copy() for k, v in shared.items()}
        z4.prefix_call(runner, [13, 14, 15, 16], shared)
        for name, value in before.items():
            np.testing.assert_array_equal(shared[name], value)
        self.assertEqual(z4.check_prefix(self.reference, runner)["rows_outside_tolerance"], 0)

    def test_interface_marks_fp16_kv(self):
        self.assertIn("float16 (9,1,6,P,64)", z4.interface("score", "fp16")["prefix_key/prefix_value"])
        self.assertIn("float16", z4.interface("scored", "fp16")["new_key/new_value"])
        self.assertIn("float32", z4.interface("scored", "fp32")["key_cache/value_cache"])


if __name__ == "__main__":
    unittest.main()
