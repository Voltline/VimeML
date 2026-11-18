import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vimeml.deployment import zh_search as zs, zh_v1_batch2 as z2, zh_v1_batch3 as z3  # noqa: E402
from vimeml.training.model_zh13 import GPTZh13Config, TinyGPTZh13  # noqa: E402

from test_deployment_zh_v1_batch import ToySP, ToyTable  # noqa: E402


def model():
    torch.manual_seed(1118)
    return TinyGPTZh13(GPTZh13Config()).float().eval()


class GraphV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = model()
        cls.reference = z3.torch_reference(cls.model)

    def test_shared_prefix_steps_match_full_forward(self):
        runner = z3.TorchRunner(self.model)
        for function in ("logits", "scored"):
            for case in z3.CASES:
                z3.check_step(self.reference, runner, function, **case)

    def test_score_matches_full_forward_with_and_without_vocab_chunks(self):
        for chunks in (1, 4):
            stats = z3.check_score(self.reference, z3.TorchRunner(self.model, vocab_chunks=chunks))
            self.assertEqual(stats["rows_outside_tolerance"], 0, stats)

    def test_traced_functions_keep_flexible_shapes(self):
        graphs = {f: z3.trace(self.model, f) for f in z3.FUNCTIONS}
        runner = z3.TorchRunner(graphs=graphs)
        z3.check_step(self.reference, runner, "scored", batch=16, prefix=40)
        self.assertEqual(z3.check_score(self.reference, runner)["rows_outside_tolerance"], 0)
        z3.trace(self.model, "score", vocab_chunks=4)

    def test_inputs_are_not_modified(self):
        runner = z3.TorchRunner(self.model)
        _, shared = z3.prompt_call(runner, "scored", [2, 100, 200, 300], [5])
        before = {k: v.copy() for k, v in shared.items()}
        k, v, position = z3.pack([(np.ones((9, 6, 2, 64), np.float32),) * 2], 3)
        k0, v0 = k.copy(), v.copy()
        runner("scored", {"input_ids": np.array([[9]], np.int32), "position": position, "key_cache": k,
                          "value_cache": v, "allowed": np.array([5], np.int32), **shared})
        for name, value in before.items():
            np.testing.assert_array_equal(shared[name], value)
        np.testing.assert_array_equal(k, k0)
        np.testing.assert_array_equal(v, v0)

    def test_search_equals_v2_backend(self):
        sp, table = ToySP(), ToyTable()
        index = zs.Index(sp, table)
        for scored in (False, True):
            v2 = z2.TorchBatchBackendV2(self.model, scored=scored)
            v3 = z3.BatchBackendV3(z3.TorchRunner(self.model), scored=scored)
            for keys in ("zhongwen", "zw", "zhong"):
                a, _ = zs.search(v2, sp, index, keys, beam=4, topk=5)
                b, _ = zs.search(v3, sp, index, keys, beam=4, topk=5)
                self.assertTrue(a)
                self.assertEqual([x[0] for x in a], [x[0] for x in b])
                for x, y in zip(a, b):
                    self.assertAlmostEqual(x[1], y[1], places=3)


if __name__ == "__main__":
    unittest.main()
