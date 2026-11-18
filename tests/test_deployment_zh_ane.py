import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vimeml.deployment import zh_ane as za, zh_v1_batch3 as z3  # noqa: E402
from vimeml.training.model_zh13 import GPTZh13Config, TinyGPTZh13  # noqa: E402


def model():
    torch.manual_seed(1118)
    return TinyGPTZh13(GPTZh13Config()).float().eval()


class ANEProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = model()

    @torch.inference_mode()
    def test_both_layouts_equal_v3_in_fp32(self):
        v3_score, v3_step = z3.graph(self.model, "score"), z3.graph(self.model, "scored")
        for name, (kind, shape) in za.functions().items():
            ex = za.example(kind, shape, "bt")
            want = v3_score(*ex) if kind == "score" else v3_step(*ex)
            for layout in za.LAYOUTS:
                got = za.graph(self.model, kind, layout)(*za.example(kind, shape, layout))
                if kind == "score":
                    torch.testing.assert_close(got, want, atol=2e-4, rtol=2e-4, msg=f"{name} {layout}")
                    continue
                lse, picked, k, v = got
                torch.testing.assert_close(lse, want[0], atol=2e-4, rtol=2e-4)
                torch.testing.assert_close(picked, want[1], atol=2e-4, rtol=2e-4)
                if layout == "ane":
                    k, v = k, v
                    torch.testing.assert_close(k, za.key_to_ane(want[2]), atol=2e-4, rtol=2e-4)
                    torch.testing.assert_close(v, za.value_to_ane(want[3]), atol=2e-4, rtol=2e-4)
                else:
                    torch.testing.assert_close(k, want[2], atol=2e-4, rtol=2e-4)
                    torch.testing.assert_close(v, want[3], atol=2e-4, rtol=2e-4)

    def test_traced_functions_match_eager(self):
        for layout in za.LAYOUTS:
            for name in (za.score_name(*za.SCORE_SHAPES[0]), za.step_name(*za.STEP_SHAPES[0])):
                kind, shape = za.functions()[name]
                ex = za.example(kind, shape, layout, seed=3)
                traced = za.trace(self.model, name, layout)
                with torch.inference_mode():
                    a, b = traced(*ex), za.graph(self.model, kind, layout)(*ex)
                for x, y in zip(a if isinstance(a, tuple) else (a,), b if isinstance(b, tuple) else (b,)):
                    torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)

    def test_numpy_layout_conversion_matches_torch(self):
        k = torch.randn(9, 2, 6, 5, 64)
        torch.testing.assert_close(torch.from_numpy(za.key_to_ane(k.numpy())), za.key_to_ane(k))
        torch.testing.assert_close(torch.from_numpy(za.value_to_ane(k.numpy())), za.value_to_ane(k))


if __name__ == "__main__":
    unittest.main()
