"""Decision rule and blind gating of the vimeml-tiny-zh-v1 confirmation."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.zh13_lm_scores import run  # noqa: E402
from vimeml.benchmarks.zh_v1_confirm import SOURCES, decide, wilson  # noqa: E402


def compare(net=300, p=1e-40, by_source=100, top3_net=50, top3_p=1e-5):
    return {"top1": {"net": net, "exact_mcnemar_p": p}, "top3": {"net": top3_net, "exact_mcnemar_p": top3_p},
            "top1_by_source": {s: {"net": by_source} for s in SOURCES}}


class Decision(unittest.TestCase):
    def test_passes(self):
        self.assertTrue(decide(compare())["passed"])

    def test_needs_significant_gain(self):
        self.assertFalse(decide(compare(net=5, p=0.2))["h1_confirmed"])
        self.assertFalse(decide(compare(net=-300))["h1_confirmed"])

    def test_every_source_must_gain(self):
        case = compare()
        case["top1_by_source"]["kdconv"]["net"] = 0
        self.assertFalse(decide(case)["passed"])

    def test_top3_guard_only_blocks_a_significant_drop(self):
        self.assertFalse(decide(compare(top3_net=-20, top3_p=0.01))["passed"])
        self.assertTrue(decide(compare(top3_net=-1, top3_p=0.9))["passed"])

    def test_wilson_contains_the_rate(self):
        low, high = wilson(936, 1000)
        self.assertLess(low, 0.936)
        self.assertGreater(high, 0.936)


class BlindGate(unittest.TestCase):
    def test_blind_needs_a_plan_before_any_file_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "blind needs"):
                run("missing.pt", "missing.model", "missing", Path(directory) / "out.jsonl", split="blind")

    def test_unknown_split_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "split"):
                run("missing.pt", "missing.model", "missing", Path(directory) / "out.jsonl", split="train")


if __name__ == "__main__":
    unittest.main()
