"""Decision rules and test-split gating of the zh v2 confirmation (synthetic rows only)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks import pinyin_gen_zh as pg  # noqa: E402
from vimeml.benchmarks.zh_v2_confirm import (check_identities, decide, preflight_check,  # noqa: E402
                                             production_rows)


def flip(diff, p):
    return {"diff": diff, "p": p}


def standard(net=0, p=1.0):
    return {"top1": {"net": net, "exact_mcnemar_p": p}}


def row(i, hit, kind="full"):
    return {"id": f"s{i // 3}", "kind": kind, "input": f"k{i}", "text": "对",
            "gen": ["对", "错"] if hit else ["错", "对"], "genScores": [0.0, -1.0], "prior": None, "rows": 1}


class Decision(unittest.TestCase):
    def test_20m_passes(self):
        self.assertTrue(decide("v2-20m", flip(2.0, 0.001), flip(-0.5, 0.4), standard(-3, 0.3))["passed"])

    def test_h1_needs_a_significant_gain(self):
        self.assertFalse(decide("v2-200m", flip(1.0, 0.2), flip(1.0, 0.2))["passed"])
        self.assertFalse(decide("v2-200m", flip(-1.0, 0.001), flip(1.0, 0.2))["passed"])
        self.assertTrue(decide("v2-200m", flip(1.0, 0.01), flip(-3.0, 0.001))["passed"])  # no guards for 200M

    def test_20m_guards(self):
        self.assertFalse(decide("v2-20m", flip(2.0, 0.001), flip(-2.0, 0.01), standard())["passed"])
        self.assertFalse(decide("v2-20m", flip(2.0, 0.001), flip(0, 1), standard(-20, 0.01))["passed"])


class Preflight(unittest.TestCase):
    def test_flips_and_movement(self):
        judged = [row(i, i % 2 == 0) for i in range(1000)]
        self.assertTrue(preflight_check(judged, judged)["passed"])
        moved = [row(i, i % 2 == 0 or i < 6) for i in range(1000)]  # 3 flips, +0.3 points
        check = preflight_check(judged, moved)
        self.assertEqual(check["flips"], 3)
        self.assertFalse(check["passed"])
        with self.assertRaisesRegex(ValueError, "differ"):
            preflight_check(judged, judged[:-1])


class Production(unittest.TestCase):
    def test_rows_follow_the_records(self):
        record = {"id": "a", "text": "你好", "full_input": "nihao", "words": ["你好"],
                  "abbreviations": [{"input": "nh", "abbreviated_word_indices": [0]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prod.jsonl"
            pg.write_jsonl(path, [{"id": "a", "input": "nihao", "v1-prod": ["你好", "拟好"]},
                                  {"id": "a", "input": "nh", "v1-prod": ["内耗"]}])
            rows = production_rows(path, [record])
            self.assertEqual([r["kind"] for r in rows], ["full", "initials"])
            self.assertEqual([pg.hit(r, 0) for r in rows], [True, False])
            pg.write_jsonl(path, [{"id": "a", "input": "nihao", "v1-prod": ["你好"]}])
            with self.assertRaisesRegex(ValueError, "miss"):
                production_rows(path, [record])


class Gating(unittest.TestCase):
    def test_unregistered_model_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "not registered"):
            check_identities(ROOT, "v2-20m")

    def test_test_split_needs_a_plan_and_contexts(self):
        frozen = ROOT / "outputs/benchmarks/ime-pinyin-zh-v1/frozen-r1"
        with self.assertRaisesRegex(ValueError, "preregistered"):
            pg.load_set(frozen, split="test")
        with self.assertRaisesRegex(ValueError, "preregistered"):
            pg.load_set(frozen, split="test", contexts="x.jsonl", confirmation={"path": "p"})
        with self.assertRaisesRegex(ValueError, "split"):
            pg.load_set(frozen, split="train")


if __name__ == "__main__":
    unittest.main()
