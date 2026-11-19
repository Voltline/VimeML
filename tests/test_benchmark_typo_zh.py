import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vimeml.benchmarks import typo_zh as B  # noqa: E402
from vimeml.training.data import file_sha  # noqa: E402

SYLLABLES = {"ni", "hao", "ma", "a", "n", "zhong", "guo", "zhon", "gu", "o"} - {"n", "zhon"}


def item(id_, variant, text, base, corrections):
    return {"id": id_, "variant": variant, "input": "x", "edits": [], "text": text, "source": "t",
            "base": base, "corrections": [{"spelling": f"s{k}", "text": t} for k, t in enumerate(corrections)]}


class EditsTest(unittest.TestCase):
    def test_inverse_edits(self):
        out = B.single_edits("nihap", SYLLABLES)                 # o -> p substitute (neighbours)
        self.assertIn("nihao", out)
        self.assertNotIn("nihap", out)
        self.assertIn("nihao", B.single_edits("nihaoo", SYLLABLES))       # double press: delete
        self.assertIn("nihao", B.single_edits("nihoa", SYLLABLES))        # swap
        self.assertIn("zhongguo", B.single_edits("zongguo", SYLLABLES))   # dropped h: insert any letter
        self.assertNotIn("nihao", B.single_edits("nihax", SYLLABLES))     # x is not next to o
        self.assertTrue(all(len(s) > 0 for s in B.single_edits("a", SYLLABLES)))


class RulesTest(unittest.TestCase):
    def setUp(self):
        self.items = [item("1", "clean", "你好", ["你好", "拟好"], ["你号"]),
                      item("1", "typo1", "你好", ["你哈", "拟哈"], ["你好", "你号"]),
                      item("1", "typo2", "你好", ["你哈", "你好"], []),
                      item("2", "clean", "中国", ["中国", "种果"], ["中过"]),
                      item("2", "typo1", "中国", ["宗国", "棕果"], ["中国"]),
                      item("2", "typo2", "中国", ["宗国", "棕果"], ["中过"])]
        self.rows = {("1", "clean"): {"lm": [-10.0, -12.0]}, ("1", "typo1"): {"lm": [-20.0, -11.0, -13.0]},
                     ("1", "typo2"): {"lm": None}, ("2", "clean"): {"lm": [-8.0, -4.0]},
                     ("2", "typo1"): {"lm": [-9.0, -7.0]}, ("2", "typo2"): {"lm": [-9.0, -15.0]}}

    def test_decide(self):
        self.assertEqual(B.decide(self.items[1], self.rows[("1", "typo1")], "lm", 3), ("你好", ["你哈", "你好"]))
        self.assertEqual(B.decide(self.items[4], self.rows[("2", "typo1")], "lm", 3), (None, ["宗国", "棕果"]))
        self.assertEqual(B.decide(self.items[2], self.rows[("1", "typo2")], "lm", 3), (None, ["你哈", "你好"]))

    def test_metrics_and_calibration(self):
        m = B.metrics(self.items, self.rows, "lm", 3)
        self.assertEqual(m["clean"]["shown"], 50.0)                    # 2/clean: delta 4
        self.assertEqual(m["typo1"]["at2"], 50.0)
        self.assertEqual(m["typo2"]["at2"], 50.0)                      # base[1] is the reference
        self.assertEqual(m["typo"]["at2"], 50.0)
        self.assertEqual(m["typo"]["shown_correct"], 25.0)
        # false display 0 needs margin > 4; at 4.0001 the 9-point and 2-point typo deltas: only 9 passes
        margin = B.calibrate(self.items, self.rows, "lm", limit=0.0)
        self.assertGreater(margin, 4)
        self.assertEqual(B.metrics(self.items, self.rows, "lm", margin)["clean"]["shown"], 0.0)
        self.assertEqual(B.calibrate(self.items, self.rows, "lm", limit=0.5), 2.0)   # largest margin of the best @2

    def test_calibration_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "set").mkdir()
            B.write_jsonl(tmp / "set" / "development.jsonl", self.items)
            (tmp / "set" / "manifest.json").write_text('{"version": "ime-typo-zh-v1-r1"}', "utf-8")
            (tmp / "rows").mkdir()
            B.write_jsonl(tmp / "rows" / "rows.jsonl", [{"id": k[0], "variant": k[1], **v} for k, v in self.rows.items()])
            meta = {"checkpoint_sha256": "abc", "step": 7, "rules": ["lm"], "limit": 0,
                    "set_manifest_sha256": file_sha(tmp / "set" / "manifest.json")}
            (tmp / "rows" / "meta.json").write_text(json.dumps(meta), "utf-8")
            c = B.calibration(tmp / "set", tmp / "rows", 1.6, "self")
            self.assertEqual(c["format"], B.CALIBRATION_FORMAT)
            self.assertEqual((c["model"]["checkpoint_sha256"], c["search"], c["rescoring"]),
                             ("abc", {"beam": 16, "topk": 10}, {"lambda": 1.6, "prior": "self"}))
            self.assertEqual(c["correction"]["margin"], 9.0)          # clean deltas 4 and -2: only 9 keeps 0 % shown
            self.assertEqual(c["correction"]["at_margin"], {"typo_at2": 50.0, "clean_false_display": 0.0})
            self.assertEqual(c["correction"]["at_fixed_margin"]["clean_false_display"], 50.0)
            json.dumps(c, allow_nan=False)
            meta["limit"] = 10
            (tmp / "rows" / "meta.json").write_text(json.dumps(meta), "utf-8")
            with self.assertRaises(ValueError):
                B.calibration(tmp / "set", tmp / "rows", 1.6, "self")

    def test_sign_flip(self):
        a = B.hits_by_item(self.items, self.rows, "lm", 100)
        b = B.hits_by_item(self.items, self.rows, "lm", 2)
        s = B.sign_flip(self.items, a, b)
        self.assertEqual((s["gained"], s["lost"]), (2, 0))
        self.assertEqual(s["diff"], 50.0)
        self.assertGreater(s["p"], 0.2)                                # two sentences cannot reach significance


class MergeTest(unittest.TestCase):
    def test_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frozen = tmp / "frozen"
            frozen.mkdir()
            records = [{"id": str(n), "split": "development", "text": "t"} for n in range(3)]
            B.write_jsonl(frozen / "development.jsonl", records)
            B.write_jsonl(frozen / "development-contexts.jsonl", [{"id": r["id"], "infield": ""} for r in records])
            (frozen / "manifest.json").write_text("{}", "utf-8")
            common = {"version": B.VERSION, "limit": 0, "typo_config": {}, "pinyin_set_manifest_sha256":
                      file_sha(frozen / "manifest.json"), "readings_sha256": "r", "code_sha256": {}}
            shards = []
            for k, ids in enumerate((["0", "2"], ["1"])):
                d = tmp / f"shard-{k}"
                d.mkdir()
                B.write_jsonl(d / "development.jsonl", [item(i, v, "t", ["t"], []) for i in ids
                                                        for v in reversed(B.VARIANTS)])
                (d / "manifest.json").write_text(json.dumps({**common, "shard": [k, 2], "seconds": 1.0,
                                                             "pool": {"beam": 16, "searches": 3}}), "utf-8")
                shards.append(d)
            m = B.merge(shards, tmp / "set", frozen)
            got = [(i["id"], i["variant"]) for i in B.read_jsonl(tmp / "set" / "development.jsonl")]
            self.assertEqual(got, [(str(n), v) for n in range(3) for v in B.VARIANTS])
            self.assertEqual(m["pool"], {"beam": 16, "searches": 6})
            with self.assertRaises(ValueError):
                B.merge(shards[:1], tmp / "set2", frozen)


if __name__ == "__main__":
    unittest.main()
