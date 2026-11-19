"""arch2 batch precheck support in joint_zh13: budget_fraction truncation and lm_microbatch pieces."""
import copy
import sys
import tomllib
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.joint_zh13 import split_rows, truncate, update_joint  # noqa: E402
from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2  # noqa: E402


def batch(rows, width, seed):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(2, width + 1, (rows,), generator=g)
    ids = torch.randint(1, 64, (rows, width), generator=g)
    labels = ids.clone()
    for r, n in enumerate(lengths.tolist()):
        ids[r, n:] = 0
        labels[r, n:] = -100
    return {"input_ids": ids, "labels": labels, "lengths": lengths}


class Truncate(unittest.TestCase):
    def test_cut_inside_an_epoch(self):
        s = {"budget_fraction": 0.1, "lm_batch": 10, "max_steps": 300,
             "updates_per_epoch": [100, 100, 100], "windows_per_epoch": [995, 995, 995]}
        truncate(s)
        self.assertEqual((s["max_steps"], s["epochs"], s["updates_per_epoch"], s["windows_per_epoch"]),
                         (30, 1, [30], [300]))
        s = {"budget_fraction": 0.5, "lm_batch": 10, "max_steps": 300,
             "updates_per_epoch": [100, 100, 100], "windows_per_epoch": [995, 995, 995]}
        truncate(s)
        self.assertEqual((s["updates_per_epoch"], s["windows_per_epoch"]), ([100, 50], [995, 500]))

    def test_fraction_must_be_inside(self):
        for value in (0.0, 1.0, 1, "0.1"):
            with self.assertRaises(ValueError):
                truncate({"budget_fraction": value, "max_steps": 10, "lm_batch": 1,
                          "updates_per_epoch": [10], "windows_per_epoch": [10]})

    def test_precheck_configs(self):
        base = tomllib.loads((ROOT / "configs/train-zh200m-12k-pinyin-pt.toml").read_text("utf-8"))
        for k in (1, 4, 8):
            c = tomllib.loads((ROOT / f"configs/train-zh200m-batch-precheck-{k}x.toml").read_text("utf-8"))
            t, b = c["training"], base["training"]
            self.assertEqual(t["lm_batch"], b["lm_batch"] * k)
            self.assertEqual(t["bucket_multiplier"] * k, b["bucket_multiplier"])
            self.assertEqual(t["warmup_steps"] * k, b["warmup_steps"])
            self.assertAlmostEqual(t["learning_rate"], b["learning_rate"] * k ** 0.5, delta=5e-6)  # rounded to 2 digits as in the plan
            self.assertAlmostEqual(t["min_learning_rate"] * 10, t["learning_rate"])
            self.assertEqual(c["pinyin"]["mix"], {q: v * k for q, v in base["pinyin"]["mix"].items()})
            self.assertEqual(t["budget_fraction"], 0.1)
            self.assertEqual((c["model"], c["dataset_sha256"], t["seed"]), (base["model"], base["dataset_sha256"], b["seed"]))


class Pieces(unittest.TestCase):
    def test_split_rows(self):
        b = batch(10, 6, 1)
        parts = split_rows(b, 4)
        self.assertEqual([len(p["lengths"]) for p in parts], [4, 4, 2])
        self.assertTrue(torch.equal(torch.cat([p["input_ids"] for p in parts]), b["input_ids"]))
        self.assertIs(split_rows(b, 10)[0], b)

    def test_pieces_keep_the_gradient(self):
        config = GPTV2Config(vocab_size=64, context_length=16, d_model=32, n_heads=2, n_layers=2, d_ff=64)
        lm, extra = batch(16, 12, 2), {"full": batch(6, 10, 3), "error": batch(3, 8, 4)}
        weights = {"lm": 16 / 25, "full": 6 / 25, "error": 3 / 25}
        grads = []
        for micro in (16, 4):
            torch.manual_seed(0)
            model = TinyGPTV2(config)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            s = {"lm_batch": 16, "lm_microbatch": micro, "grad_clip": 1e9}
            out = update_joint(model, optimizer, copy.deepcopy(lm), copy.deepcopy(extra), weights, s,
                               torch.device("cpu"), "fp32")
            grads.append(([p.grad.clone() for p in model.parameters()], out))
        for a, b in zip(grads[0][0], grads[1][0]):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        self.assertAlmostEqual(grads[0][1]["loss"], grads[1][1]["loss"], places=5)
        for kind in extra:
            self.assertAlmostEqual(grads[0][1]["pinyin"][kind]["loss"], grads[1][1]["pinyin"][kind]["loss"], places=5)


if __name__ == "__main__":
    unittest.main()
