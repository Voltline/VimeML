"""Unit tests for joint_ja_kana: extra data streams, ce_only rows, and online distillation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from vimeml.training.distributed import Context
from vimeml.training.joint_ja_kana import (
    Schedule,
    distilled_loss,
    update,
)
from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2


def write_store(root: Path, kind: str, split: str, rows):
    """Write a tiny vimeml_ja_kana_v1 kind store from [(token_ids, first), ...]."""
    d = root / kind
    d.mkdir(parents=True, exist_ok=True)
    tokens, offsets, firsts = [], [0], []
    for ids, first in rows:
        tokens.extend(ids)
        offsets.append(len(tokens))
        firsts.append(first)
    np.array(tokens, dtype="<u2").tofile(d / f"{split}.tokens.bin")
    np.array(offsets, dtype="<u8").tofile(d / f"{split}.offsets.bin")
    np.array(firsts, dtype="u1").tofile(d / f"{split}.first.bin")


class TestJointJaKana(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.primary = self.root / "primary"
        self.extra = self.root / "extra"
        plain_rows = [([2, 10 + i, 11 + i, 12 + i, 13 + i, 14 + i, 15 + i, 16 + i, 17 + i, 18 + i, 3], 1)
                      for i in range(12)]
        span_rows = [([2, 4, 10 + i, 5, 11 + i, 6, 12 + i, 13 + i, 3], 6) for i in range(12)]
        extra_span_rows = [([2, 4, 20 + i, 5, 21 + i, 6, 22 + i, 23 + i, 3], 6) for i in range(8)]
        write_store(self.primary, "plain", "train", plain_rows)
        write_store(self.primary, "span", "train", span_rows)
        write_store(self.extra, "span", "train", extra_span_rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_schedule_without_extra_has_no_hard_and_is_rank_partitioned(self):
        full = Schedule(self.primary, {"plain": 4, "span": 4}, seed=42, crop_probability=0.0, rank=0, world=1)
        r0 = Schedule(self.primary, {"plain": 4, "span": 4}, seed=42, crop_probability=0.0, rank=0, world=2)
        r1 = Schedule(self.primary, {"plain": 4, "span": 4}, seed=42, crop_probability=0.0, rank=1, world=2)
        b_full = full[0]
        self.assertNotIn("hard", b_full["plain"])
        self.assertEqual(b_full["plain"]["input_ids"].shape[0], 4)
        self.assertEqual(r0[0]["plain"]["input_ids"].shape[0], 2)
        self.assertEqual(r1[0]["plain"]["input_ids"].shape[0], 2)
        self.assertTrue(torch.equal(b_full["plain"]["input_ids"][0::2], r0[0]["plain"]["input_ids"]))
        self.assertTrue(torch.equal(b_full["plain"]["input_ids"][1::2], r1[0]["plain"]["input_ids"]))

    def test_schedule_extra_and_ce_only_flags(self):
        sched = Schedule(
            self.primary,
            {"plain": 2, "span": 2},
            seed=42,
            crop_probability=0.0,
            rank=0,
            world=1,
            extra=[(self.extra, {"span": 2})],
            ce_only_sources=(1,),
        )
        batch = sched[0]
        self.assertEqual(batch["span"]["input_ids"].shape[0], 4)
        self.assertEqual(batch["plain"]["hard"].tolist(), [False, False])
        self.assertEqual(batch["span"]["hard"].tolist(), [False, False, True, True])
        passes = sched.passes(10)
        self.assertIn("span", passes)
        self.assertIn("span@1", passes)

    def test_distilled_update_with_mixed_hard_and_soft_rows(self):
        torch.manual_seed(0)
        cfg = GPTV2Config(vocab_size=64, context_length=32, d_model=32, n_heads=2, n_layers=2, d_ff=64, dropout=0.0)
        student = TinyGPTV2(cfg)
        teacher = TinyGPTV2(cfg).eval().requires_grad_(False)
        opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
        sched = Schedule(
            self.primary,
            {"plain": 2, "span": 2},
            seed=42,
            crop_probability=0.0,
            extra=[(self.extra, {"span": 2})],
            ce_only_sources=(1,),
        )
        plan = {
            "settings": {"normalization": "per_kind", "grad_clip": 1.0},
            "mix": {"plain": 2, "span": 4},
            "batch": 6,
            "distillation": {"alpha": 0.5, "temperature": 1.0, "ce_only_kinds": []},
        }
        ctx = Context()
        res = update(student, student, opt, sched[0], plan, torch.device("cpu"), "fp32", ctx, teacher=teacher)
        self.assertIn("kl", res["kinds"]["plain"])
        self.assertIn("kl", res["kinds"]["span"])
        self.assertGreaterEqual(res["kinds"]["span"]["kl"], 0.0)

    def test_distilled_loss_alpha_zero_equals_ce(self):
        torch.manual_seed(1)
        logits = torch.randn(2, 5, 16)
        t_logits = torch.randn(2, 5, 16)
        labels = torch.tensor([[-100, 1, 2, 3, -100], [4, 5, -100, 6, 7]])
        mixed, ce, kl = distilled_loss(logits, t_logits, labels, alpha=0.0, temperature=1.0)
        self.assertAlmostEqual(float(mixed), float(ce), places=5)
        self.assertGreater(float(kl), 0.0)


if __name__ == "__main__":
    unittest.main()
