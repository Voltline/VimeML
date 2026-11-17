"""Data-parallel updates of the zh trainers (vimeml.training.distributed).

No process group is started: each rank's share runs in turn on one model with gradients
accumulating, which is what DDP's all-reduce sums. Divided by the world size (DDP averages),
that must equal the one-device gradient of the same global batch.
"""
from contextlib import nullcontext
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vimeml.training import joint_zh13, joint_zh200m
from vimeml.training.distributed import Context
from vimeml.training.model_v2 import token_loss

VOCAB = 50


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.embed = nn.Embedding(VOCAB, 16)
        self.head = nn.Linear(16, VOCAB)

    def forward(self, ids, labels):
        logits = self.head(torch.tanh(self.embed(ids)))
        valid = labels != -100
        return {"loss_sum": nn.functional.cross_entropy(logits[valid], labels[valid], reduction="sum")}


class TinyWeighted(Tiny):
    """Tiny with the token_weights loss of model_v2 and a context length (joint_zh200m packing)."""
    class config:
        context_length = 24

    def forward(self, ids, labels, token_weights=None):
        hidden = torch.tanh(self.embed(ids))
        return token_loss(self.head, hidden, ids, labels, token_weights)


class LocalRank(Context):
    """A rank without communication: sums are its own values, no DDP wrapper."""
    def no_sync(self, module, last):
        return nullcontext()

    def sum(self, values, device):
        return list(values)


class Keep:
    """Optimizer stand-in that leaves gradients in place for inspection."""
    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


def batch(rows, width, seed):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(3, width + 1, (rows,), generator=g)
    ids = torch.randint(4, VOCAB, (rows, width), generator=g)
    labels = torch.randint(4, VOCAB, (rows, width), generator=g)
    for r, n in enumerate(lengths.tolist()):
        ids[r, n:] = 0
        labels[r, n:] = -100
    return {"input_ids": ids, "labels": labels, "lengths": lengths}


SETTINGS = {"lm_microbatch": 4, "grad_clip": 1e9}
CPU = torch.device("cpu")


def gradients(model):
    return [p.grad.detach().clone() for p in model.parameters()]


class DataParallelUpdateTest(unittest.TestCase):
    def check(self, update, world, model=Tiny):
        one = model()
        update(one, Context())
        many = model()
        for rank in range(world):
            update(many, LocalRank(world, rank))
        for a, b in zip(gradients(one), gradients(many)):
            torch.testing.assert_close(b / world, a, atol=1e-6, rtol=1e-5)

    def test_lm_update_matches_one_device(self):
        lm = batch(11, 24, 1)
        for module in (joint_zh13, joint_zh200m):
            for world in (2, 3, 4):
                self.check(lambda m, ctx: module.update_lm(m, Keep(), [lm], SETTINGS, CPU, "fp32", ctx, m), world)

    def test_joint_update_matches_one_device(self):
        lm = batch(16, 24, 2)
        extra = {"full": batch(5, 20, 3), "mixed": batch(6, 20, 4), "initials": batch(3, 12, 5),
                 "error": batch(2, 18, 6)}
        weights = {"lm": 16 / 32, "full": 5 / 32, "mixed": 6 / 32, "initials": 3 / 32, "error": 2 / 32}
        for world in (2, 4):     # world 4 leaves some ranks without error rows
            self.check(lambda m, ctx: joint_zh13.update_joint(m, Keep(), lm, extra, weights, SETTINGS, CPU,
                                                              "fp32", ctx, m), world)

    def test_packed_joint_update_matches_one_device_and_the_unpacked_update(self):
        lm = batch(16, 24, 2)
        extra = {"full": batch(5, 20, 3), "mixed": batch(6, 20, 4), "initials": batch(3, 12, 5),
                 "error": batch(2, 18, 6)}
        weights = {"lm": 16 / 32, "full": 5 / 32, "mixed": 6 / 32, "initials": 3 / 32, "error": 2 / 32}
        for microbatch in (1, 4, 64):    # 1: a row per microbatch; 64: one microbatch per rank
            s = {**SETTINGS, "lm_microbatch": microbatch}

            def packed(m, ctx):
                return joint_zh200m.update_joint(m, Keep(), {"lm": lm, **extra}, weights, s, CPU, "fp32", ctx, m)
            for world in (2, 4):    # ranks get different numbers of microbatches; world 4 leaves some without error rows
                self.check(packed, world, TinyWeighted)
            reference = TinyWeighted()
            want = joint_zh13.update_joint(reference, Keep(), lm, extra, weights, SETTINGS, CPU, "fp32")
            model = TinyWeighted()
            got = packed(model, Context())
            for a, b in zip(gradients(reference), gradients(model)):
                torch.testing.assert_close(b, a, atol=1e-6, rtol=1e-5)
            self.assertAlmostEqual(got["loss"], want["loss"], places=5)
            for kind in extra:
                self.assertAlmostEqual(got["pinyin"][kind]["loss"], want["pinyin"][kind]["loss"], places=5)

    def test_reported_totals_do_not_depend_on_the_world(self):
        lm = batch(9, 16, 8)
        one = joint_zh13.update_lm(Tiny(), Keep(), [lm], SETTINGS, CPU, "fp32")
        share = joint_zh13.update_lm(Tiny(), Keep(), [lm], SETTINGS, CPU, "fp32", LocalRank(3, 1))
        for key in ("supervised_tokens", "input_tokens", "examples"):
            self.assertEqual(one[key], share[key])

    def test_shares_cover_the_batch_in_order(self):
        rows = batch(7, 10, 9)
        parts = [Context(3, r).share(rows) for r in range(3)]
        self.assertEqual(torch.cat([p["lengths"] for p in parts]).tolist(), rows["lengths"].tolist())
        for p in parts:
            self.assertLessEqual(int(p["lengths"].max()), p["input_ids"].shape[1])
        self.assertIsNone(Context(8, 0).share(batch(3, 10, 10)))   # rows 0..0 of 3


if __name__ == "__main__":
    unittest.main()
