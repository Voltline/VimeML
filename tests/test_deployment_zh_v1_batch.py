"""zh-v1 batched KV step and the per-position beam search (random weights; no checkpoint needed)."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.deployment import zh_search as zs, zh_v1_batch as zb, zh_v1_batch2 as z2  # noqa: E402
from vimeml.training.model_zh13 import GPTZh13Config, TinyGPTZh13  # noqa: E402


def random_model():
    torch.manual_seed(1115)
    return TinyGPTZh13(GPTZh13Config()).float().eval()


class BatchGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = random_model()

    def test_rows_at_different_positions_match_full_forward(self):
        for case in ({"batch": 6}, {"batch": 1}, {"batch": 3, "slots": zb.CONTEXT}):
            stats = zb.check_graph(self.model, zb.KVBatchGraphZh(self.model), **case)
            self.assertEqual(stats["rows_outside_tolerance"], 0)
            self.assertEqual(stats["argmax_equal"], stats["rows"])

    def test_traced_graph_keeps_flexible_batch_and_slots(self):
        traced = torch.jit.trace(zb.KVBatchGraphZh(self.model), (
            torch.tensor([[7], [9]], dtype=torch.int32), torch.tensor([0, 2], dtype=torch.int32),
            torch.zeros(zb.LAYERS, 2, zb.HEADS, 3, zb.HEAD_DIM), torch.zeros(zb.LAYERS, 2, zb.HEADS, 3, zb.HEAD_DIM)))
        stats = zb.check_graph(self.model, traced, batch=4)
        self.assertEqual(stats["rows_outside_tolerance"], 0)

    def test_padding_row_does_not_change_real_rows(self):
        graph = zb.KVBatchGraphZh(self.model)
        k = torch.randn(zb.LAYERS, 2, zb.HEADS, 5, zb.HEAD_DIM)
        v = torch.randn_like(k)
        with torch.inference_mode():
            alone = graph(torch.tensor([[11]], dtype=torch.int32), torch.tensor([4], dtype=torch.int32), k[:, :1], v[:, :1])
            padded = graph(torch.tensor([[11], [0]], dtype=torch.int32), torch.tensor([4, 0], dtype=torch.int32), k, v)
        for a, b in zip(alone, padded):
            torch.testing.assert_close(a[:1] if a.dim() == 2 else a[:, :1], b[:1] if b.dim() == 2 else b[:, :1])


class BatchGraphV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = random_model()

    def test_read_only_caches_match_full_forward(self):
        for case in ({"batch": 6}, {"batch": 1}, {"batch": 3, "slots": zb.CONTEXT}):
            stats = zb.check_graph(self.model, z2.step_view(z2.KVBatchGraphZhV2(self.model)), **case)
            self.assertEqual(stats["rows_outside_tolerance"], 0)

    def test_multi_token_prefill_from_different_positions(self):
        for case in ({"batch": 3}, {"batch": 1, "tokens": 1}, {"batch": 5, "tokens": 4}):
            stats = z2.check_prefill(self.model, z2.KVBatchGraphZhV2(self.model), **case)
            self.assertEqual(stats["rows_outside_tolerance"], 0)

    def test_traced_flexible_tokens(self):
        traced = z2.trace(self.model, "logits", "flex")
        stats = z2.check_prefill(self.model, traced, batch=4, tokens=5)
        self.assertEqual(stats["rows_outside_tolerance"], 0)

    def test_scored_head_is_log_softmax_on_allowed(self):
        self.assertLess(z2.check_scored(self.model)["max_abs_log_prob"], zb.TOLERANCE)

    def test_sequence_head_scores_every_new_position(self):
        step = z2.torch_sequence_step(z2.KVBatchGraphZhV2(self.model, "sequence"))
        for case in z2.SEQUENCE_CASES:
            stats = z2.check_sequence(self.model, step, **case)
            self.assertEqual(stats["positions_outside_tolerance"], 0)

    def test_traced_sequence_head_and_continuation_sums(self):
        traced = z2.trace(self.model, "sequence", "flex")
        stats = z2.check_continuations(self.model, z2.torch_sequence_step(traced), count=20)
        self.assertTrue(stats["passed"], stats)
        self.assertEqual(stats["calls"], 1 + 2)     # prefill + ceil(20 / 16)
        with self.assertRaises(ValueError):
            z2.trace(self.model, "sequence", 1)

    def test_cache_is_not_modified(self):
        k = torch.randn(zb.LAYERS, 2, zb.HEADS, 4, zb.HEAD_DIM)
        before = k.clone()
        with torch.inference_mode():
            z2.KVBatchGraphZhV2(self.model)(torch.tensor([[7], [9]], dtype=torch.int32),
                                            torch.tensor([1, 3], dtype=torch.int32), k, k)
        torch.testing.assert_close(k, before)


class ToySP:
    """Three pieces: 中 (zhong), 文 (wen), 中文 as one token; ids 10..12."""
    pieces = {10: "中", 11: "文", 12: "中文"}

    def get_piece_size(self):
        return 13

    def is_control(self, i):
        return i < 8

    def is_unknown(self, i):
        return False

    def is_byte(self, i):
        return False

    def id_to_piece(self, i):
        return self.pieces.get(i, "")

    def encode(self, text):
        return []

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


class ToyTable:
    words = {"中": [(["zhong"], 1.0)], "文": [(["wen"], 1.0)]}

    def characters(self):
        return set(self.words)


class FixedBackend:
    """Log-probs that favour single characters, as a plain LM does at the first step."""
    def __init__(self):
        self.rows = 0

    def logits(self):
        x = torch.full((13,), -20.0)
        x[10], x[11], x[12] = 0.0, 0.0, -1.0
        return x

    def prefill(self, ids):
        return self.logits(), None

    def decode(self, tokens, states):
        self.rows += len(tokens)
        return torch.stack([self.logits() for _ in tokens]), [None] * len(tokens)


class ScoredBackend(FixedBackend):
    """Same scores, delivered as log-probs on the allowed ids only (the v2 scored head)."""
    scored = True

    def prefill(self, ids, allowed):
        return self.logits().log_softmax(-1)[allowed], None

    def decode(self, tokens, states, allowed):
        self.rows += len(tokens)
        return torch.stack([self.logits().log_softmax(-1)[allowed] for _ in tokens]), [None] * len(tokens)


class SearchTest(unittest.TestCase):
    def test_multi_key_token_survives_beam_one(self):
        index = zs.Index(ToySP(), ToyTable())
        ranked, stats = zs.search(FixedBackend(), ToySP(), index, "zhongwen", beam=1, topk=5)
        self.assertEqual([t for t, _, _ in ranked], ["中文"])
        self.assertEqual(stats.buckets, [1])     # one forward: the 中 hypothesis at position 5

    def test_shared_beam_failure_mode_is_gone(self):
        """Single-key initials score higher per step; they must not evict the 8-key token."""
        index = zs.Index(ToySP(), ToyTable())
        ranked, _ = zs.search(FixedBackend(), ToySP(), index, "zhongwen", beam=2, topk=5)
        self.assertEqual(ranked[0][0], "中文")

    def test_initials_and_full_readings(self):
        index = zs.Index(ToySP(), ToyTable())
        for keys in ("zw", "zhw", "zhongw", "zwen"):
            ranked, _ = zs.search(FixedBackend(), ToySP(), index, keys, beam=4, topk=5)
            self.assertIn("中文", [t for t, _, _ in ranked], keys)

    def test_scored_backend_gives_the_same_ranking(self):
        index = zs.Index(ToySP(), ToyTable())
        for keys in ("zhongwen", "zw", "zhw"):
            a, _ = zs.search(FixedBackend(), ToySP(), index, keys, beam=4, topk=5)
            b, _ = zs.search(ScoredBackend(), ToySP(), index, keys, beam=4, topk=5)
            self.assertEqual([t for t, _, _ in a], [t for t, _, _ in b])
            for (_, x, _), (_, y, _) in zip(a, b):
                self.assertAlmostEqual(x, y, places=5)


if __name__ == "__main__":
    unittest.main()
