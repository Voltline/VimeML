"""Targeted unit tests for ja-gen stage 2 beam search, union rescoring, and decision logic."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from vimeml.benchmarks.kana_gen_ja import (
    evaluate_stage21_decision,
    is_d5_item,
    load_dev_pool,
    merge_and_rank,
    rank_lm_only,
    task_scores,
)
from vimeml.deployment.ja_search import (
    BOS,
    CTX,
    EOS,
    OUT,
    READ,
    KanaIndex,
    prompt_ids,
    search,
    search_constrained,
    search_free,
)
from vimeml.deployment.zh_torch_backend import ModelBatchBackend
from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2


class ToySP:
    """Minimal deterministic SentencePiece-like tokenizer for unit tests."""

    def __init__(self):
        self.pieces = [
            "<pad>", "<unk>", "<s>", "</s>", "<CTX>", "<READ>", "<OUT>", "\n",
            "き", "ょ", "う", "は", "い", "今", "日", "今日", "行", "く", "わ", "た", "し", "私",
        ]
        self.p2i = {p: i for i, p in enumerate(self.pieces)}

    def get_piece_size(self):
        return len(self.pieces)

    def is_control(self, tid):
        return tid in (0, 2, 3, 4, 5, 6)

    def is_unknown(self, tid):
        return tid == 1

    def is_byte(self, tid):
        return False

    def id_to_piece(self, tid):
        return self.pieces[tid]

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            if text[i:i + 2] in self.p2i:
                out.append(self.p2i[text[i:i + 2]])
                i += 2
            elif text[i] in self.p2i:
                out.append(self.p2i[text[i]])
                i += 1
            else:
                out.append(1)
                i += 1
        return out

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids if i >= 8)


class TestJaGenStage2(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.sp = ToySP()
        cfg = GPTV2Config(
            vocab_size=self.sp.get_piece_size(),
            context_length=64,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            dropout=0.0,
        )
        self.model = TinyGPTV2(cfg).eval()
        self.backend = ModelBatchBackend(self.model, device="cpu")
        self.index = KanaIndex({
            "きょう": [self.sp.p2i["今日"]],
            "きょ": [self.sp.p2i["今"]],
            "う": [self.sp.p2i["日"], self.sp.p2i["う"]],
            "は": [self.sp.p2i["は"]],
            "い": [self.sp.p2i["行"], self.sp.p2i["い"]],
            "く": [self.sp.p2i["く"]],
            "わたし": [self.sp.p2i["私"]],
        })

    def test_prompt_ids_format_and_left_trim(self):
        ids = prompt_ids(self.sp, "きょう", "わたし")
        self.assertEqual(ids[:2], [BOS, CTX])
        self.assertIn(READ, ids)
        self.assertEqual(ids[-1], OUT)
        trimmed = prompt_ids(self.sp, "きょう", "わ" * 100, limit=16, min_output_room=4)
        self.assertLessEqual(len(trimmed), 12)
        self.assertEqual(trimmed[:2], [BOS, CTX])
        self.assertEqual(trimmed[-1], OUT)

    def test_constrained_search_matches_task_scores(self):
        ranked, stats = search_constrained(self.backend, self.sp, self.index, "きょうは", context="", beam=8, topk=5)
        self.assertGreater(len(ranked), 0)
        self.assertGreater(stats.calls, 0)
        texts = [t for t, _, _ in ranked]
        beam_scores = [s for _, s, _ in ranked]
        exact_scores = task_scores(self.model, self.sp, "きょうは", "", texts, device="cpu")
        for text, bs, es in zip(texts, beam_scores, exact_scores):
            if self.sp.encode(text) == [ids for t, _, ids in ranked if t == text][0]:
                self.assertAlmostEqual(bs, es, places=4)

    def test_unreachable_falls_back_to_free_search(self):
        ranked, stats = search(self.backend, self.sp, self.index, "わた", context="", beam=4, topk=3,
                               search_mode="constrained", fallback_free=True)
        self.assertTrue(stats.fallback)
        self.assertEqual(stats.mode, "free")
        self.assertGreater(len(ranked), 0)

    def test_free_search_runs(self):
        ranked, stats = search_free(self.backend, self.sp, "きょう", context="", beam=4, topk=3, max_tokens=5)
        self.assertGreater(len(ranked), 0)
        self.assertEqual(stats.mode, "free")

    def test_merge_and_rank_and_lm_only(self):
        az = ["今日は", "きょうは"]
        gen = ["今日わ", "今日は"]
        # Union order: ["今日は", "きょうは", "今日わ"]
        res = merge_and_rank(az, gen, [-2.0, -3.0, -1.0], [-5.0, -1.0, -10.0], lam=1.6)
        # Scores: 今日は = -10.0, きょうは = -4.6, 今日わ = -17.0
        self.assertEqual(res, ["きょうは", "今日は", "今日わ"])
        self.assertEqual(rank_lm_only(az, [-5.0, -1.0]), ["きょうは", "今日は"])

    def test_dev_pool_refuses_test_and_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp) / "kana-ja-v1-test-n20"
            tdir.mkdir()
            with self.assertRaises(ValueError):
                load_dev_pool(tdir)
            sdir = Path(tmp) / "kana-ja-v1-development-n20"
            sdir.mkdir()
            (sdir / "manifest.json").write_text(json.dumps({"split": "development", "sealed": True}), "utf-8")
            (sdir / "evaluation_items.json").write_text("[]", "utf-8")
            (sdir / "candidates.jsonl").write_text("", "utf-8")
            with self.assertRaises(ValueError):
                load_dev_pool(sdir)

    def test_d5_and_stage21_decision(self):
        self.assertTrue(is_d5_item({"query": "ワタクシハ", "answers": ["私は"]}))
        self.assertTrue(is_d5_item({"input_hiragana": "わたくしは", "answers": ["私は"]}))
        self.assertFalse(is_d5_item({"query": "ワタシハ", "answers": ["私は"]}))
        read_s = [{"kana_development": {"notation_lenient_top1": {"rate": 86.5}},
                   "standard_development": {"exact_top1": {"hits": 510}}}] * 4
        lm_s = [{"kana_development": {"notation_lenient_top1": {"rate": 85.0}},
                 "standard_development": {"exact_top1": {"hits": 515}}}] * 4
        dec = evaluate_stage21_decision(read_s, lm_s)
        self.assertTrue(dec["keep_reading_recipe"])
        read_low_std = [{"kana_development": {"notation_lenient_top1": {"rate": 87.0}},
                         "standard_development": {"exact_top1": {"hits": 495}}}] * 4
        self.assertFalse(evaluate_stage21_decision(read_low_std, lm_s)["keep_reading_recipe"])


if __name__ == "__main__":
    unittest.main()
