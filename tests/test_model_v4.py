"""Fixed V4 parameter contract, causal loss, incremental/rollback and graph paths."""
import sys
import unittest
from pathlib import Path

import sentencepiece as spm
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.model_v4 import GPTV4Config, TinyGPTV4, PARAMETERS, LAYER_PARAMETERS
from vimeml.training.infer_v4 import JapaneseLMV4, score_candidates_cached
from vimeml.training.evaluate_ime import score_candidates, beam_suggestions
from vimeml.deployment.v4 import trace_kv_graph, CACHE_SHAPE
from vimeml.deployment.graph import trace_graph


class V4ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(944)
        cls.model = TinyGPTV4().eval()

    def test_exact_parameter_contract_and_rejects_architecture_changes(self):
        model = self.model
        self.assertEqual(model.parameter_count(), PARAMETERS)
        self.assertEqual(model.token_embedding.weight.numel(), 3_145_728)
        self.assertEqual(model.position_embedding.weight.numel(), 49_152)
        self.assertEqual(model.final_norm.weight.numel(), 384)
        for block in model.blocks:
            self.assertEqual(sum(p.numel() for p in block.parameters()), LAYER_PARAMETERS)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        self.assertTrue(all(m.bias is None for m in model.modules() if isinstance(m, torch.nn.Linear)))
        for key, value in (("vocab_size", 12288), ("n_layers", 6), ("dropout", .1), ("norm_eps", 1e-6)):
            with self.assertRaises(ValueError):
                GPTV4Config(**{key: value})

    def test_causality_padding_and_already_shifted_loss(self):
        ids = torch.tensor([[2, 10, 20, 30]])
        padded = torch.tensor([[2, 10, 20, 30, 0, 0, 0, 0]])
        labels = torch.tensor([[10, 20, 30, 3, -100, -100, -100, -100]])
        with torch.no_grad():
            short, full = self.model(ids), self.model(padded)
            torch.testing.assert_close(short, full[:, :4], atol=3e-4, rtol=3e-4)
            future = self.model(torch.tensor([[2, 10, 20, 30, 111, 112, 113, 114]]))
            torch.testing.assert_close(short, future[:, :4], atol=3e-4, rtol=3e-4)
            actual = self.model(padded, labels)
            expected = F.cross_entropy(full.reshape(-1, 8192), labels.reshape(-1), reduction="sum", ignore_index=-100)
            torch.testing.assert_close(actual["loss_sum"], expected)
            self.assertEqual(int(actual["token_count"]), 4)

    def test_incremental_chunks_full_context_and_aba_rollback(self):
        ids = torch.randint(7, 8192, (1, 128), generator=torch.Generator().manual_seed(948))
        cache = self.model.new_cache()
        self.assertEqual(tuple(cache.key.shape), CACHE_SHAPE)
        self.assertEqual(cache.key.numel() * 4 * 2, 2_752_512)
        with torch.no_grad():
            full = self.model(ids)
            start = 0
            for size in (1, 7, 3, 20, 97):
                actual = self.model.forward_cached(ids[:, start:start + size], cache)
                torch.testing.assert_close(actual, full[:, start:start + size], atol=3e-4, rtol=3e-4)
                start += size
        with self.assertRaises(ValueError):
            self.model.forward_cached(torch.tensor([[10]]), cache)
        branch = self.model.new_cache()
        self.model.forward_cached(ids[:, :7], branch)
        saved = branch.clone()
        a = self.model.forward_cached(torch.tensor([[100, 101, 102]]), branch)
        branch.restore(saved)
        self.model.forward_cached(torch.tensor([[200, 201]]), branch)
        branch.restore(saved)
        again = self.model.forward_cached(torch.tensor([[100, 101, 102]]), branch)
        torch.testing.assert_close(a, again, atol=0, rtol=0)
        self.assertEqual(saved.length, 7)
        self.assertEqual(branch.length, 10)
        with torch.inference_mode():
            inference_cache = self.model.new_cache()
            inference_snapshot = inference_cache.clone()
        inference_cache.restore(inference_snapshot)
        self.assertEqual(inference_cache.length, 0)

    def test_joint_candidate_score_next_token_and_beam_paths(self):
        lm = JapaneseLMV4.__new__(JapaneseLMV4)
        lm.model, lm.device = self.model, torch.device("cpu")
        lm.processor = spm.SentencePieceProcessor(model_file=str(ROOT / "artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model"))
        lm.special = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
        lm.forbidden = (0, 1, 2, 4, 5, 6)
        for context, candidates in [("昨日友達と", ["話した", "離した", "放した", "はなした"]),
                                    ("私は", ["学校に行きます", "学校へ行きます", "学生です"]),
                                    ("", ["ありがとうございます", "ありがとう"])]:
            original = score_candidates(lm, context, candidates)
            cached = score_candidates_cached(lm, context, candidates)
            self.assertEqual(original["common_prefix_tokens_including_bos"], cached["common_prefix_tokens_including_bos"])
            for left, right in zip(original["candidates"], cached["candidates"], strict=True):
                self.assertEqual(left["token_ids"], right["token_ids"])
                self.assertAlmostEqual(left["log_probability_sum"], right["log_probability_sum"], delta=3e-4)
            order = lambda data: sorted(candidates, key=lambda c: next(r["log_probability_sum"] for r in data["candidates"] if r["text"] == c), reverse=True)
            self.assertEqual(order(original), order(cached))
        self.assertTrue(all(item["id"] not in lm.forbidden for item in lm.next_pieces("今日は", 5)))
        self.assertTrue(lm.generate("今日は", max_new_tokens=2, temperature=0)["prefix_preserved"])
        self.assertTrue(beam_suggestions(lm, "今日は", max_tokens=3, width=3, count=2))
        with self.assertRaises(ValueError):
            lm.prefix_ids("<CTX>今日は")

    def test_dynamic_full_graph_and_functional_kv_trace(self):
        trace_graph(self.model)
        trace_kv_graph(self.model)


if __name__ == "__main__":
    unittest.main()
