"""vimeml-tiny-zh-v1 deployment: chunk graph, cached scoring adapter, ranking diffs and bundle identity."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.zh13_lm_scores import score_item  # noqa: E402
from vimeml.deployment import v4_coreml as v4, zh_v1 as zh  # noqa: E402
from vimeml.deployment.v4 import CACHE_SHAPE as V4_CACHE_SHAPE  # noqa: E402
from vimeml.deployment.v4_chunk import KVChunkGraphV4  # noqa: E402
from vimeml.training.model_zh13 import GPTZh13Config, TinyGPTZh13  # noqa: E402

CHECKPOINT = ROOT / "artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/deployment.pt"
TOKENIZER = ROOT / "artifacts/tokenizers/zh-v13/zh-unigram-12288"


class EagerChunkRuntime:
    """Same run() contract as ChunkRuntimeZh (one call per chunk), eager graph."""
    def __init__(self, model):
        self.graph, self.config, self.calls = KVChunkGraphV4(model).eval(), model.config, 0

    @torch.inference_mode()
    def run(self, ids, state=None, start=0, chunks=None):
        state = zh.empty_cache() if state is None else state
        key, value = (torch.from_numpy(a.copy()) for a in state)
        ids = [int(i) for i in ids]
        parts, offset = [], 0
        for size in chunks or [len(ids)]:
            logits, key, value = self.graph(torch.tensor([ids[offset:offset + size]], dtype=torch.int32),
                                            torch.tensor([start + offset], dtype=torch.int32), key, value)
            parts.append(logits.numpy())
            offset += size
            self.calls += 1
        return np.concatenate(parts, axis=1), (key.numpy(), value.numpy())


class FakeProcessor:
    """Character-level stand-in: one id per character; round-trips exactly."""
    def __init__(self):
        self.ids = {}

    def encode(self, text):
        return [self.ids.setdefault(c, 8 + len(self.ids)) for c in text]

    def decode(self, ids):
        chars = {i: c for c, i in self.ids.items()}
        return "".join(chars[i] for i in ids)


def small_model():
    torch.manual_seed(1111)
    return TinyGPTZh13(GPTZh13Config()).eval()


class ZhContractTests(unittest.TestCase):
    def test_contract_differs_from_v4_only_in_vocab_and_layers(self):
        self.assertEqual(zh.CACHE_SHAPE, (9, *V4_CACHE_SHAPE[1:]))
        self.assertEqual((zh.VOCAB, v4.VOCAB), (12288, 8192))
        self.assertEqual(zh.RECIPES[zh.DEPLOYED_RECIPE], {"method": "linear_symmetric", "bits": 8,
                                                           "granularity": "per_block", "block_size": 32})
        self.assertEqual(zh.WEIGHT_THRESHOLD, 2048)

    @torch.inference_mode()
    def test_chunk_graph_matches_full_forward_and_cache(self):
        model = small_model()
        traced = zh.trace_chunk_graph(model)  # asserts every schedule internally
        ids = torch.randint(8, zh.VOCAB, (1, 20), generator=torch.Generator().manual_seed(3), dtype=torch.int32)
        zeros = torch.zeros(zh.CACHE_SHAPE)
        logits, key, _ = traced(ids, torch.tensor([0], dtype=torch.int32), zeros, zeros)
        self.assertEqual(tuple(key.shape), zh.CACHE_SHAPE)
        torch.testing.assert_close(logits, model(ids.long()), atol=3e-4, rtol=3e-4)

    @torch.inference_mode()
    def test_cached_forward_reproduces_stateless_scores(self):
        model = small_model()
        runtime = EagerChunkRuntime(model)
        cached = zh.CachedChunkForwardZh(runtime, pad=0)
        processor = FakeProcessor()
        candidates = ["你好", "你们好呀", "您", "拟好"]
        expected = score_item(model, processor, "我想说", candidates)
        actual = score_item(cached, processor, "我想说", candidates)
        np.testing.assert_allclose(actual["scores"], expected["scores"], atol=1e-4, rtol=1e-5)
        self.assertEqual(actual["scored_tokens"], expected["scored_tokens"])
        # One call for the shared prefix plus one per row extending past it; the
        # one-token candidate's input row is the shared prefix itself.
        self.assertEqual(runtime.calls, 1 + len(candidates) - 1)


class RankingTests(unittest.TestCase):
    def write(self, directory, name, rows):
        path = Path(directory) / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    def test_rank_differences_use_stable_ties_and_ids_only(self):
        with tempfile.TemporaryDirectory() as directory:
            a = self.write(directory, "a.jsonl", [{"id": "x", "candidates": ["p", "q", "r"], "scores": [-1.0, -1.0, -3.0]},
                                                  {"id": "y", "candidates": ["p", "q"], "scores": [-2.0, -1.0]},
                                                  {"id": "z", "candidates": [], "scores": []}])
            b = self.write(directory, "b.jsonl", [{"id": "x", "candidates": ["p", "q", "r"], "scores": [-1.0, -1.0, -0.5]},
                                                  {"id": "y", "candidates": ["p", "q"], "scores": [-2.0, -1.5]},
                                                  {"id": "z", "candidates": [], "scores": []}])
            result = zh.rank_differences(a, b)
        self.assertEqual(zh.ranking({"candidates": ["p", "q"], "scores": [-1.0, -1.0]}), [0, 1])
        self.assertEqual(result["order_changed_ids"], ["x"])
        self.assertEqual(result["top1_changed_ids"], ["x"])
        self.assertAlmostEqual(result["max_score_abs_delta"], 2.5)

    def test_changed_items_from_result_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / "a", Path(directory) / "b"
            before.mkdir(), after.mkdir()
            self.write(before, "items.jsonl", [{"id": "i1", "rank": 1}, {"id": "i2", "rank": 2}, {"id": "i3", "rank": None}])
            self.write(after, "items.jsonl", [{"id": "i1", "rank": 3}, {"id": "i2", "rank": 1}, {"id": "i3", "rank": None}])
            self.assertEqual(zh.changed_items(before, after), {"improved_ids": ["i2"], "regressed_ids": ["i1"]})


@unittest.skipUnless(CHECKPOINT.is_file() and (TOKENIZER / "tokenizer.model").is_file(), "zh-v1 payload not restored")
class BundleTests(unittest.TestCase):
    def test_bundle_stores_tied_head_once_and_matches_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            manifest = zh.export_bundle(CHECKPOINT, TOKENIZER, output)
            self.assertEqual(manifest["parameter_count"], 20_036_736)
            self.assertEqual(manifest["cache_shape"], list(zh.CACHE_SHAPE))
            self.assertNotIn("lm_head.weight", torch.load(output / "weights.pt", weights_only=True))
            lm = zh.BundleLMZh(output)
            saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
            for name, value in lm.model.state_dict().items():
                self.assertTrue(torch.equal(value, saved["model"][name]), name)
            with self.assertRaises(ValueError):
                zh.export_bundle(CHECKPOINT, TOKENIZER, output)


if __name__ == "__main__":
    unittest.main()
