"""V4 bundle identity, tied-weight storage and KV evaluation adapter checks."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.deployment import v4_chunk, v4_coreml as v4
from vimeml.deployment.bundle import read_json, write_json
from vimeml.deployment.v4 import CACHE_SHAPE, KVStepGraphV4
from vimeml.training.data import file_sha
from vimeml.training.evaluate_ime import score_candidates
from vimeml.training.model_v4 import TinyGPTV4

TOKENIZER = ROOT / "artifacts/tokenizers/v4-a/ja-unigram-8192"
TOKENS = ROOT / "artifacts/token-data/corpus-v4-8k/manifest.json"


class EagerKVRuntime:
    """Same run() contract as KVRuntimeV4, using the eager single-token graph."""
    def __init__(self, model):
        self.graph, self.config, self.calls = KVStepGraphV4(model).eval(), model.config, 0

    @torch.inference_mode()
    def run(self, ids, state=None, start=0):
        state = v4.empty_cache() if state is None else state
        key, value = (torch.from_numpy(a.copy()) for a in state)
        parts = []
        for offset, token in enumerate(ids):
            logits, key, value = self.graph(torch.tensor([[int(token)]], dtype=torch.int32),
                                            torch.tensor([start + offset], dtype=torch.int32), key, value)
            parts.append(logits.numpy())
            self.calls += 1
        return np.concatenate(parts, axis=1), (key.numpy(), value.numpy())


class EagerChunkRuntime:
    """Same run() contract as KVChunkRuntimeV4 (one call per chunk), eager graph."""
    def __init__(self, model):
        self.graph, self.config, self.calls = v4_chunk.KVChunkGraphV4(model).eval(), model.config, 0

    @torch.inference_mode()
    def run(self, ids, state=None, start=0, chunks=None):
        state = v4.empty_cache() if state is None else state
        key, value = (torch.from_numpy(a.copy()) for a in state)
        parts, offset = [], 0
        for size in chunks or [len(ids)]:
            logits, key, value = self.graph(torch.tensor([list(ids[offset:offset + size])], dtype=torch.int32),
                                            torch.tensor([start + offset], dtype=torch.int32), key, value)
            parts.append(logits.numpy())
            offset += size
            self.calls += 1
        return np.concatenate(parts, axis=1), (key.numpy(), value.numpy())


@unittest.skipUnless(TOKENS.is_file() and (TOKENIZER / "tokenizer.model").is_file(), "V4 tokenizer artifacts absent")
class V4DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(1020)
        cls.model = TinyGPTV4().eval()

    def checkpoint(self, folder):
        tokens = read_json(TOKENS)
        saved = {"format": "vimeml_tiny_gpt_v4", "architecture": "tiny_gpt_v4",
                 "model_config": self.model.configuration(), "model": self.model.state_dict(),
                 "step": 7, "epoch": 0, "deployment_only": True, "source_checkpoint_sha256": "a" * 64,
                 "config": {"architecture": "tiny_gpt_v4", "token_dir": "artifacts/token-data/corpus-v4-8k"},
                 "signatures": {"tokens": file_sha(TOKENS)},
                 "tokenizer_identity": {"tokenizer_model_sha256": tokens["tokenizer_model_sha256"],
                                        "vocab_size": 8192, "special_ids": tokens["special_ids"]}}
        path = Path(folder) / "deployment.pt"
        torch.save(saved, path)
        return path, saved

    def test_bundle_roundtrip_binds_origin_ties_head_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, saved = self.checkpoint(folder)
            bundle = Path(folder) / "bundle"
            report = v4.export_bundle(checkpoint, TOKENIZER, bundle)
            self.assertEqual(report["checkpoint_sha256"], "a" * 64)
            self.assertEqual(report["deployment_checkpoint_sha256"], file_sha(checkpoint))
            self.assertNotIn("lm_head.weight", torch.load(bundle / "weights.pt", weights_only=True))
            lm = v4.BundleLMV4(bundle)
            self.assertIs(lm.model.lm_head.weight, lm.model.token_embedding.weight)
            self.assertEqual(lm.forbidden, (0, 1, 2, 4, 5, 6))
            ids = torch.tensor([[2, 70, 900, 31]])
            with torch.no_grad():
                torch.testing.assert_close(lm.model(ids), self.model(ids), atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "already exists"):
                v4.export_bundle(checkpoint, TOKENIZER, bundle)
            (bundle / "tokenizer.model").write_bytes((bundle / "tokenizer.model").read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                v4.verify_bundle(bundle)
            manifest = read_json(bundle / "manifest.json")
            manifest["architecture"] = "tiny_gpt_v2"
            write_json(bundle / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                v4.verify_bundle(bundle, hashes=False)
            saved["model"]["lm_head.weight"] = saved["model"]["lm_head.weight"].clone() + 1
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, "tied"):
                v4.export_bundle(checkpoint, TOKENIZER, Path(folder) / "untied")

    def test_cached_adapter_matches_full_scoring_and_skips_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, _ = self.checkpoint(folder)
            v4.export_bundle(checkpoint, TOKENIZER, Path(folder) / "bundle")
            full = v4.BundleLMV4(Path(folder) / "bundle")
            cached = v4.BundleLMV4(Path(folder) / "bundle")
        runtime = EagerKVRuntime(cached.model)
        cached.model = v4.CachedForwardV4(runtime, cached.special["pad"])
        for context, candidates in [("昨日友達と", ["話した", "離した", "はなした"]), ("", ["ありがとう", "有難う"])]:
            expected = score_candidates(full, context, candidates)
            actual = score_candidates(cached, context, candidates)
            self.assertEqual(expected["common_prefix_tokens_including_bos"], actual["common_prefix_tokens_including_bos"])
            for left, right in zip(expected["candidates"], actual["candidates"], strict=True):
                self.assertEqual(left["token_ids"], right["token_ids"])
                self.assertAlmostEqual(left["log_probability_sum"], right["log_probability_sum"], delta=3e-4)
        # Shared prefix once plus each suffix; trailing PAD positions are not computed.
        runtime.calls = 0
        inputs = torch.tensor([[2, 10, 11, 12, 0, 0], [2, 10, 11, 13, 14, 15]])
        logits = cached.model(inputs)
        self.assertEqual(runtime.calls, 3 + 1 + 3)
        self.assertTrue(torch.equal(logits[0, 4:], torch.zeros(2, 8192)))
        with torch.no_grad():
            torch.testing.assert_close(logits[1], self.model(inputs[1:])[0], atol=3e-4, rtol=3e-4)

    def test_chunk_graph_trace_matches_full_and_forward_cached_and_masks_dirty_tail(self):
        v4_chunk.trace_kv_chunk_graph(self.model)
        runtime = EagerChunkRuntime(self.model)
        ids = [2, *range(100, 130)]
        with torch.no_grad():
            full = self.model(torch.tensor([ids])).numpy()
        for chunks in ([31], [1] * 31, [3, 28], [15, 16]):
            actual, _ = runtime.run(ids, chunks=chunks)
            np.testing.assert_allclose(actual, full, atol=3e-4, rtol=3e-4)
        _, state = runtime.run(ids[:7])
        dirty = tuple(a.copy() for a in state)
        for a in dirty:
            a[:, :, :, 7:, :] = 999
        clean, _ = runtime.run(ids[7:10], state, 7)
        polluted, _ = runtime.run(ids[7:10], dirty, 7)
        np.testing.assert_array_equal(clean, polluted)

    def test_chunk_adapter_uses_one_call_per_prefix_and_suffix(self):
        runtime = EagerChunkRuntime(self.model)
        forward = v4_chunk.CachedChunkForwardV4(runtime, 0)
        inputs = torch.tensor([[2, 10, 11, 12, 0, 0], [2, 10, 11, 13, 14, 15]])
        logits = forward(inputs)
        self.assertEqual(runtime.calls, 1 + 1 + 1)
        with torch.no_grad():
            torch.testing.assert_close(logits[1], self.model(inputs[1:])[0], atol=3e-4, rtol=3e-4)
            torch.testing.assert_close(logits[0, :4], self.model(inputs[:1, :4])[0], atol=3e-4, rtol=3e-4)

    def test_error_distribution_reports_quantiles_and_argmax(self):
        expected = np.zeros((1, 2, 4), dtype=np.float32)
        expected[0, :, 1] = 1
        actual = expected.copy()
        actual[0, 1, 2] = 2
        report = v4.error_distribution(expected, actual)
        self.assertEqual(report["outside_tolerance"], 1)
        self.assertEqual(report["argmax_equal_positions"], 1)
        self.assertEqual(report["quantiles_abs"]["0.5"], 0.0)
        self.assertEqual(tuple(v4.empty_cache()[0].shape), CACHE_SHAPE)


if __name__ == "__main__":
    unittest.main()
