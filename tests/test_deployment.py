"""Small synthetic graphs only; never export/convert/evaluate frozen production weights."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.deployment.bundle import (BundleLM, FORMAT, export_bundle, source_fingerprints,
                                      tree_inventory, verify_bundle, write_json)
from vimeml.training.data import file_sha
from vimeml.deployment.graph import ConversionGraph, trace_graph
from vimeml.deployment.validation import compare_rows, numeric_error
from vimeml.training.model import GPTConfig, TinyGPT


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = TinyGPT(GPTConfig(vocab_size=32, context_length=16, d_model=16,
                                      n_heads=4, n_layers=2, d_ff=32)).eval()
        (ROOT / "outputs").mkdir(exist_ok=True)

    def test_adapter_trace_preserves_lengths_causality_and_padding(self):
        traced = trace_graph(self.model)
        graph = ConversionGraph(self.model)
        short = torch.tensor([[2, 5, 7, 9, 11]], dtype=torch.int32)
        padded = torch.nn.functional.pad(short, (0, 11))
        changed = padded.clone()
        changed[:, 5:] = 19
        for forward in (self.model, graph, traced):
            with torch.inference_mode():
                expected = forward(short)
                torch.testing.assert_close(expected, forward(padded)[:, :5], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(expected, forward(changed)[:, :5], atol=1e-6, rtol=1e-5)

    def make_bundle(self, path):
        weights = {name: value for name, value in self.model.state_dict().items() if name != "lm_head.weight"}
        torch.save(weights, path / "weights.pt")
        write_json(path / "config.json", self.model.configuration())
        (path / "tokenizer.model").write_bytes(b"mock tokenizer")
        (path / "token-manifest.json").write_text("{}", encoding="utf-8")
        write_json(path / "manifest.json", {"format": FORMAT, "status": "complete", "files": tree_inventory(path),
            "source_fingerprints": source_fingerprints(), "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
            "checkpoint_sha256": "f" * 64, "parameter_count": self.model.parameter_count()})

    def test_bundle_preserves_tied_weight_and_logits_and_detects_tampering(self):
        class Processor:
            def __init__(self, **kwargs): pass
            def pad_id(self): return 0
            def unk_id(self): return 1
            def bos_id(self): return 2
            def eos_id(self): return 3
            def vocab_size(self): return 32
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as folder:
            path = Path(folder)
            self.make_bundle(path)
            with patch("vimeml.deployment.bundle.spm.SentencePieceProcessor", Processor):
                lm = BundleLM(path)
            self.assertIs(lm.model.token_embedding.weight, lm.model.lm_head.weight)
            ids = torch.tensor([[2, 5, 7]])
            torch.testing.assert_close(lm.model(ids), self.model(ids), rtol=0, atol=0)
            (path / "tokenizer.model").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_bundle(path)

    def test_export_strips_training_state_and_refuses_existing_output(self):
        class Processor:
            def __init__(self, **kwargs): pass
            def pad_id(self): return 0
            def unk_id(self): return 1
            def bos_id(self): return 2
            def eos_id(self): return 3
            def vocab_size(self): return 32
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as folder:
            path = Path(folder)
            tokenizer = path / "tokenizer.model"
            tokenizer.write_bytes(b"test tokenizer")
            token_manifest = path / "tokens.json"
            write_json(token_manifest, {"input_sha256": {"tokenizer.model": file_sha(tokenizer)},
                                       "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3}})
            checkpoint = path / "checkpoint.pt"
            signatures = {"tokens": file_sha(token_manifest), "training_code": {
                name: file_sha(ROOT / "src/vimeml/training" / name) for name in ("model.py", "data.py", "train.py")}}
            torch.save({"format": "vimeml_tiny_gpt_v1", "model_config": self.model.configuration(),
                        "model": self.model.state_dict(), "signatures": signatures, "step": 1,
                        "optimizer": {"secret_training_state": torch.ones(1)}, "rng": torch.ones(1)}, checkpoint)
            source_sha = file_sha(checkpoint)
            with patch("vimeml.deployment.bundle.spm.SentencePieceProcessor", Processor):
                export_bundle(checkpoint, tokenizer, token_manifest, path / "export")
                lm = BundleLM(path / "export")
                with self.assertRaisesRegex(ValueError, "already exists"):
                    export_bundle(checkpoint, tokenizer, token_manifest, path / "export")
            weights = torch.load(path / "export/weights.pt", weights_only=True)
            self.assertEqual(set(weights), set(self.model.state_dict()) - {"lm_head.weight"})
            self.assertTrue(all(isinstance(value, torch.Tensor) for value in weights.values()))
            self.assertIs(lm.model.lm_head.weight, lm.model.token_embedding.weight)
            self.assertEqual(file_sha(checkpoint), source_sha)

    def test_comparison_rejects_candidate_changes_and_reports_order_changes(self):
        def row(order, score):
            return {"id": "1", "query": "x", "left_context": "", "answers": ["a"],
                    "candidates": [{"text": "a", "score": 0}, {"text": "b", "score": 0}],
                    "orders": {"lm_context_sum": order}, "fallbacks": {},
                    "lm_scores": {"contextual": {"common_prefix_tokens_including_bos": 1,
                        "candidates": [{"text": "a", "token_ids": [4], "log_probability_sum": score},
                                       {"text": "b", "token_ids": [5], "log_probability_sum": -1.}]}}}
        before, after = row(["a", "b"], -.9), row(["b", "a"], -1.1)
        result = compare_rows([before], [after])
        self.assertEqual(result["top1_changed_cases"], 1)
        self.assertAlmostEqual(result["max_score_sum_abs"], .2)
        after["candidates"][0]["text"] = "changed"
        with self.assertRaisesRegex(ValueError, "input mismatch"):
            compare_rows([before], [after])

    def test_numerical_gate_counts_failures_and_rejects_nan(self):
        import numpy as np
        expected = np.array([1., 2.])
        result = numeric_error(expected, np.array([1.01, 2.5]), .1, 0)
        self.assertEqual(result["outside_tolerance"], 1)
        with self.assertRaises(ValueError):
            numeric_error(expected, np.array([1., np.nan]), .1, 0)


if __name__ == "__main__":
    unittest.main()
