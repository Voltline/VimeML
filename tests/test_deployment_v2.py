"""V2 architecture/identity rejection and causal parameter selection checks."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vimeml.deployment import v2
from vimeml.deployment.bundle import write_json
from vimeml.deployment.graph import trace_graph
from vimeml.training.data import file_sha
from vimeml.training.infer import ROOT
from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2


class Processor:
    def __init__(self, **kwargs): pass
    def pad_id(self): return 0
    def unk_id(self): return 1
    def bos_id(self): return 2
    def eos_id(self): return 3
    def vocab_size(self): return 128


class V2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(716)
        self.model = TinyGPTV2(GPTV2Config(vocab_size=128, context_length=32, d_model=32,
                                         n_heads=4, n_layers=1, d_ff=64)).eval()

    def test_export_load_preserves_origin_identity_tying_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            tokenizer = path / "tokenizer.model"
            tokenizer.write_bytes(b"synthetic tokenizer")
            tokens = path / "tokens.json"
            write_json(tokens, {"input_sha256": {"tokenizer.model": file_sha(tokenizer)},
                                "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3}})
            saved = {"format": "vimeml_tiny_gpt_v2", "architecture": "tiny_gpt_v2",
                "model_config": self.model.configuration(), "model": self.model.state_dict(), "step": 40000,
                "source_checkpoint_sha256": "a" * 64, "deployment_only": True,
                "signatures": {"tokens": file_sha(tokens), "training_code": {
                    name: file_sha(ROOT / "src/vimeml/training" / name)
                    for name in ("model.py", "model_v2.py", "model_factory.py")}}}
            checkpoint = path / "deployment.pt"
            torch.save(saved, checkpoint)
            with patch.object(v2.spm, "SentencePieceProcessor", Processor):
                report = v2.export_bundle(checkpoint, tokenizer, tokens, path / "bundle")
                lm = v2.BundleLM(path / "bundle")
                self.assertEqual(report["checkpoint_sha256"], "a" * 64)
                self.assertNotEqual(report["deployment_checkpoint_sha256"], report["checkpoint_sha256"])
                self.assertIs(lm.model.token_embedding.weight, lm.model.lm_head.weight)
                ids = torch.tensor([[2, 7, 19]])
                torch.testing.assert_close(lm.model(ids), self.model(ids), atol=0, rtol=0)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    v2.export_bundle(checkpoint, tokenizer, tokens, path / "bundle")
            manifest = path / "bundle/manifest.json"
            value = v2.read_json(manifest)
            value["architecture"] = "tiny_gpt_v1"
            write_json(manifest, value)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                v2.verify_bundle(path / "bundle")
            value["architecture"] = "tiny_gpt_v2"
            write_json(manifest, value)
            (path / "bundle/tokenizer.model").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                v2.verify_bundle(path / "bundle")
            saved["model"]["lm_head.weight"] = saved["model"]["lm_head.weight"].clone() + 1
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, "tied"):
                v2.export_bundle(checkpoint, tokenizer, tokens, path / "bad")

    def test_v2_trace_generalizes_lengths_padding_and_causality(self):
        forward = trace_graph(self.model)
        ids = torch.tensor([[2, 17, 25, 9]], dtype=torch.int32)
        pad = torch.nn.functional.pad(ids, (0, 28))
        future = pad.clone()
        future[:, 4:] = 71
        with torch.inference_mode():
            expected = self.model(ids)
            torch.testing.assert_close(forward(ids), expected, atol=3e-5, rtol=3e-4)
            torch.testing.assert_close(forward(pad)[:, :4], expected, atol=3e-5, rtol=3e-4)
            torch.testing.assert_close(forward(future)[:, :4], expected, atol=3e-5, rtol=3e-4)

    def test_int8_selection_preserves_mask_and_checks_finite_parameters(self):
        def entry(value, op, parameter, name):
            return SimpleNamespace(val=value, child_ops=[SimpleNamespace(name="consumer", op_type=op,
                                        params_name_mapping={parameter: name})])
        matrix = np.ones((64, 64), np.float32)
        metadata = {"causal_mask": entry(np.triu(np.full((64,64), -np.inf), 1), "slice_by_index", "x", "causal_mask"),
            "model_token_embedding_weight": entry(matrix, "gather", "x", "model_token_embedding_weight"),
            "linear_weight": entry(matrix, "linear", "weight", "linear_weight")}
        opt = SimpleNamespace(get_weights_metadata=lambda *a, **k: metadata,
                              OptimizationConfig=lambda **k: SimpleNamespace(**k))
        config, audit = v2.parameter_selection(opt, None, SimpleNamespace(weight_threshold=2048))
        self.assertIsNone(config.global_config)
        self.assertEqual(set(config.op_name_configs), {"linear_weight", "model_token_embedding_weight"})
        self.assertEqual(audit["excluded_constants"][0]["nonfinite_elements"], 2016)
        metadata["linear_weight"].val = matrix.copy()
        metadata["linear_weight"].val[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite learned parameter"):
            v2.parameter_selection(opt, None, SimpleNamespace(weight_threshold=2048))


if __name__ == "__main__":
    unittest.main()
