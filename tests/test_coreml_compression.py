"""Compression selection and synthetic causal-mask regression checks."""
import platform
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/deployment"))

from coreml_conservative import parameter_compression_config, weight_storage_graph
from vimeml.deployment.coreml import inspect_spec
from vimeml.deployment.graph import trace_graph
from vimeml.training.model import GPTConfig, TinyGPT


def entry(value, op_type, parameter, name):
    return SimpleNamespace(val=value, child_ops=[SimpleNamespace(
        name="consumer", op_type=op_type, params_name_mapping={parameter: name})])


class SelectionTests(unittest.TestCase):
    def config(self, metadata):
        opt = SimpleNamespace(get_weights_metadata=lambda *args, **kwargs: metadata,
                              OptimizationConfig=lambda **kwargs: SimpleNamespace(**kwargs))
        return parameter_compression_config(opt, None, SimpleNamespace(weight_threshold=2048))

    def test_only_parameter_matrices_selected_and_infinite_mask_excluded(self):
        finite = np.ones((64, 64), dtype=np.float32)
        mask = np.triu(np.full((64, 64), -np.inf, dtype=np.float32), 1)
        metadata = {
            "causal_mask": entry(mask, "slice_by_index", "x", "causal_mask"),
            "linear_weight": entry(finite, "linear", "weight", "linear_weight"),
            "model_token_embedding_weight": entry(finite, "gather", "x", "model_token_embedding_weight"),
            "lookup_table": entry(finite, "gather", "x", "lookup_table"),
        }
        config, audit = self.config(metadata)
        self.assertIsNone(config.global_config)
        self.assertEqual(set(config.op_name_configs), {"linear_weight", "model_token_embedding_weight"})
        self.assertEqual({item["name"] for item in audit["excluded_constants"]}, {"causal_mask", "lookup_table"})
        excluded_mask = next(item for item in audit["excluded_constants"] if item["name"] == "causal_mask")
        self.assertEqual(excluded_mask["nonfinite_elements"], 2016)

    def test_nonfinite_parameter_is_rejected_instead_of_silently_skipped(self):
        value = np.ones((64, 64), dtype=np.float32)
        value[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite learned parameter linear_weight"):
            self.config({"linear_weight": entry(value, "linear", "weight", "linear_weight")})

    def test_shared_constant_with_structural_use_is_excluded(self):
        value = np.ones((64, 64), dtype=np.float32)
        shared = entry(value, "linear", "weight", "shared")
        shared.child_ops += entry(value, "add", "y", "shared").child_ops
        config, audit = self.config({"shared": shared, "weight": entry(value, "linear", "weight", "weight")})
        self.assertEqual(set(config.op_name_configs), {"weight"})
        self.assertEqual(audit["excluded_constants"][0]["name"], "shared")


@unittest.skipUnless(platform.system() == "Darwin", "Core ML integration requires Mac")
class CausalMaskCompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import coremltools as ct
        cls.ct = ct
        torch.set_num_threads(1)
        torch.manual_seed(716)
        tiny = TinyGPT(GPTConfig(vocab_size=128, context_length=64, d_model=32,
                                n_heads=4, n_layers=1, d_ff=64)).eval()
        program, _ = weight_storage_graph(ct, trace_graph(tiny), tiny.config.context_length)
        cls.model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                               compute_precision=ct.precision.FLOAT32, skip_model_load=True)

    def test_palette_and_linear_preserve_mask_and_create_compressed_weights(self):
        opt = self.ct.optimize.coreml
        before = opt.get_weights_metadata(self.model, weight_threshold=2048)["causal_mask"].val
        self.assertEqual(int(np.isneginf(before).sum()), 2016)
        variants = [
            ("palette", opt.OpPalettizerConfig(mode="kmeans", nbits=4,
                granularity="per_grouped_channel", group_size=16, weight_threshold=2048), opt.palettize_weights),
            ("linear-int8", opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                granularity="per_block", block_size=32, weight_threshold=2048), opt.linear_quantize_weights),
            ("linear", opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4",
                granularity="per_block", block_size=32, weight_threshold=2048), opt.linear_quantize_weights),
        ]
        for name, op_config, run in variants:
            with self.subTest(method=name):
                config, audit = parameter_compression_config(opt, self.model, op_config)
                self.assertNotIn("causal_mask", config.op_name_configs)
                self.assertGreater(len(audit["selected_constants"]), 0)
                result = run(self.model, config=config)
                after = opt.get_weights_metadata(result, weight_threshold=2048)["causal_mask"].val
                np.testing.assert_array_equal(after, before)
                self.assertTrue(any(name.startswith("constexpr_") for name in inspect_spec(result)["operations"]))


if __name__ == "__main__":
    unittest.main()
