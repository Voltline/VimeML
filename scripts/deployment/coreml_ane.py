"""Neural Engine variant: FP16 computation, enumerated input lengths, optional INT8 weights.

The conservative package computes in FP32, which the Neural Engine cannot execute, so every
op falls back to CPU. This entry converts the same traced graph with FP16 computation and a
small set of fixed lengths (callers right-pad to the next length; PAD never affects valid
positions under the causal mask), then optionally applies per-channel INT8 weight quantization.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from vimeml.deployment.bundle import BundleLM
from vimeml.deployment.coreml import coremltools, finish, inspect_spec, verify_package
from vimeml.deployment.graph import trace_graph
from vimeml.training.data import file_sha
from coreml_conservative import parameter_compression_config

LENGTHS = (16, 32, 64, 128)


def convert(bundle, output, lengths):
    ct = coremltools()
    lm = BundleLM(bundle)
    shapes = ct.EnumeratedShapes(shapes=[(1, length) for length in lengths], default=(1, lengths[0]))
    model = ct.convert(trace_graph(lm.model), source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16,
        inputs=[ct.TensorType(name="input_ids", shape=shapes, dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)], skip_model_load=True)
    model.short_description = "Frozen TinyGPT; FP16 computation for the Neural Engine; enumerated lengths; no KV cache."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    finish(output, model, {"kind": "fp16_ane", "minimum_ios": 18, "bundle": lm.metadata, "lengths": list(lengths),
        "conversion_script_sha256": file_sha(Path(__file__)),
        "interface": {"input_ids": f"int32 [1,T], T in {list(lengths)}; right-pad with PAD=0",
                      "logits": "float32 [1,T,16384]; FP16 computation; no softmax"}})


def compress(source, output):
    ct = coremltools()
    original = verify_package(source)
    if original["kind"] != "fp16_ane":
        raise ValueError("Compress an uncompressed fp16_ane conversion.")
    from coremltools.optimize import coreml as opt
    model = ct.models.MLModel(str(source / "model.mlpackage"), skip_model_load=True)
    config = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                         granularity="per_channel", weight_threshold=2048)
    optimization_config, selection = parameter_compression_config(opt, model, config)
    result = opt.linear_quantize_weights(model, config=optimization_config)
    if not any(name.startswith("constexpr_") for name in inspect_spec(result)["operations"]):
        raise ValueError("No compressed constexpr weights found.")
    finish(output, result, {"kind": "linear8_fp16_ane", "minimum_ios": 18, "bundle": original["bundle"],
        "lengths": original["lengths"], "interface": original["interface"],
        "source_manifest_sha256": file_sha(source / "manifest.json"),
        "conversion_script_sha256": file_sha(Path(__file__)),
        "compression": {"method": "linear", "bits": 8, "granularity": "per_channel", "weight_threshold": 2048,
                        "selection": selection,
                        "scope": "Learned matrices only; masks and small constants unchanged; FP16 activations."}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    convert_parser = sub.add_parser("convert")
    convert_parser.add_argument("--bundle", type=Path, required=True)
    convert_parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    compress_parser = sub.add_parser("compress")
    compress_parser.add_argument("--source", type=Path, required=True)
    for command in (convert_parser, compress_parser):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new versioned directory.")
    torch.set_num_threads(4)
    if args.command == "convert":
        lengths = sorted(set(args.lengths))
        if lengths[0] < 1 or lengths[-1] != 128:
            parser.error("Lengths must be positive and include 128.")
        convert(args.bundle, args.output, lengths)
    else:
        compress(args.source, args.output)
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
