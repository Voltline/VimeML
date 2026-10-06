"""FP16 weight-storage baseline and weight-only quantization with FP32 computation.

Kept outside the frozen deployment sources so an existing inference bundle
remains verifiable. This converter's hash and exact precision policy are saved
in the package manifest. Stages preserve prior experiment directories.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from vimeml.deployment.bundle import BundleLM, read_json
from vimeml.deployment.coreml import coremltools, finish, inspect_spec, verify_package
from vimeml.deployment.graph import trace_graph
from vimeml.training.data import file_sha


def weight_storage_graph(ct, traced, context_length, target=18):
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil.scope import ScopeInfo

    program = ct.convert(traced, source="pytorch", convert_to="milinternal",
        minimum_deployment_target=getattr(ct.target, f"iOS{target}"), compute_precision=ct.precision.FLOAT32,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(lower_bound=1,
            upper_bound=context_length, default=min(16, context_length))), dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)])
    changed = {"fp16_linear_layers": 0, "fp16_token_embedding_tables": 0,
               "fp32_qkv_layers": 0, "fp32_position_embedding_tables": 0}
    for function in program.functions.values():
        with function:
            for op in list(function.operations):
                if op.op_type == "linear":
                    if "attention.qkv" in op.weight.name:
                        changed["fp32_qkv_layers"] += 1
                        continue
                    with mb.scope(*[ScopeInfo(source=source, data=data)
                                    for source, data in op.scopes.items()]), mb.set_before_op(op):
                        weight = mb.const(val=op.weight.val.astype(np.float16),
                                          name=op.weight.name + "_storage_fp16")
                        arguments = {"weight": weight}
                        if op.bias is not None:
                            arguments["bias"] = mb.const(val=op.bias.val.astype(np.float16),
                                                         name=op.bias.name + "_storage_fp16")
                        result = mb.linear(x=op.x, name=op.name + "_compute_fp32", **arguments)
                    function.replace_uses_of_var_after_op(anchor_op=op,
                        old_var=op.outputs[0], new_var=result)
                    function.remove_ops([op])
                    changed["fp16_linear_layers"] += 1
                elif (op.op_type == "gather" and op.x.val is not None and
                      op.x.val.dtype == np.float32):
                    if "position_embedding" in op.x.name:
                        changed["fp32_position_embedding_tables"] += 1
                        continue
                    if "token_embedding" not in op.x.name:
                        raise ValueError("Unexpected floating-point gather table; inspect the graph.")
                    with mb.scope(*[ScopeInfo(source=source, data=data)
                                    for source, data in op.scopes.items()]), mb.set_before_op(op):
                        table = mb.const(val=op.x.val.astype(np.float16),
                                         name=op.x.name + "_storage_fp16")
                        gathered = mb.gather(x=table, indices=op.indices, axis=op.axis,
                                             name=op.name + "_storage_fp16")
                        result = mb.cast(x=gathered, dtype="fp32", name=op.name + "_compute_fp32")
                    function.replace_uses_of_var_after_op(anchor_op=op,
                        old_var=op.outputs[0], new_var=result)
                    function.remove_ops([op])
                    changed["fp16_token_embedding_tables"] += 1
    if not all(changed.values()):
        raise ValueError(f"Expected TinyGPT precision boundaries were not found: {changed}")
    program.functions["main"].outputs[0].set_name("logits")
    return program, changed


def convert(bundle, output, target=18):
    ct = coremltools()
    if output.exists():
        raise ValueError("Output exists; choose a new conversion experiment.")
    lm = BundleLM(bundle)
    program, changed = weight_storage_graph(ct, trace_graph(lm.model), lm.model.config.context_length, target)
    model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=getattr(ct.target, f"iOS{target}"),
                       compute_precision=ct.precision.FLOAT32, skip_model_load=True)
    policy = {"recipe": "fp16_weights_fp32_compute_preserve_positions_qkv_v1",
              "computation": "FP32; token embedding gather is FP16 then cast to FP32",
              "fp16_parameters": "Token embedding; non-QKV linear weights and biases, including LM head",
              "fp32_parameters": "Position embedding; attention QKV weights/biases; LayerNorm parameters",
              "graph_changes": changed}
    model.short_description = "Frozen TinyGPT; mixed FP16/FP32 weights, FP32 computation; no KV cache."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    model.user_defined_metadata["precision_recipe"] = policy["recipe"]
    finish(output, model, {"kind": "fp16_weights_fp32_compute", "minimum_ios": target, "bundle": lm.metadata,
        "precision_policy": policy, "conversion_script_sha256": file_sha(Path(__file__)),
        "interface": {"input_ids": "int32 [1,T], 1<=T<=128; right PAD only",
                      "logits": "float32 [1,T,16384]; FP32 computation; no softmax"}})


def parameter_compression_config(opt, model, config):
    """Allow only parameter matrices; global defaults must leave other consts intact."""
    metadata = opt.get_weights_metadata(model, weight_threshold=config.weight_threshold)
    selected, excluded = [], []
    configs = {}
    for name, entry in metadata.items():
        value = entry.val
        consumers = [{"name": child.name, "type": child.op_type,
                      "inputs": dict(child.params_name_mapping)} for child in entry.child_ops]
        record = {"name": name, "shape": [int(dim) for dim in value.shape],
                  "dtype": str(value.dtype), "elements": int(value.size), "consumers": consumers}
        normalized_name = name.replace(".", "_")
        embedding = normalized_name.startswith(("model_token_embedding_weight",
                                               "model_position_embedding_weight"))
        parameter_consumers = all(
            (child.op_type == "linear" and child.params_name_mapping.get("weight") == name) or
            (embedding and child.op_type == "gather" and child.params_name_mapping.get("x") == name)
            for child in entry.child_ops)
        if (not consumers or not parameter_consumers or value.ndim != 2 or
                not np.issubdtype(value.dtype, np.floating)):
            record.update(reason="Not a linear weight or named embedding parameter",
                          nonfinite_elements=int((~np.isfinite(value)).sum()))
            excluded.append(record)
            continue
        if value.size <= config.weight_threshold:
            record["reason"] = "Does not exceed weight threshold"
            excluded.append(record)
            continue
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite learned parameter {name}; refusing compression.")
        selected.append(record)
        configs[name] = config
    if not selected:
        raise ValueError("No eligible learned parameter matrices found; inspect the model graph.")
    selection = {"policy": "linear_weights_and_named_embedding_tables_v1",
                 "selected_constants": selected, "excluded_constants": excluded,
                 "selected_elements": sum(item["elements"] for item in selected)}
    print(f"Compression selection: {len(selected)} parameter matrices; "
          f"{len(excluded)} large constants excluded.", flush=True)
    for item in excluded:
        print(f"  Excluded {item['name']}: {item['reason']}", flush=True)
    return opt.OptimizationConfig(global_config=None, op_name_configs=configs), selection


def compress(source, output, validation, method, bits, group_size, block_size, granularity="per_block"):
    """Same exact-package alignment gate, extended to this declared source kind."""
    ct = coremltools()
    if output.exists():
        raise ValueError("Output exists; choose a new compression experiment.")
    original = verify_package(source)
    if original["kind"] != "fp16_weights_fp32_compute":
        raise ValueError("Use this command only with an uncompressed conservative conversion.")
    gate = read_json(validation)
    if (gate.get("format") != "vimeml_alignment_v1" or not gate.get("passed") or
            gate.get("coreml_manifest_sha256") != file_sha(source / "manifest.json")):
        raise ValueError("First pass alignment for this exact source and supply its alignment.json.")
    if method == "palette" and group_size and original["minimum_ios"] < 18:
        raise ValueError("Grouped palette requires iOS18; use per-tensor palette or linear INT8 on iOS17.")
    if method == "linear" and (granularity == "per_block" or bits == 4) and original["minimum_ios"] < 18:
        raise ValueError("Blockwise/4bit linear quantization requires iOS18; use INT8 per-channel on iOS17.")
    from coremltools.optimize import coreml as opt
    if method == "palette":
        try:
            from sklearn.cluster import KMeans  # noqa: F401: required by Core ML k-means fallback
        except ImportError as error:
            raise RuntimeError("Install the Mac requirements first; palette compression needs "
                               "scikit-learn==1.5.1 (uv pip install --python venv/coreml/bin/python "
                               "scikit-learn==1.5.1).") from error
    model = ct.models.MLModel(str(source / "model.mlpackage"), skip_model_load=True)
    if method == "palette":
        config = opt.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=2048,
            granularity="per_grouped_channel" if group_size else "per_tensor", group_size=group_size or 32)
    else:
        config = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=f"int{bits}",
            granularity=granularity, block_size=block_size if granularity == "per_block" else 0, weight_threshold=2048)
    optimization_config, selection = parameter_compression_config(opt, model, config)
    if method == "palette":
        result = opt.palettize_weights(model, config=optimization_config)
    else:
        result = opt.linear_quantize_weights(model, config=optimization_config)
    if not any(name.startswith("constexpr_") for name in inspect_spec(result)["operations"]):
        raise ValueError("No compressed constexpr weights found.")
    finish(output, result, {"kind": f"{method}{bits}_fp32_compute", "minimum_ios": original["minimum_ios"],
        "bundle": original["bundle"], "interface": original["interface"],
        "source_manifest_sha256": file_sha(source / "manifest.json"),
        "fp16_alignment_sha256": file_sha(validation),
        "conversion_script_sha256": file_sha(Path(__file__)),
        "precision_policy": {"source_recipe": original["precision_policy"]["recipe"],
                             "computation": original["precision_policy"]["computation"]},
        "compression": {"method": method, "bits": bits, "group_size": group_size,
                        "block_size": block_size if method == "linear" and granularity == "per_block" else None,
                        "granularity": granularity if method == "linear" else "per_grouped_channel" if group_size else "per_tensor",
                        "weight_threshold": 2048,
                        "selection": selection,
                        "scope": "Eligible learned matrices, including FP32 QKV; structural constants/masks unchanged; no activation quantization or retraining."}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    convert_parser = sub.add_parser("convert", help="Mixed weight storage and FP32 computation, iOS18.")
    convert_parser.add_argument("--bundle", type=Path, required=True)
    convert_parser.add_argument("--target", type=int, choices=(17, 18), default=18)
    compress_parser = sub.add_parser("compress", help="Compress this exact validated conservative baseline.")
    compress_parser.add_argument("--source", type=Path, required=True)
    compress_parser.add_argument("--fp16-alignment", type=Path, required=True)
    compress_parser.add_argument("--method", choices=("palette", "linear"), default="palette")
    compress_parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    compress_parser.add_argument("--group-size", type=int, choices=(0, 8, 16, 32), default=16)
    compress_parser.add_argument("--block-size", type=int, choices=(16, 32, 64, 128), default=32)
    compress_parser.add_argument("--granularity", choices=("per_channel", "per_block"), default="per_block")
    for command in (convert_parser, compress_parser):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive.")
    if args.output.exists():
        parser.error("Output exists; use a new versioned directory.")
    torch.set_num_threads(args.threads)
    if args.command == "convert":
        convert(args.bundle, args.output, args.target)
    else:
        compress(args.source, args.output, args.fp16_alignment, args.method, args.bits,
                 args.group_size, args.block_size, args.granularity)
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
