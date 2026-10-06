"""Mac-only Core ML conversion/compression/runtime, imported on demand."""
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from vimeml.deployment.bundle import (BundleLM, environment, fresh_directory, read_json,
                                      tree_inventory, write_json)
from vimeml.deployment.graph import trace_graph
from vimeml.training.data import file_sha


def coremltools():
    if platform.system() != "Darwin":
        raise RuntimeError("Run Core ML conversion, compression and prediction manually on Mac.")
    import coremltools as ct
    return ct


def inspect_spec(model):
    spec = model.get_spec()
    operations = Counter()
    def walk(block):
        for op in block.operations:
            operations[op.type] += 1
            for child in op.blocks:
                walk(child)
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            walk(block)
    return {"specification_version": spec.specificationVersion, "operations": dict(operations),
            "inputs": [f.name for f in spec.description.input], "outputs": [f.name for f in spec.description.output]}


def finish(output, model, metadata):
    fresh_directory(output)
    package = output / "model.mlpackage"
    model.save(str(package))
    inventory = tree_inventory(package)
    write_json(output / "manifest.json", {"format": "vimeml_coreml_v1", "status": "complete", **metadata,
        "environment": environment(), "spec": inspect_spec(model), "package_files": inventory,
        "package_bytes": sum(entry["bytes"] for entry in inventory.values()),
        "note": "Logical file bytes, not compiled size, IPA size or resident memory. Inspect weight.bin and const/constexpr ops for tied-weight duplication."})


def verify_package(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("format") != "vimeml_coreml_v1" or manifest.get("status") != "complete":
        raise ValueError("Incomplete/unsupported Core ML experiment.")
    if tree_inventory(directory / "model.mlpackage") != manifest["package_files"]:
        raise ValueError("Core ML package changed.")
    return manifest


def convert(bundle, output, target):
    ct = coremltools()
    if output.exists():
        raise ValueError("Output exists; choose a new conversion experiment.")
    lm = BundleLM(bundle)
    traced = trace_graph(lm.model)
    model = ct.convert(traced, source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, f"iOS{target}"),
        compute_precision=ct.precision.FLOAT16,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(lower_bound=1,
            upper_bound=lm.model.config.context_length, default=min(16, lm.model.config.context_length))), dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)], skip_model_load=True)
    model.short_description = "Frozen Tiny Japanese GPT; LM-only reranking and phrase continuation. No KV cache."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    finish(output, model, {"kind": "fp16", "minimum_ios": target, "bundle": lm.metadata,
                         "interface": {"input_ids": "int32 [1,T], 1<=T<=128; right PAD only",
                                       "logits": "float32 [1,T,16384]; FP16 computation; no softmax"}})


def compress(source, output, method, bits, group_size, block_size, validation):
    ct = coremltools()
    if output.exists():
        raise ValueError("Output exists; choose a new compression experiment.")
    original = verify_package(source)
    if original["kind"] != "fp16":
        raise ValueError("Compress the uncompressed FP16 source, not another compressed package.")
    gate = read_json(validation)
    if (gate.get("format") != "vimeml_alignment_v1" or not gate.get("passed") or
            gate.get("coreml_manifest_sha256") != file_sha(source / "manifest.json")):
        raise ValueError("First pass alignment for this exact FP16 model and supply its alignment.json.")
    from coremltools.optimize import coreml as opt
    model = ct.models.MLModel(str(source / "model.mlpackage"), skip_model_load=True)
    if method == "palette":
        grouped = group_size > 0
        if grouped and original["minimum_ios"] < 18:
            raise ValueError("Grouped-channel palettization requires an iOS18 source conversion.")
        config = opt.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=2048,
            granularity="per_grouped_channel" if grouped else "per_tensor", group_size=group_size or 32)
        result = opt.palettize_weights(model, config=opt.OptimizationConfig(global_config=config))
    else:
        if original["minimum_ios"] < 18:
            raise ValueError("This blockwise linear compression experiment requires an iOS18 source.")
        config = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=f"int{bits}",
            granularity="per_block", block_size=block_size, weight_threshold=2048)
        result = opt.linear_quantize_weights(model, config=opt.OptimizationConfig(global_config=config))
    ops = inspect_spec(result)["operations"]
    if not any(name.startswith("constexpr_") for name in ops):
        raise ValueError("No compressed constexpr weights found; refusing to label this a compressed model.")
    finish(output, result, {"kind": f"{method}{bits}", "minimum_ios": original["minimum_ios"],
        "bundle": original["bundle"], "interface": original["interface"],
        "source_manifest_sha256": file_sha(source / "manifest.json"),
        "fp16_alignment_sha256": file_sha(validation),
        "compression": {"method": method, "bits": bits, "group_size": group_size,
                        "block_size": block_size, "weight_threshold": 2048,
                        "scope": "Eligible constants; small constants remain uncompressed; no activation quantization or retraining."}})


class CoreMLForward:
    def __init__(self, directory, config, compute_units):
        ct = coremltools()
        self.config = config
        self.calls = 0
        self.predict_seconds = 0.0
        self.compute_units = compute_units
        started = time.perf_counter()
        self.model = ct.models.MLModel(str(Path(directory) / "model.mlpackage"),
                                      compute_units=getattr(ct.ComputeUnit, compute_units))
        self.load_seconds = time.perf_counter() - started

    def __call__(self, inputs):
        ids = inputs.detach().cpu().numpy()
        if ids.ndim != 2 or ids.shape[0] < 1 or not 1 <= ids.shape[1] <= self.config.context_length:
            raise ValueError("Expected nonempty batch with 1..128 tokens.")
        if np.any(ids < 0) or np.any(ids >= self.config.vocab_size):
            raise ValueError("Token ID outside vocabulary.")
        outputs = []
        # First deployment contract uses batch=1; preserve batched scoring semantics on host.
        for row in ids:
            started = time.perf_counter()
            logits = np.array(self.model.predict({"input_ids": row[None].astype(np.int32)})["logits"],
                              dtype=np.float32, copy=True)
            self.predict_seconds += time.perf_counter() - started
            self.calls += 1
            if logits.shape != (1, len(row), self.config.vocab_size) or not np.isfinite(logits).all():
                raise ValueError("Core ML returned wrong shape or nonfinite logits.")
            outputs.append(torch.from_numpy(logits))
        return torch.cat(outputs)


class CoreMLLM(BundleLM):
    def __init__(self, bundle, directory, compute_units="CPU_ONLY"):
        super().__init__(bundle)
        manifest = verify_package(directory)
        if manifest["bundle"]["bundle_manifest_sha256"] != self.metadata["bundle_manifest_sha256"]:
            raise ValueError("Core ML package belongs to a different inference bundle.")
        config = self.model.config
        self.model = CoreMLForward(directory, config, compute_units)
        self.metadata.update(precision=manifest["kind"], compute_units=compute_units,
                             coreml_manifest_sha256=file_sha(Path(directory) / "manifest.json"))
