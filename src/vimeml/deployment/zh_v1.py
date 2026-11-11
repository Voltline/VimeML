"""vimeml-tiny-zh-v1 inference bundle, FP32/compressed Core ML packages and gates.

Thin module beside the V4 deployment code, which keeps the Japanese contract
(8K vocabulary, seven layers, `JapaneseLMV4`). The block is the V4 family; only
the vocabulary (12,288), layer count (nine) and loader differ, so the graphs are
reused: the V2 explicit-attention `ConversionGraph` for the stateless package and
`v4_chunk.KVChunkGraphV4` (layer-count agnostic) for the chunk-KV package.

Stateless: int32 [1,T], 1<=T<=128 -> float32 [1,T,12288].
Chunk KV: input_ids int32 [1,T], position int32 [1], key/value caches float32
(9,1,6,128,64) -> logits [1,T,12288] plus updated caches. Caller owns snapshots.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from vimeml.benchmarks import zh13_lm_scores as zh13
from vimeml.deployment import v2
from vimeml.deployment.bundle import environment, fresh_directory, read_json, write_json
from vimeml.deployment.coreml import CoreMLForward, coremltools, inspect_spec
from vimeml.deployment.graph import trace_graph
from vimeml.deployment.v4_chunk import KVChunkGraphV4
from vimeml.deployment.v4_coreml import error_distribution, summarize_errors
from vimeml.deployment.validation import numeric_error
from vimeml.training.data import file_sha
from vimeml.training.evaluate_ime import common_prefix_length
from vimeml.training.joint_zh13 import ARCHITECTURE, FORMAT as CHECKPOINT_FORMAT, SPECIAL
from vimeml.training.model_zh13 import GPTZh13Config, TinyGPTZh13

RELEASE = "vimeml-tiny-zh-v1"
VOCAB = 12288
LAYERS = 9
CACHE_SHAPE = (LAYERS, 1, 6, 128, 64)
CONTEXT = 128
BUNDLE_FORMAT = "vimeml_inference_bundle_zh_v1"
PACKAGE_FORMAT = "vimeml_coreml_zh_v1"
CHUNK_FORMAT = "vimeml_coreml_zh_v1_kv_chunk"
ALIGNMENT_FORMAT = "vimeml_alignment_zh_v1"
CHUNK_ALIGNMENT_FORMAT = "vimeml_kv_chunk_alignment_zh_v1"
REFERENCE_FORMAT = "vimeml_zh_v1_mac_preparation_reference_v1"
TOLERANCE = 3e-4

# Task-4 recipe (unchanged V4.2/V2.1) plus the task-5 exploratory survey.
RECIPES = {
    "int8-b32": {"method": "linear_symmetric", "bits": 8, "granularity": "per_block", "block_size": 32},
    "int4-b32": {"method": "linear_symmetric", "bits": 4, "granularity": "per_block", "block_size": 32},
    "palette6-gc32": {"method": "kmeans_palettization", "bits": 6, "granularity": "per_grouped_channel",
                      "group_size": 32},
    "palette4-gc32": {"method": "kmeans_palettization", "bits": 4, "granularity": "per_grouped_channel",
                      "group_size": 32},
    "int8-b32-embfp16": {"method": "linear_symmetric", "bits": 8, "granularity": "per_block", "block_size": 32,
                         "tied_embedding": "fp16"},
}
DEPLOYED_RECIPE = "int8-b32"
WEIGHT_THRESHOLD = 2048
EMBEDDING = "model_token_embedding_weight"


def export_bundle(checkpoint, tokenizer_dir, output):
    """Unique FP32 inference tensors (tied head stored once), config, tokenizer and identity."""
    checkpoint, tokenizer_dir, output = map(Path, (checkpoint, tokenizer_dir, output))
    if output.exists():
        raise ValueError("Bundle output already exists.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved.get("format") != CHECKPOINT_FORMAT or saved.get("architecture") != ARCHITECTURE
            or not saved.get("deployment_only") or not saved.get("source_checkpoint_sha256")
            or saved.get("release_name") != RELEASE):
        raise ValueError("Expected the vimeml-tiny-zh-v1 deployment checkpoint with source provenance.")
    model, processor, _ = zh13.load(checkpoint, tokenizer_dir / "tokenizer.model")
    state = saved["model"]
    if not torch.equal(state["lm_head.weight"], state["token_embedding.weight"]):
        raise ValueError("Expected identical tied embedding/head tensors.")
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite learned weight.")
    if processor.vocab_size() != VOCAB or model.config.n_layers != LAYERS:
        raise ValueError("Unexpected zh-v1 vocabulary or layer count.")
    weights = {name: value.detach().cpu().float().contiguous() for name, value in state.items()
               if name != "lm_head.weight"}
    fresh_directory(output)
    torch.save(weights, output / "weights.pt")
    write_json(output / "config.json", model.configuration())
    shutil.copyfile(tokenizer_dir / "tokenizer.model", output / "tokenizer.model")
    identity = saved["tokenizer_identity"]
    result = {"format": BUNDLE_FORMAT, "status": "complete", "architecture": ARCHITECTURE,
              "release_name": RELEASE, "precision": "fp32", "parameter_count": model.parameter_count(),
              "vocab_size": VOCAB, "n_layers": LAYERS, "cache_shape": list(CACHE_SHAPE),
              "special_ids": dict(SPECIAL), "reserved_ids": identity.get("reserved_ids", {}),
              "checkpoint_sha256": saved["source_checkpoint_sha256"],
              "deployment_checkpoint_sha256": file_sha(checkpoint),
              "checkpoint_step": saved["step"], "checkpoint_epoch_index": saved["epoch"],
              "arm": saved["config"].get("arm"), "training_signatures": saved["signatures"],
              "tokenizer_identity": identity, "tied_lm_head": True,
              "files": v2.inventory(output, hashes=True), "environment": environment(),
              "policy": "Unique FP32 inference weights; no optimizer/RNG. BOS + joint context/candidate "
                        "tokenization; common-prefix suffix logP sum; no EOS/truncation."}
    write_json(output / "manifest.json", result)
    return result


def verify_bundle(path, hashes=True):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if (manifest.get("format") != BUNDLE_FORMAT or manifest.get("status") != "complete"
            or manifest.get("architecture") != ARCHITECTURE or not manifest.get("tied_lm_head")):
        raise ValueError("Unsupported/incomplete zh-v1 bundle.")
    if set(manifest["files"]) != {"weights.pt", "config.json", "tokenizer.model"}:
        raise ValueError("Unexpected inference file list.")
    for name, entry in manifest["files"].items():
        if (path / name).stat().st_size != entry["bytes"] or (hashes and file_sha(path / name) != entry["sha256"]):
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return manifest


class BundleLMZh:
    """PyTorch CPU FP32 model from a zh-v1 bundle; `.loaded()` feeds `zh13_lm_scores.run`."""
    def __init__(self, path):
        path = Path(path)
        manifest = verify_bundle(path, hashes=False)
        self.processor = spm.SentencePieceProcessor(model_file=str(path / "tokenizer.model"))
        self.special = {name: getattr(self.processor, f"{name}_id")() for name in ("pad", "unk", "bos", "eos")}
        self.model = TinyGPTZh13(GPTZh13Config(**read_json(path / "config.json"))).eval()
        state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        if "lm_head.weight" in state:
            raise ValueError("Shared embedding must be stored only once.")
        loaded = self.model.load_state_dict(state, strict=False)
        if loaded.missing_keys != ["lm_head.weight"] or loaded.unexpected_keys:
            raise ValueError(f"Unexpected inference weights: {loaded}")
        if (self.special != manifest["special_ids"] or self.special != SPECIAL
                or self.processor.vocab_size() != VOCAB or self.model.config.n_layers != LAYERS
                or self.model.parameter_count() != manifest["parameter_count"]):
            raise ValueError("Bundle model/tokenizer contract mismatch.")
        self.metadata = {"architecture": ARCHITECTURE, "release_name": RELEASE,
                         "checkpoint_sha256": manifest["checkpoint_sha256"],
                         "checkpoint_step": manifest["checkpoint_step"],
                         "deployment_checkpoint_sha256": manifest["deployment_checkpoint_sha256"],
                         "tokenizer_sha256": manifest["files"]["tokenizer.model"]["sha256"],
                         "bundle_manifest_sha256": file_sha(path / "manifest.json"),
                         "model": self.model.configuration(), "precision": "fp32", "device": "cpu",
                         "environment": environment()}

    def loaded(self):
        meta = {"checkpoint": None, "checkpoint_sha256": self.metadata["checkpoint_sha256"],
                "step": self.metadata["checkpoint_step"],
                "tokenizer_model_sha256": self.metadata["tokenizer_sha256"],
                "forward": {k: v for k, v in self.metadata.items() if k not in ("model", "environment")}}
        return self.model, self.processor, meta


def finish(output, model, fmt, metadata):
    output = Path(output)
    fresh_directory(output)
    model.save(str(output / "model.mlpackage"))
    files = v2.inventory(output / "model.mlpackage")
    write_json(output / "manifest.json", {"format": fmt, "status": "complete", "architecture": ARCHITECTURE,
        "release_name": RELEASE, **metadata, "environment": environment(), "spec": inspect_spec(model),
        "package_files": files, "package_bytes": sum(entry["bytes"] for entry in files.values()),
        "note": "Logical package bytes; compiled size, resident memory and device footprint are separate."})


def verify_package(path, fmt=PACKAGE_FORMAT):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if (manifest.get("format") != fmt or manifest.get("status") != "complete"
            or manifest.get("architecture") != ARCHITECTURE):
        raise ValueError("Incomplete/unsupported zh-v1 Core ML package.")
    if v2.inventory(path / "model.mlpackage") != manifest["package_files"]:
        raise ValueError("Package file inventory changed.")
    return manifest


def convert(bundle, output):
    ct = coremltools()
    if Path(output).exists():
        raise ValueError("Conversion output exists.")
    lm = BundleLMZh(bundle)
    model = ct.convert(trace_graph(lm.model), source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(lower_bound=1, upper_bound=CONTEXT, default=16)),
                              dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)], skip_model_load=True)
    model.short_description = "vimeml-tiny-zh-v1 RMSNorm/SwiGLU 12K L9; FP32 computation; no KV cache."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    finish(output, model, PACKAGE_FORMAT, {"kind": "fp32", "minimum_ios": 18, "bundle": lm.metadata,
        "compute_units": "CPU_ONLY",
        "interface": {"input_ids": "int32 [1,T], 1<=T<=128; right PAD",
                      "logits": f"float32 [1,T,{VOCAB}]; full vocabulary; no softmax"},
        "precision_policy": "All learned parameters and computation FP32; explicit causal attention.",
        "conversion_source_sha256": file_sha(Path(__file__))})


# Chunk schedules exercised before export: start 0, nonzero starts and the last slot.
TRACE_SCHEDULES = ([1, 3, 124], [128], [3, 1, 124], [124, 3, 1], [127, 1])


@torch.inference_mode()
def trace_chunk_graph(model):
    graph = KVChunkGraphV4(model).eval()
    zeros = (torch.zeros(CACHE_SHAPE), torch.zeros(CACHE_SHAPE))
    traced = torch.jit.trace(graph, (torch.tensor([[2, 4, 7]], dtype=torch.int32),
                                     torch.tensor([0], dtype=torch.int32), *zeros))
    ids = torch.randint(8, VOCAB, (1, CONTEXT), generator=torch.Generator().manual_seed(1111), dtype=torch.int32)
    expected = model(ids.long())
    for fn in (graph, traced):
        for schedule in TRACE_SCHEDULES:
            state, start, cache = zeros, 0, model.new_cache()
            for size in schedule:
                chunk = ids[:, start:start + size]
                actual, key, value = fn(chunk, torch.tensor([start], dtype=torch.int32), *state)
                reference = model.forward_cached(chunk, cache)
                torch.testing.assert_close(actual, expected[:, start:start + size], atol=3e-4, rtol=3e-4)
                torch.testing.assert_close(actual, reference, atol=3e-4, rtol=3e-4)
                torch.testing.assert_close(key, cache.key, atol=3e-4, rtol=3e-4)
                torch.testing.assert_close(value, cache.value, atol=3e-4, rtol=3e-4)
                state, start = (key, value), start + size
    return traced


def convert_chunk(bundle, output):
    ct = coremltools()
    if Path(output).exists():
        raise ValueError("Conversion output exists.")
    lm = BundleLMZh(bundle)
    model = ct.convert(trace_chunk_graph(lm.model), source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(lower_bound=1, upper_bound=CONTEXT, default=1)),
                              dtype=np.int32),
                ct.TensorType(name="position", shape=(1,), dtype=np.int32),
                ct.TensorType(name="key_cache", shape=CACHE_SHAPE, dtype=np.float32),
                ct.TensorType(name="value_cache", shape=CACHE_SHAPE, dtype=np.float32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32),
                 ct.TensorType(name="new_key_cache", dtype=np.float32),
                 ct.TensorType(name="new_value_cache", dtype=np.float32)], skip_model_load=True)
    model.short_description = "vimeml-tiny-zh-v1 chunked KV step; FP32 computation and caches."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    finish(output, model, CHUNK_FORMAT, {"kind": "fp32", "minimum_ios": 18, "bundle": lm.metadata,
        "cache_shape": list(CACHE_SHAPE), "cache_dtype": "float32", "compute_units": "CPU_ONLY",
        "cache_bytes_per_snapshot": 2 * int(np.prod(CACHE_SHAPE)) * 4,
        "interface": {"input_ids": "int32 [1,T], 1<=T<=128",
                      "position": "int32 [1]; current cache length; caller guarantees position+T<=128",
                      "key_cache/value_cache": "float32 (9,1,6,128,64); slots >= position+T are ignored",
                      "logits": f"float32 [1,T,{VOCAB}] for the new tokens; no softmax",
                      "new_key_cache/new_value_cache": "inputs with slots position..position+T-1 replaced"},
        "cache_write": "index_put (scatter) as in V2.1 kv_cache.CacheGraph; static causal mask gathered by absolute position",
        "graph": "vimeml.deployment.v4_chunk.KVChunkGraphV4 (layer-count agnostic)",
        "trace_schedules": [list(s) for s in TRACE_SCHEDULES],
        "precision_policy": "All learned parameters, computation and caches FP32.",
        "policy": "Batch1; caller owns immutable snapshots and rollback; no sliding or cross-request retention.",
        "conversion_source_sha256": file_sha(Path(__file__))})


def _fp16_embedding(model, ct, name=EMBEDDING):
    """Store the tied embedding/head constant as FP16 behind constexpr_cast (FP32 compute unchanged)."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.milproto.load import load as load_milproto
    from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline
    spec = model.get_spec()
    program = load_milproto(spec, specification_version=spec.specificationVersion,
                            file_weights_dir=model.weights_dir)
    replaced = []
    for function in program.functions.values():
        for op in list(function.operations):
            if op.op_type == "const" and op.outputs[0].name == name:
                value = op.outputs[0].val
                if value.dtype != np.float32:
                    raise ValueError("Expected the FP32 tied embedding constant.")
                with function:
                    cast = mb.constexpr_cast(source_val=value.astype(np.float16), output_dtype="fp32",
                                             before_op=op, name=name + "_fp16")
                function.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=cast,
                                                      no_check_var_types=True)
                function.remove_ops([op])
                replaced.append({"name": name, "shape": list(value.shape),
                                 "max_abs_fp16_rounding": float(np.abs(value - value.astype(np.float16)
                                                                       .astype(np.float32)).max())})
    if len(replaced) != 1:
        raise ValueError(f"Expected exactly one tied embedding constant, found {len(replaced)}.")
    # Re-state the stateless interface: a MIL program does not carry the flexible T range.
    result = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32, pass_pipeline=PassPipeline.EMPTY, skip_model_load=True,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(lower_bound=1, upper_bound=CONTEXT, default=16)),
                              dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)])
    result.short_description = model.short_description
    result.user_defined_metadata.update(model.user_defined_metadata)
    if result.get_spec().description.input[0] != model.get_spec().description.input[0]:
        raise ValueError("FP16-embedding rewrite changed the stateless input contract.")
    return result, replaced[0]


def compress(source, output, alignment, recipe=DEPLOYED_RECIPE, chunk=False):
    """Compress an aligned FP32 package. `int8-b32` is the deployed recipe; others are survey-only."""
    ct = coremltools()
    source, output, alignment = map(Path, (source, output, alignment))
    if output.exists():
        raise ValueError("Compression output exists.")
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe {recipe}")
    if chunk and recipe != DEPLOYED_RECIPE:
        raise ValueError("The quantization survey is stateless only.")
    fmt = CHUNK_FORMAT if chunk else PACKAGE_FORMAT
    original = verify_package(source, fmt)
    gate = read_json(alignment)
    expected = CHUNK_ALIGNMENT_FORMAT if chunk else ALIGNMENT_FORMAT
    if (original["kind"] != "fp32" or gate.get("format") != expected or not gate.get("passed")
            or gate.get("coreml_manifest_sha256") != file_sha(source / "manifest.json")):
        raise ValueError("First pass alignment for this exact uncompressed zh-v1 FP32 package.")
    from coremltools.optimize import coreml as opt
    model = ct.models.MLModel(str(source / "model.mlpackage"), skip_model_load=True)
    spec = RECIPES[recipe]
    if spec["method"] == "linear_symmetric":
        config = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=f"int{spec['bits']}",
            granularity="per_block", block_size=spec["block_size"], weight_threshold=WEIGHT_THRESHOLD)
    else:
        config = opt.OpPalettizerConfig(mode="kmeans", nbits=spec["bits"], granularity=spec["granularity"],
                                        group_size=spec["group_size"], weight_threshold=WEIGHT_THRESHOLD)
    optimization, selection = v2.parameter_selection(opt, model, config)
    embedding = None
    if spec.get("tied_embedding") == "fp16":
        names = [n for n in optimization.op_name_configs if n.replace(".", "_") == EMBEDDING]
        if len(names) != 1:
            raise ValueError("Tied embedding constant not found in the parameter selection.")
        optimization = opt.OptimizationConfig(global_config=None, op_name_configs={
            k: v for k, v in optimization.op_name_configs.items() if k != names[0]})
        selection["selected_constants"] = [s for s in selection["selected_constants"] if s["name"] != names[0]]
        selection["fp16_constants"] = names
        selection["selected_elements"] = sum(s["elements"] for s in selection["selected_constants"])
    print(f"{recipe}: selected {len(selection['selected_constants'])} parameter matrices; "
          f"excluded {len(selection['excluded_constants'])} structural constants.", flush=True)
    if spec["method"] == "linear_symmetric":
        result = opt.linear_quantize_weights(model, config=optimization)
    else:
        result = opt.palettize_weights(model, config=optimization)
    if spec.get("tied_embedding") == "fp16":
        result, embedding = _fp16_embedding(result, ct)
    if not any(name.startswith("constexpr_") for name in inspect_spec(result)["operations"]):
        raise ValueError("No compressed weights found.")
    carried = {key: original[key] for key in original if key in (
        "minimum_ios", "bundle", "interface", "cache_shape", "cache_dtype", "compute_units",
        "cache_bytes_per_snapshot", "cache_write", "graph", "policy")}
    finish(output, result, fmt, {**carried, "kind": f"{recipe}_fp32_compute", "recipe": recipe,
        "deployed_recipe": recipe == DEPLOYED_RECIPE, "survey_only": recipe != DEPLOYED_RECIPE,
        "source_manifest_sha256": file_sha(source / "manifest.json"),
        "fp32_alignment_sha256": file_sha(alignment), "conversion_source_sha256": file_sha(Path(__file__)),
        "compression": {**spec, "weight_threshold": WEIGHT_THRESHOLD, "selection": selection,
                        **({"fp16_embedding": embedding} if embedding else {})},
        "precision_policy": "Compressed parameter matrices, FP32 compute; norms/masks unchanged; "
                            "no activation quantization."})


class CoreMLLMZh(BundleLMZh):
    def __init__(self, bundle, directory):
        super().__init__(bundle)
        manifest = verify_package(directory)
        if manifest["bundle"]["bundle_manifest_sha256"] != self.metadata["bundle_manifest_sha256"]:
            raise ValueError("Core ML package belongs to another bundle.")
        self.model = CoreMLForward(directory, self.model.config, "CPU_ONLY")
        self.metadata.update(precision=manifest["kind"], compute_units="CPU_ONLY", graph="stateless",
                             package=Path(directory).name,
                             coreml_manifest_sha256=file_sha(Path(directory) / "manifest.json"))


def empty_cache():
    return tuple(np.zeros(CACHE_SHAPE, dtype=np.float32) for _ in range(2))


class ChunkRuntimeZh:
    """Chunked Core ML KV calls. Inputs are never modified; outputs are owned copies."""
    def __init__(self, bundle, directory):
        lm = BundleLMZh(bundle)
        self.config, self.metadata = lm.model.config, dict(lm.metadata)
        self.manifest = verify_package(directory, CHUNK_FORMAT)
        if self.manifest["bundle"]["bundle_manifest_sha256"] != self.metadata["bundle_manifest_sha256"]:
            raise ValueError("KV chunk model belongs to another bundle.")
        if (self.manifest["cache_shape"] != list(CACHE_SHAPE) or self.manifest["cache_dtype"] != "float32"
                or self.manifest["compute_units"] != "CPU_ONLY"):
            raise ValueError("KV chunk package cache/compute contract differs.")
        ct = coremltools()
        self.model = ct.models.MLModel(str(Path(directory) / "model.mlpackage"),
                                       compute_units=ct.ComputeUnit.CPU_ONLY)
        self.metadata.update(precision=self.manifest["kind"], compute_units="CPU_ONLY", graph="kv_chunk",
                             package=Path(directory).name,
                             coreml_manifest_sha256=file_sha(Path(directory) / "manifest.json"))
        self.calls = 0

    def step(self, ids, position, state=None):
        ids = np.asarray(ids).reshape(-1)
        if (not ids.size or not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any() or (ids >= VOCAB).any()
                or type(position) is not int or position < 0 or position + ids.size > CONTEXT
                or (state is None and position != 0)):
            raise ValueError("Invalid KV chunk input/position.")
        state = empty_cache() if state is None else state
        if len(state) != 2 or any(not isinstance(a, np.ndarray) or a.shape != CACHE_SHAPE or a.dtype != np.float32
                                  for a in state):
            raise ValueError("Invalid cache tensor.")
        result = self.model.predict({"input_ids": ids.astype(np.int32).reshape(1, -1),
                                     "position": np.array([position], np.int32),
                                     "key_cache": state[0], "value_cache": state[1]})
        self.calls += 1
        logits = np.array(result["logits"], dtype=np.float32, copy=True)
        caches = tuple(np.array(result[name], dtype=np.float32, copy=True)
                       for name in ("new_key_cache", "new_value_cache"))
        if (logits.shape != (1, ids.size, VOCAB) or not np.isfinite(logits).all()
                or any(a.shape != CACHE_SHAPE for a in caches)):
            raise ValueError("Invalid Core ML cache/logits output.")
        return logits, caches

    def run(self, ids, state=None, start=0, chunks=None):
        """Process `ids` from `start` in the given chunk sizes (default: one call)."""
        ids = [int(i) for i in np.asarray(ids).reshape(-1)]
        chunks = chunks or [len(ids)]
        if sum(chunks) != len(ids) or any(size < 1 for size in chunks):
            raise ValueError("Chunk sizes must cover the tokens.")
        parts, offset = [], 0
        for size in chunks:
            logits, state = self.step(ids[offset:offset + size], start + offset, state)
            parts.append(logits)
            offset += size
        return np.concatenate(parts, axis=1), state


class CachedChunkForwardZh:
    """Scoring adapter: one chunk call for the exact common token prefix of the batch,
    then one chunk call per row suffix from that immutable snapshot.

    Rows arrive right-padded with PAD (id 0, never produced by encode). Trailing PAD
    positions are not computed; their logits are zero and never scored.
    """
    def __init__(self, runtime, pad):
        self.runtime, self.config, self.pad = runtime, runtime.config, pad

    def __call__(self, inputs):
        rows = [list(row) for row in inputs.cpu().numpy().astype(np.int64)]
        width = len(rows[0])
        for row in rows:
            while len(row) > 1 and row[-1] == self.pad:
                row.pop()
        common = common_prefix_length(rows)
        prefix = state = None
        if common:
            prefix, state = self.runtime.run(rows[0][:common])
        output = np.zeros((len(rows), width, VOCAB), dtype=np.float32)
        for index, row in enumerate(rows):
            if common == len(row):
                logits = prefix
            elif common:
                suffix, _ = self.runtime.run(row[common:], state, common)
                logits = np.concatenate((prefix, suffix), axis=1)
            else:
                logits, _ = self.runtime.run(row)
            output[index, :len(row)] = logits[0]
        return torch.from_numpy(output)


class CachedChunkLMZh(BundleLMZh):
    def __init__(self, bundle, directory):
        super().__init__(bundle)
        self.runtime = ChunkRuntimeZh(bundle, directory)
        self.model = CachedChunkForwardZh(self.runtime, self.special["pad"])
        self.metadata = self.runtime.metadata


def read_reference(reference, lm):
    reference = Path(reference)
    info = read_json(reference / "reference.json")
    if (info.get("format") != REFERENCE_FORMAT or info["checkpoint_step"] != lm.metadata["checkpoint_step"]
            or info["checkpoint_sha256"] != lm.metadata["checkpoint_sha256"]
            or info["tokenizer_sha256"] != lm.metadata["tokenizer_sha256"]):
        raise ValueError("Reference belongs to another checkpoint/tokenizer.")
    with np.load(reference / "logits.npz", allow_pickle=False) as arrays:
        return info, {name: arrays[name] for name in arrays.files}


@torch.inference_mode()
def align(lm, reference, output):
    """Stateless fixture gate: valid prefixes within 3e-4 and padding/future invariants."""
    output = Path(output)
    if output.exists():
        raise ValueError("Alignment output exists.")
    info, arrays = read_reference(reference, lm)
    errors, actuals = [], {}
    for name, case in info["cases"].items():
        ids = arrays[name + "_input_ids"]
        actual = lm.model(torch.from_numpy(ids.copy()).long()).numpy()
        actuals[name] = actual
        valid = case["valid_prefix"]
        expected = arrays[name + "_logits"]
        error = error_distribution(expected[:, :valid], actual[:, :valid])
        error.update(id=name, length=case["length"], valid_prefix=valid,
                     last_top1_equal=int(actual[0, valid - 1].argmax()) == int(expected[0, valid - 1].argmax()))
        errors.append(error)
    invariants = {name: numeric_error(actuals["short"], actuals[name][:, :16], TOLERANCE, TOLERANCE)
                  for name in ("right_pad", "changed_future")}
    report = {"format": ALIGNMENT_FORMAT,
        "passed": all(item["outside_tolerance"] == 0 for item in [*errors, *invariants.values()]),
        "model": lm.metadata, "coreml_manifest_sha256": lm.metadata.get("coreml_manifest_sha256"),
        "reference": info, "tolerances": {"atol": TOLERANCE, "rtol": TOLERANCE},
        "summary": summarize_errors(errors), "logits": errors, "invariants": invariants,
        "note": "Strict PyTorch FP32 fixture comparison. Quantized ranking quality is evaluated separately; failures are preserved."}
    fresh_directory(output)
    write_json(output / "alignment.json", report)
    return report


@torch.inference_mode()
def align_chunk(runtime, bundle, reference, output, stateless=None):
    """Chunk-KV gate vs frozen PyTorch fixtures, or vs a same-precision stateless package.

    Patterns per case: one call, single tokens, [3, n-3] and half/half; `long` single
    tokens also vs `long_kv_step_logits`. Branches share an immutable snapshot; A/B/A
    rollback and a dirty unused cache tail must be exact.
    """
    output = Path(output)
    if output.exists():
        raise ValueError("Alignment output exists.")
    lm = BundleLMZh(bundle)
    info, arrays = read_reference(reference, lm)
    old = None
    if stateless:
        old = CoreMLLMZh(bundle, stateless)
        if old.metadata["precision"] != runtime.metadata["precision"]:
            raise ValueError("Compare chunk KV with the stateless package of the same precision.")
    full = (lambda ids: old.model(torch.tensor([list(ids)])).numpy()) if old else (
        lambda ids: lm.model(torch.tensor([list(ids)])).numpy())
    errors, invariants, saved = [], [], {}
    for name, case in info["cases"].items():
        ids = [int(i) for i in arrays[name + "_input_ids"][0]]
        n, valid = len(ids), case["valid_prefix"]
        expected = full(ids) if old else arrays[name + "_logits"]
        patterns = {"one_shot": [n], "single_tokens": [1] * n}
        if n > 3:
            patterns.update({"split_3": [3, n - 3], "half_half": [n // 2, n - n // 2]})
        outputs = {}
        for label, chunks in patterns.items():
            actual, _ = runtime.run(ids, chunks=chunks)
            outputs[label] = actual
            errors.append({"case": name, "check": label, "chunks": chunks if label != "single_tokens" else f"[1]*{n}",
                           **error_distribution(expected[:, :valid], actual[:, :valid])})
        saved[name] = outputs["one_shot"]
        for label in patterns:
            if label != "one_shot":
                invariants.append({"case": name, "check": f"{label}_vs_one_shot_within_tolerance",
                                   **numeric_error(outputs["one_shot"], outputs[label], TOLERANCE, TOLERANCE)})
        if name == "long" and not old:
            errors.append({"case": name, "check": "single_tokens_vs_forward_cached_steps",
                           **error_distribution(arrays["long_kv_step_logits"], outputs["single_tokens"])})
    for name in ("right_pad", "changed_future"):
        invariants.append({"case": name, "check": "valid_prefix_vs_short",
                           **numeric_error(saved["short"], saved[name][:, :16], TOLERANCE, TOLERANCE)})
    prefix = [int(i) for i in arrays["short_input_ids"][0][:7]]
    _, state = runtime.run(prefix)
    original = tuple(a.copy() for a in state)
    branches = []
    for suffix in ([9], [47, 81, 102], [9]):
        actual, _ = runtime.run(suffix, state, len(prefix))
        errors.append({"check": "branch", "suffix": suffix,
                       **error_distribution(full([*prefix, *suffix])[:, len(prefix):], actual)})
        branches.append(actual)
    immutable = all(np.array_equal(a, b) for a, b in zip(state, original, strict=True))
    invariants.append({"check": "rollback_A_B_A", **numeric_error(branches[0], branches[2], 0, 0)})
    dirty = tuple(a.copy() for a in state)
    for a in dirty:
        a[:, :, :, len(prefix):, :] = 999
    for suffix in ([9], [47, 81, 102]):
        clean, _ = runtime.run(suffix, state, len(prefix))
        polluted, _ = runtime.run(suffix, dirty, len(prefix))
        invariants.append({"check": "unused_tail", "suffix": suffix, **numeric_error(clean, polluted, 0, 0)})
    report = {"format": CHUNK_ALIGNMENT_FORMAT,
        "passed": immutable and all(e["outside_tolerance"] == 0 for e in errors + invariants),
        "coreml_manifest_sha256": runtime.metadata["coreml_manifest_sha256"], "model": runtime.metadata,
        "comparison": "same-weight stateless Core ML" if old else "frozen PyTorch FP32 fixtures",
        "baseline_manifest_sha256": old.metadata["coreml_manifest_sha256"] if old else None,
        "tolerances": {"atol": TOLERANCE, "rtol": TOLERANCE}, "immutable_branch_inputs": immutable,
        "summary": summarize_errors(errors), "errors": errors, "invariants": invariants,
        "kv_calls": runtime.calls}
    fresh_directory(output)
    write_json(output / "alignment.json", report)
    return report


def read_scores(path):
    return {row["id"]: row for row in zh13.read_jsonl(path)}


def ranking(row):
    """Pool indices by descending score; ties keep pool order (as score_standard_ime_zh)."""
    scores = row.get("scores")
    if not scores:
        return list(range(len(row["candidates"])))
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def rank_differences(before_path, after_path):
    """Per-item ranking differences between two score files; ids only, no benchmark text."""
    before, after = read_scores(before_path), read_scores(after_path)
    if set(before) != set(after):
        raise ValueError("Score files cover different items.")
    order_changed, top1_changed, max_delta = [], [], 0.0
    for item_id in sorted(before):
        a, b = before[item_id], after[item_id]
        if a["candidates"] != b["candidates"]:
            raise ValueError(f"Candidate pools differ: {item_id}")
        if a.get("scores"):
            max_delta = max(max_delta, max(abs(x - y) for x, y in zip(a["scores"], b["scores"], strict=True)))
        ra, rb = ranking(a), ranking(b)
        if ra != rb:
            order_changed.append(item_id)
            if ra[0] != rb[0]:
                top1_changed.append(item_id)
    return {"items": len(before), "order_changed_items": len(order_changed), "order_changed_ids": order_changed,
            "top1_changed_items": len(top1_changed), "top1_changed_ids": top1_changed,
            "max_score_abs_delta": max_delta}


def item_outcomes(result_dir):
    """id -> exact Top-1 correctness from score_standard_ime_zh.py `items.jsonl`."""
    rows = zh13.read_jsonl(Path(result_dir) / "items.jsonl")
    return {row["id"]: row["rank"] == 1 for row in rows}


def changed_items(before_dir, after_dir):
    before, after = item_outcomes(before_dir), item_outcomes(after_dir)
    if set(before) != set(after):
        raise ValueError("Result directories cover different items.")
    return {"improved_ids": sorted(i for i in before if after[i] and not before[i]),
            "regressed_ids": sorted(i for i in before if before[i] and not after[i])}


def score_development(lm, frozen, output, tracks=("full", "abbrev")):
    """Write `zh13_lm_scores.run` score files for `lm` (PyTorch or Core ML) on development."""
    output = Path(output)
    summaries = {}
    for track in tracks:
        summaries[track] = zh13.run(None, None, frozen, output / f"{track}.jsonl", track, loaded=lm.loaded())
        print(json.dumps({"track": track, "items": summaries[track]["items"],
                          "scores_sha256": summaries[track]["scores_sha256"]}), flush=True)
    return summaries
