"""V4 chunked KV graph: T new tokens at absolute positions with explicit FP32 caches.

input_ids int32[1,T] (1<=T<=128), position int32[1] (current cache length,
position+T<=128), key_cache/value_cache float32 (7,1,6,128,64) -> logits
float32[1,T,8192] for the new tokens plus updated caches. Cache writes reuse the
V2.1 `kv_cache.CacheGraph` index_put pattern; a static causal mask gathered by
absolute query position hides future and unused tail slots. The single-token
`KVStepGraphV4`, its packages and records are unchanged.
"""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vimeml.deployment import v2
from vimeml.deployment import v4_coreml as base
from vimeml.deployment.bundle import fresh_directory, read_json, write_json
from vimeml.deployment.coreml import coremltools, inspect_spec
from vimeml.deployment.v4 import CACHE_SHAPE
from vimeml.deployment.validation import numeric_error
from vimeml.training.data import file_sha

FORMAT = "vimeml_coreml_v4_kv_chunk_v1"
ALIGNMENT_FORMAT = "vimeml_kv_chunk_alignment_v4"


class KVChunkGraphV4(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        mask = torch.full((128, 128), -float("inf"))
        self.register_buffer("causal_mask", torch.triu(mask, diagonal=1))

    def forward(self, input_ids, position, key_cache, value_cache):
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device, dtype=torch.int32) + position[0]
        hidden = self.model.token_embedding(input_ids) + self.model.position_embedding(positions)
        # Rows are absolute query positions: slot s is visible to p+i iff s <= p+i.
        mask = self.causal_mask[positions.long()]
        indices = (positions.long(),)
        keys, values = [], []
        for layer, block in enumerate(self.model.blocks):
            qkv = block.attention.qkv(block.attention_norm(hidden)).reshape(1, length, 3, 6, 64)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            key = key_cache[layer].permute(2, 0, 1, 3).index_put(indices, key.permute(2, 0, 1, 3)).permute(1, 2, 0, 3)
            value = value_cache[layer].permute(2, 0, 1, 3).index_put(indices, value.permute(2, 0, 1, 3)).permute(1, 2, 0, 3)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(64) + mask
            attended = torch.matmul(torch.softmax(scores, dim=-1), value).transpose(1, 2).reshape(1, length, 384)
            hidden = hidden + block.attention.projection(attended)
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
            keys.append(key)
            values.append(value)
        return self.model.lm_head(self.model.final_norm(hidden)), torch.stack(keys), torch.stack(values)


# Chunk schedules exercised before export: start 0, nonzero starts and the last slot.
TRACE_SCHEDULES = ([1, 3, 124], [128], [3, 1, 124], [124, 3, 1], [127, 1])


@torch.inference_mode()
def trace_kv_chunk_graph(model):
    graph = KVChunkGraphV4(model).eval()
    zeros = (torch.zeros(CACHE_SHAPE), torch.zeros(CACHE_SHAPE))
    traced = torch.jit.trace(graph, (torch.tensor([[2, 4, 7]], dtype=torch.int32),
                                     torch.tensor([0], dtype=torch.int32), *zeros))
    ids = torch.randint(7, 8192, (1, 128), generator=torch.Generator().manual_seed(1020), dtype=torch.int32)
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


def convert(bundle, output):
    ct = coremltools()
    if Path(output).exists():
        raise ValueError("Conversion output exists.")
    lm = base.BundleLMV4(bundle)
    model = ct.convert(trace_kv_chunk_graph(lm.model), source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(lower_bound=1, upper_bound=128, default=1)),
                              dtype=np.int32),
                ct.TensorType(name="position", shape=(1,), dtype=np.int32),
                ct.TensorType(name="key_cache", shape=CACHE_SHAPE, dtype=np.float32),
                ct.TensorType(name="value_cache", shape=CACHE_SHAPE, dtype=np.float32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32),
                 ct.TensorType(name="new_key_cache", dtype=np.float32),
                 ct.TensorType(name="new_value_cache", dtype=np.float32)], skip_model_load=True)
    model.short_description = "VimeML V4 chunked KV step; FP32 computation and caches."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    base.finish(output, model, FORMAT, {"kind": "fp32", "minimum_ios": 18, "bundle": lm.metadata,
        "cache_shape": list(CACHE_SHAPE), "cache_dtype": "float32", "compute_units": "CPU_ONLY",
        "cache_bytes_per_snapshot": 2 * int(np.prod(CACHE_SHAPE)) * 4,
        "interface": {"input_ids": "int32 [1,T], 1<=T<=128",
                      "position": "int32 [1]; current cache length; caller guarantees position+T<=128",
                      "key_cache/value_cache": "float32 (7,1,6,128,64); slots >= position+T are ignored",
                      "logits": "float32 [1,T,8192] for the new tokens; no softmax",
                      "new_key_cache/new_value_cache": "inputs with slots position..position+T-1 replaced"},
        "cache_write": "index_put (scatter) as in V2.1 kv_cache.CacheGraph; static causal mask gathered by absolute position",
        "trace_schedules": [list(s) for s in TRACE_SCHEDULES],
        "precision_policy": "All learned parameters, computation and caches FP32.",
        "policy": "Batch1; caller owns immutable snapshots and rollback; no sliding or cross-request retention.",
        "conversion_source_sha256": file_sha(Path(__file__))})


def compress(source, output, alignment):
    ct = coremltools()
    source, output, alignment = map(Path, (source, output, alignment))
    if output.exists():
        raise ValueError("Compression output exists.")
    original = base.verify_package(source, FORMAT)
    gate = read_json(alignment)
    if (original["kind"] != "fp32" or gate.get("format") != ALIGNMENT_FORMAT or not gate.get("passed")
            or gate.get("coreml_manifest_sha256") != file_sha(source / "manifest.json")):
        raise ValueError("First pass alignment for this exact uncompressed V4 chunk FP32 package.")
    from coremltools.optimize import coreml as opt
    model = ct.models.MLModel(str(source / "model.mlpackage"), skip_model_load=True)
    config = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                        granularity="per_block", block_size=32, weight_threshold=2048)
    optimization, selection = v2.parameter_selection(opt, model, config)
    print(f"Selected {len(selection['selected_constants'])} parameter matrices; "
          f"excluded {len(selection['excluded_constants'])} structural constants.", flush=True)
    result = opt.linear_quantize_weights(model, config=optimization)
    if not any(name.startswith("constexpr_") for name in inspect_spec(result)["operations"]):
        raise ValueError("No compressed weights found.")
    carried = {key: original[key] for key in ("minimum_ios", "bundle", "interface", "cache_shape", "cache_dtype",
                                              "compute_units", "cache_bytes_per_snapshot", "cache_write", "policy")}
    base.finish(output, result, FORMAT, {**carried, "kind": "linear8_fp32_compute",
        "source_manifest_sha256": file_sha(source / "manifest.json"),
        "fp32_alignment_sha256": file_sha(alignment), "conversion_source_sha256": file_sha(Path(__file__)),
        "compression": {"method": "linear_symmetric", "bits": 8, "block_size": 32, "granularity": "per_block",
                        "weight_threshold": 2048, "selection": selection},
        "precision_policy": "INT8 parameter matrices, FP32 compute; norms/masks unchanged; no activation quantization."})


class KVChunkRuntimeV4:
    """Chunked Core ML KV calls. Inputs are never modified; outputs are owned copies."""
    def __init__(self, bundle, directory):
        lm = base.BundleLMV4(bundle)
        self.config, self.metadata = lm.model.config, dict(lm.metadata)
        self.manifest = base.verify_package(directory, FORMAT)
        if self.manifest["bundle"]["bundle_manifest_sha256"] != self.metadata["bundle_manifest_sha256"]:
            raise ValueError("KV chunk model belongs to another bundle.")
        if (self.manifest["cache_shape"] != list(CACHE_SHAPE) or self.manifest["cache_dtype"] != "float32"
                or self.manifest["compute_units"] != "CPU_ONLY"):
            raise ValueError("KV chunk package cache/compute contract differs.")
        ct = coremltools()
        self.model = ct.models.MLModel(str(Path(directory) / "model.mlpackage"),
                                       compute_units=ct.ComputeUnit.CPU_ONLY)
        self.metadata.update(precision=self.manifest["kind"], compute_units="CPU_ONLY", graph="kv_chunk",
                             coreml_manifest_sha256=file_sha(Path(directory) / "manifest.json"))
        self.calls = 0

    def step(self, ids, position, state=None):
        ids = np.asarray(ids).reshape(-1)
        if (not ids.size or not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any() or (ids >= base.VOCAB).any()
                or type(position) is not int or position < 0 or position + ids.size > 128
                or (state is None and position != 0)):
            raise ValueError("Invalid KV chunk input/position.")
        state = base.empty_cache() if state is None else state
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
        if (logits.shape != (1, ids.size, base.VOCAB) or not np.isfinite(logits).all()
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


class CachedChunkForwardV4(base.CachedForwardV4):
    """One chunk call for the exact common prefix, then one call per candidate suffix."""


class CachedChunkLMV4(base.BundleLMV4):
    def __init__(self, bundle, directory):
        super().__init__(bundle)
        self.runtime = KVChunkRuntimeV4(bundle, directory)
        self.model = CachedChunkForwardV4(self.runtime, self.special["pad"])
        self.metadata = self.runtime.metadata


@torch.inference_mode()
def align(runtime, bundle, reference, output, stateless=None, single=None):
    """Chunk KV gate vs frozen PyTorch fixtures, or vs a same-precision stateless package.

    Also compares T=1 chunk decode with the existing single-token KV package when given.
    """
    output = Path(output)
    if output.exists():
        raise ValueError("Alignment output exists.")
    lm = base.BundleLMV4(bundle)
    info, arrays = base.read_reference(reference, lm)
    old = None
    if stateless:
        old = base.CoreMLLMV4(bundle, stateless)
        if old.metadata["precision"] != runtime.metadata["precision"]:
            raise ValueError("Compare chunk KV with the stateless package of the same precision.")
    step = base.KVRuntimeV4(bundle, single) if single else None
    if step and step.metadata["precision"] != runtime.metadata["precision"]:
        raise ValueError("Compare with the single-token package of the same precision.")
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
                           **base.error_distribution(expected[:, :valid], actual[:, :valid])})
        saved[name] = outputs["one_shot"]
        for label in patterns:
            if label != "one_shot":
                invariants.append({"case": name, "check": f"{label}_vs_one_shot_within_tolerance",
                                   **numeric_error(outputs["one_shot"], outputs[label], base.TOLERANCE, base.TOLERANCE)})
        if name == "long" and not old:
            errors.append({"case": name, "check": "single_tokens_vs_forward_cached_steps",
                           **base.error_distribution(arrays["long_kv_step_logits"], outputs["single_tokens"])})
        if step:
            reference_steps, _ = step.run(ids)
            errors.append({"case": name, "check": "T1_decode_vs_single_token_package",
                           **base.error_distribution(reference_steps[:, :valid], outputs["single_tokens"][:, :valid])})
    for name in ("right_pad", "changed_future"):
        invariants.append({"case": name, "check": "valid_prefix_vs_short",
                           **numeric_error(saved["short"], saved[name][:, :16], base.TOLERANCE, base.TOLERANCE)})
    prefix = [int(i) for i in arrays["short_input_ids"][0][:7]]
    _, state = runtime.run(prefix)
    original = tuple(a.copy() for a in state)
    branches = []
    for suffix in ([9], [47, 81, 102], [9]):
        actual, _ = runtime.run(suffix, state, len(prefix))
        errors.append({"check": "branch", "suffix": suffix,
                       **base.error_distribution(full([*prefix, *suffix])[:, len(prefix):], actual)})
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
    report = {"format": ALIGNMENT_FORMAT,
        "passed": immutable and all(e["outside_tolerance"] == 0 for e in errors + invariants),
        "coreml_manifest_sha256": runtime.metadata["coreml_manifest_sha256"], "model": runtime.metadata,
        "comparison": "same-weight stateless Core ML" if old else "frozen PyTorch FP32 fixtures",
        "baseline_manifest_sha256": old.metadata["coreml_manifest_sha256"] if old else None,
        "single_token_manifest_sha256": step.metadata["coreml_manifest_sha256"] if step else None,
        "tolerances": {"atol": base.TOLERANCE, "rtol": base.TOLERANCE}, "immutable_branch_inputs": immutable,
        "summary": base.summarize_errors(errors), "errors": errors, "invariants": invariants,
        "kv_calls": runtime.calls}
    fresh_directory(output)
    write_json(output / "alignment.json", report)
    return report
