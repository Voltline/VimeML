"""vimeml-tiny-zh-v1 batched single-step KV decoding for constrained beam search.

One call advances B independent hypotheses by one token each. Rows may have different
cache lengths (different tokenization paths consume different numbers of tokens); row b
attends only to its own slots 0..position[b].

Interface (Core ML names):
  input_ids    int32 [B,1]             one new token per row
  position     int32 [B]               row cache length before this token; position[b] < S
  key_cache    float32 (9,B,6,S,64)    row b slots >= position[b] are ignored (may hold garbage)
  value_cache  float32 (9,B,6,S,64)
  -> logits    float32 [B,12288]       last (only) position, no softmax
     new_key   float32 (9,B,6,64)      key/value of the new token; the caller writes them into
     new_value float32 (9,B,6,64)      slot position[b] of its own row buffer

S (cache slots passed in) is flexible, 1<=S<=128: the caller passes only max(position)+1 slots,
so input bytes and attention cost follow the real lengths instead of the 128-slot context.
Returning only the new slot avoids writing a full cache copy per call (B=16, S=128 would be
28 MiB per cache). B is flexible 1..16, or fixed (8, 16) in the fixed variants.

The prompt is not handled here: run it once with the existing batch-1 chunk package (or the
`prefill` function of the merged package) and copy its cache rows into the batch buffer.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vimeml.deployment import v2
from vimeml.deployment import zh_v1
from vimeml.deployment.bundle import read_json, write_json
from vimeml.deployment.coreml import coremltools, inspect_spec
from vimeml.deployment.v4_chunk import KVChunkGraphV4
from vimeml.training.data import file_sha

FORMAT = "vimeml_coreml_zh_v1_kv_batch"
ALIGNMENT_FORMAT = "vimeml_kv_batch_alignment_zh_v1"
MERGED_FORMAT = "vimeml_coreml_zh_v1_kv_multifunction"
VOCAB, LAYERS, CONTEXT, HEADS, HEAD_DIM = zh_v1.VOCAB, zh_v1.LAYERS, zh_v1.CONTEXT, 6, 64
MAX_BATCH = 16
TOLERANCE = zh_v1.TOLERANCE
MASKED = -1e9


class KVBatchGraphZh(nn.Module):
    """Batched one-token step over per-row caches; see the module docstring."""

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, input_ids, position, key_cache, value_cache):
        batch, slots = input_ids.shape[0], key_cache.shape[3]
        index = torch.arange(slots, dtype=torch.int32, device=input_ids.device)
        hidden = self.model.token_embedding(input_ids[:, 0]) + self.model.position_embedding(position)
        hidden = hidden[:, None, :]                                               # [B,1,384]
        write = (index[None, :] == position[:, None]).to(hidden.dtype)[:, None, :, None]   # [B,1,S,1]
        mask = (index[None, :] > position[:, None]).to(hidden.dtype)[:, None, None, :] * MASKED  # [B,1,1,S]
        keys, values = [], []
        for layer, block in enumerate(self.model.blocks):
            qkv = block.attention.qkv(block.attention_norm(hidden)).reshape(batch, 1, 3, HEADS, HEAD_DIM)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)              # [B,6,1,64]
            full_key = key_cache[layer] * (1 - write) + key * write                # [B,6,S,64]
            full_value = value_cache[layer] * (1 - write) + value * write
            scores = torch.matmul(query, full_key.transpose(-2, -1)) / math.sqrt(HEAD_DIM) + mask
            attended = torch.matmul(torch.softmax(scores, dim=-1), full_value)     # [B,6,1,64]
            hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(batch, 1, 384))
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
            keys.append(key[:, :, 0])
            values.append(value[:, :, 0])
        logits = self.model.lm_head(self.model.final_norm(hidden[:, 0]))
        return logits, torch.stack(keys), torch.stack(values)


def _sequences(rows, lengths, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(8, VOCAB, (rows, max(lengths)), generator=g, dtype=torch.int32)


ROW_LENGTHS = lambda batch: [3 + (7 * b) % 23 for b in range(batch)]     # 3..25 tokens per row
ROW_OFFSETS = lambda batch: [(5 * b) % 9 for b in range(batch)]         # rows start at different steps


@torch.inference_mode()
def check_graph(model, fn, batch=MAX_BATCH, seed=1115, slots=None, strict=True):
    """Step `batch` rows of different lengths through `fn` from empty caches. Rows start at different
    steps, so positions differ within every call. Each step's logits must match `model`'s full forward
    of that row's own sequence (`model(ids [1,T]) -> [1,T,V]`). Returns error statistics; with
    `strict`, raises on the first row outside the tolerance."""
    lengths, offset = ROW_LENGTHS(batch), ROW_OFFSETS(batch)
    ids = _sequences(batch, lengths, seed)
    expected = [torch.as_tensor(model(ids[b:b + 1, :lengths[b]].long()))[0].float() for b in range(batch)]
    cache = [torch.zeros(LAYERS, HEADS, 0, HEAD_DIM) for _ in range(2 * batch)]
    worst, outside, rows, calls, argmax = 0.0, 0, 0, 0, 0
    for step in range(max(o + n for o, n in zip(offset, lengths))):
        live = [b for b in range(batch) if offset[b] <= step < offset[b] + lengths[b]]
        if not live:
            continue
        # Unused tail slots hold large finite junk: the mask must hide them.
        width = slots or (max(cache[2 * b].shape[2] for b in live) + 1)
        k = torch.full((LAYERS, len(live), HEADS, width, HEAD_DIM), 30.0)
        v = torch.full_like(k, -30.0)
        for r, b in enumerate(live):
            n = cache[2 * b].shape[2]
            k[:, r, :, :n] = cache[2 * b]
            v[:, r, :, :n] = cache[2 * b + 1]
        position = torch.tensor([cache[2 * b].shape[2] for b in live], dtype=torch.int32)
        tokens = torch.stack([ids[b, step - offset[b]] for b in live])[:, None]
        logits, new_k, new_v = (torch.as_tensor(x).float() for x in fn(tokens, position, k, v))
        calls += 1
        for r, b in enumerate(live):
            want = expected[b][step - offset[b]]
            error = float((logits[r] - want).abs().max())
            worst = max(worst, error)
            rows += 1
            argmax += int(logits[r].argmax() == want.argmax())
            if not torch.allclose(logits[r], want, atol=TOLERANCE, rtol=TOLERANCE):
                outside += 1
                if strict:
                    torch.testing.assert_close(logits[r], want, atol=TOLERANCE, rtol=TOLERANCE)
            cache[2 * b] = torch.cat([cache[2 * b], new_k[:, r, :, None]], dim=2)
            cache[2 * b + 1] = torch.cat([cache[2 * b + 1], new_v[:, r, :, None]], dim=2)
    return {"batch": batch, "slots": slots or "max(position)+1", "calls": calls, "rows": rows,
            "max_abs": worst, "rows_outside_tolerance": outside, "argmax_equal": argmax}


@torch.inference_mode()
def trace_batch_graph(model):
    graph = KVBatchGraphZh(model).eval()
    example = (torch.tensor([[7], [9]], dtype=torch.int32), torch.tensor([0, 2], dtype=torch.int32),
               torch.zeros(LAYERS, 2, HEADS, 3, HEAD_DIM), torch.zeros(LAYERS, 2, HEADS, 3, HEAD_DIM))
    traced = torch.jit.trace(graph, example)
    for fn in (graph, traced):
        check_graph(model, fn)
        check_graph(model, fn, batch=1)
        check_graph(model, fn, batch=5, slots=CONTEXT)     # garbage-free tail of a full 128-slot buffer
    return traced


def convert(bundle, output, batch=None, cache_dtype="float32"):
    """batch=None: flexible 1..16; an int fixes B. cache_dtype float16 halves cache input bytes."""
    ct = coremltools()
    if Path(output).exists():
        raise ValueError("Conversion output exists.")
    if batch is not None and not 1 <= batch <= MAX_BATCH:
        raise ValueError("Batch out of range.")
    lm = zh_v1.BundleLMZh(bundle)
    b = batch or ct.RangeDim(lower_bound=1, upper_bound=MAX_BATCH, default=MAX_BATCH)
    s = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT, default=32)
    dtype = {"float32": np.float32, "float16": np.float16}[cache_dtype]
    model = ct.convert(trace_batch_graph(lm.model), source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32,
        inputs=[ct.TensorType(name="input_ids", shape=(b, 1), dtype=np.int32),
                ct.TensorType(name="position", shape=(b,), dtype=np.int32),
                ct.TensorType(name="key_cache", shape=(LAYERS, b, HEADS, s, HEAD_DIM), dtype=dtype),
                ct.TensorType(name="value_cache", shape=(LAYERS, b, HEADS, s, HEAD_DIM), dtype=dtype)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32),
                 ct.TensorType(name="new_key", dtype=dtype),
                 ct.TensorType(name="new_value", dtype=dtype)], skip_model_load=True)
    model.short_description = "vimeml-tiny-zh-v1 batched one-token KV step for beam search; FP32 compute."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    batch_text = f"fixed {batch}" if batch else f"flexible 1..{MAX_BATCH} (RangeDim)"
    zh_v1.finish(output, model, FORMAT, {"kind": "fp32", "minimum_ios": 18, "bundle": lm.metadata,
        "batch": batch_text, "batch_range": [batch or 1, batch or MAX_BATCH],
        "cache_shape": [LAYERS, "B", HEADS, "S", HEAD_DIM], "cache_slots": f"S flexible 1..{CONTEXT}",
        "cache_dtype": cache_dtype, "compute_units": "CPU_ONLY",
        "interface": {"input_ids": "int32 [B,1]; one new token per row",
                      "position": "int32 [B]; row cache length before this token; position[b] < S",
                      "key_cache/value_cache": f"{cache_dtype} ({LAYERS},B,6,S,64), 1<=S<={CONTEXT}; "
                                               "row b slots >= position[b] are ignored",
                      "logits": f"float32 [B,{VOCAB}] for the new token; no softmax",
                      "new_key/new_value": f"{cache_dtype} ({LAYERS},B,6,64); the new token's key/value, "
                                           "to be written by the caller into slot position[b] of row b"},
        "rows": "independent; no attention across rows; padding rows (any token, position 0) do not affect others",
        "prompt": "not in this package: run the batch-1 chunk package once and copy its cache rows",
        "graph": "vimeml.deployment.zh_v1_batch.KVBatchGraphZh (one-hot slot write, per-row length mask)",
        "precision_policy": f"FP32 parameters and compute; caches {cache_dtype}.",
        "policy": "Caller owns all cache buffers; inputs are never modified.",
        "conversion_source_sha256": file_sha(Path(__file__))})


def compress(source, output, alignment):
    """INT8 block32 (the deployed zh-v1 recipe) of an aligned FP32 batch package."""
    ct = coremltools()
    source, output, alignment = map(Path, (source, output, alignment))
    if output.exists():
        raise ValueError("Compression output exists.")
    original = zh_v1.verify_package(source, FORMAT)
    gate = read_json(alignment)
    if (original["kind"] != "fp32" or gate.get("format") != ALIGNMENT_FORMAT or not gate.get("passed")
            or gate.get("coreml_manifest_sha256") != file_sha(source / "manifest.json")):
        raise ValueError("First pass alignment for this exact FP32 batch package.")
    from coremltools.optimize import coreml as opt
    model = ct.models.MLModel(str(source / "model.mlpackage"), skip_model_load=True)
    spec = zh_v1.RECIPES[zh_v1.DEPLOYED_RECIPE]
    config = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_block",
                                         block_size=spec["block_size"], weight_threshold=zh_v1.WEIGHT_THRESHOLD)
    optimization, selection = v2.parameter_selection(opt, model, config)
    print(f"selected {len(selection['selected_constants'])} parameter matrices; "
          f"excluded {len(selection['excluded_constants'])} structural constants.", flush=True)
    result = opt.linear_quantize_weights(model, config=optimization)
    if not any(name.startswith("constexpr_") for name in inspect_spec(result)["operations"]):
        raise ValueError("No compressed weights found.")
    carried = {k: original[k] for k in ("minimum_ios", "bundle", "batch", "batch_range", "cache_shape",
                                        "cache_slots", "cache_dtype", "compute_units", "interface", "rows",
                                        "prompt", "graph", "policy")}
    zh_v1.finish(output, result, FORMAT, {**carried, "kind": "int8-b32_fp32_compute", "recipe": zh_v1.DEPLOYED_RECIPE,
        "source_manifest_sha256": file_sha(source / "manifest.json"), "fp32_alignment_sha256": file_sha(alignment),
        "compression": {**spec, "weight_threshold": zh_v1.WEIGHT_THRESHOLD, "selection": selection},
        "precision_policy": "INT8 block32 parameter matrices, FP32 compute; norms unchanged; no activation quantization.",
        "conversion_source_sha256": file_sha(Path(__file__))})


def merge(chunk, batch, output):
    """One multifunction package: `prefill` (batch-1 chunk) and `decode` (batch step). Identical
    weight constants are stored once, so loading both functions costs one copy of the weights."""
    ct = coremltools()
    chunk, batch, output = map(Path, (chunk, batch, output))
    if output.exists():
        raise ValueError("Merge output exists.")
    a = zh_v1.verify_package(chunk, zh_v1.CHUNK_FORMAT)
    b = zh_v1.verify_package(batch, FORMAT)
    if a["bundle"]["bundle_manifest_sha256"] != b["bundle"]["bundle_manifest_sha256"] or a["kind"] != b["kind"]:
        raise ValueError("Chunk and batch packages differ in bundle or precision.")
    output.mkdir(parents=True)
    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(str(chunk / "model.mlpackage"), src_function_name="main", target_function_name="prefill")
    desc.add_function(str(batch / "model.mlpackage"), src_function_name="main", target_function_name="decode")
    desc.default_function_name = "decode"
    ct.utils.save_multifunction(desc, str(output / "model.mlpackage"))
    files = v2.inventory(output / "model.mlpackage")
    write_json(output / "manifest.json", {"format": MERGED_FORMAT, "status": "complete",
        "architecture": zh_v1.ARCHITECTURE, "release_name": zh_v1.RELEASE, "kind": b["kind"],
        "bundle": b["bundle"], "functions": {
            "prefill": {"source_manifest_sha256": file_sha(chunk / "manifest.json"), "interface": a["interface"]},
            "decode": {"source_manifest_sha256": file_sha(batch / "manifest.json"), "interface": b["interface"],
                       "batch_range": b["batch_range"], "cache_slots": b["cache_slots"]}},
        "default_function": "decode", "minimum_ios": 18, "compute_units": "CPU_ONLY",
        "loading": "MLModelConfiguration.functionName = \"prefill\" or \"decode\" (iOS 18); one MLModel per function",
        "package_files": files, "package_bytes": sum(e["bytes"] for e in files.values()),
        "conversion_source_sha256": file_sha(Path(__file__))})


class TorchBatchBackend:
    """PyTorch reference for the two calls the search makes (FP32; CPU or CUDA)."""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device).float().eval()
        self.device = device
        self.chunk = KVChunkGraphV4(self.model).to(device).eval()
        self.batch = KVBatchGraphZh(self.model).to(device).eval()
        self.calls = self.rows = 0

    @torch.inference_mode()
    def prefill(self, ids):
        zeros = torch.zeros(LAYERS, 1, HEADS, CONTEXT, HEAD_DIM, device=self.device)
        logits, k, v = self.chunk(torch.tensor([ids], dtype=torch.int32, device=self.device),
                                  torch.tensor([0], dtype=torch.int32, device=self.device), zeros, zeros.clone())
        n = len(ids)
        return logits[0, -1].float(), (k[:, 0, :, :n].clone(), v[:, 0, :, :n].clone())

    @torch.inference_mode()
    def decode(self, tokens, states):
        """tokens [B]; states: per row (key, value) of shape (9,6,n_b,64). Returns logits [B,V] and new states."""
        width = max(s[0].shape[2] for s in states) + 1
        k = torch.zeros(LAYERS, len(states), HEADS, width, HEAD_DIM, device=self.device)
        v = torch.zeros_like(k)
        for r, (sk, sv) in enumerate(states):
            n = sk.shape[2]
            k[:, r, :, :n] = sk
            v[:, r, :, :n] = sv
        position = torch.tensor([s[0].shape[2] for s in states], dtype=torch.int32, device=self.device)
        logits, nk, nv = self.batch(torch.tensor(tokens, dtype=torch.int32, device=self.device)[:, None],
                                    position, k, v)
        self.calls += 1
        self.rows += len(states)
        new = [(torch.cat([sk, nk[:, r, :, None]], 2), torch.cat([sv, nv[:, r, :, None]], 2))
               for r, (sk, sv) in enumerate(states)]
        return logits.float(), new


class CoreMLBatchBackend:
    """Same two calls on Core ML: the batch-1 chunk package for the prompt, the batch package per step."""

    def __init__(self, chunk_dir, batch_dir, compute_units="CPU_ONLY"):
        ct = coremltools()
        unit = getattr(ct.ComputeUnit, compute_units)
        self.manifest = zh_v1.verify_package(batch_dir, FORMAT)
        zh_v1.verify_package(chunk_dir, zh_v1.CHUNK_FORMAT)
        self.chunk = ct.models.MLModel(str(Path(chunk_dir) / "model.mlpackage"), compute_units=unit)
        self.batch = ct.models.MLModel(str(Path(batch_dir) / "model.mlpackage"), compute_units=unit)
        self.dtype = np.float16 if self.manifest["cache_dtype"] == "float16" else np.float32
        lo, hi = self.manifest["batch_range"]
        self.fixed = hi if lo == hi else None
        self.calls = self.rows = 0

    def prefill(self, ids):
        zeros = np.zeros((LAYERS, 1, HEADS, CONTEXT, HEAD_DIM), np.float32)
        out = self.chunk.predict({"input_ids": np.array([ids], np.int32), "position": np.array([0], np.int32),
                                  "key_cache": zeros, "value_cache": zeros})
        n = len(ids)
        return (torch.from_numpy(np.array(out["logits"][0, -1], np.float32)),
                (np.array(out["new_key_cache"][:, 0, :, :n], self.dtype),
                 np.array(out["new_value_cache"][:, 0, :, :n], self.dtype)))

    def decode(self, tokens, states):
        rows = len(states)
        size = self.fixed or rows
        if rows > size:
            raise ValueError("More rows than the fixed batch.")
        width = max(s[0].shape[2] for s in states) + 1
        k = np.zeros((LAYERS, size, HEADS, width, HEAD_DIM), self.dtype)
        v = np.zeros_like(k)
        position = np.zeros(size, np.int32)
        ids = np.zeros((size, 1), np.int32)     # padding rows: token 0 at position 0
        for r, (sk, sv) in enumerate(states):
            n = sk.shape[2]
            k[:, r, :, :n] = sk
            v[:, r, :, :n] = sv
            position[r] = n
            ids[r, 0] = tokens[r]
        out = self.batch.predict({"input_ids": ids, "position": position, "key_cache": k, "value_cache": v})
        self.calls += 1
        self.rows += rows
        nk, nv = out["new_key"], out["new_value"]
        new = [(np.concatenate([sk, np.asarray(nk[:, r, :, None], self.dtype)], 2),
                np.concatenate([sv, np.asarray(nv[:, r, :, None], self.dtype)], 2)) for r, (sk, sv) in enumerate(states)]
        return torch.from_numpy(np.array(out["logits"][:rows], np.float32)), new


def coreml_step(model, manifest):
    """check_graph-style step function over a Core ML batch package (pads to a fixed batch)."""
    lo, hi = manifest["batch_range"]
    fixed = hi if lo == hi else None
    dtype = np.float16 if manifest["cache_dtype"] == "float16" else np.float32

    def fn(tokens, position, k, v):
        rows = tokens.shape[0]
        size = fixed or rows
        ids = np.zeros((size, 1), np.int32)
        pos = np.zeros(size, np.int32)
        kk = np.zeros((LAYERS, size, HEADS, k.shape[3], HEAD_DIM), dtype)
        vv = np.zeros_like(kk)
        ids[:rows] = tokens.numpy()
        pos[:rows] = position.numpy()
        kk[:, :rows] = k.numpy()
        vv[:, :rows] = v.numpy()
        out = model.predict({"input_ids": ids, "position": pos, "key_cache": kk, "value_cache": vv})
        return (np.asarray(out["logits"][:rows], np.float32), np.asarray(out["new_key"][:, :rows], np.float32),
                np.asarray(out["new_value"][:, :rows], np.float32))
    return fn


def coreml_full(model):
    """Stateless package as a full-forward reference: ids [1,T] -> logits [1,T,V]."""
    return lambda ids: np.asarray(model.predict({"input_ids": ids.numpy().astype(np.int32)})["logits"], np.float32)


ALIGN_CASES = ({"batch": 16}, {"batch": 1}, {"batch": 5, "slots": CONTEXT}, {"batch": 16, "seed": 2026})


def align(batch_dir, bundle, output, stateless=None):
    """Gate a batch package. Reference: PyTorch FP32 from the bundle, or (`stateless`) a stateless
    Core ML package of the same precision, as the Vime contract asks for INT8."""
    ct = coremltools()
    batch_dir, output = Path(batch_dir), Path(output)
    if output.exists():
        raise ValueError("Alignment output exists.")
    manifest = zh_v1.verify_package(batch_dir, FORMAT)
    lm = zh_v1.BundleLMZh(bundle)
    if manifest["bundle"]["bundle_manifest_sha256"] != lm.metadata["bundle_manifest_sha256"]:
        raise ValueError("Batch package belongs to another bundle.")
    unit = ct.ComputeUnit.CPU_ONLY
    model = ct.models.MLModel(str(batch_dir / "model.mlpackage"), compute_units=unit)
    if stateless:
        ref_manifest = zh_v1.verify_package(stateless)
        reference = coreml_full(ct.models.MLModel(str(Path(stateless) / "model.mlpackage"), compute_units=unit))
        ref_name = f"stateless Core ML {ref_manifest['kind']} ({Path(stateless).name})"
    else:
        reference = lambda ids: lm.model(ids).detach()
        ref_name = "PyTorch FP32 full forward (bundle)"
    lo, hi = manifest["batch_range"]
    checks = []
    for case in ALIGN_CASES:
        case = dict(case)
        if case["batch"] > hi:
            continue
        stats = check_graph(reference, coreml_step(model, manifest), strict=False, **case)
        if lo == hi and case["batch"] < hi:
            stats["padding_rows"] = hi - case["batch"]
        checks.append(stats)
        print(json.dumps(stats), flush=True)
    passed = all(c["rows_outside_tolerance"] == 0 for c in checks)
    write_json(output, {"format": ALIGNMENT_FORMAT, "passed": passed, "reference": ref_name,
        "coreml_manifest_sha256": file_sha(batch_dir / "manifest.json"), "package": batch_dir.name,
        "kind": manifest["kind"], "tolerance": {"atol": TOLERANCE, "rtol": TOLERANCE}, "checks": checks,
        "method": "rows of 3..25 tokens starting at different steps (positions differ in every call); each "
                  "step's logits vs the reference's full forward of the row's own sequence; unused cache slots "
                  "filled with +/-30 junk; fixed-batch packages padded with token 0 at position 0",
        "conversion_source_sha256": file_sha(Path(__file__))})
    print(f"alignment {'passed' if passed else 'FAILED'}: max {max(c['max_abs'] for c in checks):.3g}")
    return passed
