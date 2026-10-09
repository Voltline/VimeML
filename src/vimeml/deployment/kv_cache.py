"""V2 incremental attention with explicit, immutable FP32 cache snapshots.

Caches belong to one request. Reuse only exact token prefixes, with absolute
learned positions; never slide, persist across requests, or mutate a branch.
"""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vimeml.deployment import v2
from vimeml.deployment.bundle import fresh_directory, read_json, write_json
from vimeml.deployment.coreml import coremltools, inspect_spec
from vimeml.training.data import file_sha

FORMAT = "vimeml_coreml_v2_kv_v1"


class CacheGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        c = model.config
        self.width, self.heads, self.head_dim = c.d_model, c.n_heads, c.d_model // c.n_heads
        mask = torch.full((c.context_length, c.context_length), -float("inf"))
        self.register_buffer("causal_mask", torch.triu(mask, diagonal=1))

    def forward(self, input_ids, cache_length, key_cache, value_cache):
        length = input_ids.shape[1]
        past = cache_length[0]
        positions = torch.arange(length, device=input_ids.device, dtype=torch.int32) + past
        hidden = self.model.token_embedding(input_ids) + self.model.position_embedding(positions)
        # Rectangular mask: queries have absolute positions past..end-1.
        mask = self.causal_mask[positions.long()]
        keys, values = [], []
        for i, block in enumerate(self.model.blocks):
            normalized = block.attention_norm(hidden)
            qkv = block.attention.qkv(normalized).reshape(1, length, 3, self.heads, self.head_dim)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            indices = (positions.long(),)
            key = key_cache[i].permute(2, 0, 1, 3).index_put(indices, key.permute(2, 0, 1, 3)).permute(1, 2, 0, 3)
            value = value_cache[i].permute(2, 0, 1, 3).index_put(indices, value.permute(2, 0, 1, 3)).permute(1, 2, 0, 3)
            attention = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim) + mask, dim=-1)
            attended = torch.matmul(attention, value).transpose(1, 2).reshape(1, length, self.width)
            hidden = hidden + block.attention.projection(attended)
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
            keys.append(key)
            values.append(value)
        # Output exactly the new chunk. A one-token decode emits one row;
        # avoid a second dynamic output dimension from runtime logit slicing.
        logits = self.model.lm_head(self.model.final_norm(hidden))
        return logits, torch.stack(keys), torch.stack(values)


def cache_shape(config):
    return (config.n_layers, 1, config.n_heads, config.context_length, config.d_model // config.n_heads)


def empty_cache(config):
    return tuple(np.zeros(cache_shape(config), dtype=np.float32) for _ in range(2))


@torch.inference_mode()
def trace(model):
    graph = CacheGraph(model).eval()
    keys, values = (torch.from_numpy(a) for a in empty_cache(model.config))
    traced = torch.jit.trace(graph, (torch.tensor([[2, 4, 7]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32), keys, values))
    # Exercise a nonzero past, single-token decode and the full capacity,
    # rather than checking only the trace example.
    generator = torch.Generator().manual_seed(716)
    ids = torch.randint(model.config.vocab_size, (1, model.config.context_length), generator=generator, dtype=torch.int32)
    expected = model(ids)
    for fn in (graph, traced):
        state = (keys, values)
        start = 0
        for size in (1, 3, model.config.context_length - 4):
            actual, k, v = fn(ids[:, start:start+size], torch.tensor([start], dtype=torch.int32),
                             *state)
            torch.testing.assert_close(actual, expected[:, start:start+size], atol=3e-5, rtol=3e-4)
            state = k, v
            start += size
        actual, _, _ = fn(ids, torch.tensor([0], dtype=torch.int32), keys, values)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    return traced


def finish(output, model, metadata):
    fresh_directory(output)
    model.save(str(Path(output) / "model.mlpackage"))
    files = v2.inventory(Path(output) / "model.mlpackage")
    write_json(Path(output) / "manifest.json", {"format": FORMAT, "status": "complete",
        "architecture": v2.ARCHITECTURE, **metadata, "environment": v2.environment(),
        "spec": inspect_spec(model), "package_files": files,
        "package_bytes": sum(item["bytes"] for item in files.values())})


def verify_package(directory):
    directory = Path(directory)
    m = read_json(directory / "manifest.json")
    if (m.get("format") != FORMAT or m.get("status") != "complete"
            or m.get("architecture") != v2.ARCHITECTURE or v2.inventory(directory / "model.mlpackage") != m["package_files"]):
        raise ValueError("Invalid KV package or changed file inventory.")
    return m


def convert(bundle, output):
    if Path(output).exists():
        raise ValueError("Output exists; choose a new experiment.")
    lm, ct = v2.BundleLM(bundle), coremltools()
    c = lm.model.config
    result = ct.convert(trace(lm.model), source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, c.context_length, default=1)), dtype=np.int32),
                ct.TensorType(name="cache_length", shape=(1,), dtype=np.int32),
                ct.TensorType(name="key_cache", shape=cache_shape(c), dtype=np.float32),
                ct.TensorType(name="value_cache", shape=cache_shape(c), dtype=np.float32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32), ct.TensorType(name="new_key_cache", dtype=np.float32),
                 ct.TensorType(name="new_value_cache", dtype=np.float32)], skip_model_load=True)
    finish(output, result, {"kind": "fp32", "minimum_ios": 18, "bundle": lm.metadata,
        "cache_shape": cache_shape(c), "cache_dtype": "float32", "compute_units": "CPU_ONLY",
        "cache_bytes_per_snapshot": 2 * int(np.prod(cache_shape(c))) * 4,
        "interface": "batch1; absolute learned positions; 0<=past, 1<=Q, past+Q<=128; logits [1,Q,V] for new tokens only. Explicit immutable K/V snapshots.",
        "conversion_source_sha256": file_sha(Path(__file__)),
        "policy": "No training/weight changes. Request-local exact token prefix reuse; no sliding or cross-request retention."})


def compress(source, output, alignment):
    if Path(output).exists():
        raise ValueError("Output exists; choose a new experiment.")
    m, gate = verify_package(source), read_json(alignment)
    if (m["kind"] != "fp32" or not gate.get("passed") or gate.get("format") != "vimeml_kv_alignment_v1"
            or gate.get("coreml_manifest_sha256") != file_sha(Path(source) / "manifest.json")):
        raise ValueError("A passed alignment report for this exact FP32 KV package is required.")
    ct = coremltools()
    from coremltools.optimize import coreml as opt
    model = ct.models.MLModel(str(Path(source) / "model.mlpackage"), skip_model_load=True)
    config = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_block", block_size=32, weight_threshold=2048)
    optimization, selection = v2.parameter_selection(opt, model, config)
    result = opt.linear_quantize_weights(model, config=optimization)
    finish(output, result, {key: m[key] for key in ("minimum_ios", "bundle", "cache_shape", "cache_dtype", "compute_units", "cache_bytes_per_snapshot", "interface", "policy")} | {
        "kind": "linear8_fp32_compute", "source_manifest_sha256": file_sha(Path(source) / "manifest.json"),
        "fp32_alignment_sha256": file_sha(alignment), "compression": {"method": "linear_symmetric", "bits": 8,
        "block_size": 32, "selection": selection}, "conversion_source_sha256": file_sha(Path(__file__))})


class Runtime:
    def __init__(self, bundle, directory):
        lm = v2.BundleLM(bundle)
        self.config, self.metadata = lm.model.config, dict(lm.metadata)
        self.manifest = verify_package(directory)
        if self.manifest["bundle"]["bundle_manifest_sha256"] != self.metadata["bundle_manifest_sha256"]:
            raise ValueError("KV model belongs to another bundle.")
        if (self.manifest["cache_shape"] != list(cache_shape(self.config))
                or self.manifest["cache_dtype"] != "float32" or self.manifest["compute_units"] != "CPU_ONLY"):
            raise ValueError("KV package cache/compute contract differs.")
        ct = coremltools()
        self.model = ct.models.MLModel(str(Path(directory) / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
        self.metadata.update(precision=self.manifest["kind"], compute_units="CPU_ONLY",
            coreml_manifest_sha256=file_sha(Path(directory) / "manifest.json"))
        self.calls = self.processed_tokens = 0

    def step(self, ids, state=None, past=0, last_only=False):
        raw = np.asarray(ids)
        if (not raw.size or not np.issubdtype(raw.dtype, np.integer)
                or raw.ndim not in (1, 2) or (raw.ndim == 2 and raw.shape[0] != 1)
                or (raw < 0).any() or (raw >= self.config.vocab_size).any()):
            raise ValueError("Invalid incremental token tensor.")
        ids = raw.astype(np.int32).reshape(1, -1)
        if (not isinstance(past, int) or past < 0 or past + ids.shape[1] > self.config.context_length
                or (state is None and past != 0)):
            raise ValueError("Invalid incremental input/cache position.")
        state = empty_cache(self.config) if state is None else state
        if len(state) != 2 or any(not isinstance(a, np.ndarray) or a.shape != cache_shape(self.config)
                or a.dtype != np.float32 or not np.isfinite(a).all() for a in state):
            raise ValueError("Invalid cache tensor.")
        result = self.model.predict({"input_ids": ids, "cache_length": np.array([past], np.int32),
            "key_cache": state[0], "value_cache": state[1]})
        self.calls += 1
        self.processed_tokens += ids.size
        # Own prediction output memory, as in the existing stateless adapter.
        # A retained branch snapshot must survive subsequent runtime calls.
        logits = np.array(result["logits"], dtype=np.float32, copy=True)
        caches = tuple(np.array(result[name], dtype=np.float32, copy=True)
                       for name in ("new_key_cache", "new_value_cache"))
        if (logits.shape != (1, ids.size, self.config.vocab_size)
                or not np.isfinite(logits).all()
                or any(a.shape != cache_shape(self.config) or not np.isfinite(a).all() for a in caches)):
            raise ValueError("Invalid Core ML cache/logits output.")
        return (logits[:, -1:].copy() if last_only else logits), caches
