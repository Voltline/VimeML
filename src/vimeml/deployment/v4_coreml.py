"""V4 inference bundle, FP32/INT8 Core ML packages and gates.

Stateless graph: int32 [1,T], 1<=T<=128 -> raw float32 [1,T,8192] logits.
KV graph: one token at an absolute position with explicit FP32 caches
(7,1,6,128,64) from `vimeml.deployment.v4.KVStepGraphV4`. The caller owns
snapshots and rollback. V1/V2 formats and entry points remain unchanged.
"""
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from vimeml.benchmarks.comparison import paired
from vimeml.deployment import v2
from vimeml.deployment.bundle import environment, fresh_directory, read_json, write_json
from vimeml.deployment.coreml import CoreMLForward, coremltools, inspect_spec
from vimeml.deployment.graph import trace_graph
from vimeml.deployment.v4 import CACHE_SHAPE, trace_kv_graph
from vimeml.deployment.validation import compare_rows, numeric_error
from vimeml.training.data import file_sha
from vimeml.training.evaluate_ime import common_prefix_length
from vimeml.training.infer import ROOT
from vimeml.training.infer_v4 import JapaneseLMV4
from vimeml.training.model_v4 import GPTV4Config, TinyGPTV4

BUNDLE_FORMAT = "vimeml_inference_bundle_v4"
PACKAGE_FORMAT = "vimeml_coreml_v4"
KV_FORMAT = "vimeml_coreml_v4_kv_v1"
ARCHITECTURE = "tiny_gpt_v4"
REFERENCE_FORMAT = "vimeml_v4_mac_preparation_reference_v1"
VOCAB = 8192
TOLERANCE = 3e-4
QUANTILES = (0.5, 0.9, 0.99, 0.999)


def export_bundle(checkpoint, tokenizer_dir, output):
    checkpoint, tokenizer_dir, output = map(Path, (checkpoint, tokenizer_dir, output))
    if output.exists():
        raise ValueError("Bundle output already exists.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved.get("format") != "vimeml_tiny_gpt_v4" or saved.get("architecture") != ARCHITECTURE
            or not saved.get("deployment_only") or not saved.get("source_checkpoint_sha256")):
        raise ValueError("Expected a V4 deployment checkpoint with source provenance.")
    # JapaneseLMV4 binds token data, tokenizer and checkpoint tokenizer identity.
    lm = JapaneseLMV4(checkpoint, tokenizer_dir, "cpu")
    state = saved["model"]
    if not torch.equal(state["lm_head.weight"], state["token_embedding.weight"]):
        raise ValueError("Expected identical tied embedding/head tensors.")
    if any(not torch.isfinite(p).all() for p in lm.model.parameters()):
        raise ValueError("Nonfinite learned weight.")
    token_manifest = ROOT / saved["config"]["token_dir"] / "manifest.json"
    weights = {name: value.detach().cpu().float().contiguous() for name, value in state.items()
               if name != "lm_head.weight"}
    fresh_directory(output)
    torch.save(weights, output / "weights.pt")
    write_json(output / "config.json", lm.model.configuration())
    shutil.copyfile(tokenizer_dir / "tokenizer.model", output / "tokenizer.model")
    shutil.copyfile(token_manifest, output / "token-manifest.json")
    result = {"format": BUNDLE_FORMAT, "status": "complete", "architecture": ARCHITECTURE,
              "precision": "fp32", "parameter_count": lm.model.parameter_count(), "special_ids": lm.special,
              "forbidden_generation_ids": list(lm.forbidden),
              "checkpoint_sha256": saved["source_checkpoint_sha256"],
              "deployment_checkpoint_sha256": file_sha(checkpoint),
              "checkpoint_step": saved["step"], "checkpoint_epoch_index": saved["epoch"],
              "training_signatures": saved["signatures"], "tokenizer_identity": saved["tokenizer_identity"],
              "tied_lm_head": True, "files": v2.inventory(output, hashes=True), "environment": environment(),
              "policy": "Unique FP32 inference weights; no optimizer/RNG. Joint suffix logP sum; no EOS/truncation."}
    write_json(output / "manifest.json", result)
    return result


def verify_bundle(path, hashes=True):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if (manifest.get("format") != BUNDLE_FORMAT or manifest.get("status") != "complete"
            or manifest.get("architecture") != ARCHITECTURE or not manifest.get("tied_lm_head")):
        raise ValueError("Unsupported/incomplete V4 bundle.")
    if set(manifest["files"]) != {"weights.pt", "config.json", "tokenizer.model", "token-manifest.json"}:
        raise ValueError("Unexpected inference file list.")
    for name, entry in manifest["files"].items():
        if (path / name).stat().st_size != entry["bytes"] or (hashes and file_sha(path / name) != entry["sha256"]):
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return manifest


class BundleLMV4(JapaneseLMV4):
    """PyTorch CPU FP32 model from a V4 bundle; V4 prefix and forbidden-ID rules."""
    def __init__(self, path):
        path = Path(path)
        manifest = verify_bundle(path, hashes=False)
        self.device = torch.device("cpu")
        self.processor = spm.SentencePieceProcessor(model_file=str(path / "tokenizer.model"))
        self.special = {name: getattr(self.processor, f"{name}_id")() for name in ("pad", "unk", "bos", "eos")}
        self.model = TinyGPTV4(GPTV4Config(**read_json(path / "config.json"))).eval()
        state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        if "lm_head.weight" in state:
            raise ValueError("Shared embedding must be stored only once.")
        loaded = self.model.load_state_dict(state, strict=False)
        if loaded.missing_keys != ["lm_head.weight"] or loaded.unexpected_keys:
            raise ValueError(f"Unexpected inference weights: {loaded}")
        if (self.special != manifest["special_ids"] or self.processor.vocab_size() != VOCAB
                or self.model.parameter_count() != manifest["parameter_count"]):
            raise ValueError("Bundle model/tokenizer contract mismatch.")
        self.forbidden = tuple(manifest["forbidden_generation_ids"])
        self.metadata = {"architecture": ARCHITECTURE, "checkpoint_sha256": manifest["checkpoint_sha256"],
                         "checkpoint_step": manifest["checkpoint_step"],
                         "deployment_checkpoint_sha256": manifest["deployment_checkpoint_sha256"],
                         "tokenizer_sha256": manifest["files"]["tokenizer.model"]["sha256"],
                         "bundle_manifest_sha256": file_sha(path / "manifest.json"),
                         "model": self.model.configuration(), "precision": "fp32", "device": "cpu",
                         "environment": environment()}


def finish(output, model, fmt, metadata):
    output = Path(output)
    fresh_directory(output)
    model.save(str(output / "model.mlpackage"))
    files = v2.inventory(output / "model.mlpackage")
    write_json(output / "manifest.json", {"format": fmt, "status": "complete", "architecture": ARCHITECTURE,
        **metadata, "environment": environment(), "spec": inspect_spec(model), "package_files": files,
        "package_bytes": sum(entry["bytes"] for entry in files.values()),
        "note": "Logical package bytes; compiled size, resident memory and device footprint are separate."})


def verify_package(path, fmt=PACKAGE_FORMAT):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if (manifest.get("format") != fmt or manifest.get("status") != "complete"
            or manifest.get("architecture") != ARCHITECTURE):
        raise ValueError("Incomplete/unsupported V4 Core ML package.")
    if v2.inventory(path / "model.mlpackage") != manifest["package_files"]:
        raise ValueError("Package file inventory changed.")
    return manifest


def convert(bundle, output):
    ct = coremltools()
    if Path(output).exists():
        raise ValueError("Conversion output exists.")
    lm = BundleLMV4(bundle)
    model = ct.convert(trace_graph(lm.model), source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(lower_bound=1, upper_bound=128, default=16)),
                              dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)], skip_model_load=True)
    model.short_description = "VimeML V4 RMSNorm/SwiGLU 8K; FP32 computation; no KV cache."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    finish(output, model, PACKAGE_FORMAT, {"kind": "fp32", "minimum_ios": 18, "bundle": lm.metadata,
        "interface": {"input_ids": "int32 [1,T], 1<=T<=128; right PAD",
                      "logits": "float32 [1,T,8192]; full vocabulary; no softmax"},
        "precision_policy": "All learned parameters and computation FP32; explicit causal attention.",
        "conversion_source_sha256": file_sha(Path(__file__))})


def convert_kv(bundle, output):
    ct = coremltools()
    if Path(output).exists():
        raise ValueError("Conversion output exists.")
    lm = BundleLMV4(bundle)
    model = ct.convert(trace_kv_graph(lm.model), source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32,
        inputs=[ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
                ct.TensorType(name="position", shape=(1,), dtype=np.int32),
                ct.TensorType(name="key_cache", shape=CACHE_SHAPE, dtype=np.float32),
                ct.TensorType(name="value_cache", shape=CACHE_SHAPE, dtype=np.float32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32),
                 ct.TensorType(name="new_key_cache", dtype=np.float32),
                 ct.TensorType(name="new_value_cache", dtype=np.float32)], skip_model_load=True)
    model.short_description = "VimeML V4 single-token KV step; FP32 computation and caches."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    finish(output, model, KV_FORMAT, {"kind": "fp32", "minimum_ios": 18, "bundle": lm.metadata,
        "cache_shape": list(CACHE_SHAPE), "cache_dtype": "float32", "compute_units": "CPU_ONLY",
        "cache_bytes_per_snapshot": 2 * int(np.prod(CACHE_SHAPE)) * 4,
        "interface": {"input_ids": "int32 [1,1]", "position": "int32 [1], 0<=position<=127; absolute slot",
                      "key_cache/value_cache": "float32 (7,1,6,128,64); slots >= position are ignored",
                      "logits": "float32 [1,1,8192] for the new token; no softmax",
                      "new_key_cache/new_value_cache": "inputs with slot `position` replaced"},
        "precision_policy": "All learned parameters, computation and caches FP32.",
        "policy": "Batch1; caller owns immutable snapshots and rollback; no sliding or cross-request retention.",
        "conversion_source_sha256": file_sha(Path(__file__))})


def compress(source, output, alignment, kv=False):
    ct = coremltools()
    source, output, alignment = map(Path, (source, output, alignment))
    if output.exists():
        raise ValueError("Compression output exists.")
    fmt = KV_FORMAT if kv else PACKAGE_FORMAT
    original = verify_package(source, fmt)
    gate = read_json(alignment)
    expected = "vimeml_kv_alignment_v4" if kv else "vimeml_alignment_v4"
    if (original["kind"] != "fp32" or gate.get("format") != expected or not gate.get("passed")
            or gate.get("coreml_manifest_sha256") != file_sha(source / "manifest.json")):
        raise ValueError("First pass alignment for this exact uncompressed V4 FP32 package.")
    from coremltools.optimize import coreml as opt
    model = ct.models.MLModel(str(source / "model.mlpackage"), skip_model_load=True)
    # Identical to the deployed V2.1 recipe.
    config = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                        granularity="per_block", block_size=32, weight_threshold=2048)
    optimization, selection = v2.parameter_selection(opt, model, config)
    print(f"Selected {len(selection['selected_constants'])} parameter matrices; "
          f"excluded {len(selection['excluded_constants'])} structural constants.", flush=True)
    result = opt.linear_quantize_weights(model, config=optimization)
    if not any(name.startswith("constexpr_") for name in inspect_spec(result)["operations"]):
        raise ValueError("No compressed weights found.")
    carried = {key: original[key] for key in original if key in (
        "minimum_ios", "bundle", "interface", "cache_shape", "cache_dtype", "compute_units",
        "cache_bytes_per_snapshot", "policy")}
    finish(output, result, fmt, {**carried, "kind": "linear8_fp32_compute",
        "source_manifest_sha256": file_sha(source / "manifest.json"),
        "fp32_alignment_sha256": file_sha(alignment), "conversion_source_sha256": file_sha(Path(__file__)),
        "compression": {"method": "linear_symmetric", "bits": 8, "block_size": 32, "granularity": "per_block",
                        "weight_threshold": 2048, "selection": selection},
        "precision_policy": "INT8 parameter matrices, FP32 compute; norms/masks unchanged; no activation quantization."})


class CoreMLLMV4(BundleLMV4):
    def __init__(self, bundle, directory):
        super().__init__(bundle)
        manifest = verify_package(directory)
        if manifest["bundle"]["bundle_manifest_sha256"] != self.metadata["bundle_manifest_sha256"]:
            raise ValueError("Core ML package belongs to another bundle.")
        self.model = CoreMLForward(directory, self.model.config, "CPU_ONLY")
        self.metadata.update(precision=manifest["kind"], compute_units="CPU_ONLY",
                             coreml_manifest_sha256=file_sha(Path(directory) / "manifest.json"))


def empty_cache():
    return tuple(np.zeros(CACHE_SHAPE, dtype=np.float32) for _ in range(2))


class KVRuntimeV4:
    """Single-token Core ML KV steps. Inputs are never modified; outputs are owned copies."""
    def __init__(self, bundle, directory):
        lm = BundleLMV4(bundle)
        self.config, self.metadata = lm.model.config, dict(lm.metadata)
        self.manifest = verify_package(directory, KV_FORMAT)
        if self.manifest["bundle"]["bundle_manifest_sha256"] != self.metadata["bundle_manifest_sha256"]:
            raise ValueError("KV model belongs to another bundle.")
        if (self.manifest["cache_shape"] != list(CACHE_SHAPE) or self.manifest["cache_dtype"] != "float32"
                or self.manifest["compute_units"] != "CPU_ONLY"):
            raise ValueError("KV package cache/compute contract differs.")
        ct = coremltools()
        self.model = ct.models.MLModel(str(Path(directory) / "model.mlpackage"),
                                       compute_units=ct.ComputeUnit.CPU_ONLY)
        self.metadata.update(precision=self.manifest["kind"], compute_units="CPU_ONLY",
                             coreml_manifest_sha256=file_sha(Path(directory) / "manifest.json"))
        self.calls = 0

    def step(self, token, position, state):
        if (type(position) is not int or not 0 <= position < 128
                or not 0 <= int(token) < VOCAB or len(state) != 2
                or any(not isinstance(a, np.ndarray) or a.shape != CACHE_SHAPE or a.dtype != np.float32
                       for a in state)):
            raise ValueError("Invalid KV step input.")
        result = self.model.predict({"input_ids": np.array([[token]], np.int32),
                                     "position": np.array([position], np.int32),
                                     "key_cache": state[0], "value_cache": state[1]})
        self.calls += 1
        logits = np.array(result["logits"], dtype=np.float32, copy=True)
        caches = tuple(np.array(result[name], dtype=np.float32, copy=True)
                       for name in ("new_key_cache", "new_value_cache"))
        if (logits.shape != (1, 1, VOCAB) or not np.isfinite(logits).all()
                or any(a.shape != CACHE_SHAPE for a in caches)):
            raise ValueError("Invalid Core ML cache/logits output.")
        return logits, caches

    def run(self, ids, state=None, start=0):
        """Process tokens one at a time from `start`; returns [1,Q,V] logits and the final snapshot."""
        ids = [int(i) for i in np.asarray(ids).reshape(-1)]
        if not ids or start < 0 or start + len(ids) > 128 or (state is None and start != 0):
            raise ValueError("Invalid KV position range.")
        state = empty_cache() if state is None else state
        parts = []
        for offset, token in enumerate(ids):
            logits, state = self.step(token, start + offset, state)
            parts.append(logits)
        return np.concatenate(parts, axis=1), state


class CachedForwardV4:
    """Evaluation adapter: share the exact common token prefix within one batch.

    Rows arrive right-padded with PAD (id 0, never produced by encode). Trailing
    PAD positions are not computed; their logits are zero and never scored.
    """
    def __init__(self, runtime, pad):
        self.runtime, self.config, self.pad = runtime, runtime.config, pad

    def __call__(self, inputs):
        rows = [list(row) for row in inputs.numpy().astype(np.int64)]
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


class CachedLMV4(BundleLMV4):
    def __init__(self, bundle, directory):
        super().__init__(bundle)
        self.runtime = KVRuntimeV4(bundle, directory)
        self.model = CachedForwardV4(self.runtime, self.special["pad"])
        self.metadata = self.runtime.metadata


def error_distribution(expected, actual, atol=TOLERANCE, rtol=TOLERANCE):
    """Strict error plus quantiles and per-position argmax agreement."""
    result = numeric_error(expected, actual, atol, rtol)
    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    result["quantiles_abs"] = {str(q): float(np.quantile(delta, q)) for q in QUANTILES}
    result["outside_fraction"] = result["outside_tolerance"] / result["values"]
    top_e, top_a = expected.argmax(-1), actual.argmax(-1)
    result["positions"] = int(top_e.size)
    result["argmax_equal_positions"] = int((top_e == top_a).sum())
    return result


def read_reference(reference, lm):
    reference = Path(reference)
    info = read_json(reference / "reference.json")
    if (info.get("format") != REFERENCE_FORMAT or info["checkpoint_step"] != lm.metadata["checkpoint_step"]
            or info["checkpoint_sha256"] != lm.metadata["checkpoint_sha256"]
            or info["tokenizer_sha256"] != lm.metadata["tokenizer_sha256"]):
        raise ValueError("Reference belongs to another checkpoint/tokenizer.")
    with np.load(reference / "logits.npz", allow_pickle=False) as arrays:
        return info, {name: arrays[name] for name in arrays.files}


def summarize_errors(items):
    deltas = [item["max_abs"] for item in items]
    return {"checks": len(items), "max_abs": max(deltas), "failed_checks": sum(
        item["outside_tolerance"] > 0 for item in items)}


@torch.inference_mode()
def align(lm, reference, output):
    """Stateless fixture gate: valid prefixes within 3e-4 and causality invariants."""
    output = Path(output)
    if output.exists():
        raise ValueError("Alignment output exists.")
    info, arrays = read_reference(reference, lm)
    errors, actuals = [], {}
    for name, case in info["cases"].items():
        ids = arrays[name + "_input_ids"]
        actual = lm.model(torch.from_numpy(ids.copy())).numpy()
        actuals[name] = actual
        valid = case["valid_prefix"]
        expected = arrays[name + "_logits"]
        error = error_distribution(expected[:, :valid], actual[:, :valid])
        error.update(id=name, length=case["length"], valid_prefix=valid,
                     last_top1_equal=int(actual[0, valid - 1].argmax()) == int(expected[0, valid - 1].argmax()))
        errors.append(error)
    invariants = {name: numeric_error(actuals["short"], actuals[name][:, :16], TOLERANCE, TOLERANCE)
                  for name in ("right_pad", "changed_future")}
    report = {"format": "vimeml_alignment_v4",
        "passed": all(item["outside_tolerance"] == 0 for item in [*errors, *invariants.values()]),
        "model": lm.metadata, "coreml_manifest_sha256": lm.metadata.get("coreml_manifest_sha256"),
        "reference": info, "tolerances": {"atol": TOLERANCE, "rtol": TOLERANCE},
        "summary": summarize_errors(errors), "logits": errors, "invariants": invariants,
        "note": "Strict PyTorch FP32 fixture comparison. Quantized ranking quality is evaluated separately; failures are preserved."}
    fresh_directory(output)
    write_json(output / "alignment.json", report)
    return report


@torch.inference_mode()
def align_kv(runtime, bundle, reference, output, stateless=None):
    """KV gate against frozen PyTorch fixtures, or against a same-weight stateless package.

    one_shot: uninterrupted single-token prefill vs full-sequence logits.
    one_token: the same steps vs PyTorch `forward_cached` single-token logits.
    split_chunk: prefill resumed from copied intermediate snapshots.
    Branches: shared immutable snapshot, A/B/A rollback, dirty unused tail.
    """
    output = Path(output)
    if output.exists():
        raise ValueError("Alignment output exists.")
    lm = BundleLMV4(bundle)
    info, arrays = read_reference(reference, lm)
    old = None
    if stateless:
        old = CoreMLLMV4(bundle, stateless)
        if old.metadata["precision"] != runtime.metadata["precision"]:
            raise ValueError("Compare KV with the stateless package of the same precision.")
    full = (lambda ids: old.model(torch.tensor([list(ids)])).numpy()) if old else (
        lambda ids: lm.model(torch.tensor([list(ids)])).numpy())
    errors, invariants, saved = [], [], {}
    for name, case in info["cases"].items():
        ids = [int(i) for i in arrays[name + "_input_ids"][0]]
        valid = case["valid_prefix"]
        expected = full(ids) if old else arrays[name + "_logits"]
        actual, _ = runtime.run(ids)
        saved[name] = actual
        errors.append({"case": name, "check": "one_shot", **error_distribution(expected[:, :valid], actual[:, :valid])})
        if name == "long" and not old:
            errors.append({"case": name, "check": "one_token",
                           **error_distribution(arrays["long_kv_step_logits"], actual)})
        if len(ids) > 3:
            for split in (3, len(ids) // 2):
                first, state = runtime.run(ids[:split])
                snapshot = tuple(a.copy() for a in state)
                second, _ = runtime.run(ids[split:], snapshot, split)
                chunked = np.concatenate((first, second), axis=1)
                errors.append({"case": name, "check": "split_chunk", "chunks": [split, len(ids) - split],
                               **error_distribution(expected[:, :valid], chunked[:, :valid])})
                invariants.append({"case": name, "check": "split_equals_one_shot", "chunks": [split, len(ids) - split],
                                   **numeric_error(actual, chunked, 0, 0)})
    for name in ("right_pad", "changed_future"):
        invariants.append({"case": name, "check": "valid_prefix_vs_short",
                           **numeric_error(saved["short"], saved[name][:, :16], TOLERANCE, TOLERANCE)})
    prefix = [int(i) for i in arrays["short_input_ids"][0][:7]]
    _, state = runtime.run(prefix)
    original = tuple(a.copy() for a in state)
    branches = []
    for suffix in ([9], [47, 81, 102], [9]):
        actual, _ = runtime.run(suffix, state, len(prefix))
        expected = full([*prefix, *suffix])[:, len(prefix):]
        errors.append({"check": "branch", "suffix": suffix, **error_distribution(expected, actual)})
        branches.append(actual)
    immutable = all(np.array_equal(a, b) for a, b in zip(state, original, strict=True))
    invariants.append({"check": "rollback_A_B_A", **numeric_error(branches[0], branches[2], 0, 0)})
    dirty = tuple(a.copy() for a in state)
    for a in dirty:
        a[:, :, :, len(prefix):, :] = 999
    clean_logits, _ = runtime.run([9], state, len(prefix))
    dirty_logits, _ = runtime.run([9], dirty, len(prefix))
    invariants.append({"check": "unused_tail", **numeric_error(clean_logits, dirty_logits, 0, 0)})
    report = {"format": "vimeml_kv_alignment_v4",
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


def top1(rows):
    return sum(bool(r["orders"]["lm_context_sum"]) and r["orders"]["lm_context_sum"][0] in r["answers"] for r in rows)


def evaluate_tracks(lm, output, reference=None, reference_label=None):
    """Score all four frozen pools; optionally compare per case with a reference run."""
    from vimeml.benchmarks.evaluate_ajimee import rerank, summarize
    from vimeml.posttraining.infer import TRACKS, read_snapshot
    output = Path(output)
    if output.exists():
        raise ValueError("Evaluation output exists.")
    frozen = {track: read_snapshot(track) for track in TRACKS}
    fresh_directory(output)
    summary = {"format": "vimeml_v4_track_evaluation_v1", "model": lm.metadata,
               "reference": str(reference) if reference else None, "reference_label": reference_label,
               "tracks": {}, "blind_read": False, "test_read": False,
               "semantics": "full-vocabulary suffix logP sum; joint context+candidate tokenization; no final EOS, truncation, or tuning; stable ties"}
    for track, original in frozen.items():
        rows = copy.deepcopy(original)
        started = time.perf_counter()
        rerank(lm, rows)
        elapsed = time.perf_counter() - started
        destination = output / track
        destination.mkdir()
        (destination / "scores.jsonl").write_text("".join(
            json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
        result = {"cases": len(rows), "top1": top1(rows), "elapsed_seconds": elapsed,
                  "metrics": summarize(rows)}
        if reference:
            before = [json.loads(line) for line in (Path(reference) / track / "scores.jsonl")
                      .read_text(encoding="utf-8").splitlines() if line.strip()]
            orders = compare_rows(before, rows)
            pairs = paired(before, rows)
            write_json(destination / "comparison.json", {"reference_label": reference_label,
                                                        "orders": orders, "paired": pairs})
            result["comparison"] = {"reference_top1": top1(before),
                **{k: v for k, v in orders.items() if k != "changes"},
                "order_changed_ids": [c["id"] for c in orders["changes"]],
                "top1_changed_ids": [c["id"] for c in orders["changes"] if c["top1_changed"]],
                "paired": pairs["paired"],
                "improved_ids": [c["id"] for c in pairs["changed_cases"] if c["change"] == "improved"],
                "regressed_ids": [c["id"] for c in pairs["changed_cases"] if c["change"] == "regressed"]}
        write_json(destination / "metrics.json", result)
        summary["tracks"][track] = {k: v for k, v in result.items() if k != "metrics"}
        print(json.dumps({"track": track, "top1": result["top1"], "cases": len(rows),
                          "comparison": {k: result["comparison"][k] for k in (
                              "reference_top1", "order_changed_cases", "top1_changed_cases", "max_score_sum_abs")}
                          if reference else None}, ensure_ascii=False), flush=True)
    for track in TRACKS:
        read_snapshot(track)
    write_json(output / "summary.json", summary)
    return summary
