"""vimeml-tiny-zh batched KV step, version 4: v3 plus a `prefix` function and FP16 KV I/O variants.

Brief: docs/zh/mac/mac-zh-kvbatch-v4.md. Changes against `zh_v1_batch3` (v3), after Vime's iPhone memory
profile of v3 (2026-11-20):

1. **`prefix` function.** The client builds the shared prefix KV (and extends it key by key) with `scored`
   and a dummy `allowed`. `prefix` returns only the new tokens' keys and values: no final norm, no head, no
   logsumexp. Inputs are those of `score`: `input_ids` [1,T] and the prefix so far. Its new_key/new_value
   are appended to the prefix (or are the prefix, for the first call with prefix_length 0).
2. **Variants** (the same four functions each):
   - `fp32kv`: v3's FP32 KV inputs and outputs, FP32 compute;
   - `fp16kv`: FP16 `key_cache`, `value_cache`, `prefix_key`, `prefix_value`, `new_key`, `new_value`; FP32
     compute (the graph casts KV to FP32 on the way in and back to FP16 on the way out);
   - `fp16kv-fp16c`: FP16 KV and FP16 compute, except reductions, exp/log and norms, which stay FP32.
   Every variant compresses to the deployed INT8 block-32 weights.

The interface is v3's (zh_v1_batch3 docstring) plus:
  prefix   input_ids int32 [1,T], prefix_length int32 [1], prefix_key/prefix_value (9,1,6,P,64)
           -> new_key, new_value (9,1,6,T,64); token t sits at prefix_length + t
KV tensors are float16 in the fp16kv variants.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vimeml.deployment import v2, zh_v1
from vimeml.deployment import zh_v1_batch3 as z3
from vimeml.deployment.bundle import read_json, write_json
from vimeml.deployment.coreml import coremltools, inspect_spec
from vimeml.training.data import file_sha

FORMAT = "vimeml_coreml_zh_kv_batch_v4"
MERGED_FORMAT = "vimeml_coreml_zh_kv_batch_v4_merged"
ALIGNMENT_FORMAT = "vimeml_kv_batch_alignment_zh_v4"
FUNCTIONS = ("logits", "scored", "score", "prefix")
LAYERS, HEADS, HEAD_DIM, VOCAB, CONTEXT, MAX_BATCH = z3.LAYERS, z3.HEADS, z3.HEAD_DIM, z3.VOCAB, z3.CONTEXT, z3.MAX_BATCH
PREFIX_INPUTS = ("input_ids", "prefix_length", "prefix_key", "prefix_value")
INPUTS = {"logits": z3.STEP_INPUTS, "scored": z3.STEP_INPUTS + ("allowed",), "score": z3.SCORE_INPUTS, "prefix": PREFIX_INPUTS}
KV_INPUTS = ("key_cache", "value_cache", "prefix_key", "prefix_value")
KV_OUTPUTS = ("new_key", "new_value")
# Alignment tolerances per variant: per value (logits, log-probs) and per sentence sum. fp32kv keeps v3's.
# fp16kv: PyTorch with FP16-rounded KV on the zh-v1 weights gave at most 3.6e-3 per value and 1.8e-3 per sum
# (2026-11-20); the bound is about 3x that. fp16kv-fp16c could not be emulated on Windows; its bound only
# catches a broken conversion. For both fp16 variants the deciding gate is the search check.
VARIANTS = {
    "fp32kv": {"kv": "fp32", "compute": "fp32", "tolerance": z3.TOLERANCE, "sum_tolerance": z3.SUM_TOLERANCE},
    "fp16kv": {"kv": "fp16", "compute": "fp32", "tolerance": 1e-2, "sum_tolerance": 1e-2},
    "fp16kv-fp16c": {"kv": "fp16", "compute": "fp16", "tolerance": 1e-1, "sum_tolerance": 2.5e-1},
}
# ops kept in FP32 under FP16 compute: softmax pieces, logsumexp, norms
FP32_OPS = {"reduce_log_sum_exp", "reduce_max", "reduce_sum", "reduce_mean", "exp", "log", "layer_norm", "rsqrt",
            "sqrt", "real_div", "pow", "softmax"}


class PrefixGraphV4(nn.Module):
    """`prefix` function: keys and values of new tokens after the prefix so far; nothing else."""

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, input_ids, prefix_length, prefix_key, prefix_value):
        _, keys, values = z3._layers(self.model, input_ids, prefix_length, (prefix_length, prefix_key, prefix_value), None)
        return keys, values


def graph(model, function, vocab_chunks=1):
    return PrefixGraphV4(model) if function == "prefix" else z3.graph(model, function, vocab_chunks)


def _fp16(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


class TorchRunner:
    """runner(function, feed) -> outputs over PyTorch graphs. kv="fp16" rounds every KV input and output to
    FP16, as the fp16kv packages store them (compute stays FP32)."""

    def __init__(self, model=None, kv="fp32", graphs=None):
        self.graphs = graphs or {f: graph(model, f) for f in FUNCTIONS}
        self.kv = kv
        self.inner = z3.TorchRunner(graphs={f: g for f, g in self.graphs.items() if f != "prefix"})

    @torch.inference_mode()
    def __call__(self, function, feed):
        if self.kv == "fp16":
            feed = {k: (_fp16(v) if k in KV_INPUTS else v) for k, v in feed.items()}
        if function == "prefix":
            k, v = self.graphs["prefix"](*(torch.as_tensor(np.ascontiguousarray(feed[n])) for n in PREFIX_INPUTS))
            out = {"new_key": k.numpy(), "new_value": v.numpy()}
        else:
            out = self.inner(function, feed)
        if self.kv == "fp16":
            out = {k: (_fp16(v) if k in KV_OUTPUTS else v) for k, v in out.items()}
        return out


class CoreMLRunner:
    """The same calls over a merged v4 package: one MLModel per function, KV fed in the package's dtype."""

    def __init__(self, path, compute_units="CPU_ONLY", functions=FUNCTIONS):
        ct = coremltools()
        path = Path(path)
        self.manifest = verify_merged(path)
        self.kv = np.float16 if self.manifest["variant"]["kv"] == "fp16" else np.float32
        unit = getattr(ct.ComputeUnit, compute_units)
        self.models = {f: ct.models.MLModel(str(path / "model.mlpackage"), compute_units=unit, function_name=f)
                       for f in functions if f in self.manifest["functions"]}

    def __call__(self, function, feed):
        feed = {k: np.ascontiguousarray(v.astype(self.kv) if k in KV_INPUTS else v)
                for k, v in feed.items() if k in INPUTS[function]}
        out = self.models[function].predict(feed)
        return {k: np.asarray(v, np.float32) for k, v in out.items()}


def prefix_call(runner, ids, shared=None):
    """`prefix` over ids after `shared` (prefix inputs, or None for none). Returns the prefix inputs for
    everything so far: the old valid slots followed by the new keys and values."""
    feed = {"input_ids": np.array([ids], np.int32), **(shared or z3.no_prefix())}
    out = runner("prefix", feed)
    n = int(feed["prefix_length"][0])
    key = np.concatenate([feed["prefix_key"][:, :, :, :n], out["new_key"]], 3)
    value = np.concatenate([feed["prefix_value"][:, :, :, :n], out["new_value"]], 3)
    return z3.as_prefix(key, value)


def check_prefix(reference, runner, tolerance=z3.TOLERANCE, seed=1121):
    """The prefix built with `prefix` in one call and in two (an extension after junk-padded slots) must give
    the full forward's next-token logits through `logits`; and the two prefixes must agree."""
    rng = np.random.default_rng(seed)
    judge = z3.Judge(False, tolerance)
    worst_kv = 0.0
    for first, more in ((11, 1), (11, 5), (1, 40), (64, 1), (3, 0)):
        ids = rng.integers(8, VOCAB, first + more + 1).astype(np.int32)
        one = prefix_call(runner, ids[:first + more].tolist())
        built = prefix_call(runner, ids[:first].tolist())
        if more:
            padded = z3.as_prefix(built["prefix_key"], built["prefix_value"], slots=first + 2, fill=30.0)
            built = prefix_call(runner, ids[first:first + more].tolist(), padded)
        worst_kv = max(worst_kv, float(np.abs(one["prefix_key"] - built["prefix_key"]).max()),
                       float(np.abs(one["prefix_value"] - built["prefix_value"]).max()))
        want = z3._full(reference, ids)[-1]
        for shared in (one, built):
            padded = z3.as_prefix(shared["prefix_key"], shared["prefix_value"], slots=first + more + 3, fill=30.0)
            out = runner("logits", {"input_ids": ids[None, -1:], "position": np.zeros(1, np.int32),
                                    "key_cache": z3.empty_cache(1, 30.0), "value_cache": z3.empty_cache(1, -30.0), **padded})
            judge(out["logits"][0], want)
    return {"kind": "prefix", "rows": judge.rows, "max_abs": judge.worst, "max_abs_kv_one_vs_extended": worst_kv,
            "rows_outside_tolerance": judge.outside}


def run_checks(reference, runner, functions=FUNCTIONS, variant="fp32kv", strict=False):
    tol = VARIANTS[variant]
    checks = []
    for function in ("logits", "scored"):
        if function in functions:
            for case in z3.CASES:
                checks.append(z3.check_step(reference, runner, function, strict=strict, tolerance=tol["tolerance"], **case))
    if "score" in functions:
        for prefill in [f for f in ("scored", "prefix") if f in functions]:
            checks.append({**z3.check_score(reference, runner, prefill=prefill, sum_tolerance=tol["sum_tolerance"]),
                           "prefill": prefill})
    if "prefix" in functions and "logits" in functions:
        checks.append(check_prefix(reference, runner, tol["tolerance"]))
    return checks


# ---- tracing and conversion -------------------------------------------------------------------------------

def example_inputs(function, **shape):
    if function != "prefix":
        return z3.example_inputs(function, **shape)
    ids, _, plen, pk, pv = z3.example_inputs("score", **{**shape, "batch": 1})
    return ids, plen, pk, pv


@torch.inference_mode()
def trace(model, function, vocab_chunks=1):
    g = graph(model, function, vocab_chunks).eval()
    traced = torch.jit.trace(g, example_inputs(function))
    for shape in ({"batch": 3, "tokens": 1, "slots": 7, "prefix": 2}, {"batch": 1, "tokens": 6, "slots": 1, "prefix": 9}):
        ex = example_inputs(function, **shape)
        for a, b in zip(z3._tuple(traced(*ex)), z3._tuple(g(*ex))):
            torch.testing.assert_close(a, b, atol=z3.TOLERANCE, rtol=z3.TOLERANCE)
    return traced


def _fp16_selector(op):
    return op.op_type not in FP32_OPS


def convert(bundle, output, function, variant):
    ct = coremltools()
    output = Path(output)
    if output.exists():
        raise ValueError("Conversion output exists.")
    if function not in FUNCTIONS or variant not in VARIANTS:
        raise ValueError((function, variant))
    spec = VARIANTS[variant]
    kv = np.float16 if spec["kv"] == "fp16" else np.float32
    lm = zh_v1.BundleLMZh(bundle)
    b = ct.RangeDim(lower_bound=1, upper_bound=MAX_BATCH, default=MAX_BATCH)
    t = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT, default={"score": 8, "prefix": 16}.get(function, 1))
    s = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT, default=8)
    p = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT, default=32)
    tensor = lambda name, shape, dtype=np.float32: ct.TensorType(name=name, shape=shape, dtype=dtype)
    prefix = [tensor("prefix_length", (1,), np.int32), tensor("prefix_key", (LAYERS, 1, HEADS, p, HEAD_DIM), kv),
              tensor("prefix_value", (LAYERS, 1, HEADS, p, HEAD_DIM), kv)]
    if function == "prefix":
        inputs = [tensor("input_ids", (1, t), np.int32)] + prefix
        outputs = [("new_key", kv), ("new_value", kv)]
    elif function == "score":
        inputs = [tensor("input_ids", (b, t), np.int32), tensor("targets", (b, t), np.int32)] + prefix
        outputs = [("target_logprobs", np.float32)]
    else:
        inputs = [tensor("input_ids", (b, t), np.int32), tensor("position", (b,), np.int32),
                  tensor("key_cache", (LAYERS, b, HEADS, s, HEAD_DIM), kv),
                  tensor("value_cache", (LAYERS, b, HEADS, s, HEAD_DIM), kv)] + prefix
        if function == "scored":
            inputs.append(ct.TensorType(name="allowed", shape=(ct.RangeDim(1, z3.MAX_ALLOWED, 160),), dtype=np.int32))
            outputs = [("logsumexp", np.float32), ("allowed_logits", np.float32), ("new_key", kv), ("new_value", kv)]
        else:
            outputs = [("logits", np.float32), ("new_key", kv), ("new_value", kv)]
    precision = (ct.transform.FP16ComputePrecision(op_selector=_fp16_selector) if spec["compute"] == "fp16"
                 else ct.precision.FLOAT32)
    model = ct.convert(trace(lm.model, function), source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=precision, inputs=inputs,
        outputs=[ct.TensorType(name=n, dtype=d) for n, d in outputs], skip_model_load=True)
    model.short_description = f"vimeml-tiny-zh batched KV step v4 ({function}, {variant}); shared prefix."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    zh_v1.finish(output, model, FORMAT, {"kind": "source", "variant": {"name": variant, **spec}, "minimum_ios": 18,
        "bundle": lm.metadata, "function": function, "vocab_chunks": 1, "batch_range": [1, 1 if function == "prefix" else MAX_BATCH],
        "compute_units": "CPU_ONLY", "interface": interface(function, spec["kv"]),
        "graph": f"vimeml.deployment.zh_v1_batch4.{type(graph(lm.model, function)).__name__}",
        "precision_policy": f"KV I/O {spec['kv']}; compute {spec['compute']}"
                            + (f" except {sorted(FP32_OPS)}" if spec["compute"] == "fp16" else ""),
        "policy": "Caller owns all buffers; inputs are never modified.", "conversion_source_sha256": file_sha(Path(__file__))})


def interface(function, kv="fp32"):
    kvt = "float16" if kv == "fp16" else "float32"
    if function == "prefix":
        return {"input_ids": "int32 [1,T]; token t sits at prefix_length + t (< 128)",
                "prefix_length": "int32 [1]; valid prefix slots, 0 = no prefix (P=1 of zeros)",
                "prefix_key/prefix_value": f"{kvt} (9,1,6,P,64); slots >= prefix_length masked but finite",
                "new_key/new_value": f"{kvt} (9,1,6,T,64); append to the prefix's valid slots"}
    out = {k: v.replace("float32 (9", f"{kvt} (9") for k, v in z3.interface(function).items()}
    return out


def compress(source, output, alignment):
    """INT8 block32 (deployed recipe) of a single-function v4 source package whose merged source package passed."""
    ct = coremltools()
    source, output, alignment = map(Path, (source, output, alignment))
    if output.exists():
        raise ValueError("Compression output exists.")
    original = zh_v1.verify_package(source, FORMAT)
    gate = read_json(alignment)
    if (original["kind"] != "source" or gate.get("format") != ALIGNMENT_FORMAT or not gate.get("passed")
            or gate.get("source_manifests", {}).get(original["function"]) != file_sha(source / "manifest.json")):
        raise ValueError("First pass alignment of a merged source package built from this exact package.")
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
    carried = {k: original[k] for k in ("variant", "minimum_ios", "bundle", "function", "vocab_chunks", "batch_range",
                                        "compute_units", "interface", "graph", "policy")}
    zh_v1.finish(output, result, FORMAT, {**carried, "kind": "int8-b32", "recipe": zh_v1.DEPLOYED_RECIPE,
        "source_manifest_sha256": file_sha(source / "manifest.json"), "source_alignment_sha256": file_sha(alignment),
        "compression": {**spec, "weight_threshold": zh_v1.WEIGHT_THRESHOLD, "selection": selection},
        "precision_policy": original["precision_policy"] + "; INT8 block32 parameter matrices; norms unchanged",
        "conversion_source_sha256": file_sha(Path(__file__))})


def merge(packages, output):
    """One multifunction package from single-function v4 packages of one bundle, variant and precision."""
    ct = coremltools()
    output = Path(output)
    if output.exists():
        raise ValueError("Merge output exists.")
    manifests = {f: zh_v1.verify_package(Path(p), FORMAT) for f, p in packages.items()}
    for f, m in manifests.items():
        if m["function"] != f:
            raise ValueError(f"{packages[f]} holds the {m['function']} function, not {f}.")
    first = next(iter(manifests.values()))
    if any(m["bundle"]["bundle_manifest_sha256"] != first["bundle"]["bundle_manifest_sha256"] or m["kind"] != first["kind"]
           or m["variant"] != first["variant"] for m in manifests.values()):
        raise ValueError("Packages differ in bundle, variant or precision.")
    output.mkdir(parents=True)
    desc = ct.utils.MultiFunctionDescriptor()
    for f, p in packages.items():
        desc.add_function(str(Path(p) / "model.mlpackage"), src_function_name="main", target_function_name=f)
    desc.default_function_name = "score" if "score" in packages else next(iter(packages))
    ct.utils.save_multifunction(desc, str(output / "model.mlpackage"))
    files = v2.inventory(output / "model.mlpackage")
    write_json(output / "manifest.json", {"format": MERGED_FORMAT, "status": "complete",
        "architecture": zh_v1.ARCHITECTURE, "kind": first["kind"], "variant": first["variant"],
        "bundle": first["bundle"], "minimum_ios": 18, "compute_units": "CPU_ONLY",
        "functions": {f: {"source_manifest_sha256": file_sha(Path(packages[f]) / "manifest.json"),
                          "batch_range": m["batch_range"], "interface": m["interface"]} for f, m in manifests.items()},
        **({"recipe": first["recipe"]} if "recipe" in first else {}),
        "default_function": desc.default_function_name,
        "loading": "MLModelConfiguration.functionName = " + " | ".join(f'"{f}"' for f in packages) + " (iOS 18); one MLModel per function",
        "package_files": files, "package_bytes": sum(e["bytes"] for e in files.values()),
        "separate_package_bytes": sum(m["package_bytes"] for m in manifests.values()),
        "conversion_source_sha256": file_sha(Path(__file__))})


def verify_merged(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if (manifest.get("format") != MERGED_FORMAT or manifest.get("status") != "complete"
            or manifest.get("architecture") != zh_v1.ARCHITECTURE):
        raise ValueError("Incomplete/unsupported merged v4 package.")
    if v2.inventory(path / "model.mlpackage") != manifest["package_files"]:
        raise ValueError("Package file inventory changed.")
    return manifest


def align(package, bundle, output, stateless=None):
    """Gate a merged v4 package, every function, at its variant's tolerance. Reference: PyTorch FP32 from the
    bundle, or (`stateless`) the stateless INT8 Core ML package (Vime's gate for INT8 packages)."""
    ct = coremltools()
    package, output = Path(package), Path(output)
    if output.exists():
        raise ValueError("Alignment output exists.")
    manifest = verify_merged(package)
    lm = zh_v1.BundleLMZh(bundle)
    if manifest["bundle"]["bundle_manifest_sha256"] != lm.metadata["bundle_manifest_sha256"]:
        raise ValueError("Package belongs to another bundle.")
    if stateless:
        ref_manifest = zh_v1.verify_package(stateless)
        model = ct.models.MLModel(str(Path(stateless) / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
        reference = lambda ids: np.asarray(model.predict({"input_ids": np.asarray(ids, np.int32)})["logits"], np.float32)
        ref_name = f"stateless Core ML {ref_manifest['kind']} ({Path(stateless).name})"
    else:
        reference = z3.torch_reference(lm.model.float().eval())
        ref_name = "PyTorch FP32 full forward (bundle)"
    variant = manifest["variant"]["name"]
    checks = run_checks(reference, CoreMLRunner(package), tuple(manifest["functions"]), variant)
    for c in checks:
        print(json.dumps(c), flush=True)
    passed = bool(checks) and all(c["rows_outside_tolerance"] == 0 for c in checks)
    write_json(output, {"format": ALIGNMENT_FORMAT, "passed": passed, "reference": ref_name, "variant": variant,
        "coreml_manifest_sha256": file_sha(package / "manifest.json"), "package": package.name, "kind": manifest["kind"],
        "source_manifests": {f: e["source_manifest_sha256"] for f, e in manifest["functions"].items()},
        "tolerance": {"atol": VARIANTS[variant]["tolerance"], "rtol": VARIANTS[variant]["tolerance"],
                      "per_sum": VARIANTS[variant]["sum_tolerance"]}, "checks": checks,
        "method": "zh_v1_batch3 checks (logits/scored with junk in masked slots; score sums with the prefix from "
                  "scored and from prefix) plus check_prefix (one-call vs extended prefix, next-token logits)",
        "conversion_source_sha256": file_sha(Path(__file__))})
    print(f"alignment {'passed' if passed else 'FAILED'}: max {max(c['max_abs'] for c in checks):.3g}")
    return passed
