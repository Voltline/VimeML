"""vimeml-tiny-zh-v1 Neural Engine probe (step 0): placement, latency, memory, power and ranking of FP16 graphs.

Brief: docs/zh/mac/mac-zh-v1-ane-probe.md. Every stage writes a fresh output. Not a deployment interface.

  check     PyTorch: both layouts equal v3 in FP32 (bundle weights); FP16 PyTorch ranking preview
  convert   one multifunction FP16 package of the probe grid: --layout bt|ane --weights fp16|int8c|int8b32 --io fp16|fp32
  plan      MLComputePlan placement per function (which ops would run on the ANE)
  accuracy  sentence-scoring rankings and beam-step top-5 vs PyTorch FP32, for a package on a compute unit
  measure   warm latency per function and compute unit, interleaved with the shipped v3 package on CPU
  memory    phys_footprint per configuration, fresh process each, models compiled outside the measured process
  loop      run one function for N seconds (for powermetrics in another terminal)
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/zh/deployment"))

from vimeml.deployment import v2, zh_ane as za, zh_v1, zh_v1_batch3 as z3  # noqa: E402
from vimeml.deployment.bundle import environment, read_json, write_json  # noqa: E402
from vimeml.training.data import file_sha  # noqa: E402
from coreml_zh_v1_batch import BUNDLE, RELEASE, fixture_rows, fresh  # noqa: E402

FORMAT = "vimeml_zh_v1_ane_probe_v1"
V3 = RELEASE / "coreml/ios18-int8-block32-kvbatch-v3"
CASES = ROOT / "outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl"
SCORE_IN = ("input_ids", "targets", "prefix_length", "prefix_key", "prefix_value")
STEP_IN = ("input_ids", "position", "key_cache", "value_cache", "prefix_length", "prefix_key", "prefix_value", "allowed")
UNITS = ("CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU", "ALL")


# ---- runners -----------------------------------------------------------------------------------------------

class TorchProbe:
    """runner(name, feed) over the PyTorch probe graphs; dtype float32 or float16 (FP16 preview on CPU)."""

    def __init__(self, model, layout, half=False):
        import torch
        self.torch, self.layout, self.half = torch, layout, half
        import copy
        self.model = copy.deepcopy(model)
        self.model = (self.model.half() if half else self.model.float()).eval()
        self.graphs = {}

    def __call__(self, name, feed):
        torch = self.torch
        kind, _ = za.functions()[name]
        g = self.graphs.setdefault(kind, za.graph(self.model, kind, self.layout))
        dt = torch.float16 if self.half else torch.float32
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x)).to(dt if x.dtype.kind == "f" else torch.int32)
        with torch.inference_mode():
            out = g(*(t(feed[n]) for n in (SCORE_IN if kind == "score" else STEP_IN)))
        if kind == "score":
            return {"target_logprobs": out.float().numpy()}
        return dict(zip(("logsumexp", "allowed_logits", "new_key", "new_value"), (o.float().numpy() for o in out)))


class CoreMLProbe:
    def __init__(self, path, unit="CPU_AND_NE", names=None, compiled=None):
        import coremltools as ct
        self.manifest = read_json(Path(path) / "manifest.json")
        if self.manifest.get("format") != FORMAT:
            raise ValueError("Not a probe package.")
        self.layout, self.io = self.manifest["layout"], self.manifest["io"]
        cu = getattr(ct.ComputeUnit, unit)
        names = names or list(self.manifest["functions"])
        if compiled:
            self.models = {n: ct.models.CompiledMLModel(str(compiled), compute_units=cu, function_name=n) for n in names}
        else:
            self.models = {n: ct.models.MLModel(str(Path(path) / "model.mlpackage"), compute_units=cu, function_name=n)
                           for n in names}

    def cast(self, feed):
        f = np.float16 if self.io == "fp16" else np.float32
        return {k: np.ascontiguousarray(v.astype(f) if v.dtype.kind == "f" else v.astype(np.int32)) for k, v in feed.items()}

    def __call__(self, name, feed):
        out = self.models[name].predict(self.cast(feed))
        return {k: np.asarray(v, np.float32) for k, v in out.items()}


# ---- feeds ------------------------------------------------------------------------------------------------

def prefix_feed(key, value, length, slots, layout):
    """Prefix inputs from v3-layout (9,1,6,n,64) prompt keys/values, zero-padded to `slots`."""
    n = key.shape[3]
    k = np.zeros((z3.LAYERS, 1, z3.HEADS, slots, z3.HEAD_DIM), np.float32)
    v = np.zeros_like(k)
    k[:, :, :, :n], v[:, :, :, :n] = key, value
    if layout == "ane":
        k, v = za.key_to_ane(k), za.value_to_ane(v)
    return {"prefix_length": np.array([length], np.int32), "prefix_key": k, "prefix_value": v}


def random_feed(name, layout, rng):
    kind, shape = za.functions()[name]
    r = lambda *s: rng.standard_normal(s).astype(np.float32)
    if kind == "score":
        b, t, p = shape
        k, v = r(z3.LAYERS, 1, z3.HEADS, p, z3.HEAD_DIM), r(z3.LAYERS, 1, z3.HEADS, p, z3.HEAD_DIM)
        feed = {"input_ids": rng.integers(8, z3.VOCAB, (b, t)).astype(np.int32),
                "targets": rng.integers(8, z3.VOCAB, (b, t)).astype(np.int32), "prefix_length": np.array([p - 2], np.int32)}
        pk = {"prefix_key": k, "prefix_value": v}
    else:
        b, p, s = shape
        k, v = r(z3.LAYERS, 1, z3.HEADS, p, z3.HEAD_DIM), r(z3.LAYERS, 1, z3.HEADS, p, z3.HEAD_DIM)
        ck, cv = r(z3.LAYERS, b, z3.HEADS, s, z3.HEAD_DIM), r(z3.LAYERS, b, z3.HEADS, s, z3.HEAD_DIM)
        feed = {"input_ids": rng.integers(8, z3.VOCAB, (b, 1)).astype(np.int32),
                "position": np.full(b, s - 1, np.int32), "prefix_length": np.array([p - 2], np.int32),
                "allowed": rng.choice(np.arange(8, z3.VOCAB), za.ALLOWED, replace=False).astype(np.int32)}
        pk = {"prefix_key": k, "prefix_value": v, "key_cache": ck, "value_cache": cv}
    if layout == "ane":
        pk = {n: (za.key_to_ane(x) if "key" in n else za.value_to_ane(x)) for n, x in pk.items()}
    return {**feed, **pk}


def v3_feed(name, rng):
    """The same work for the shipped v3 package (flexible shapes): score -> `score`, step -> `scored`."""
    feed = random_feed(name, "bt", rng)
    kind, _ = za.functions()[name]
    return ("score" if kind == "score" else "scored"), feed


# ---- ranking cases -----------------------------------------------------------------------------------------

def ranking_cases(processor, limit=None):
    """(id, context, candidates): the search fixture's top lists plus the sentence-scoring fixture."""
    from coreml_zh_v1_batch2 import sequence_fixture_cases
    if not CASES.exists():
        raise SystemExit(f"Missing {CASES} (the search fixture of the v3 task).")
    out = []
    for line in open(CASES, encoding="utf-8"):
        c = json.loads(line)
        texts = list(dict.fromkeys(x["text"] for x in c["top"]))
        if len(texts) > 1:
            out.append((c["id"], c["context"], texts))
    out += [(f"sequence/{i}", ctx, cands) for i, (ctx, cands) in enumerate(sequence_fixture_cases())]
    return out[:limit] if limit else out


def split_case(processor, context, candidates):
    seqs = [[2] + processor.encode(context + c) for c in candidates]
    common = 0
    while all(len(s) > common + 1 for s in seqs) and len({s[common] for s in seqs}) == 1:
        common += 1
    common = max(common, 1)
    return seqs[0][:common], [s[common:] for s in seqs]


def score_shape(prompt_len, cont_len):
    fits = [s for s in za.SCORE_SHAPES if prompt_len <= s[2] and cont_len <= s[1]]
    return min(fits, key=lambda s: (s[2], s[1])) if fits else None


class PromptCache:
    """Prompt keys/values from PyTorch FP32 (v3 layout); the probe measures the scoring and step graphs only."""

    def __init__(self, model):
        self.runner = z3.TorchRunner(model)

    def __call__(self, ids):
        out, _ = z3.prompt_call(self.runner, "scored", ids)
        return out["new_key"], out["new_value"]


def probe_scores(runner, layout, prompt, head, conts):
    """Sums of target log-probs per continuation through the probe's score functions (rows padded to B)."""
    shape = score_shape(len(head) - 1, max(len(c) for c in conts))
    if shape is None:
        return None
    b, t, p = shape
    if len(head) > 1:
        k, v = prompt(head[:-1])
        pre = prefix_feed(k, v, len(head) - 1, p, layout)
    else:
        pre = prefix_feed(np.zeros((z3.LAYERS, 1, z3.HEADS, 0, z3.HEAD_DIM), np.float32),
                          np.zeros((z3.LAYERS, 1, z3.HEADS, 0, z3.HEAD_DIM), np.float32), 0, p, layout)
    sums = []
    for start in range(0, len(conts), b):
        part = conts[start:start + b]
        ids, tg = np.zeros((b, t), np.int32), np.zeros((b, t), np.int32)
        for r in range(b):
            c = part[r] if r < len(part) else part[0]
            ids[r, :len(c)] = [head[-1]] + list(c[:-1])
            tg[r, :len(c)] = c
        lp = runner(za.score_name(b, t, p), {"input_ids": ids, "targets": tg, **pre})["target_logprobs"].astype(np.float64)
        sums += [float(lp[r, :len(c)].sum()) for r, c in enumerate(part)]
    return sums


def compare_rank(got, want):
    order = lambda xs: sorted(range(len(xs)), key=lambda i: -xs[i])
    g, w = order(got), order(want)
    flips = sum(1 for i in range(len(got)) for j in range(i + 1, len(got)) if (got[i] - got[j]) * (want[i] - want[j]) < 0)
    return {"top1": g[0] == w[0], "exact": g == w, "flipped_pairs": flips,
            "max_abs": max(abs(x - y) for x, y in zip(got, want))}


def summarize(rows):
    if not rows:
        return {}
    return {"cases": len(rows), "top1_equal": sum(r["top1"] for r in rows), "exact_order": sum(r["exact"] for r in rows),
            "flipped_pairs": sum(r["flipped_pairs"] for r in rows),
            "max_abs_sum_diff": max(r["max_abs"] for r in rows),
            "median_max_abs_sum_diff": statistics.median(r["max_abs"] for r in rows)}


def ranking_check(processor, model, runner, layout, limit=None, baseline=None):
    """Probe (and optionally the shipped v3 runner) vs PyTorch FP32 on every ranking case that fits the grid."""
    prompt, reference = PromptCache(model), z3.TorchRunner(model)
    rows, base_rows, skipped = [], [], []
    for cid, context, cands in ranking_cases(processor, limit):
        head, conts = split_case(processor, context, cands)
        want, _ = z3.score_continuations(reference, head, conts)
        got = probe_scores(runner, layout, prompt, head, conts)
        if got is None:
            skipped.append(cid)
            continue
        rows.append({"id": cid, "candidates": len(cands), **compare_rank(got, want)})
        if baseline is not None:
            b, _ = z3.score_continuations(baseline, head, conts)
            base_rows.append({"id": cid, **compare_rank(b, want)})
    return {"probe": summarize(rows), "v3_int8_cpu": summarize(base_rows), "skipped": skipped, "rows": rows}


def step_check(processor, model, runner, layout, context="晚上一起吃饭吗"):
    """16 beam rows after one prompt (fixture_rows): log-probs over 256 allowed ids vs PyTorch FP32."""
    ref = z3.TorchRunner(model)
    prompt = [2] + processor.encode(context)
    rows = [r["prefix_ids"][1:] + [r["token"]] for r in fixture_rows(processor)]
    out, shared = z3.prompt_call(ref, "scored", prompt)
    rng = np.random.default_rng(5)
    states = []
    for r in rows:
        own = r[:-1]
        if not own:
            states.append((np.zeros((z3.LAYERS, z3.HEADS, 0, z3.HEAD_DIM), np.float32),) * 2)
            continue
        o = ref("scored", {"input_ids": np.array([own], np.int32), "position": np.zeros(1, np.int32),
                           "key_cache": z3.empty_cache(), "value_cache": z3.empty_cache(), "allowed": np.array([0], np.int32), **shared})
        states.append((o["new_key"][:, 0], o["new_value"][:, 0]))
    longest = max(s[0].shape[2] for s in states)
    fits = [s for s in za.STEP_SHAPES if s[2] >= longest and s[1] >= len(prompt)]
    b, p, s = min(fits, key=lambda x: (x[1], x[2]))
    k, v, position = z3.pack(states, s)
    ids = np.array([[r[-1]] for r in rows], np.int32)
    full = ref("logits", {"input_ids": ids, "position": position, "key_cache": k, "value_cache": v, **shared})["logits"]
    allowed = np.unique(np.concatenate([np.argsort(-full[i])[:16] for i in range(len(rows))]))
    allowed = np.concatenate([allowed, rng.choice(np.setdiff1d(np.arange(8, z3.VOCAB), allowed), za.ALLOWED - len(allowed), replace=False)]).astype(np.int32)
    pre = prefix_feed(shared["prefix_key"][:, :, :, :len(prompt)], shared["prefix_value"][:, :, :, :len(prompt)], len(prompt), p, layout)
    if layout == "ane":
        k, v = za.key_to_ane(k), za.value_to_ane(v)
    o = runner(za.step_name(b, p, s), {"input_ids": ids, "position": position, "key_cache": k, "value_cache": v,
                                       "allowed": allowed, **pre})
    got = o["allowed_logits"] - o["logsumexp"][:, None]
    want = z3._log_softmax(full)[:, allowed]
    top = lambda x: list(np.argsort(-x)[:5])
    return {"rows": len(rows), "function": za.step_name(b, p, s), "max_abs_logprob": float(np.abs(got - want).max()),
            "top1_equal": int(sum(top(got[i])[0] == top(want[i])[0] for i in range(len(rows)))),
            "top5_order_equal": int(sum(top(got[i]) == top(want[i]) for i in range(len(rows))))}


# ---- stages ------------------------------------------------------------------------------------------------

def load_model(bundle):
    lm = zh_v1.BundleLMZh(bundle)
    return lm, lm.model.float().eval()


def stage_check(a):
    import torch
    torch.set_num_threads(4)
    out = fresh(a.output)
    lm, model = load_model(a.bundle)
    result = {"fp32_equal_v3": {}, "fp16_preview": {}}
    v3s, v3t = z3.graph(model, "score"), z3.graph(model, "scored")
    with torch.inference_mode():
        for name, (kind, shape) in za.functions().items():
            want = v3s(*za.example(kind, shape, "bt")) if kind == "score" else v3t(*za.example(kind, shape, "bt"))[:2]
            for layout in za.LAYOUTS:
                got = za.graph(model, kind, layout)(*za.example(kind, shape, layout))
                got = got if kind == "score" else got[:2]
                diff = (got - want).abs().max() if kind == "score" else max((x - y).abs().max() for x, y in zip(got, want))
                result["fp32_equal_v3"][f"{layout}/{name}"] = float(diff)
    worst = max(result["fp32_equal_v3"].values())
    if worst > 3e-4:
        raise AssertionError(result["fp32_equal_v3"])
    for layout in za.LAYOUTS:
        try:
            runner = TorchProbe(lm.model, layout, half=True)
            result["fp16_preview"][layout] = {"ranking": ranking_check(lm.processor, model, runner, layout, a.limit),
                                              "step": step_check(lm.processor, model, runner, layout)}
            for r in result["fp16_preview"][layout]["ranking"]["rows"]:
                r.pop("id", None)
        except (RuntimeError, NotImplementedError) as e:
            result["fp16_preview"][layout] = {"error": str(e)[:300]}
    write_json(out, {"format": "vimeml_zh_v1_ane_probe_check", "passed": True, **result,
                     "note": "FP16 preview is PyTorch half on CPU, not Core ML; indicative only.", "environment": environment()})
    print(f"FP32 equal to v3: max {worst:.3g}")
    for layout, r in result["fp16_preview"].items():
        print(layout, json.dumps(r if "error" in r else {"ranking": r["ranking"]["probe"], "step": r["step"]}, ensure_ascii=False))


def stage_convert(a):
    import coremltools as ct
    import torch
    from coremltools.optimize import coreml as opt
    out = fresh(a.output)
    lm, model = load_model(a.bundle)
    parts = out / "parts"
    parts.mkdir(parents=True)
    fdt = np.float16 if a.io == "fp16" else np.float32
    names = a.functions or list(za.functions())
    records = {}
    for name in names:
        kind, shape = za.functions()[name]
        ex = za.example(kind, shape, a.layout)
        labels = SCORE_IN if kind == "score" else STEP_IN
        inputs = [ct.TensorType(name=n, shape=tuple(x.shape), dtype=np.int32 if x.dtype == torch.int32 else fdt)
                  for n, x in zip(labels, ex)]
        outputs = ["target_logprobs"] if kind == "score" else ["logsumexp", "allowed_logits", "new_key", "new_value"]
        t0 = time.perf_counter()
        mlm = ct.convert(za.trace(model, name, a.layout), source="pytorch", convert_to="mlprogram",
                         minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16,
                         inputs=inputs, outputs=[ct.TensorType(name=n, dtype=fdt) for n in outputs], skip_model_load=True)
        if a.weights != "fp16":
            cfg = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", weight_threshold=zh_v1.WEIGHT_THRESHOLD,
                                              **({"granularity": "per_channel"} if a.weights == "int8c"
                                                 else {"granularity": "per_block", "block_size": 32}))
            mlm = opt.linear_quantize_weights(mlm, config=opt.OptimizationConfig(global_config=cfg))
        mlm.save(str(parts / f"{name}.mlpackage"))
        records[name] = {"kind": kind, "shape": list(shape), "inputs": {n: list(x.shape) for n, x in zip(labels, ex)},
                         "convert_s": round(time.perf_counter() - t0, 1)}
        print(name, records[name]["convert_s"], "s", flush=True)
    desc = ct.utils.MultiFunctionDescriptor()
    for name in names:
        desc.add_function(str(parts / f"{name}.mlpackage"), src_function_name="main", target_function_name=name)
    desc.default_function_name = names[0]
    ct.utils.save_multifunction(desc, str(out / "model.mlpackage"))
    files = v2.inventory(out / "model.mlpackage")
    write_json(out / "manifest.json", {"format": FORMAT, "layout": a.layout, "weights": a.weights, "io": a.io,
        "compute_precision": "fp16", "minimum_ios": 18, "functions": records, "bundle": lm.metadata,
        "package_files": files, "package_bytes": sum(e["bytes"] for e in files.values()),
        "separate_bytes": {n: sum(e["bytes"] for e in v2.inventory(parts / f"{n}.mlpackage").values()) for n in names},
        "source_sha256": {"zh_ane.py": file_sha(ROOT / "src/vimeml/deployment/zh_ane.py"), "cli": file_sha(Path(__file__))},
        "environment": environment()})
    print(f"{out.name}: {sum(e['bytes'] for e in files.values()):,} bytes")


def stage_plan(a):
    import shutil
    import tempfile
    import coremltools as ct
    from collections import Counter
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.compute_device import MLCPUComputeDevice, MLGPUComputeDevice, MLNeuralEngineComputeDevice
    out = fresh(a.output)
    kinds = {MLCPUComputeDevice: "cpu", MLGPUComputeDevice: "gpu", MLNeuralEngineComputeDevice: "ane"}
    tmp = Path(tempfile.mkdtemp())
    result = {}
    try:
        compiled = ct.utils.compile_model(str(Path(a.package) / "model.mlpackage"), str(tmp / "m.mlmodelc"))
        for unit in a.units:
            plan = MLComputePlan.load_from_path(path=str(compiled), compute_units=getattr(ct.ComputeUnit, unit))
            for fname, fn in plan.model_structure.program.functions.items():
                where, cost, off = Counter(), Counter(), Counter()
                for op in fn.block.operations:
                    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    if usage is None:
                        continue
                    dev = kinds.get(type(usage.preferred_compute_device), "other")
                    est = plan.get_estimated_cost_for_mlprogram_operation(op)
                    where[dev] += 1
                    cost[dev] += est.weight if est else 0.0
                    if dev != "ane":
                        ane_ok = any(isinstance(d, MLNeuralEngineComputeDevice) for d in usage.supported_compute_devices)
                        off[f"{op.operator_name}{'' if ane_ok else ' (ANE unsupported)'}"] += 1
                result[f"{unit}/{fname}"] = {"ops_by_device": dict(where),
                                             "cost_by_device": {k: round(v, 4) for k, v in cost.items()},
                                             "not_on_ane": dict(off.most_common())}
                print(unit, fname, dict(where), {k: round(v, 3) for k, v in cost.items()}, flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    write_json(out, {"format": "vimeml_zh_v1_ane_probe_plan", "package": Path(a.package).name, "plans": result,
                     "environment": environment()})


def stage_accuracy(a):
    out = fresh(a.output)
    lm, model = load_model(a.bundle)
    runner = CoreMLProbe(a.package, a.unit)
    baseline = z3.CoreMLRunner(a.v3, functions=("scored", "score")) if a.v3 else None
    ranking = ranking_check(lm.processor, model, runner, runner.layout, a.limit, baseline)
    step = step_check(lm.processor, model, runner, runner.layout)
    write_json(out, {"format": "vimeml_zh_v1_ane_probe_accuracy", "package": Path(a.package).name, "unit": a.unit,
                     "ranking": ranking, "step": step, "reference": "PyTorch FP32 (bundle); prompt keys/values from "
                     "PyTorch FP32 for the probe, from the package itself for v3", "environment": environment()})
    print(json.dumps({"probe": ranking["probe"], "v3_int8_cpu": ranking["v3_int8_cpu"], "skipped": len(ranking["skipped"]),
                      "step": step}, ensure_ascii=False))


def timed(fn, feed, repeats, warmup):
    for _ in range(warmup):
        fn(feed)
    xs = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn(feed)
        xs.append((time.perf_counter() - t) * 1000)
    xs.sort()
    return {"p50_ms": round(statistics.median(xs), 3), "p95_ms": round(xs[int(0.95 * (len(xs) - 1))], 3)}


def stage_measure(a):
    import coremltools as ct
    out = fresh(a.output)
    rng = np.random.default_rng(1119)
    names = a.functions or list(za.functions())
    loaded, loads = {}, {}
    for item in a.packages:
        label, _, path = item.partition("=")
        for unit in a.units:
            t = time.perf_counter()
            probe = CoreMLProbe(path, unit, names)
            first = {}
            for n in names:
                t1 = time.perf_counter()
                probe(n, random_feed(n, probe.layout, rng))
                first[n] = round((time.perf_counter() - t1) * 1000, 1)
            loads[f"{label}/{unit}"] = {"load_ms": round((time.perf_counter() - t) * 1000 - sum(first.values()), 1),
                                        "first_call_ms": first}
            loaded[f"{label}/{unit}"] = probe
    base = {f: ct.models.MLModel(str(Path(a.v3) / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY, function_name=f)
            for f in ("score", "scored")}
    results = []
    for n in names:
        fn, v3feed = v3_feed(n, rng)
        feeds = {k: p.cast(random_feed(n, p.layout, rng)) for k, p in loaded.items()}
        for rep in range(3):
            row = {"function": n, "pass": rep, "v3_int8_cpu": timed(base[fn].predict, v3feed, a.repeats, a.warmup)}
            for k, p in loaded.items():
                row[k] = timed(p.models[n].predict, feeds[k], a.repeats, a.warmup)
            results.append(row)
            print(json.dumps({k: (v["p50_ms"] if isinstance(v, dict) else v) for k, v in row.items()}), flush=True)
    write_json(out, {"format": "vimeml_zh_v1_ane_probe_latency", "loads": loads, "results": results,
                     "note": "Mac python predict() wall time incl. input conversion, ms; v3 = shipped INT8 package, CPU_ONLY, "
                             "same B/T/P/S; 3 interleaved passes", "environment": environment()})


def memory_child(spec):
    import coremltools as ct
    rng = np.random.default_rng(1)
    step, score = spec["step"], spec["score"]
    layout = spec.get("layout", "bt")
    steps = [random_feed(step, layout, rng) for _ in range(spec["steps"])]
    scores = [random_feed(score, layout, rng) for _ in range(2)]
    cu = getattr(ct.ComputeUnit, spec["unit"])
    from coreml_zh_v1_batch2 import snap
    mem = {"start": snap()}
    if spec["kind"] == "v3":
        m_step = ct.models.CompiledMLModel(spec["compiled"], compute_units=cu, function_name="scored")
        m_score = ct.models.CompiledMLModel(spec["compiled"], compute_units=cu, function_name="score")
        call_step, call_score = m_step.predict, m_score.predict
    else:
        f = np.float16 if spec["io"] == "fp16" else np.float32
        cast = lambda d: {k: (v.astype(f) if v.dtype.kind == "f" else v) for k, v in d.items()}
        steps, scores = [cast(x) for x in steps], [cast(x) for x in scores]
        m_step = ct.models.CompiledMLModel(spec["compiled"], compute_units=cu, function_name=step)
        m_score = ct.models.CompiledMLModel(spec["compiled"], compute_units=cu, function_name=score)
        call_step, call_score = m_step.predict, m_score.predict
    mem["after_load"] = snap()
    for r in range(spec["rounds"]):
        for x in steps:
            call_step(x)
        call_score(scores[r % 2])
        if r == 0:
            mem["after_first_round"] = snap()
    mem["after_rounds"] = snap()
    return mem


def stage_memory(a):
    import coremltools as ct
    out = fresh(a.output)
    cache = Path(a.compiled)
    cache.mkdir(parents=True, exist_ok=True)
    configs = [{"label": "v3-int8/CPU_ONLY", "kind": "v3", "package": str(a.v3), "unit": "CPU_ONLY"}]
    for item in a.packages:
        label, _, rest = item.partition("=")
        path, _, unit = rest.rpartition(":")
        m = read_json(Path(path) / "manifest.json")
        configs.append({"label": f"{label}/{unit}", "kind": "probe", "package": path, "unit": unit,
                        "layout": m["layout"], "io": m["io"]})
    for c in configs:
        p = Path(c["package"])
        target = cache / f"{p.name}-{file_sha(p / 'manifest.json')[:12]}.mlmodelc"
        if not target.exists():
            ct.utils.compile_model(str(p / "model.mlpackage"), str(target))
        c.update(compiled=str(target), step=a.step, score=a.score, steps=a.steps, rounds=a.rounds)
    results = {}
    for c in configs:
        subprocess.run([sys.executable, __file__, "memory-child", json.dumps({**c, "rounds": 1})], capture_output=True,
                       text=True, check=True, cwd=ROOT)            # warm-up: the ANE compiles and caches its program
        runs = []
        for _ in range(a.runs):
            res = subprocess.run([sys.executable, __file__, "memory-child", json.dumps(c)], capture_output=True, text=True,
                                 check=True, cwd=ROOT)
            runs.append(json.loads(res.stdout.strip().splitlines()[-1]))
        rel = lambda r, k, f="phys_footprint_mib": r[k][f] - r["start"]["phys_footprint_mib"]
        med = lambda f: round(statistics.median(f(r) for r in runs), 1)
        row = {k: med(lambda r, k=k: rel(r, k)) for k in runs[0] if k != "start"}
        row["peak"] = med(lambda r: rel(r, "after_rounds", "peak_phys_footprint_mib"))
        row["min_peak"] = round(min(rel(r, "after_rounds", "peak_phys_footprint_mib") for r in runs), 1)
        results[c["label"]] = {"median_mib": row, "runs": runs, "spec": c}
        print(c["label"], json.dumps(row), flush=True)
    write_json(out, {"format": "vimeml_zh_v1_ane_probe_memory", "results": results,
                     "workload": f"per round: {a.steps} x {a.step}, 1 x {a.score} (v3: scored/score on the same shapes)",
                     "note": "phys_footprint of this process relative to `start` (coremltools imported, inputs built); "
                             "ANE programs live partly in system daemons and are not counted here; one unmeasured "
                             "warm-up process per configuration first", "environment": environment()})


def stage_loop(a):
    rng = np.random.default_rng(3)
    if a.package == "v3":
        import coremltools as ct
        fn, feed = v3_feed(a.function, rng)
        m = ct.models.MLModel(str(Path(a.v3) / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY, function_name=fn)
        call = lambda: m.predict(feed)
    else:
        p = CoreMLProbe(a.package, a.unit, [a.function])
        feed = p.cast(random_feed(a.function, p.layout, rng))
        call = lambda: p.models[a.function].predict(feed)
    for _ in range(20):
        call()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < a.seconds:
        call()
        n += 1
    el = time.perf_counter() - t0
    print(json.dumps({"calls": n, "seconds": round(el, 2), "calls_per_s": round(n / el, 1), "ms_per_call": round(el / n * 1000, 3)}))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "memory-child":
        print(json.dumps(memory_child(json.loads(sys.argv[2]))))
        return
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("check"); s.add_argument("output"); s.add_argument("--limit", type=int)
    s = sub.add_parser("convert"); s.add_argument("output")
    s.add_argument("--layout", required=True, choices=za.LAYOUTS)
    s.add_argument("--weights", required=True, choices=("fp16", "int8c", "int8b32"))
    s.add_argument("--io", default="fp16", choices=("fp16", "fp32"))
    s.add_argument("--functions", nargs="+", choices=list(za.functions()))
    s = sub.add_parser("plan"); s.add_argument("package"); s.add_argument("output")
    s.add_argument("--units", nargs="+", default=["CPU_AND_NE"], choices=UNITS)
    s = sub.add_parser("accuracy"); s.add_argument("package"); s.add_argument("output")
    s.add_argument("--unit", default="CPU_AND_NE", choices=UNITS); s.add_argument("--limit", type=int)
    s = sub.add_parser("measure"); s.add_argument("output"); s.add_argument("packages", nargs="+", help="label=<package>")
    s.add_argument("--units", nargs="+", default=["CPU_ONLY", "CPU_AND_NE"], choices=UNITS)
    s.add_argument("--functions", nargs="+", choices=list(za.functions()))
    s.add_argument("--repeats", type=int, default=100); s.add_argument("--warmup", type=int, default=20)
    s = sub.add_parser("memory"); s.add_argument("output"); s.add_argument("packages", nargs="+", help="label=<package>:<unit>")
    s.add_argument("--step", default=za.step_name(*za.STEP_SHAPES[0]), choices=[n for n, (k, _) in za.functions().items() if k == "step"])
    s.add_argument("--score", default=za.score_name(16, 16, 32), choices=[n for n, (k, _) in za.functions().items() if k == "score"])
    s.add_argument("--steps", type=int, default=16); s.add_argument("--rounds", type=int, default=20)
    s.add_argument("--runs", type=int, default=5)
    s.add_argument("--compiled", default=str(ROOT / "outputs/deployment/compiled-for-memory"))
    s = sub.add_parser("loop"); s.add_argument("package", help="probe package dir, or 'v3'"); s.add_argument("function")
    s.add_argument("--unit", default="CPU_AND_NE", choices=UNITS); s.add_argument("--seconds", type=float, default=30)
    for sp in sub.choices.values():
        sp.add_argument("--bundle", default=str(BUNDLE))
        sp.add_argument("--v3", default=str(V3))
    a = p.parse_args()
    {"check": stage_check, "convert": stage_convert, "plan": stage_plan, "accuracy": stage_accuracy,
     "measure": stage_measure, "memory": stage_memory, "loop": stage_loop}[a.stage](a)


if __name__ == "__main__":
    main()
