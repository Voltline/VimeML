"""vimeml-tiny-zh batched KV step v4 (v3 + `prefix` function, FP16 KV variants): Mac stages.

Brief: docs/zh/mac/mac-zh-kvbatch-v4.md. Every stage writes a fresh output.

  check      PyTorch: the four functions (eager and traced) vs the full forward, FP32 and FP16-rounded KV
  build      one variant end to end: convert the four functions, merge, align (PyTorch), compress to INT8,
             merge, align against the stateless INT8 package
  loadorder  macOS phys_footprint while loading functions in a given order, then a fixed workload; one fresh
             process per run, models from .mlmodelc compiled outside the measured process
  measure    latency of prefix building, prefix extension, a beam step and sentence scoring, v3 vs v4 variants
  search     the per-key-position beam search with a v4 package vs the FP32 fixture and a reference package
  fixture    Vime fixtures: prefix in one and two calls, the next step after it, sentence scoring
  ios        client resource (compiled package + manifest section) for Vime
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/zh/deployment"))

from vimeml.deployment import v2, zh_v1, zh_v1_batch3 as z3, zh_v1_batch4 as z4  # noqa: E402
from vimeml.deployment.bundle import environment, fresh_directory, read_json, write_json  # noqa: E402
from vimeml.training.data import file_sha  # noqa: E402

RELEASE = ROOT / "artifacts/releases/vimeml-tiny-zh-v1-hf-v1"
BUNDLE = RELEASE / "inference"
STATELESS_INT8 = RELEASE / "coreml/ios18-int8-block32"
V3_INT8 = RELEASE / "coreml/ios18-int8-block32-kvbatch-v3"
READINGS = ROOT / "outputs/zh-readings/dict/vime_pinyin.dict.yaml"
CASES = ROOT / "outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl"
LAYERS, HEADS, HEAD_DIM, VOCAB = z3.LAYERS, z3.HEADS, z3.HEAD_DIM, z3.VOCAB
ORDERS = ("score>scored", "scored>score", "score>logits", "logits>score", "score>prefix", "prefix>score",
          "score>prefix>scored")


def fresh(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} exists; choose a new output.")
    return path


def footprint():
    """macOS phys_footprint and its lifetime maximum of this process, MiB (task_info TASK_VM_INFO)."""
    from coreml_zh_v1_batch2 import footprint as f
    return f()


def snap():
    now, peak = footprint()
    return {"phys_footprint_mib": round(now, 1), "peak_phys_footprint_mib": round(peak, 1)}


def timed(model, feed, repeats, warmup):
    for _ in range(warmup):
        model.predict(feed)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.predict(feed)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"p50_ms": round(statistics.median(times), 3), "p95_ms": round(times[int(0.95 * (len(times) - 1))], 3)}


# ---- check and build --------------------------------------------------------------------------------------

def stage_check(a):
    import torch
    torch.set_num_threads(4)
    model = zh_v1.BundleLMZh(a.bundle).model.float().eval()
    reference = z3.torch_reference(model)
    out = {}
    traced = {f: z4.trace(model, f) for f in z4.FUNCTIONS}
    for label, runner, variant in (("eager_fp32kv", z4.TorchRunner(model), "fp32kv"),
                                   ("traced_fp32kv", z4.TorchRunner(graphs=traced), "fp32kv"),
                                   ("eager_fp16kv", z4.TorchRunner(model, kv="fp16"), "fp16kv")):
        out[label] = z4.run_checks(reference, runner, z4.FUNCTIONS, variant)
    failed = [k for k, v in out.items() if any(c["rows_outside_tolerance"] for c in v)]
    write_json(fresh(a.output), {"format": "vimeml_kv_batch_v4_torch_check_zh", "passed": not failed, **out,
                                 "environment": environment()})
    worst = {k: max(c["max_abs"] for c in v) for k, v in out.items()}
    print(("passed " if not failed else f"FAILED {failed} ") + json.dumps(worst))
    sys.exit(1 if failed else 0)


def stage_build(a):
    """$OUT/<variant>/: src-<f>, src-merged, src-merged-align.json, int8-<f>, int8-merged, int8-merged-align.json."""
    out = fresh(Path(a.output) / a.variant)
    out.mkdir(parents=True)
    for f in z4.FUNCTIONS:
        print(f"convert {f}", flush=True)
        z4.convert(a.bundle, out / f"src-{f}", f, a.variant)
    z4.merge({f: out / f"src-{f}" for f in z4.FUNCTIONS}, out / "src-merged")
    if not z4.align(out / "src-merged", a.bundle, out / "src-merged-align.json"):
        raise SystemExit(f"{a.variant}: source alignment failed; INT8 not built (this is a result, report it).")
    for f in z4.FUNCTIONS:
        z4.compress(out / f"src-{f}", out / f"int8-{f}", out / "src-merged-align.json")
    z4.merge({f: out / f"int8-{f}" for f in z4.FUNCTIONS}, out / "int8-merged")
    passed = z4.align(out / "int8-merged", a.bundle, out / "int8-merged-align.json", stateless=a.stateless)
    for name in ("src-merged", "int8-merged"):
        m = read_json(out / name / "manifest.json")
        print(f"{name}: {m['package_bytes']:,} bytes (separate {m['separate_package_bytes']:,})")
    sys.exit(0 if passed else 1)


# ---- shared workload ----------------------------------------------------------------------------------------

def feeds(rng, prompt=48, steps=16, rows=16, score_tokens=16, kv=np.float32):
    """Random but finite inputs; shapes are what matters."""
    k = lambda *shape: rng.standard_normal(shape).astype(kv)
    shared = lambda n: {"prefix_length": np.array([n], np.int32), "prefix_key": k(LAYERS, 1, HEADS, n, HEAD_DIM),
                        "prefix_value": k(LAYERS, 1, HEADS, n, HEAD_DIM)}
    allowed = rng.choice(np.arange(8, VOCAB), 160, replace=False).astype(np.int32)
    empty = {"prefix_length": np.zeros(1, np.int32), "prefix_key": np.zeros((LAYERS, 1, HEADS, 1, HEAD_DIM), kv),
             "prefix_value": np.zeros((LAYERS, 1, HEADS, 1, HEAD_DIM), kv)}
    one_row = {"position": np.zeros(1, np.int32), "key_cache": np.zeros((LAYERS, 1, HEADS, 1, HEAD_DIM), kv),
               "value_cache": np.zeros((LAYERS, 1, HEADS, 1, HEAD_DIM), kv)}
    ids = lambda b, t: rng.integers(8, VOCAB, (b, t)).astype(np.int32)
    return {
        "prompt": {"input_ids": ids(1, prompt), **empty},                               # prefix from nothing
        "extend": {"input_ids": ids(1, 1), **shared(prompt)},                           # one more key
        "decode": [{"input_ids": ids(rows, 1), "position": np.full(rows, own, np.int32),
                    "key_cache": k(LAYERS, rows, HEADS, own, HEAD_DIM), "value_cache": k(LAYERS, rows, HEADS, own, HEAD_DIM),
                    "allowed": allowed, **shared(prompt)} for own in range(1, steps + 1)],
        "score": {"input_ids": ids(rows, score_tokens), "targets": ids(rows, score_tokens), **shared(prompt)},
        "one_row": one_row, "allowed": allowed,
    }


def prefix_feed(function, feed, f):
    """The prompt or extension call through `prefix`, or through `scored`/`logits` as v3 clients do it."""
    if function == "prefix":
        return feed
    extra = {"allowed": f["allowed"][:1]} if function == "scored" else {}
    return {**feed, **f["one_row"], **extra}


# ---- load order (memory) ------------------------------------------------------------------------------------

def loadorder_child(spec):
    import coremltools as ct
    kv = np.float16 if spec["kv"] == "fp16" else np.float32
    rng = np.random.default_rng(1)
    f = feeds(rng, spec["prompt"], spec["steps"], 16, spec["score_tokens"], kv)
    order = spec["order"].split(">")
    mem = {"start": snap()}
    models = {}
    for name in order:
        models[name] = ct.models.CompiledMLModel(spec["compiled"], compute_units=ct.ComputeUnit.CPU_ONLY, function_name=name)
        mem[f"after_load_{name}"] = snap()
    pre = next(n for n in ("prefix", "logits", "scored") if n in models)
    for round_ in range(spec["rounds"]):
        models[pre].predict(prefix_feed(pre, f["prompt"], f))
        for _ in range(8):
            models[pre].predict(prefix_feed(pre, f["extend"], f))
        if "scored" in models:
            for d in f["decode"]:
                models["scored"].predict(d)
        models["score"].predict(f["score"])
        if round_ == 0:
            mem["after_first_round"] = snap()
    mem["after_rounds"] = snap()
    return mem


def stage_loadorder(a):
    out = fresh(a.output)
    import coremltools as ct
    cache = Path(a.compiled)
    cache.mkdir(parents=True, exist_ok=True)
    results = {}
    for item in a.packages:
        label, _, path = item.partition("=")
        manifest = read_json(Path(path) / "manifest.json")
        kv = manifest.get("variant", {}).get("kv", "fp32")
        functions = set(manifest["functions"])
        target = cache / f"{Path(path).name}-{file_sha(Path(path) / 'manifest.json')[:12]}.mlmodelc"
        if not target.exists():
            ct.utils.compile_model(str(Path(path) / "model.mlpackage"), str(target))
        for order in a.orders:
            if not set(order.split(">")) <= functions:
                continue
            spec = {"order": order, "compiled": str(target), "kv": kv, "prompt": a.prompt, "steps": a.steps,
                    "score_tokens": a.score_tokens, "rounds": a.rounds}
            runs = []
            for _ in range(a.runs):
                res = subprocess.run([sys.executable, __file__, "loadorder-child", json.dumps(spec)], capture_output=True,
                                     text=True, check=True, cwd=ROOT)
                runs.append(json.loads(res.stdout.strip().splitlines()[-1]))
            rel = lambda r, k, fld="phys_footprint_mib": r[k][fld] - r["start"]["phys_footprint_mib"]
            row = {k: round(statistics.median(rel(r, k) for r in runs), 1) for k in runs[0] if k != "start"}
            row["peak"] = round(statistics.median(rel(r, "after_rounds", "peak_phys_footprint_mib") for r in runs), 1)
            loads = [k for k in runs[0] if k.startswith("after_load_")]
            row["load_peak"] = round(statistics.median(max(rel(r, k, "peak_phys_footprint_mib") for k in loads) for r in runs), 1)
            results[f"{label}:{order}"] = {"package": label, "order": order, "median_mib": row, "runs": runs}
            print(json.dumps({f"{label}:{order}": row}), flush=True)
    write_json(out, {"format": "vimeml_kv_batch_v4_loadorder_zh", "results": results,
                     "workload": "per round: prefix of `prompt` tokens, 8 one-token extensions (through prefix, else "
                                 "logits, else scored with allowed [one id]), `steps` B=16 scored steps if scored is "
                                 "loaded, one B=16 score call of `score_tokens` tokens",
                     "note": "Mac, Python, CPU_ONLY, fresh process per run, .mlmodelc compiled outside; MiB relative to "
                             "`start`, medians over runs; load_peak = highest lifetime peak seen right after a load. "
                             "Not the keyboard-extension footprint.", "environment": environment()})


# ---- latency ---------------------------------------------------------------------------------------------

def stage_measure(a):
    out = fresh(a.output)
    import coremltools as ct
    rows = []
    for item in a.packages:
        label, _, path = item.partition("=")
        manifest = read_json(Path(path) / "manifest.json")
        kv = np.float16 if manifest.get("variant", {}).get("kv") == "fp16" else np.float32
        fns = manifest["functions"]
        m = {fn: ct.models.MLModel(str(Path(path) / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY,
                                   function_name=fn) for fn in fns}
        pre = "prefix" if "prefix" in fns else "scored"
        f = feeds(np.random.default_rng(1120), a.prompt, a.own, 16, a.score_tokens, kv)
        row = {"package": label, "prefix_via": pre,
               "prompt_prefix": timed(m[pre], prefix_feed(pre, f["prompt"], f), a.repeats, a.warmup),
               "extend_prefix_1": timed(m[pre], prefix_feed(pre, f["extend"], f), a.repeats, a.warmup),
               "scored_step_b16": timed(m["scored"], f["decode"][-1], a.repeats, a.warmup),
               "score_b16": timed(m["score"], f["score"], a.repeats, a.warmup)}
        rows.append(row)
        print(json.dumps({k: (v["p50_ms"] if isinstance(v, dict) else v) for k, v in row.items()}), flush=True)
    write_json(out, {"format": "vimeml_kv_batch_v4_latency_zh", "results": rows,
                     "shapes": {"prompt": a.prompt, "own": a.own, "rows": 16, "score_tokens": a.score_tokens},
                     "note": "Mac python predict() wall time incl. input conversion.", "environment": environment()})


# ---- search, fixtures, client resource --------------------------------------------------------------------

def compare(got, want):
    ref = dict(want)
    common = [abs(s - ref[t]) for t, s in got if t in ref]
    return {"exact": [t for t, _ in got] == [t for t, _ in want], "top1_equal": bool(got) and got[0][0] == want[0][0],
            "max_common_score_diff": max(common or [0.0])}


def stage_search(a):
    import sentencepiece as spm
    from vimeml.deployment import zh_search as zs
    from vimeml.posttraining.zh_readings import ReadingTable
    out = fresh(a.output)
    sp = spm.SentencePieceProcessor(model_file=str(Path(a.bundle) / "tokenizer.model"))
    index = zs.Index(sp, ReadingTable(str(a.readings)))
    be = z3.BatchBackendV3(z4.CoreMLRunner(a.package, functions=("scored",)))
    ref_manifest = read_json(Path(a.reference) / "manifest.json")
    ref_runner = (z4.CoreMLRunner(a.reference, functions=("scored",)) if ref_manifest["format"] == z4.MERGED_FORMAT
                  else z3.CoreMLRunner(a.reference, functions=("scored",)))
    ref = z3.BatchBackendV3(ref_runner)
    meta = json.loads((Path(a.cases).parent / "meta.json").read_text(encoding="utf-8"))["search"]
    rows, times = [], []
    for line in open(a.cases, encoding="utf-8"):
        c = json.loads(line)
        t = time.perf_counter()
        ranked, st = zs.search(be, sp, index, c["keys"], c["context"], meta["mode"], meta["beam"], meta["topk"])
        times.append((time.perf_counter() - t) * 1000)
        r_ranked, _ = zs.search(ref, sp, index, c["keys"], c["context"], meta["mode"], meta["beam"], meta["topk"])
        got = [(x[0], x[1]) for x in ranked]
        rows.append({"id": c["id"], "top": got, "ms": round(times[-1], 1),
                     "vs_fp32_fixture": compare(got, [(x["text"], x["gen_score"]) for x in c["top"]]),
                     "vs_reference": compare(got, [(x[0], x[1]) for x in r_ranked])})
    summary = {"cases": len(rows), **{k: {m: sum(r[k][m] for r in rows) for m in ("exact", "top1_equal")}
                                      | {"max_common_score_diff": max(r[k]["max_common_score_diff"] for r in rows)}
                                      for k in ("vs_fp32_fixture", "vs_reference")},
               "p50_ms": statistics.median(times)}
    out.mkdir(parents=True)
    write_json(out / "summary.json", {"format": "vimeml_kv_batch_v4_search_check_zh", "package": str(a.package),
        "package_manifest_sha256": file_sha(Path(a.package) / "manifest.json"), "reference": str(a.reference),
        "fixture": str(a.cases), "summary": summary, "environment": environment()})
    with open(out / "cases.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


def stage_fixture(a):
    """prefix.jsonl: per context, the prefix built in one `prefix` call and in two (all but the last token, then
    the last), then one `logits` and one `scored` step for the next token; score.jsonl: the v3 sentence cases with
    the prefix from `prefix`. Expected values from the stateless package of the same precision."""
    import torch
    from coreml_zh_v1_batch2 import sequence_fixture_cases
    from vimeml.deployment.coreml import coremltools
    ct = coremltools()
    out = fresh(a.output)
    lm = zh_v1.BundleLMZh(a.bundle)
    package = Path(a.package)
    manifest = z4.verify_merged(package)
    runner = z4.CoreMLRunner(package)
    st = ct.models.MLModel(str(Path(a.stateless) / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    stateless = lambda ids: np.asarray(st.predict({"input_ids": np.asarray(ids, np.int32)})["logits"], np.float32)
    top = lambda x, ids: [[int(ids[i]), round(float(x[i]), 5)] for i in np.argsort(-np.asarray(x))[:5]]
    cases, worst = [], 0.0
    for context in ("晚上一起吃饭吗", "我在地铁上马上到", "明天上午去故宫参观下午逛街", "好"):
        ids = [2] + lm.processor.encode(context)
        one = z4.prefix_call(runner, ids[:-1])
        two = z4.prefix_call(runner, [ids[-2]], z4.prefix_call(runner, ids[:-2])) if len(ids) > 2 else one
        want = stateless([ids])[0, -1]
        allowed = np.sort(np.argsort(-want)[:20]).astype(np.int32)
        step = {"input_ids": np.array([[ids[-1]]], np.int32), "position": np.zeros(1, np.int32),
                "key_cache": z3.empty_cache(), "value_cache": z3.empty_cache()}
        got_l = runner("logits", {**step, **one})["logits"][0]
        got_l2 = runner("logits", {**step, **two})["logits"][0]
        got_s = z3.first_scores("scored", runner("scored", {**step, **one, "allowed": allowed}))[0]
        want_s = z3._log_softmax(want)[allowed]
        worst = max(worst, float(np.abs(got_l - want).max()), float(np.abs(got_l2 - want).max()),
                    float(np.abs(got_s - want_s).max()))
        cases.append({"context": context, "ids": ids, "prefix_ids": ids[:-1], "step_token": ids[-1],
                      "allowed": allowed.tolist(), "logits_top5_package": top(got_l, range(VOCAB)),
                      "logits_top5_package_two_calls": top(got_l2, range(VOCAB)), "logits_top5_stateless": top(want, range(VOCAB)),
                      "scored_top5_package": top(got_s, allowed), "scored_top5_stateless": top(want_s, allowed)})
    score_cases, worst_score = [], 0.0
    from vimeml.benchmarks.zh13_lm_scores import score_item
    model = lm.model.float().eval()
    for context, candidates in sequence_fixture_cases():
        with torch.inference_mode():
            rule = score_item(model, lm.processor, context, candidates)
        common = rule["common_prefix_tokens_including_bos"]
        seqs = [[2] + lm.processor.encode(context + c) for c in candidates]
        head, conts = seqs[0][:common], [s[common:] for s in seqs]
        got, calls = z3.score_continuations(runner, head, conts, prefill="prefix")
        ref = []
        for s in seqs:
            lp = z3._log_softmax(stateless([s])[0])
            ref.append(float(sum(lp[j - 1, s[j]] for j in range(common, len(s)))))
        worst_score = max(worst_score, max(abs(x - y) for x, y in zip(got, ref)))
        score_cases.append({"context": context, "candidates": candidates, "prefix_ids": head, "continuation_ids": conts,
                            "scores_package": [round(x, 5) for x in got], "scores_stateless": [round(x, 5) for x in ref],
                            "same_order": list(np.argsort(got)) == list(np.argsort(ref)), "calls": calls})
    out.mkdir(parents=True)
    for name, rows in (("prefix.jsonl", cases), ("score.jsonl", score_cases)):
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            for c in rows:
                f.write(json.dumps(c, ensure_ascii=False, default=int) + "\n")
    write_json(out / "meta.json", {"format": "vime_zh_kv_batch_v4_fixture_v1", "package": str(package),
        "package_manifest_sha256": file_sha(package / "manifest.json"), "variant": manifest["variant"],
        "stateless_package": Path(a.stateless).name, "functions": list(manifest["functions"]),
        "prefix_how": "prefix_ids through `prefix` (input_ids [1,T], prefix_length 0) -> prefix; two_calls: prefix_ids[:-1] "
                      "first, then the last prefix id with that prefix, KV appended; then step_token through `logits` "
                      "and `scored` (B=1, position 0, row cache one zero slot)",
        "score_how": "zh_v1_batch3.score_continuations with the prefix from `prefix`",
        "max_abs_step_vs_stateless": worst, "max_abs_score_sum_vs_stateless": worst_score,
        "score_same_order": sum(c["same_order"] for c in score_cases), "score_cases": len(score_cases),
        "tolerance": z4.VARIANTS[manifest["variant"]["name"]], "tokenizer_sha256": lm.metadata["tokenizer_sha256"]})
    print(f"fixtures written; step vs stateless max {worst:.3g}, score sums max {worst_score:.3g}, "
          f"same order {sum(c['same_order'] for c in score_cases)}/{len(score_cases)}")


def stage_ios(a):
    from package_zh_v1_ios import compile_package
    package, alignment, search, fixture, output = map(Path, (a.package, a.alignment, a.search, a.fixture, a.output))
    manifest = z4.verify_merged(package)
    sha = file_sha(package / "manifest.json")
    gate = read_json(alignment)
    if (manifest.get("recipe") != zh_v1.DEPLOYED_RECIPE or gate.get("format") != z4.ALIGNMENT_FORMAT or not gate.get("passed")
            or gate["coreml_manifest_sha256"] != sha or not gate["reference"].startswith("stateless Core ML")):
        raise ValueError("Passed stateless alignment of this INT8 package required.")
    if read_json(search / "summary.json")["package_manifest_sha256"] != sha or read_json(fixture / "meta.json")["package_manifest_sha256"] != sha:
        raise ValueError("Search check and fixture must belong to this package.")
    variant = manifest["variant"]["name"]
    # v1 keeps its resource name; v2-20m gets its own so both can ship side by side; the Japanese
    # vimeml-ja-v6-20m (same block) gets a Japanese name and manifest.
    release = manifest["bundle"].get("release_name", zh_v1.RELEASE)
    family = {zh_v1.RELEASE: "TinyChinese", "vimeml-ja-v6-20m": "TinyJapaneseV6"}.get(release, "TinyChineseV2")
    lang = "Ja" if release.startswith("vimeml-ja") else "Zh"
    resource = f"{family}KVBatchV4{variant.replace('-', '').upper()}INT8.mlmodelc"
    fresh_directory(fresh(output))
    resources = output / "Resources"
    resources.mkdir()
    compiled = compile_package(package, output / "compile")
    shutil.copytree(compiled, resources / resource)
    shutil.rmtree(output / "compile")
    for name in ("prefix.jsonl", "score.jsonl", "meta.json"):
        shutil.copyfile(fixture / name, output / f"KVBatchV4Fixture-{variant}-{name}")
    section = {"resource": resource, "variant": manifest["variant"], "source_manifest_sha256": sha,
               "compiled_files_sha256": {n: e["sha256"] for n, e in v2.inventory(resources / resource, hashes=True).items()},
               "functions": {f: {"interface": e["interface"], "batch_range": e["batch_range"]} for f, e in manifest["functions"].items()},
               "loading": manifest["loading"], "alignment_tolerance": gate["tolerance"], "alignment_sha256": file_sha(alignment),
               "search_check": read_json(search / "summary.json")["summary"],
               "policy": "Caller owns all buffers. `prefix` returns only the new tokens' KV: append it to the prefix's "
                         "valid slots. Masked slots must hold finite values. KV tensors are "
                         + ("float16." if manifest["variant"]["kv"] == "fp16" else "float32.")}
    write_json(resources / f"VimeLMManifest{lang}KVBatchV4-{variant}.json", {"format": f"vime_ios_lm_{lang.lower()}_kv_batch_v4",
        "bundle_manifest_sha256": manifest["bundle"]["bundle_manifest_sha256"],
        "tokenizer_sha256": manifest["bundle"]["tokenizer_sha256"], "kv_batch_v4": section})
    write_json(output / "manifest.json", {"format": "vimeml_zh_ios_kv_batch_v4_resources_v1", "files": v2.inventory(output),
                                          "note": "Add Resources/ beside the existing zh resources."})
    print(f"{resource} ready")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "loadorder-child":
        print(json.dumps(loadorder_child(json.loads(sys.argv[2]))))
        return
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("check"); s.add_argument("output")
    s = sub.add_parser("build"); s.add_argument("output"); s.add_argument("--variant", required=True, choices=list(z4.VARIANTS))
    s.add_argument("--stateless", default=str(STATELESS_INT8))
    s = sub.add_parser("loadorder"); s.add_argument("output"); s.add_argument("packages", nargs="+", help="label=<merged package>")
    s.add_argument("--orders", nargs="+", default=list(ORDERS))
    s.add_argument("--prompt", type=int, default=48); s.add_argument("--steps", type=int, default=16)
    s.add_argument("--score-tokens", type=int, default=16); s.add_argument("--rounds", type=int, default=10)
    s.add_argument("--runs", type=int, default=5)
    s.add_argument("--compiled", default=str(ROOT / "outputs/deployment/compiled-for-memory"))
    s = sub.add_parser("measure"); s.add_argument("output"); s.add_argument("packages", nargs="+", help="label=<merged package>")
    s.add_argument("--prompt", type=int, default=48); s.add_argument("--own", type=int, default=16)
    s.add_argument("--score-tokens", type=int, default=16)
    s.add_argument("--repeats", type=int, default=100); s.add_argument("--warmup", type=int, default=10)
    s = sub.add_parser("search"); s.add_argument("package"); s.add_argument("output")
    s.add_argument("--reference", default=str(V3_INT8), help="merged v3 or v4 package to compare rankings with")
    s.add_argument("--cases", default=str(CASES)); s.add_argument("--readings", default=str(READINGS))
    s = sub.add_parser("fixture"); s.add_argument("package"); s.add_argument("output")
    s.add_argument("--stateless", default=str(STATELESS_INT8))
    s = sub.add_parser("ios"); s.add_argument("package"); s.add_argument("output")
    for name in ("alignment", "search", "fixture"):
        s.add_argument("--" + name, required=True)
    for sp in sub.choices.values():
        sp.add_argument("--bundle", default=str(BUNDLE))
    a = p.parse_args()
    {"check": stage_check, "build": stage_build, "loadorder": stage_loadorder, "measure": stage_measure,
     "search": stage_search, "fixture": stage_fixture, "ios": stage_ios}[a.stage](a)


if __name__ == "__main__":
    main()
