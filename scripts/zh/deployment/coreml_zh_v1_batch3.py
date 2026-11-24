"""vimeml-tiny-zh-v1 batched KV step v3 (shared prefix, lean scoring, one multifunction package): Mac stages.

Brief: docs/zh/mac/mac-zh-v1-kvbatch-v3.md. Every stage writes a fresh output.

  check     PyTorch: eager and traced functions vs the full forward (bundle weights)
  convert   FP32 single-function package: --function logits|scored|score [--vocab-chunks N]
  merge     one multifunction package from single-function packages (FP32 or INT8)
  align     a merged package, every function, vs PyTorch FP32 or vs the stateless package (--stateless)
  compress  INT8 block32 of a single-function FP32 package (gate: the merged FP32 alignment)
  measure   v3 vs the shipped v2 packages on the same work: beam decode steps and sentence scoring
  memory    macOS phys_footprint, one fresh process per configuration, the same session workload
  search    the per-key-position beam search with the v3 package vs the FP32 fixture and the v2 package
  fixture   Vime fixtures: beam step after a shared prompt, and sentence scoring
  ios       client resource (compiled package + manifest section) for Vime
"""
import argparse
import json
import os
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

from vimeml.deployment import v2, zh_v1, zh_v1_batch2 as z2, zh_v1_batch3 as z3  # noqa: E402
from vimeml.deployment.bundle import environment, fresh_directory, read_json, write_json  # noqa: E402
from vimeml.training.data import file_sha  # noqa: E402
from coreml_zh_v1_batch import BUNDLE, READINGS, RELEASE, STATELESS_INT8, fixture_rows, fresh  # noqa: E402
from coreml_zh_v1_batch2 import snap, sequence_fixture_cases, timed  # noqa: E402

LAYERS, HEADS, HEAD_DIM, VOCAB = z3.LAYERS, z3.HEADS, z3.HEAD_DIM, z3.VOCAB
V2_DECODE = RELEASE / "coreml/ios18-int8-block32-kvbatch-v2"        # unified-scored, shipped
V2_SEQUENCE = RELEASE / "coreml/ios18-int8-block32-kvsequence"      # sequence head, shipped
RESOURCE = "TinyChineseV1KVBatchV3INT8.mlmodelc"


def stage_check(a):
    import torch
    torch.set_num_threads(4)
    lm = zh_v1.BundleLMZh(a.bundle)
    model = lm.model.float().eval()
    reference = z3.torch_reference(model)
    out = {}
    for label, runner in (("eager", z3.TorchRunner(model)),
                          ("traced", z3.TorchRunner(graphs={f: z3.trace(model, f) for f in z3.FUNCTIONS}))):
        out[label] = z3.run_checks(reference, runner)
    out["score_vocab_chunks_4"] = z3.check_score(reference, z3.TorchRunner(graphs={
        "scored": z3.trace(model, "scored"), "score": z3.trace(model, "score", vocab_chunks=4)}))
    worst = max(c["max_abs"] for v in (out["eager"], out["traced"]) for c in v)
    if out["score_vocab_chunks_4"]["rows_outside_tolerance"]:
        raise AssertionError(out["score_vocab_chunks_4"])
    write_json(fresh(a.output), {"format": "vimeml_kv_batch_v3_torch_check_zh_v1", "passed": True, **out,
                                 "environment": environment()})
    print(f"passed, max {worst:.3g}; score with 4 vocab chunks {out['score_vocab_chunks_4']['max_abs']:.3g} per sum")


def stage_convert(a):
    z3.convert(a.bundle, fresh(a.output), a.function, a.vocab_chunks)


def stage_merge(a):
    packages = dict(item.split("=", 1) for item in a.functions)
    z3.merge(packages, fresh(a.output))
    m = read_json(Path(a.output) / "manifest.json")
    print(f"merged {list(packages)}: {m['package_bytes']:,} bytes (separate packages {m['separate_package_bytes']:,})")


def stage_align(a):
    sys.exit(0 if z3.align(a.package, a.bundle, fresh(a.output), stateless=a.stateless) else 1)


def stage_compress(a):
    z3.compress(a.package, fresh(a.output), a.alignment)


# ---- the shared workload (latency and memory) ---------------------------------------------------------------

def session_feeds(rng, prompt, own, rows, score_tokens):
    """Inputs for one beam step and one scoring call, in the v2 layout (prompt copied into every row) and in
    the v3 layout (prompt as the shared prefix). Random but finite caches; shapes are what matters."""
    k = lambda *shape: rng.standard_normal(shape).astype(np.float32)
    allowed = rng.choice(np.arange(8, VOCAB), 160, replace=False).astype(np.int32)
    ids1 = rng.integers(8, VOCAB, (rows, 1)).astype(np.int32)
    ids_t = rng.integers(8, VOCAB, (rows, score_tokens)).astype(np.int32)
    shared = {"prefix_length": np.array([prompt], np.int32), "prefix_key": k(LAYERS, 1, HEADS, prompt, HEAD_DIM),
              "prefix_value": k(LAYERS, 1, HEADS, prompt, HEAD_DIM)}
    width = max(own, 1)
    return {
        "v2_decode": {"input_ids": ids1, "position": np.full(rows, prompt + own, np.int32),
                      "key_cache": k(LAYERS, rows, HEADS, prompt + own, HEAD_DIM),
                      "value_cache": k(LAYERS, rows, HEADS, prompt + own, HEAD_DIM), "allowed": allowed},
        "v3_decode": {"input_ids": ids1, "position": np.full(rows, own, np.int32),
                      "key_cache": k(LAYERS, rows, HEADS, width, HEAD_DIM), "value_cache": k(LAYERS, rows, HEADS, width, HEAD_DIM),
                      "allowed": allowed, **shared},
        "v2_score": {"input_ids": ids_t, "targets": ids_t, "position": np.full(rows, prompt, np.int32),
                     "key_cache": k(LAYERS, rows, HEADS, prompt, HEAD_DIM), "value_cache": k(LAYERS, rows, HEADS, prompt, HEAD_DIM)},
        "v3_score": {"input_ids": ids_t, "targets": ids_t, **shared},
        "prompt": {"input_ids": rng.integers(8, VOCAB, (1, prompt)).astype(np.int32), "position": np.zeros(1, np.int32),
                   "key_cache": np.zeros((LAYERS, 1, HEADS, 1, HEAD_DIM), np.float32),
                   "value_cache": np.zeros((LAYERS, 1, HEADS, 1, HEAD_DIM), np.float32), "allowed": allowed},
    }


def v3_prompt(feed):
    return {**feed, **z3.no_prefix()}


def load_set(label, spec, unit="CPU_ONLY"):
    """{"decode": model, "score": model, "prompt": model, "stateless": model or None} for v2 or v3."""
    import coremltools as ct
    cu = getattr(ct.ComputeUnit, unit)
    mlm = lambda p, f=None: ct.models.MLModel(str(Path(p) / "model.mlpackage"), compute_units=cu,
                                              **({"function_name": f} if f else {}))
    models = {"stateless": mlm(spec["stateless"]) if spec.get("stateless") else None}
    if spec["kind"] == "v2":
        models["decode"] = models["prompt"] = mlm(spec["decode"])
        models["score"] = mlm(spec["sequence"])
    else:
        models["decode"] = models["prompt"] = mlm(spec["package"], "scored")
        models["score"] = mlm(spec["package"], "score")
        models["logits"] = mlm(spec["package"], "logits") if spec.get("logits", True) else None
    return models


def stage_measure(a):
    out = fresh(a.output)
    rng = np.random.default_rng(1118)
    sets = {"v2": load_set("v2", {"kind": "v2", "decode": a.v2_decode, "sequence": a.v2_sequence}),
            "v3": load_set("v3", {"kind": "v3", "package": a.package, "logits": False})}
    results = []
    for prompt in a.prompts:
        for own in a.own:
            if prompt + own + max(a.score_tokens) > z3.CONTEXT:
                continue
            for t in a.score_tokens:
                f = session_feeds(rng, prompt, own, a.rows, t)
                for _ in range(3):                    # interleaved passes, v2 then v3
                    for kind in ("v2", "v3"):
                        m = sets[kind]
                        row = {"pass": _, "kind": kind, "prompt": prompt, "own": own, "rows": a.rows, "score_tokens": t,
                               "decode": timed(m["decode"], f[f"{kind}_decode"], a.repeats, a.warmup),
                               "score": timed(m["score"], f[f"{kind}_score"], a.repeats, a.warmup)}
                        results.append(row)
                        print(json.dumps({k: (v["p50_ms"] if isinstance(v, dict) else v) for k, v in row.items()}), flush=True)
    write_json(out, {"format": "vimeml_kv_batch_v3_latency_zh_v1", "package": Path(a.package).name,
                     "package_manifest_sha256": file_sha(Path(a.package) / "manifest.json"),
                     "v2_decode": str(a.v2_decode), "v2_sequence": str(a.v2_sequence), "results": results,
                     "note": "Mac python predict() wall time incl. input conversion; v2 copies the prompt into every row "
                             "(S = prompt + own), v3 passes it once as the prefix.", "environment": environment()})


def memory_child(spec):
    """One fresh process. coremltools is imported and every input is built (and kept alive) before `start`;
    models load from .mlmodelc compiled outside the measured process, as an app does. The first Mac run imported
    coremltools after `start` (+130-160 MiB per load) and compiled .mlpackage in-process (+-30 MiB noise); see
    docs/zh/reports/mac-20261118/zh-v1-kvbatch-v3.md, section 8."""
    import coremltools as ct
    kind = spec["kind"]
    rng = np.random.default_rng(1)
    # every input stays referenced: dropping the unused half before `start` let macOS hand that memory back
    # during the loads (-340 MiB on the Mac)
    feeds = [session_feeds(rng, spec["prompt"], own, 16, spec["score_tokens"]) for own in range(1, spec["steps"] + 1)]
    unit = ct.ComputeUnit.CPU_ONLY
    compiled = spec["compiled"]
    mlm = lambda p, f=None: ct.models.CompiledMLModel(compiled[str(p)], compute_units=unit,
                                                      **({"function_name": f} if f else {}))
    mem = {"start": snap()}
    models = {}
    if spec.get("stateless"):
        mlm(spec["stateless"]).predict({"input_ids": np.full((1, 16), 7, np.int32)})
        mem["after_stateless"] = snap()
    if kind == "v2":
        models["decode"] = models["prompt"] = mlm(spec["decode"])
        mem["after_kvbatch_v2"] = snap()
        models["score"] = mlm(spec["sequence"])
        mem["after_kvsequence"] = snap()
    else:
        for f in ("scored", "score", "logits"):
            models[f] = mlm(spec["package"], f)
            mem[f"after_fn_{f}"] = snap()
        models["decode"] = models["prompt"] = models["scored"]
    mem["after_load"] = snap()
    for round_ in range(spec["rounds"]):
        models["prompt"].predict(feeds[0]["prompt"] if kind == "v2" else v3_prompt(feeds[0]["prompt"]))
        for f in feeds:                              # one search: decode steps with growing own caches
            models["decode"].predict(f[f"{kind}_decode"])
        if models.get("logits") is not None:
            models["logits"].predict({k: v for k, v in v3_prompt(feeds[0]["prompt"]).items() if k != "allowed"})
        models["score"].predict(feeds[-1][f"{kind}_score"])
        if round_ == 0:
            mem["after_first_round"] = snap()
    mem["after_rounds"] = snap()
    return mem


def stage_memory(a):
    out = fresh(a.output)
    base = {"prompt": a.prompt, "steps": a.steps, "score_tokens": a.score_tokens, "rounds": a.rounds}
    stateless = str(a.stateless) if a.stateless else None
    configs = [{"label": "v2-shipped", "kind": "v2", "decode": str(a.v2_decode), "sequence": str(a.v2_sequence),
                "stateless": stateless, **base}]
    for item in a.packages:
        label, _, path = item.partition("=")
        configs.append({"label": label, "kind": "v3", "package": path, "stateless": stateless, **base})
        if a.without_stateless:
            configs.append({"label": label + "-alone", "kind": "v3", "package": path, "stateless": None, **base})
    import coremltools as ct
    cache = Path(a.compiled)                     # compiled once, outside the measured processes; reused across runs
    cache.mkdir(parents=True, exist_ok=True)
    compiled = {}
    for spec in configs:
        for key in ("stateless", "decode", "sequence", "package"):
            p = spec.get(key)
            if p and p not in compiled:
                target = cache / f"{Path(p).name}-{file_sha(Path(p) / 'manifest.json')[:12]}.mlmodelc"
                if not target.exists():
                    ct.utils.compile_model(str(Path(p) / "model.mlpackage"), str(target))
                compiled[p] = str(target)
    results = {}
    for spec in configs:
        spec["compiled"] = compiled
        runs = []
        for _ in range(a.runs):
            res = subprocess.run([sys.executable, __file__, "memory-child", json.dumps(spec)], capture_output=True,
                                 text=True, check=True, cwd=ROOT)
            runs.append(json.loads(res.stdout.strip().splitlines()[-1]))
        rel = lambda r, k, f="phys_footprint_mib": r[k][f] - r["start"]["phys_footprint_mib"]
        med = lambda f: round(statistics.median(f(r) for r in runs), 1)
        row = {k: med(lambda r, k=k: rel(r, k)) for k in runs[0] if k != "start"}
        row.update(peak=med(lambda r: rel(r, "after_rounds", "peak_phys_footprint_mib")),
                   min_peak=round(min(rel(r, "after_rounds", "peak_phys_footprint_mib") for r in runs), 1),
                   growth_after_first_round=med(lambda r: r["after_rounds"]["phys_footprint_mib"]
                                                - r["after_first_round"]["phys_footprint_mib"]))
        results[spec["label"]] = {"spec": {k: v for k, v in spec.items() if k != "compiled"}, "median_mib": row, "runs": runs}
        print(json.dumps({spec["label"]: row}), flush=True)
    write_json(out, {"format": "vimeml_kv_batch_v3_memory_zh_v1", "results": results, "compiled": compiled,
                     "workload": "per round: prompt prefill, `steps` B=16 decode steps with own caches 1..steps after the "
                                 "prompt, one logits call (v3), one B=16 scoring call of `score_tokens` tokens",
                     "note": "Mac, Python, CPU_ONLY, fresh process per run; coremltools imported, the configuration's "
                             "inputs built and models compiled before `start`; phys_footprint and lifetime max relative "
                             "to `start`, medians over runs. Weights are file-mapped and not in phys_footprint. Not the "
                             "keyboard-extension footprint.", "environment": environment()})


# ---- search, fixtures, client resource --------------------------------------------------------------------

def compare(got, want, tolerance=None):
    ref = dict(want)
    common = [abs(s - ref[t]) for t, s in got if t in ref]
    return {"exact": [t for t, _ in got] == [t for t, _ in want], "same_set": {t for t, _ in got} == set(ref),
            "top1_equal": bool(got) and bool(want) and got[0][0] == want[0][0],
            "max_common_score_diff": max(common or [0.0])}


def stage_search(a):
    import sentencepiece as spm
    from vimeml.deployment import zh_search as zs
    from vimeml.posttraining.zh_readings import ReadingTable
    out = fresh(a.output)
    sp = spm.SentencePieceProcessor(model_file=str(Path(a.bundle) / "tokenizer.model"))
    index = zs.Index(sp, ReadingTable(str(a.readings)))
    be = z3.BatchBackendV3(z3.CoreMLRunner(a.package, functions=("scored",)))
    ref = z2.CoreMLBatchBackendV2(a.v2_decode)
    meta = json.loads((Path(a.cases).parent / "meta.json").read_text(encoding="utf-8"))
    beam, topk, mode = meta["search"]["beam"], meta["search"]["topk"], meta["search"]["mode"]
    rows, times, times_v2 = [], [], []
    for line in open(a.cases, encoding="utf-8"):
        c = json.loads(line)
        t = time.perf_counter()
        ranked, st = zs.search(be, sp, index, c["keys"], c["context"], mode, beam, topk)
        times.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        r_ranked, _ = zs.search(ref, sp, index, c["keys"], c["context"], mode, beam, topk)
        times_v2.append((time.perf_counter() - t) * 1000)
        got = [(x[0], x[1]) for x in ranked]
        rows.append({"id": c["id"], "top": got, "calls": st.calls, "rows": st.rows, "ms": round(times[-1], 1),
                     "vs_fp32_fixture": compare(got, [(x["text"], x["gen_score"]) for x in c["top"]]),
                     "vs_v2_package": compare(got, [(x[0], x[1]) for x in r_ranked])})
    pct = lambda xs: {"p50_ms": statistics.median(xs), "p95_ms": sorted(xs)[int(0.95 * (len(xs) - 1))]}
    summary = {"cases": len(rows), **{k: {m: sum(r[k][m] for r in rows) for m in ("exact", "same_set", "top1_equal")}
                                      | {"max_common_score_diff": max(r[k]["max_common_score_diff"] for r in rows)}
                                      for k in ("vs_fp32_fixture", "vs_v2_package")},
               "v3": pct(times), "v2": pct(times_v2)}
    out.mkdir(parents=True)
    write_json(out / "summary.json", {"format": "vimeml_kv_batch_v3_search_check_zh_v1", "package": Path(a.package).name,
        "package_manifest_sha256": file_sha(Path(a.package) / "manifest.json"), "fixture": str(a.cases),
        "v2_package": str(a.v2_decode), "summary": summary, "environment": environment()})
    with open(out / "cases.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


def stage_fixture(a):
    """step.jsonl: one shared prompt (BOS + context), 16 rows with their own tokens after it, one scored step
    and one logits step. score.jsonl: the v2 sentence-scoring cases with the v3 calls. Expected values from
    the package, the stateless package of the same precision and PyTorch FP32."""
    import torch
    from vimeml.benchmarks.zh13_lm_scores import score_item
    from vimeml.deployment.coreml import coremltools
    ct = coremltools()
    out = fresh(a.output)
    lm = zh_v1.BundleLMZh(a.bundle)
    model = lm.model.float().eval()
    package = Path(a.package)
    manifest = z3.verify_merged(package)
    runner = z3.CoreMLRunner(package)
    st = ct.models.MLModel(str(Path(a.stateless) / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    stateless = lambda ids: np.asarray(st.predict({"input_ids": np.asarray(ids, np.int32)})["logits"], np.float32)
    pytorch = z3.torch_reference(model)
    full = lambda ref, ids: np.asarray(ref(np.asarray([ids], np.int32)), np.float32)[0, -1]
    # step fixture
    prompt = [2] + lm.processor.encode(a.context)
    rows = [r["prefix_ids"][1:] + [r["token"]] for r in fixture_rows(lm.processor)]      # own tokens + new token
    with torch.inference_mode():
        lasts = [full(pytorch, prompt + r) for r in rows]
    allowed = sorted({int(i) for x in lasts for i in np.argsort(-x)[:20]})
    _, shared = z3.prompt_call(runner, "scored", prompt, allowed)
    states = []
    for r in rows:
        own = r[:-1]
        if not own:
            states.append((np.zeros((LAYERS, HEADS, 0, HEAD_DIM), np.float32),) * 2)
            continue
        o = runner("scored", {"input_ids": np.array([own], np.int32), "position": np.zeros(1, np.int32),
                              "key_cache": z3.empty_cache(), "value_cache": z3.empty_cache(),
                              "allowed": np.array(allowed, np.int32), **shared})
        states.append((o["new_key"][:, 0], o["new_value"][:, 0]))
    k, v, position = z3.pack(states)
    step = {"input_ids": np.array([[r[-1]] for r in rows], np.int32), "position": position, "key_cache": k,
            "value_cache": v, **shared}
    scored = z3.first_scores("scored", runner("scored", {**step, "allowed": np.array(allowed, np.int32)}))
    logits = runner("logits", step)["logits"]
    worst = 0.0
    cases = []
    top = lambda x, ids: [[int(ids[i]), round(float(x[i]), 5)] for i in np.argsort(-np.asarray(x))[:5]]
    for i, r in enumerate(rows):
        s_last = full(stateless, prompt + r)
        s_lp = z3._log_softmax(s_last)[allowed]
        worst = max(worst, float(np.abs(scored[i] - s_lp).max()), float(np.abs(logits[i] - s_last).max()))
        cases.append({"own_ids": r[:-1], "token": r[-1],
                      "scored_top5_package": top(scored[i], allowed), "scored_top5_stateless": top(s_lp, allowed),
                      "scored_top5_pytorch_fp32": top(z3._log_softmax(lasts[i])[allowed], allowed),
                      "logits_top5_package": top(logits[i], range(VOCAB)), "logits_top5_stateless": top(s_last, range(VOCAB))})
    out.mkdir(parents=True)
    with open(out / "step.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.savez_compressed(out / "step.npz", prompt_ids=np.array(prompt, np.int32), allowed=np.array(allowed, np.int32),
                        scored=scored, logits=logits)
    # score fixture
    score_cases, worst_score = [], 0.0
    for context, candidates in sequence_fixture_cases():
        with torch.inference_mode():
            rule = score_item(model, lm.processor, context, candidates)
        common = rule["common_prefix_tokens_including_bos"]
        seqs = [[2] + lm.processor.encode(context + c) for c in candidates]
        head, conts = seqs[0][:common], [s[common:] for s in seqs]
        got, calls = z3.score_continuations(runner, head, conts)
        ref = []
        for s in seqs:
            lp = z3._log_softmax(np.asarray(stateless(np.asarray([s], np.int32)), np.float32)[0])
            ref.append(float(sum(lp[j - 1, s[j]] for j in range(common, len(s)))))
        worst_score = max(worst_score, max(abs(x - y) for x, y in zip(got, ref)))
        score_cases.append({"context": context, "candidates": candidates, "prefix_ids": head, "continuation_ids": conts,
                            "scores_package": [round(x, 5) for x in got], "scores_stateless_same_precision": [round(x, 5) for x in ref],
                            "scores_pytorch_fp32": [round(x, 5) for x in rule["scores"]], "calls": calls})
    with open(out / "score.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for c in score_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    write_json(out / "meta.json", {"format": "vime_zh_kv_batch_v3_fixture_v1", "package": package.name,
        "package_manifest_sha256": file_sha(package / "manifest.json"), "stateless_package": Path(a.stateless).name,
        "functions": list(manifest["functions"]), "step_rows": len(cases), "score_cases": len(score_cases),
        "step_how": "prompt_ids through `scored` (B=1, no prefix) -> prefix; each row's own_ids through `scored` (B=1, "
                    "after the prefix) -> its own cache; then one call with all 16 rows (input = token, position = "
                    "len(own_ids)) through `scored` (allowed = npz 'allowed'; values are log-probs) and `logits`",
        "score_how": "zh_v1_batch3.score_continuations: prompt call over prefix_ids[:-1] -> prefix; then `score` with "
                     "input = [prefix_ids[-1]] + c[:-1], targets = c, up to 16 rows per call",
        "max_abs_step_vs_stateless": worst, "max_abs_score_sum_vs_stateless": worst_score,
        "tolerance": {"atol": z3.TOLERANCE, "rtol": z3.TOLERANCE, "per_sum": z3.SUM_TOLERANCE},
        "tokenizer_sha256": lm.metadata["tokenizer_sha256"]})
    print(f"fixtures written; step vs stateless max {worst:.3g}, score sums vs stateless max {worst_score:.3g}")


def stage_ios(a):
    from package_zh_v1_ios import compile_package
    package, alignment, search, fixture, output = map(Path, (a.package, a.alignment, a.search, a.fixture, a.output))
    manifest = z3.verify_merged(package)
    sha = file_sha(package / "manifest.json")
    if manifest.get("recipe") != zh_v1.DEPLOYED_RECIPE or set(manifest["functions"]) != set(z3.FUNCTIONS):
        raise ValueError("Expected the deployed INT8 recipe with all three functions.")
    gate = read_json(alignment)
    if (gate.get("format") != z3.ALIGNMENT_FORMAT or not gate.get("passed") or gate["coreml_manifest_sha256"] != sha
            or not gate["reference"].startswith("stateless Core ML")):
        raise ValueError("Passed stateless alignment of this package required.")
    checked = read_json(search / "summary.json")
    meta = read_json(fixture / "meta.json")
    if checked["package_manifest_sha256"] != sha or meta["package_manifest_sha256"] != sha:
        raise ValueError("Search check and fixture must belong to this package.")
    if output.exists():
        raise SystemExit(f"{output} exists; choose a new output.")
    fresh_directory(output)
    resources = output / "Resources"
    resources.mkdir()
    compiled = compile_package(package, output / "compile")
    shutil.copytree(compiled, resources / RESOURCE)
    shutil.rmtree(output / "compile")
    for name in ("step.jsonl", "step.npz", "score.jsonl", "meta.json"):
        shutil.copyfile(fixture / name, output / f"KVBatchV3Fixture-{name}")
    section = {"resource": RESOURCE, "source_manifest_sha256": sha,
               "compiled_files_sha256": {n: e["sha256"] for n, e in v2.inventory(resources / RESOURCE, hashes=True).items()},
               "functions": {f: {"interface": e["interface"], "vocab_chunks": e["vocab_chunks"]} for f, e in manifest["functions"].items()},
               "loading": manifest["loading"], "batch_range": manifest["batch_range"],
               "alignment_tolerance": gate["tolerance"], "alignment_reference": gate["reference"],
               "alignment_sha256": file_sha(alignment), "search_check": checked["summary"],
               "fixture_max_abs": {"step": meta["max_abs_step_vs_stateless"], "score_sum": meta["max_abs_score_sum_vs_stateless"]},
               "policy": "Caller owns all buffers. The prompt's new_key/new_value become prefix_key/prefix_value; each row "
                         "appends its own new_key/new_value to its own cache. Masked slots must hold finite values.",
               "kind": manifest["kind"], "compute_units": manifest["compute_units"]}
    write_json(resources / "VimeLMManifestZhV1KVBatchV3.json", {"format": "vime_ios_lm_zh_v1_kv_batch_v3",
        "release_name": zh_v1.RELEASE, "bundle_manifest_sha256": manifest["bundle"]["bundle_manifest_sha256"],
        "tokenizer_sha256": manifest["bundle"]["tokenizer_sha256"], "kv_batch_v3": section})
    write_json(output / "manifest.json", {"format": "vimeml_zh_v1_ios_kv_batch_v3_resources_v1", "files": v2.inventory(output),
                                          "note": "Add Resources/ beside the existing zh-v1 resources."})
    print(f"{RESOURCE} ready")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "memory-child":
        print(json.dumps(memory_child(json.loads(sys.argv[2]))))
        return
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("check"); s.add_argument("output")
    s = sub.add_parser("convert"); s.add_argument("output"); s.add_argument("--function", required=True, choices=z3.FUNCTIONS)
    s.add_argument("--vocab-chunks", type=int, default=1)
    s = sub.add_parser("merge"); s.add_argument("output"); s.add_argument("functions", nargs="+", help="function=<package dir>")
    s = sub.add_parser("align"); s.add_argument("package"); s.add_argument("output"); s.add_argument("--stateless")
    s = sub.add_parser("compress"); s.add_argument("package"); s.add_argument("alignment"); s.add_argument("output")
    s = sub.add_parser("measure"); s.add_argument("package"); s.add_argument("output")
    s.add_argument("--prompts", type=int, nargs="+", default=[16, 48])
    s.add_argument("--own", type=int, nargs="+", default=[1, 8, 16])
    s.add_argument("--score-tokens", type=int, nargs="+", default=[8, 16])
    s.add_argument("--rows", type=int, default=16)
    s.add_argument("--repeats", type=int, default=100); s.add_argument("--warmup", type=int, default=10)
    s = sub.add_parser("memory"); s.add_argument("output")
    s.add_argument("packages", nargs="+", help="label=<merged v3 package dir>")
    s.add_argument("--prompt", type=int, default=48); s.add_argument("--steps", type=int, default=16)
    s.add_argument("--score-tokens", type=int, default=16); s.add_argument("--rounds", type=int, default=20)
    s.add_argument("--runs", type=int, default=5)
    s.add_argument("--compiled", default=str(ROOT / "outputs/deployment/compiled-for-memory"),
                   help="cache of .mlmodelc compiled outside the measured processes (keyed by manifest hash)")
    s.add_argument("--stateless", default=str(STATELESS_INT8), help="'' to leave the stateless package out")
    s.add_argument("--without-stateless", action="store_true", help="also run every v3 package without stateless")
    s = sub.add_parser("search"); s.add_argument("package"); s.add_argument("cases"); s.add_argument("output")
    s.add_argument("--readings", default=str(READINGS))
    s = sub.add_parser("fixture"); s.add_argument("package"); s.add_argument("output")
    s.add_argument("--stateless", default=str(STATELESS_INT8)); s.add_argument("--context", default="晚上一起吃饭吗")
    s = sub.add_parser("ios"); s.add_argument("package"); s.add_argument("output")
    for name in ("alignment", "search", "fixture"):
        s.add_argument("--" + name, required=True)
    for name in ("measure", "memory", "search"):
        sub.choices[name].add_argument("--v2-decode", default=str(V2_DECODE))
        if name != "search":
            sub.choices[name].add_argument("--v2-sequence", default=str(V2_SEQUENCE))
    for sp in sub.choices.values():
        sp.add_argument("--bundle", default=str(BUNDLE))
    a = p.parse_args()
    {"check": stage_check, "convert": stage_convert, "merge": stage_merge, "align": stage_align,
     "compress": stage_compress, "measure": stage_measure, "memory": stage_memory, "search": stage_search,
     "fixture": stage_fixture, "ios": stage_ios}[a.stage](a)


if __name__ == "__main__":
    main()
