"""vimeml-tiny-zh-v1 batched KV step v2 (read-only caches, optional prefill and scored head): Mac stages.

Brief: docs/reference/mac-zh-v1-kvbatch-v2.md. Every stage writes a fresh output.

  check     PyTorch: v2 graph (eager and traced) decode, prefill and scored-head checks vs full forward
  convert   FP32 v2 package: --batch flex|16|1,4,8,16  --tokens 1|flex  --slots flex|16,32,64,128  --head logits|scored
  align     vs PyTorch FP32, or vs the stateless package (--stateless)
  compress  INT8 block32 of an aligned FP32 v2 package
  measure   decode latency over the accepted B x S grid; scored head at K=149/450; prompt prefill time
  memory    macOS phys_footprint (current and lifetime peak), one fresh process per configuration
  search    the per-key-position beam search on a v2 package vs the FP32 fixture and vs a v1 INT8 batch run
  fixture   Vime fixture: 16 rows at different positions, one decode call (prompt prefilled with the package)
"""
import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/deployment"))

from vimeml.deployment import zh_v1, zh_v1_batch as zb, zh_v1_batch2 as z2  # noqa: E402
from vimeml.deployment.bundle import environment, write_json  # noqa: E402
from vimeml.training.data import file_sha  # noqa: E402
from coreml_zh_v1_batch import (BUNDLE, CHUNK_INT8, READINGS, STATELESS_INT8, fixture_rows, fresh, load,  # noqa: E402
                                rss_mib)

LAYERS, HEADS, HEAD_DIM, VOCAB = z2.LAYERS, z2.HEADS, z2.HEAD_DIM, z2.VOCAB


def parse_dim(text):
    if text == "flex":
        return "flex"
    values = [int(x) for x in text.split(",")]
    return values[0] if len(values) == 1 else values


def stage_check(a):
    import torch
    torch.set_num_threads(4)
    lm = zh_v1.BundleLMZh(a.bundle)
    model = lm.model.float().eval()
    eager = z2.run_checks(model, lambda: z2.KVBatchGraphZhV2(model))
    traced = {f"{head}-T{tokens}": "traced and checked" for head, tokens in (("logits", 1), ("logits", "flex"), ("scored", 1))
              if z2.trace(model, head, tokens) is not None}
    scored = z2.check_scored(model)
    worst = max(c["max_abs"] for v in eager.values() for c in v)
    write_json(fresh(a.output), {"format": "vimeml_kv_batch_v2_torch_check_zh_v1", "passed": True, "eager": eager,
                                 "traced": traced, "scored_head": scored, "environment": environment()})
    print(f"passed, max {worst:.3g}; scored head {scored['max_abs_log_prob']:.3g}")


def stage_convert(a):
    tokens = 1 if a.tokens == "1" else "flex"
    z2.convert(a.bundle, fresh(a.output), batch=parse_dim(a.batch), tokens=tokens, slots=parse_dim(a.slots), head=a.head)


def stage_align(a):
    sys.exit(0 if z2.align(a.package, a.bundle, fresh(a.output), stateless=a.stateless) else 1)


def stage_compress(a):
    z2.compress(a.package, fresh(a.output), a.alignment)


def timed(model, feed, repeats, warmup):
    for _ in range(warmup):
        model.predict(feed)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.predict(feed)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"p50_ms": round(statistics.median(times), 3), "p95_ms": round(times[int(0.95 * (len(times) - 1))], 3),
            "min_ms": round(times[0], 3), "repeats": repeats}


def decode_feed(rng, size, slots, tokens=1, allowed=None):
    k = rng.standard_normal((LAYERS, size, HEADS, slots, HEAD_DIM)).astype(np.float32)
    feed = {"input_ids": rng.integers(8, VOCAB, (size, tokens)).astype(np.int32),
            "position": rng.integers(0, slots, size).astype(np.int32) if tokens == 1 else np.zeros(size, np.int32),
            "key_cache": k, "value_cache": k.copy()}
    if allowed is not None:
        feed["allowed"] = rng.choice(np.arange(8, VOCAB), allowed, replace=False).astype(np.int32)
    return feed


def grid(manifest, batches, slots):
    bs, ss = z2.shapes_of(manifest)
    lo, hi = manifest["batch_range"]
    out = []
    for b in batches:
        if b > hi:
            continue
        size = min(o for o in bs if o >= b) if bs else b
        for s in slots:
            if ss and s > max(ss):
                continue
            out.append((b, size, min(o for o in ss if o >= s) if ss else s))
    return out


def stage_measure(a):
    from vimeml.deployment.coreml import coremltools
    ct = coremltools()
    out = fresh(a.output)
    stateless = load(ct, a.stateless, a.compute_units) if a.stateless else None
    if stateless is not None:
        stateless.predict({"input_ids": np.full((1, 16), 7, np.int32)})
    path = Path(a.package)
    manifest = zh_v1.verify_package(path, z2.FORMAT)
    model = load(ct, path, a.compute_units)
    scored = manifest["head"] == "scored"
    rng = np.random.default_rng(1115)
    results = []
    for rows, size, s in grid(manifest, a.batches, a.slots):
        for k in (a.allowed if scored else [None]):
            row = {"call": "decode", "rows": rows, "batch_shape": size, "slots": s, "allowed": k,
                   **timed(model, decode_feed(rng, size, s, allowed=k), a.repeats, a.warmup)}
            results.append(row)
            print(json.dumps(row), flush=True)
    bs, ss = z2.shapes_of(manifest)
    one = min(bs) if bs else 1
    for t in a.prompt_tokens:
        if manifest["tokens"].startswith("flexible"):
            feed = decode_feed(rng, one, 1, tokens=t, allowed=a.allowed[0] if scored else None)
            row = {"call": "prefill (one call)", "tokens": t, **timed(model, feed, a.repeats // 4, a.warmup)}
        else:
            feeds = [decode_feed(rng, one, (min(o for o in ss if o >= max(i, 1)) if ss else max(i, 1)),
                                 allowed=a.allowed[0] if scored else None) for i in range(t)]
            for i, f in enumerate(feeds):
                f["position"][:] = 0
                f["position"][0] = i
            for f in feeds[:2]:
                model.predict(f)
            totals = []
            for _ in range(max(a.repeats // 10, 5)):
                start = time.perf_counter()
                for f in feeds:
                    model.predict(f)
                totals.append((time.perf_counter() - start) * 1000)
            totals.sort()
            row = {"call": "prefill (one token per call, B=%d)" % one, "tokens": t,
                   "p50_ms": round(statistics.median(totals), 3), "max_ms": round(totals[-1], 3), "repeats": len(totals)}
        results.append(row)
        print(json.dumps(row), flush=True)
    write_json(out, {"format": "vimeml_kv_batch_v2_latency_zh_v1", "package": path.name, "kind": manifest["kind"],
                     "head": manifest["head"], "batch": manifest["batch"], "cache_slots": manifest["cache_slots"],
                     "tokens": manifest["tokens"], "package_manifest_sha256": file_sha(path / "manifest.json"),
                     "compute_units": a.compute_units, "stateless_loaded": str(a.stateless) if a.stateless else None,
                     "results": results, "note": "Mac python predict() wall time incl. input conversion.",
                     "environment": environment()})


# ---- memory -----------------------------------------------------------------------------------------

def footprint():
    """(phys_footprint, lifetime max phys_footprint) in MiB via proc_pid_rusage(RUSAGE_INFO_V4)."""
    class RUsageV4(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(40)]
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    info = RUsageV4()
    if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    return info.f7 / 2 ** 20, info.f28 / 2 ** 20      # ri_phys_footprint, ri_lifetime_max_phys_footprint


def snap():
    now, peak = footprint()
    return {"phys_footprint_mib": round(now, 1), "peak_phys_footprint_mib": round(peak, 1), "rss_mib": round(rss_mib(), 1)}


def memory_child(spec):
    import coremltools as ct
    unit = ct.ComputeUnit.CPU_ONLY
    rng = np.random.default_rng(1)
    mem = {"start": snap()}
    st = ct.models.MLModel(str(Path(spec["stateless"]) / "model.mlpackage"), compute_units=unit)
    st.predict({"input_ids": np.full((1, 16), 7, np.int32)})
    mem["after_stateless"] = snap()
    if spec.get("chunk"):
        chunk = ct.models.MLModel(str(Path(spec["chunk"]) / "model.mlpackage"), compute_units=unit)
        z = np.zeros((LAYERS, 1, HEADS, 128, HEAD_DIM), np.float32)
        chunk.predict({"input_ids": rng.integers(8, VOCAB, (1, 12)).astype(np.int32), "position": np.array([0], np.int32),
                       "key_cache": z, "value_cache": z})
        mem["after_chunk"] = snap()
    path = Path(spec["package"])
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    v2 = manifest["format"] == z2.FORMAT
    model = ct.models.MLModel(str(path / "model.mlpackage"), compute_units=unit)
    mem["after_load"] = snap()
    scored = v2 and manifest["head"] == "scored"
    if v2:
        bs, ss = z2.shapes_of(manifest)
    else:
        lo, hi = manifest["batch_range"]
        bs, ss = (None if lo != hi else [hi]), None
    fit = lambda n, opts: n if not opts else min(o for o in opts if o >= n)
    hi = manifest["batch_range"][1]

    def call(rows, slots, tokens=1):
        feed = decode_feed(rng, fit(rows, bs), fit(slots, ss), tokens=tokens, allowed=160 if scored else None)
        model.predict(feed)

    if v2 and manifest["tokens"].startswith("flexible"):
        call(1, 1, tokens=24)
        mem["after_prefill_24"] = snap()
    call(min(16, hi), 32)
    mem["after_first_decode_B16_S32"] = snap()
    for rows in (1, 4, 8, 16):
        for slots in (16, 32, 64):
            if rows <= hi and (not ss or slots <= max(ss)):
                call(rows, slots)
    mem["after_sweep_S_upto_64"] = snap()
    if not ss or max(ss) >= 128:
        call(min(16, hi), 128)
        mem["after_B16_S128"] = snap()
    return mem


def stage_memory(a):
    out = fresh(a.output)
    configs = []
    for item in a.configs:
        label, _, rest = item.partition("=")
        package, _, extra = rest.partition("+")
        configs.append({"label": label, "package": package, "chunk": str(CHUNK_INT8) if extra == "chunk" else None,
                        "stateless": str(a.stateless)})
    results = {}
    for spec in configs:
        runs = []
        for _ in range(a.runs):
            res = subprocess.run([sys.executable, __file__, "memory-child", json.dumps(spec)], capture_output=True,
                                 text=True, check=True, cwd=ROOT)
            runs.append(json.loads(res.stdout.strip().splitlines()[-1]))
        base = lambda r: r["after_stateless"]["phys_footprint_mib"]
        last = lambda r: r[list(r)[-1]]
        results[spec["label"]] = {
            "spec": spec, "runs": runs,
            "added_over_stateless_mib": statistics.median(last(r)["phys_footprint_mib"] - base(r) for r in runs),
            "peak_added_over_stateless_mib": statistics.median(last(r)["peak_phys_footprint_mib"] - base(r) for r in runs),
            "load_only_mib": statistics.median(r["after_load"]["phys_footprint_mib"]
                                               - r["after_chunk" if spec["chunk"] else "after_stateless"]["phys_footprint_mib"]
                                               for r in runs)}
        print(json.dumps({spec["label"]: {k: v for k, v in results[spec["label"]].items() if k not in ("runs", "spec")}}), flush=True)
    write_json(out, {"format": "vimeml_kv_batch_v2_memory_zh_v1", "results": results,
                     "note": "Mac, Python, CPU_ONLY, fresh process per run; proc_pid_rusage V4 phys_footprint and "
                             "lifetime max. Medians over runs. Not the iOS keyboard-extension footprint.",
                     "environment": environment()})


# ---- search and fixture -------------------------------------------------------------------------------

def compare(got, want):
    """got/want: [(text, score)] best first."""
    ref = dict(want)
    common = [abs(s - ref[t]) for t, s in got if t in ref]
    return {"exact": [t for t, _ in got] == [t for t, _ in want], "same_set": {t for t, _ in got} == set(ref),
            "top1_equal": bool(got) and bool(want) and got[0][0] == want[0][0],
            "max_common_score_diff": max(common or [0.0]), "dropped": [t for t, _ in want if t not in dict(got)]}


def summarize(rows, key):
    xs = [r[key] for r in rows]
    return {"exact_order": sum(x["exact"] for x in xs), "same_text_set": sum(x["same_set"] for x in xs),
            "top1_equal": sum(x["top1_equal"] for x in xs),
            "max_common_score_diff": max(x["max_common_score_diff"] for x in xs)}


def stage_search(a):
    import sentencepiece as spm
    from vimeml.deployment import zh_search as zs
    from vimeml.posttraining.zh_readings import ReadingTable
    out = fresh(a.output)
    sp = spm.SentencePieceProcessor(model_file=str(Path(a.bundle) / "tokenizer.model"))
    index = zs.Index(sp, ReadingTable(str(a.readings)))
    be = z2.CoreMLBatchBackendV2(a.package, chunk=a.chunk if a.prompt == "chunk" else None)
    ref = zb.CoreMLBatchBackend(a.chunk, a.reference_batch) if a.reference_batch else None
    meta = json.loads((Path(a.cases).parent / "meta.json").read_text(encoding="utf-8"))
    beam, topk, mode = meta["search"]["beam"], meta["search"]["topk"], meta["search"]["mode"]
    rows, times = [], []
    for line in open(a.cases, encoding="utf-8"):
        c = json.loads(line)
        calls, prefills = be.calls, be.prefill_calls
        t = time.perf_counter()
        ranked, st = zs.search(be, sp, index, c["keys"], c["context"], mode, beam, topk)
        times.append((time.perf_counter() - t) * 1000)
        got = [(x[0], x[1]) for x in ranked]
        row = {"id": c["id"], "top": got, "calls": st.calls, "rows": st.rows, "prefill_calls": be.prefill_calls - prefills,
               "ms": round(times[-1], 1), "vs_fp32_fixture": compare(got, [(x["text"], x["gen_score"]) for x in c["top"]])}
        if ref is not None:
            r_ranked, _ = zs.search(ref, sp, index, c["keys"], c["context"], mode, beam, topk)
            row["vs_reference_batch"] = compare(got, [(x[0], x[1]) for x in r_ranked])
        rows.append(row)
    summary = {"cases": len(rows), "vs_fp32_fixture": summarize(rows, "vs_fp32_fixture"),
               "mean_calls": statistics.mean(r["calls"] for r in rows), "mean_rows": statistics.mean(r["rows"] for r in rows),
               "mean_prefill_calls": statistics.mean(r["prefill_calls"] for r in rows),
               "p50_ms": statistics.median(times), "p95_ms": sorted(times)[int(0.95 * (len(times) - 1))],
               "prompt": "package (one call)" if be.flexible_tokens else ("chunk package" if be.chunk else "package, one token per call")}
    if ref is not None:
        summary["vs_reference_batch"] = summarize(rows, "vs_reference_batch")
        summary["reference_batch"] = Path(a.reference_batch).name
    out.mkdir(parents=True)
    write_json(out / "summary.json", {"format": "vimeml_kv_batch_v2_search_check_zh_v1", "package": Path(a.package).name,
        "package_manifest_sha256": file_sha(Path(a.package) / "manifest.json"), "fixture": str(a.cases),
        "summary": summary, "environment": environment()})
    with open(out / "cases.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


def stage_fixture(a):
    import torch
    from vimeml.deployment.coreml import coremltools
    ct = coremltools()
    out = fresh(a.output)
    lm = zh_v1.BundleLMZh(a.bundle)
    model = lm.model.float().eval()
    rows = fixture_rows(lm.processor)
    be = z2.CoreMLBatchBackendV2(a.package)
    scored = be.scored
    with torch.inference_mode():
        full = [model(torch.tensor([r["prefix_ids"] + [r["token"]]]))[0, -1].float() for r in rows]
    allowed = sorted({int(i) for f in full for i in torch.topk(f, 20).indices}) if scored else None
    states = [be.prefill(r["prefix_ids"], allowed)[1] for r in rows]
    got, _ = be.decode([r["token"] for r in rows], states, allowed)
    stateless = load(ct, a.stateless, "CPU_ONLY")
    s_logits = [np.asarray(stateless.predict({"input_ids": np.array([r["prefix_ids"] + [r["token"]]], np.int32)})
                           ["logits"][0, -1], np.float32) for r in rows]
    if scored:
        s_ref = [torch.from_numpy(x).log_softmax(-1)[allowed].numpy() for x in s_logits]
        p_ref = [f.log_softmax(-1)[allowed].numpy() for f in full]
    else:
        s_ref, p_ref = s_logits, [f.numpy() for f in full]
    worst = max(float(np.abs(got[i].numpy() - s_ref[i]).max()) for i in range(len(rows)))
    ids = allowed if scored else list(range(VOCAB))
    top = lambda x: [[int(ids[i]), round(float(x[i]), 5)] for i in np.argsort(-np.asarray(x))[:5]]
    cases = [{**r, "top5_package": top(got[i].numpy()), "top5_stateless_same_precision": top(s_ref[i]),
              "top5_pytorch_fp32": top(p_ref[i])} for i, r in enumerate(rows)]
    out.mkdir(parents=True)
    with open(out / "batch16.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.savez_compressed(out / "batch16_logits.npz", package=got.numpy(), stateless=np.stack(s_ref), pytorch_fp32=np.stack(p_ref),
                        **({"allowed": np.array(allowed, np.int32)} if scored else {}))
    write_json(out / "meta.json", {"format": "vime_zh_kv_batch_fixture_v2", "rows": len(rows), "head": be.manifest["head"],
        "package": Path(a.package).name, "package_manifest_sha256": file_sha(Path(a.package) / "manifest.json"),
        "stateless_package": Path(a.stateless).name, "prompt": "prefilled with the package itself (" + (
            "one call per row" if be.flexible_tokens else "one token per call") + ")",
        "how": "prefill each row's prefix_ids (B=1, position 0), copy the cache rows into a (9,16,6,S,64) buffer with "
               "S >= max(len(prefix_ids)), then one decode call with input_ids = token, position = len(prefix_ids)"
               + ("; allowed = the npz 'allowed' ids; values are log-probs (allowed_logits - logsumexp)" if scored else ""),
        "max_abs_package_vs_stateless": worst, "tolerance": {"atol": z2.TOLERANCE, "rtol": z2.TOLERANCE},
        "tokenizer_sha256": lm.metadata["tokenizer_sha256"]})
    print(f"fixture written; package vs stateless max {worst:.3g}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "memory-child":
        print(json.dumps(memory_child(json.loads(sys.argv[2]))))
        return
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("check"); s.add_argument("output")
    s = sub.add_parser("convert"); s.add_argument("output"); s.add_argument("--batch", default="flex")
    s.add_argument("--tokens", default="1", choices=("1", "flex")); s.add_argument("--slots", default="flex")
    s.add_argument("--head", default="logits", choices=("logits", "scored"))
    s = sub.add_parser("align"); s.add_argument("package"); s.add_argument("output"); s.add_argument("--stateless")
    s = sub.add_parser("compress"); s.add_argument("package"); s.add_argument("alignment"); s.add_argument("output")
    s = sub.add_parser("measure"); s.add_argument("package"); s.add_argument("output")
    s.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    s.add_argument("--slots", type=int, nargs="+", default=[16, 32, 64, 128])
    s.add_argument("--allowed", type=int, nargs="+", default=[149, 450])
    s.add_argument("--prompt-tokens", type=int, nargs="+", default=[8, 16, 32])
    s.add_argument("--repeats", type=int, default=200); s.add_argument("--warmup", type=int, default=20)
    s.add_argument("--stateless", default=str(STATELESS_INT8), help="'' to measure without it")
    s.add_argument("--compute-units", default="CPU_ONLY", choices=("CPU_ONLY", "CPU_AND_NE", "ALL"))
    s = sub.add_parser("memory"); s.add_argument("output")
    s.add_argument("configs", nargs="+", help="label=<package dir>[+chunk]  (+chunk also loads the shipped chunk package)")
    s.add_argument("--runs", type=int, default=3); s.add_argument("--stateless", default=str(STATELESS_INT8))
    s = sub.add_parser("search"); s.add_argument("package"); s.add_argument("cases"); s.add_argument("output")
    s.add_argument("--prompt", default="package", choices=("package", "chunk"),
                   help="package: the v2 package itself (one call if T is flexible, else one token per call)")
    s.add_argument("--chunk", default=str(CHUNK_INT8)); s.add_argument("--readings", default=str(READINGS))
    s.add_argument("--reference-batch", help="v1 batch package to run the same search with (chunk prefill)")
    s = sub.add_parser("fixture"); s.add_argument("package"); s.add_argument("output")
    s.add_argument("--stateless", default=str(STATELESS_INT8))
    for sp in sub.choices.values():
        sp.add_argument("--bundle", default=str(BUNDLE))
    a = p.parse_args()
    {"check": stage_check, "convert": stage_convert, "align": stage_align, "compress": stage_compress,
     "measure": stage_measure, "memory": stage_memory, "search": stage_search, "fixture": stage_fixture}[a.stage](a)


if __name__ == "__main__":
    main()
