"""vimeml-tiny-zh-v1 batched KV decoding package for Vime's constrained beam search (Mac stages).

Brief: docs/zh/mac/mac-zh-v1-kvbatch.md. Every stage writes a fresh output.

  check     PyTorch: batch graph (eager and traced) vs full forward
  convert   FP32 batch package: --batch flex|8|16, --cache-dtype float32|float16
  align     Core ML batch package vs PyTorch FP32, or vs a stateless package (--stateless)
  compress  INT8 block32 of an aligned FP32 batch package
  merge     multifunction package: prefill (batch-1 chunk) + decode (batch step), shared weights
  measure   per-predict latency over B and S (and optional stateless package loaded alongside)
  fixture   Vime fixture: B=16 rows at different positions, one decode call, per-row top-5
  search    reference beam search on Core ML (chunk prefill + batch decode) vs a generation fixture
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.deployment import zh_v1, zh_v1_batch as zb  # noqa: E402
from vimeml.deployment.bundle import environment, write_json  # noqa: E402
from vimeml.training.data import file_sha  # noqa: E402

RELEASE = ROOT / "artifacts/releases/vimeml-tiny-zh-v1-hf-v1"
BUNDLE = RELEASE / "inference"
STATELESS_INT8 = RELEASE / "coreml/ios18-int8-block32"
CHUNK_INT8 = RELEASE / "coreml/ios18-int8-block32-kvchunk"
READINGS = ROOT / "outputs/zh-readings/dict/vime_pinyin.dict.yaml"


def fresh(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} exists; choose a new output.")
    return path


def stage_check(a):
    import torch
    torch.set_num_threads(4)
    lm = zh_v1.BundleLMZh(a.bundle)
    traced = zb.trace_batch_graph(lm.model)
    stats = [zb.check_graph(lm.model, fn, **case) for fn in (zb.KVBatchGraphZh(lm.model), traced)
             for case in zb.ALIGN_CASES]
    write_json(fresh(a.output), {"format": "vimeml_kv_batch_torch_check_zh_v1", "passed": True, "checks": stats,
                                 "environment": environment()})
    print(f"passed, max {max(s['max_abs'] for s in stats):.3g}")


def stage_convert(a):
    batch = None if a.batch == "flex" else int(a.batch)
    zb.convert(a.bundle, fresh(a.output), batch=batch, cache_dtype=a.cache_dtype)


def stage_align(a):
    ok = zb.align(a.package, a.bundle, fresh(a.output), stateless=a.stateless)
    sys.exit(0 if ok else 1)


def stage_compress(a):
    zb.compress(a.package, fresh(a.output), a.alignment)


def stage_merge(a):
    zb.merge(a.chunk, a.package, fresh(a.output))


def rss_mib():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except ImportError:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2 ** 20 if sys.platform == "darwin" else peak / 1024


def load(ct, path, unit, function=None):
    kwargs = {"compute_units": getattr(ct.ComputeUnit, unit)}
    if function:
        kwargs["function_name"] = function
    return ct.models.MLModel(str(Path(path) / "model.mlpackage"), **kwargs)


def stage_measure(a):
    from vimeml.deployment.coreml import coremltools
    ct = coremltools()
    out = fresh(a.output)
    rss = {"start": rss_mib()}
    stateless = load(ct, a.stateless, a.compute_units) if a.stateless else None
    if stateless is not None:
        stateless.predict({"input_ids": np.full((1, 16), 7, np.int32)})
        rss["after_stateless"] = rss_mib()
    path = Path(a.package)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    merged = manifest["format"] == zb.MERGED_FORMAT
    model = load(ct, path, a.compute_units, "decode" if merged else None)
    info = manifest["functions"]["decode"] if merged else manifest
    lo, hi = info["batch_range"]
    dtype = np.float16 if (manifest.get("cache_dtype") or "float32") == "float16" else np.float32
    rng = np.random.default_rng(1115)
    results = []
    for b in a.batches:
        if not (lo <= b <= hi or (lo == hi and b < hi)):
            continue                     # a fixed package pads fewer rows; more rows than its batch are not measured
        size = hi if lo == hi else b
        for s in a.slots:
            k = rng.standard_normal((zb.LAYERS, size, zb.HEADS, s, zb.HEAD_DIM)).astype(dtype)
            feed = {"input_ids": rng.integers(8, zb.VOCAB, (size, 1)).astype(np.int32),
                    "position": rng.integers(0, s, size).astype(np.int32), "key_cache": k, "value_cache": k.copy()}
            for _ in range(a.warmup):
                model.predict(feed)
            times = []
            for _ in range(a.repeats):
                t = time.perf_counter()
                model.predict(feed)
                times.append((time.perf_counter() - t) * 1000)
            times.sort()
            row = {"rows": b, "batch_shape": size, "slots": s, "p50_ms": round(statistics.median(times), 3),
                   "p95_ms": round(times[int(0.95 * (len(times) - 1))], 3), "min_ms": round(times[0], 3),
                   "repeats": a.repeats}
            results.append(row)
            print(json.dumps(row), flush=True)
    rss["after_batch"] = rss_mib()
    write_json(out, {"format": "vimeml_kv_batch_latency_zh_v1", "package": path.name, "kind": manifest["kind"],
                     "package_manifest_sha256": file_sha(path / "manifest.json"), "compute_units": a.compute_units,
                     "stateless_loaded": str(a.stateless) if a.stateless else None, "results": results,
                     "rss_mib": rss, "note": "Mac python predict() wall time incl. input conversion; "
                     "RSS is process resident memory, not the iOS footprint.", "environment": environment()})


def fixture_rows(sp):
    """16 rows: real sentence prefixes of different lengths, each followed by one more token."""
    text = ("今天天气很好我们去公园散步吧", "中文输入法", "你好", "我想吃饭了", "明天上午去故宫参观下午逛街",
            "这个问题需要你看一下", "晚上一起吃火锅吗", "谢谢", "好的没问题", "我在地铁上马上到",
            "周末有空吗", "这家店的菜很好吃", "帮我订一张去上海的票", "下雨了记得带伞", "会议改到三点", "收到")
    rows = []
    for t in text:
        ids = [2] + sp.encode(t)
        rows.append({"text": t, "prefix_ids": ids[:-1], "token": ids[-1], "position": len(ids) - 1})
    return rows


def stage_fixture(a):
    import sentencepiece as spm
    import torch
    from vimeml.deployment.coreml import coremltools
    ct = coremltools()
    out = fresh(a.output)
    lm = zh_v1.BundleLMZh(a.bundle)
    rows = fixture_rows(lm.processor)
    if len({r["position"] for r in rows}) < 8:
        raise ValueError("Fixture rows should have mostly different positions.")
    width = max(r["position"] for r in rows) + 1
    torch_be = zb.TorchBatchBackend(lm.model)
    states = [torch_be.prefill(r["prefix_ids"])[1] for r in rows]
    t_logits, _ = torch_be.decode([r["token"] for r in rows], states)
    full = [lm.model(torch.tensor([r["prefix_ids"] + [r["token"]]]))[0, -1] for r in rows]
    worst_torch = max(float((t_logits[i] - full[i]).abs().max()) for i in range(len(rows)))
    be = zb.CoreMLBatchBackend(a.chunk, a.package)
    states = [be.prefill(r["prefix_ids"])[1] for r in rows]
    c_logits, _ = be.decode([r["token"] for r in rows], states)
    stateless = load(ct, a.stateless, "CPU_ONLY")
    s_logits = [np.asarray(stateless.predict({"input_ids": np.array([r["prefix_ids"] + [r["token"]]], np.int32)})
                           ["logits"][0, -1], np.float32) for r in rows]
    worst = max(float(np.abs(c_logits[i].numpy() - s_logits[i]).max()) for i in range(len(rows)))
    top = lambda x: [[int(i), round(float(x[i]), 5)] for i in np.argsort(-np.asarray(x))[:5]]
    cases = [{**r, "top5_package": top(c_logits[i].numpy()), "top5_stateless_same_precision": top(s_logits[i]),
              "top5_pytorch_fp32": top(full[i].detach().numpy())} for i, r in enumerate(rows)]
    out.mkdir(parents=True)
    with open(out / "batch16.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.savez_compressed(out / "batch16_logits.npz", package=c_logits.numpy(),
                        stateless=np.stack(s_logits), pytorch_fp32=torch.stack(full).detach().numpy())
    write_json(out / "meta.json", {"format": "vime_zh_kv_batch_fixture_v1", "rows": len(rows), "slots_needed": width,
        "package": Path(a.package).name, "package_manifest_sha256": file_sha(Path(a.package) / "manifest.json"),
        "chunk_package": Path(a.chunk).name, "stateless_package": Path(a.stateless).name,
        "how": "prefill each row's prefix_ids with the batch-1 chunk package (position 0), copy the cache rows "
               "into a (9,16,6,S,64) buffer with S >= slots_needed, then one decode call with input_ids = token, "
               "position = len(prefix_ids)",
        "max_abs_package_vs_stateless": worst, "max_abs_pytorch_batch_vs_full": worst_torch,
        "tolerance": {"atol": zb.TOLERANCE, "rtol": zb.TOLERANCE}, "tokenizer_sha256": lm.metadata["tokenizer_sha256"]})
    print(f"fixture written; package vs stateless max {worst:.3g}, PyTorch batch vs full {worst_torch:.3g}")


def stage_search(a):
    import sentencepiece as spm
    from vimeml.deployment import zh_search as zs
    from vimeml.posttraining.zh_readings import ReadingTable
    out = fresh(a.output)
    sp = spm.SentencePieceProcessor(model_file=str(Path(a.bundle) / "tokenizer.model"))
    index = zs.Index(sp, ReadingTable(str(a.readings)))
    be = zb.CoreMLBatchBackend(a.chunk, a.package)
    meta = json.loads((Path(a.cases).parent / "meta.json").read_text(encoding="utf-8"))
    beam, topk = meta["search"]["beam"], meta["search"]["topk"]
    rows, exact, same_set, top1, worst, worst_common, times = [], 0, 0, 0, 0.0, 0.0, []
    for line in open(a.cases, encoding="utf-8"):
        c = json.loads(line)
        t = time.perf_counter()
        ranked, st = zs.search(be, sp, index, c["keys"], c["context"], meta["search"]["mode"], beam, topk)
        times.append((time.perf_counter() - t) * 1000)
        got = [x[0] for x in ranked]
        want = [x["text"] for x in c["top"]]
        diff = max([abs(s - w["gen_score"]) for (_, s, _), w in zip(ranked, c["top"])] or [0.0])
        ref = {w["text"]: w["gen_score"] for w in c["top"]}
        common = max([abs(s - ref[t]) for t, s, _ in ranked if t in ref] or [0.0])
        exact += got == want
        same_set += set(got) == set(want)
        top1 += bool(got) and bool(want) and got[0] == want[0]
        worst, worst_common = max(worst, diff), max(worst_common, common)
        rows.append({"id": c["id"], "exact": got == want, "same_set": set(got) == set(want), "max_score_diff": diff,
                     "max_score_diff_common_texts": common, "dropped": [t for t in want if t not in got],
                     "calls": st.calls, "rows": st.rows, "ms": round(times[-1], 1)})
    n = len(rows)
    summary = {"cases": n, "exact_order": exact, "same_text_set": same_set, "top1_equal": top1,
               "max_score_diff": worst, "max_score_diff_common_texts": worst_common,
               "mean_calls": statistics.mean(r["calls"] for r in rows), "mean_rows": statistics.mean(r["rows"] for r in rows),
               "p50_ms": statistics.median(times), "tolerance": meta.get("tolerance")}
    out.mkdir(parents=True)
    write_json(out / "summary.json", {"format": "vimeml_kv_batch_search_check_zh_v1", "package": Path(a.package).name,
        "package_manifest_sha256": file_sha(Path(a.package) / "manifest.json"), "fixture": str(a.cases),
        "summary": summary, "environment": environment()})
    with open(out / "cases.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("check"); s.add_argument("output")
    s = sub.add_parser("convert"); s.add_argument("output"); s.add_argument("--batch", default="flex")
    s.add_argument("--cache-dtype", default="float32", choices=("float32", "float16"))
    s = sub.add_parser("align"); s.add_argument("package"); s.add_argument("output"); s.add_argument("--stateless")
    s = sub.add_parser("compress"); s.add_argument("package"); s.add_argument("alignment"); s.add_argument("output")
    s = sub.add_parser("merge"); s.add_argument("package"); s.add_argument("output")
    s.add_argument("--chunk", default=str(CHUNK_INT8))
    s = sub.add_parser("measure"); s.add_argument("package"); s.add_argument("output")
    s.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    s.add_argument("--slots", type=int, nargs="+", default=[16, 32, 64, 128])
    s.add_argument("--repeats", type=int, default=200); s.add_argument("--warmup", type=int, default=20)
    s.add_argument("--stateless", default=str(STATELESS_INT8), help="'' to measure without it")
    s.add_argument("--compute-units", default="CPU_ONLY", choices=("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"))
    s = sub.add_parser("fixture"); s.add_argument("package"); s.add_argument("output")
    s.add_argument("--chunk", default=str(CHUNK_INT8)); s.add_argument("--stateless", default=str(STATELESS_INT8))
    s = sub.add_parser("search"); s.add_argument("package"); s.add_argument("cases"); s.add_argument("output")
    s.add_argument("--chunk", default=str(CHUNK_INT8),
                   help="prompt package; an FP32 batch package needs an FP32 chunk package here to match FP32 exactly")
    s.add_argument("--readings", default=str(READINGS))
    for name, sp in sub.choices.items():
        sp.add_argument("--bundle", default=str(BUNDLE))
    a = p.parse_args()
    {"check": stage_check, "convert": stage_convert, "align": stage_align, "compress": stage_compress,
     "merge": stage_merge, "measure": stage_measure, "fixture": stage_fixture, "search": stage_search}[a.stage](a)


if __name__ == "__main__":
    main()
