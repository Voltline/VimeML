"""How often a model's Top-1 prediction appears verbatim in the training text (ime-predict-zh-v1 track A).

usage:
  python -I -X utf8 scripts/benchmarks/verbatim_predict_zh.py OUT RUN [RUN ...] [--workers 8]
      RUN = a directory with track-a.jsonl written by predict_ime_zh.py (its name labels the model).

For every development track A item and every run, the Top-1 candidate c is taken with the last k Han
characters of the prefix (k = 0, 4, 8; the item is skipped for k when the prefix's final Han run is
shorter than k), and the string tail_k(prefix) + c is searched verbatim in every document of
export-r1/train. The reference is the human continuation cut to the same length as c, with the
same left context: a model whose strings hit the training text more often than the human text
does at the same length is reproducing training text, not just writing common Chinese.

export-r1/train is a superset of what the zh models saw (P1 pool and LCCC are drawn from it), so
the rates are an upper bound for the model and the reference alike. A candidate that is not
clean Han text (e.g. a cut byte-fallback character) counts as not found.

Writes OUT/hits.json (pattern -> found, local only, holds benchmark text) and OUT/summary.json and
OUT/summary.md (statistics only).
"""
import argparse
import collections
import glob
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ITEMS = ROOT / "outputs/benchmarks/ime-predict-zh-v1/frozen-r1/items-a.jsonl"
TRAIN = ROOT / "outputs/zh-corpus/export-r1/train"
HAN = re.compile("[〇㐀-䶿一-鿿豈-﫿]+")
KS = (0, 4, 8)


def tail(prefix, k):
    runs = HAN.findall(prefix)
    if k == 0:
        return ""
    if not runs or not prefix.endswith(runs[-1]) or len(runs[-1]) < k:
        return None
    return runs[-1][-k:]


def top1(row):
    c = row["candidates"]
    if isinstance(c, str):
        c = json.loads(c)
    return c[0][0] if c else None


def texts(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=20000, columns=["text"]):
        yield from batch.column(0).to_pylist()


def scan(job):
    import ahocorasick
    path, patterns = job
    automaton = ahocorasick.Automaton()
    for p in patterns:
        automaton.add_word(p, p)
    automaton.make_automaton()
    found = set()
    for t in texts(path):
        if t:
            for _, p in automaton.iter(t):
                found.add(p)
    return Path(path).name, found


def main():
    from vimeml.benchmarks.statistics import paired
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--files", help="glob of training files to search instead of export-r1/train (smoke runs)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True)
    items = {}
    with open(ITEMS, encoding="utf-8") as stream:
        for line in stream:
            i = json.loads(line)
            if i["split"] == "development":
                items[i["id"]] = i
    runs = {}
    for run in a.runs:
        rows = {}
        with open(Path(run) / "track-a.jsonl", encoding="utf-8", errors="surrogatepass") as stream:
            for line in stream:
                r = json.loads(line)
                rows[r["id"]] = top1(r)
        if set(rows) != set(items):
            raise SystemExit(f"{run}: track A ids differ from the development items")
        runs[Path(run).name] = rows
    # (run, k, id) -> (model string, reference string); None strings are "not found" without search
    pairs = {}
    for name, rows in runs.items():
        for i, item in items.items():
            c = rows[i]
            ok = bool(c) and HAN.fullmatch(c) is not None
            for k in KS:
                t = tail(item["prefix"], k)
                if t is None:
                    continue
                ref = item["continuation"][:len(c)] if ok else None
                pairs[(name, k, i)] = (t + c if ok else None, t + ref if ok and len(ref) == len(c) else None)
    patterns = sorted({s for pair in pairs.values() for s in pair if s})
    print(json.dumps({"items": len(items), "runs": list(runs), "patterns": len(patterns)}), flush=True)
    files = sorted(glob.glob(a.files or str(TRAIN / "*.parquet")))
    found = set()
    with Pool(a.workers) as pool:
        for name, hit in pool.imap_unordered(scan, [(f, patterns) for f in files]):
            found |= hit
            print(name, len(hit), flush=True)
    (out / "hits.json").write_text(json.dumps({p: p in found for p in patterns}, ensure_ascii=False), encoding="utf-8")
    summary = dict(items=len(items), train_files=len(files), ks=list(KS), runs={})
    lines = ["| model | k | items | model found | reference found | difference (points) | gains/losses | cluster p | cluster bootstrap 95% |",
             "|---|---:|---:|---:|---:|---:|---|---:|---|"]
    for name in runs:
        summary["runs"][name] = {}
        for k in KS:
            keys = sorted(i for (n, kk, i) in pairs if n == name and kk == k)
            model = [bool(pairs[(name, k, i)][0]) and pairs[(name, k, i)][0] in found for i in keys]
            ref = [bool(pairs[(name, k, i)][1]) and pairs[(name, k, i)][1] in found for i in keys]
            clean = sum(pairs[(name, k, i)][0] is not None for i in keys)
            p = paired(ref, model, [items[i]["dialogue"] for i in keys])
            n = len(keys)
            summary["runs"][name][str(k)] = dict(items=n, clean_candidates=clean, model_rate=sum(model) / n,
                                                 reference_rate=sum(ref) / n, paired=p)
            lo, hi = p["cluster_bootstrap_95"]
            lines.append(f"| {name} | {k} | {n} | {100 * sum(model) / n:.1f}% | {100 * sum(ref) / n:.1f}% | "
                         f"{100 * p['net'] / n:+.1f} | +{p['gain']}/−{p['loss']} | {p['cluster_sign_flip_p']:.3g} | "
                         f"[{100 * lo / n:+.1f}, {100 * hi / n:+.1f}] |")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
