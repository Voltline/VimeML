"""Run and score the Chinese prediction benchmark ime-predict-zh-v1-r1 on development.

  run    decode tracks A, C (both key modes) and B for one model; development only
  score  Top-k rates (A, C), keystroke savings with the tuned threshold (B), coverage (D)

Protocol: docs/zh-CN/benchmarks/predict-zh-plan.md. Decoder: vimeml.benchmarks.predict_zh.
The blind split is never read here.
"""
import argparse
import json
import math
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from vimeml.benchmarks import predict_zh as P  # noqa: E402
from vimeml.training.data import file_sha, write_json  # noqa: E402

VERSION = "ime-predict-zh-v1-r1"
FROZEN = ROOT / "outputs/benchmarks/ime-predict-zh-v1/frozen-r1"
FROZEN_SHA = {
    "items-a.jsonl": "493fc4702b9dc37b1f645fcfdcc3efa515f840adcef7e1423c3d233b21eea87f",
    "messages-b.jsonl": "2640d6b573fb2e45996a7b6556a2aeafe9af3e42e96603588947ba75fb7a1c71",
    "items-c.jsonl": "dc88bc5f0a7e1ed2e3c0ee2e9ae7841ca2712e13a34151e293e2b205df21fb2e",
    "manifest.json": "002dc9a39113e1831b5d2352187a079e15705770310dc81de6e06a87e68f6415",
}
RIME = ROOT / "datasets/ime-standard-zh-v1/engine/rime-ice"
QUANTILES = [q / 100 for q in range(5, 100, 5)]
MODES = (("first_syllable", "syllable", "keys_first_syllable"), ("initials", "initials", "keys_initials"))


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, file=sys.stderr, flush=True)


def development(name, limit):
    rows = [r for r in P.read_jsonl(FROZEN / name) if r["split"] == "development"]
    if limit:  # smoke runs: the first `limit` items of every source
        seen = {}
        rows = [r for r in rows if seen.setdefault(r["source"], []).append(1) or len(seen[r["source"]]) <= limit]
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def quantile(values, q):
    values = sorted(values)
    position = q * (len(values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def load_model(args):
    if args.model != "vime":
        raise SystemExit(f"Unknown model adapter {args.model}")
    from vimeml.benchmarks.zh13_lm_scores import load
    from vimeml.training.joint_zh13 import SPECIAL
    model, processor, meta = load(args.checkpoint, args.tokenizer, args.device)
    return P.VimeLM(model, processor, args.device, SPECIAL), meta


def run(args):
    for name, sha in FROZEN_SHA.items():
        if file_sha(FROZEN / name) != sha:
            raise SystemExit(f"{name} differs from the frozen {VERSION}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lm, model_meta = load_model(args)
    dicts = RIME / "cn_dicts"
    readings = P.load_readings(dicts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import jieba
        import pypinyin
    jieba.setLogLevel(60)
    rime_commit = subprocess.run(["git", "-C", str(RIME), "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()
    meta = {
        "version": VERSION, "split": "development", "limit_per_source": args.limit,
        "model": {"adapter": args.model, **model_meta, "device": args.device, "dtype": "float32"},
        "decoder": {"beam": P.BEAM, "max_chars": P.MAX_CHARS, "min_chars": P.MIN_CHARS, "top": P.TOP,
                    "show": P.SHOW, "reserve": P.RESERVE, "context_length": lm.context_length},
        "frozen_sha256": FROZEN_SHA,
        "readings": {"rime_ice_commit": rime_commit,
                     **{n: file_sha(dicts / n) for n in ("8105.dict.yaml", "41448.dict.yaml")}},
        "jieba": jieba.__version__, "pypinyin": pypinyin.__version__, "torch": torch.__version__,
    }
    started = time.time()

    rows_a, items = [], development("items-a.jsonl", args.limit)
    for k, item in enumerate(items, 1):
        candidates = P.predict(lm, item["previous_turns"], item["prefix"])
        rank, chars = P.rank_of_hit(candidates, item["continuation"])
        rows_a.append({"id": item["id"], "source": item["source"], "dialogue": item["dialogue"],
                       "candidates": candidates, "rank": rank, "hit_chars": chars,
                       "confidence": candidates[0][1] if candidates else None})
        if k % 500 == 0:
            log("A", k, len(items))
    write_jsonl(out / "track-a.jsonl", rows_a)
    meta["seconds_a"] = round(time.time() - started, 1)

    rows_c, items = [], development("items-c.jsonl", args.limit)
    for k, item in enumerate(items, 1):
        for mode, kind, field in MODES:
            constraint = P.Constraint(kind, item[field])
            candidates = P.predict(lm, item["previous_turns"], item["prefix"], constraint, readings)
            rows_c.append({"id": item["id"], "source": item["source"], "dialogue": item["dialogue"],
                           "mode": mode, "keys": item[field], "candidates": candidates,
                           "rank": P.rank_exact(candidates, item["target"])})
        if k % 500 == 0:
            log("C", k, len(items))
    write_jsonl(out / "track-c.jsonl", rows_c)
    meta["seconds_c"] = round(time.time() - started - meta["seconds_a"], 1)

    scores = [r["confidence"] for r in rows_a if r["confidence"] is not None]
    grid = [P.NEG] + [quantile(scores, q) for q in QUANTILES]
    meta["threshold_grid"] = [P.finite(t) for t in grid]

    def pinyin(run):
        return pypinyin.lazy_pinyin(run, style=pypinyin.Style.NORMAL)

    def cut(message):
        return jieba.lcut(message, HMM=False)

    rows_b, items, decodes = [], development("messages-b.jsonl", args.limit), 0
    for k, item in enumerate(items, 1):
        message, memo = item["message"], {}

        def predictions(position):
            nonlocal decodes
            if position not in memo:
                memo[position] = P.predict(lm, item["previous_turns"], message[:position])
                decodes += 1
            return memo[position]

        costs, ends = P.keystrokes(message, pinyin), P.boundaries(message, cut)
        results = [P.simulate(message, costs, ends, predictions, t) for t in grid]
        rows_b.append({"id": item["id"], "source": item["source"], "dialogue": item["dialogue"],
                       "results": results})
        if k % 500 == 0:
            log("B", k, len(items), "decodes", decodes)
    write_jsonl(out / "track-b.jsonl", rows_b)
    meta["decodes_b"] = decodes
    meta["seconds_total"] = round(time.time() - started, 1)
    write_json(out / "meta.json", meta)
    log("done", meta["seconds_total"], "s")


def savings(rows, index):
    keys = sum(r["results"][index]["keys"] for r in rows)
    base = sum(r["results"][index]["baseline_keys"] for r in rows)
    shows = sum(r["results"][index]["shows"] for r in rows)
    clicks = sum(r["results"][index]["clicks"] for r in rows)
    return {"savings": 1 - keys / base, "keys": keys, "baseline_keys": base, "shows": shows,
            "clicks": clicks, "click_rate": clicks / shows if shows else None}


def score(args):
    run_dir = Path(args.run)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    rows_a = P.read_jsonl(run_dir / "track-a.jsonl")
    hits3 = [r["hit_chars"] for r in rows_a if r["rank"] and r["rank"] <= 3]
    result = {"version": meta["version"], "split": meta["split"], "model": meta["model"],
              "track_a": {**P.rates(rows_a, "rank"),
                          "mean_hit_chars_top3": sum(hits3) / len(hits3) if hits3 else None}}
    rows_c = P.read_jsonl(run_dir / "track-c.jsonl")
    result["track_c"] = {mode: P.rates([r for r in rows_c if r["mode"] == mode], "rank") for mode, _, _ in MODES}

    rows_b = P.read_jsonl(run_dir / "track-b.jsonl")
    grid = meta["threshold_grid"]
    table = [{"threshold": t, **savings(rows_b, i)} for i, t in enumerate(grid)]
    # best savings; ties go to the higher threshold (fewer shows)
    best = max(range(len(grid)), key=lambda i: (table[i]["savings"], i))
    sources = sorted({r["source"] for r in rows_b})
    result["track_b"] = {"chosen": table[best],
                         "by_source": {s: savings([r for r in rows_b if r["source"] == s], best)["savings"]
                                       for s in sources},
                         "grid": table}
    result["track_d"] = P.coverage_curve([{"id": r["id"], "confidence": r["confidence"], "rank": r["rank"]}
                                          for r in rows_a])
    write_json(run_dir / "score.json", result)
    a, b = result["track_a"], result["track_b"]["chosen"]
    print(f"A  top1 {a['top1']['micro']:.4f}  top3 {a['top3']['micro']:.4f} (macro {a['top3']['macro']:.4f})"
          f"  top5 {a['top5']['micro']:.4f}  hit chars {a['mean_hit_chars_top3']:.2f}")
    for mode in result["track_c"]:
        c = result["track_c"][mode]
        print(f"C  {mode:14s} top1 {c['top1']['micro']:.4f}  top3 {c['top3']['micro']:.4f}"
              f"  top5 {c['top5']['micro']:.4f}")
    print(f"B  savings {b['savings']:.4f}  threshold {b['threshold']}  shows {b['shows']}"
          f"  clicks {b['clicks']}  click rate {b['click_rate']}")
    print("D ", {k: round(v["top1"], 4) for k, v in result["track_d"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default="vime")
    r.add_argument("--checkpoint", default=str(ROOT / "artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/final.pt"))
    r.add_argument("--tokenizer", default=str(ROOT / "artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model"))
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.add_argument("--limit", type=int, default=0, help="items per source, for smoke runs")
    r.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    args = parser.parse_args()
    run(args) if args.command == "run" else score(args)


if __name__ == "__main__":
    main()
