"""Final-particle prediction on ime-pinyin-zh-colloq-v1 (a secondary metric of training-zh-colloq-plan.md).

For every record whose clause ends in a sentence-final particle: the cursor stands just before the particle,
the context is the text before it in the same message (infield, as the keyboard sees it; no earlier turns),
and the true continuation runs from the particle to the end of its Han run (usually the particle alone).
Decoder: vimeml.benchmarks.predict_zh as in the prediction benchmark, except that single-character candidates
are allowed (min_chars = 1). A candidate hits when the continuation starts with it.

  python -X utf8 scripts/zh/benchmarks/predict_colloq_zh.py run --checkpoint CKPT --out OUT [--split development]
  python -X utf8 scripts/zh/benchmarks/predict_colloq_zh.py score OUT [OUT ...]
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/zh/benchmarks"))

from vimeml.benchmarks import predict_zh as P  # noqa: E402
from vimeml.training.data import file_sha, write_json  # noqa: E402
from contexts_pinyin_ime_zh import infield  # noqa: E402

FROZEN = ROOT / "outputs/benchmarks/ime-pinyin-zh-colloq-v1/frozen-r1"
MIN_CHARS = 1
CHUNK = 2048


def items(frozen, split):
    out = []
    for line in open(Path(frozen) / f"{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if not r["final_particle"]:
            continue
        message, end = r["message"], r["span"][1]
        position = end - 1
        stop = position
        while stop < len(message) and P.HAN.fullmatch(message[stop]):
            stop += 1
        out.append({"id": r["id"], "source": r["source"], "group": r["group"],
                    "prefix": infield(message[:position]), "continuation": message[position:stop]})
    return out


def run(args):
    from predict_ime_zh import load_model
    P.MIN_CHARS = MIN_CHARS
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lm, model_meta = load_model(argparse.Namespace(model="vime", checkpoint=args.checkpoint,
                                                   tokenizer=args.tokenizer, device=args.device))
    rows, todo = [], items(args.frozen, args.split)
    started = time.time()
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        decoded = P.predict_many(lm, [([], item["prefix"], None) for item in chunk])
        for item, candidates in zip(chunk, decoded):
            rank, chars = P.rank_of_hit(candidates, item["continuation"])
            rows.append({**item, "candidates": candidates, "rank": rank, "hit_chars": chars})
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out / "meta.json", {
        "frozen": str(args.frozen), "split": args.split,
        "frozen_sha256": file_sha(Path(args.frozen) / f"{args.split}.jsonl"),
        "model": {**model_meta, "device": args.device},
        "decoder": {"beam": P.BEAM, "min_chars": MIN_CHARS, "max_chars": P.MAX_CHARS, "top": P.TOP},
        "items": len(rows), "seconds": round(time.time() - started, 1)})


def score(args):
    for run_dir in args.runs:
        rows = [json.loads(l) for l in open(Path(run_dir) / "rows.jsonl", encoding="utf-8")]
        by = collections.defaultdict(list)
        for r in rows:
            by["all"].append(r)
            by[r["source"]].append(r)
        line = []
        for name, group in by.items():
            top1 = sum(r["rank"] == 1 for r in group) / len(group)
            top3 = sum(r["rank"] is not None and r["rank"] <= 3 for r in group) / len(group)
            line.append(f"{name} n={len(group)} top1={top1:.3f} top3={top3:.3f}")
        print(run_dir, " | ".join(line))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("run")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--frozen", default=str(FROZEN))
    r.add_argument("--split", default="development", choices=("development",))
    r.add_argument("--tokenizer", default=str(ROOT / "artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model"))
    r.add_argument("--device", default="cuda")
    s = commands.add_parser("score")
    s.add_argument("runs", nargs="+")
    args = parser.parse_args()
    {"run": run, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
