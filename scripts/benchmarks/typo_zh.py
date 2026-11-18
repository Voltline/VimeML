"""Keyboard-typo correction set ime-typo-zh-v1 (development only): build, score, report.

usage:
  python -X utf8 scripts/benchmarks/typo_zh.py build --out DIR/shard-I --shard I/N [--limit N]
  python -X utf8 scripts/benchmarks/typo_zh.py merge DIR/shard-0 ... DIR/shard-(N-1) [--out SET]
  python -X utf8 scripts/benchmarks/typo_zh.py score --checkpoint CKPT --out DIR [--rules lm task] [--limit N]
  python -X utf8 scripts/benchmarks/typo_zh.py report NAME=DIR:RULE [...] [--base NAME/calibrated] [--out FILE]

build reads the development split of ime-pinyin-zh-v1 (frozen-r1) and Vime's readings dictionary;
both stay out of git, and so does the built set. The pool is made once by vimeml-tiny-zh-v1.
"""
import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks import typo_zh as tz  # noqa: E402

V1 = ROOT / "artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/final.pt"
TOKENIZER = ROOT / "artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model"
SET = ROOT / "outputs/benchmarks/ime-typo-zh-v1/r1"
FROZEN = ROOT / "outputs/benchmarks/ime-pinyin-zh-v1/frozen-r1"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(SET))
    b.add_argument("--checkpoint", default=str(V1))
    b.add_argument("--tokenizer", default=str(TOKENIZER))
    b.add_argument("--readings", default=str(ROOT / "outputs/zh-readings/dict/vime_pinyin.dict.yaml"))
    b.add_argument("--frozen", default=str(FROZEN))
    b.add_argument("--typo-config", default=str(ROOT / "configs/data-zh-pinyin-typo.toml"))
    b.add_argument("--device", default="cuda")
    b.add_argument("--limit", type=int, default=0, help="sentences, for smoke runs")
    b.add_argument("--shard", default="0/1", help="I/N: sentences whose index mod N is I")
    b.add_argument("--beam", type=int, default=16, help="16 for the set; smaller only for smoke runs")
    m = sub.add_parser("merge")
    m.add_argument("shards", nargs="+")
    m.add_argument("--out", default=str(SET))
    m.add_argument("--frozen", default=str(FROZEN))
    s = sub.add_parser("score")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--set", default=str(SET))
    s.add_argument("--rules", nargs="+", choices=("lm", "task"), default=["lm", "task"])
    s.add_argument("--tokenizer", default=str(TOKENIZER))
    s.add_argument("--frozen", default=str(FROZEN))
    s.add_argument("--device", default="cuda")
    s.add_argument("--limit", type=int, default=0, help="items, for smoke runs")
    r = sub.add_parser("report")
    r.add_argument("runs", nargs="+", help="NAME=DIR:RULE")
    r.add_argument("--set", default=str(SET))
    r.add_argument("--base", help="NAME/fixed or NAME/calibrated")
    r.add_argument("--margin", type=float, default=3.0, help="fixed margin of the lm rule")
    r.add_argument("--out")
    args = parser.parse_args()
    if args.command == "build":
        with open(args.typo_config, "rb") as stream:
            typo = tomllib.load(stream)["typo"]
        typo = {k: typo[k] for k in ("two_share", "weights", "repeat_share")}
        manifest = tz.build(args.frozen, args.out, args.checkpoint, args.tokenizer, args.readings, typo,
                            args.device, args.limit, args.beam, shard=tuple(map(int, args.shard.split("/"))))
        print(json.dumps({k: manifest[k] for k in ("shard", "items", "pool", "seconds")}, indent=1))
    elif args.command == "merge":
        manifest = tz.merge(args.shards, args.out, args.frozen)
        print(json.dumps({k: manifest[k] for k in ("items", "coverage", "corrections_per_input", "build_seconds")},
                         indent=1))
    elif args.command == "score":
        print(tz.score(args.set, args.checkpoint, args.tokenizer, args.out, args.frozen, tuple(args.rules),
                       args.device, args.limit))
    else:
        runs = {}
        for spec in args.runs:
            name, rest = spec.split("=", 1)
            directory, rule = rest.rsplit(":", 1)
            runs[name] = (directory, rule)
        text = tz.report(args.set, runs, args.base, args.margin)
        print(text)
        if args.out:
            Path(args.out).write_text(text, "utf-8")


if __name__ == "__main__":
    main()
