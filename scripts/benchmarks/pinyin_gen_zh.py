"""Pinyin generation on ime-pinyin-zh-v1 development (Vime search, optional prior rescoring).

usage:
  python -X utf8 scripts/benchmarks/pinyin_gen_zh.py run --checkpoint CKPT --out DIR \\
      [--mode task|lm] [--beam 16] [--prior artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/final.pt | --no-prior] \\
      [--no-context] [--device cuda] [--limit N]
  python -X utf8 scripts/benchmarks/pinyin_gen_zh.py report NAME=DIR [NAME=DIR ...] [--base NAME] [--lam 1.6]

Checkpoints: vimeml-tiny-zh-v1, its pinyin post-training stages, and 12K zh-200m (tiny_gpt_zh200m).
The readings dictionary is Vime's (outputs/zh-readings/dict/vime_pinyin.dict.yaml; CC BY-SA sources,
not committed). Development only; the test split is never read.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks import pinyin_gen_zh as pg  # noqa: E402

V1 = ROOT / "artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/final.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--mode", choices=("task", "lm"), default="task")
    r.add_argument("--beam", type=int, default=16)
    r.add_argument("--topk", type=int, default=10)
    r.add_argument("--prior", default=str(V1))
    r.add_argument("--no-prior", action="store_true")
    r.add_argument("--no-context", action="store_true")
    r.add_argument("--tokenizer", default=str(ROOT / "artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model"))
    r.add_argument("--readings", default=str(ROOT / "outputs/zh-readings/dict/vime_pinyin.dict.yaml"))
    r.add_argument("--frozen", default=str(ROOT / "outputs/benchmarks/ime-pinyin-zh-v1/frozen-r1"))
    r.add_argument("--device", default="cuda")
    r.add_argument("--limit", type=int, default=0, help="records, for smoke runs")
    q = sub.add_parser("reprior", help="recompute the prior column of a finished run with another checkpoint")
    q.add_argument("--rows", required=True, help="directory of a finished run")
    q.add_argument("--prior", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--tokenizer", default=str(ROOT / "artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model"))
    q.add_argument("--frozen", default=str(ROOT / "outputs/benchmarks/ime-pinyin-zh-v1/frozen-r1"))
    q.add_argument("--device", default="cuda")
    s = sub.add_parser("report")
    s.add_argument("runs", nargs="+", help="NAME=DIR")
    s.add_argument("--base")
    s.add_argument("--lam", type=float, default=1.6)
    s.add_argument("--out", help="also write the Markdown here")
    args = parser.parse_args()
    if args.command == "run":
        info = pg.run(args.checkpoint, args.tokenizer, args.readings, args.frozen, args.out, args.mode, args.beam,
                      args.topk, None if args.no_prior else args.prior, args.device, args.limit,
                      not args.no_context)
        print(info)
    elif args.command == "reprior":
        print(pg.reprior(args.rows, args.prior, args.tokenizer, args.frozen, args.out, args.device))
    else:
        text = pg.report(dict(item.split("=", 1) for item in args.runs), args.lam, args.base)
        print(text)
        if args.out:
            Path(args.out).write_text(text, "utf-8")


if __name__ == "__main__":
    main()
