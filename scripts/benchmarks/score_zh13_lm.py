"""Write LM rerank scores of a Vime 1.3 zh checkpoint on Standard zh development.

The output JSONL is the --scores input of score_standard_ime_zh.py; a .meta.json
next to it records the checkpoint, tokenizer and scores SHA256. Development only.

usage:
  python -X utf8 scripts/benchmarks/score_zh13_lm.py CHECKPOINT TOKENIZER_MODEL OUT.jsonl \\
      [--frozen outputs/benchmarks/ime-standard-zh-v1/frozen-r1] [--track full|abbrev] [--device cuda]
  python -I -X utf8 scripts/benchmarks/score_standard_ime_zh.py score FROZEN RESULT_DIR --track full --scores OUT.jsonl
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.zh13_lm_scores import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--frozen", type=Path, default=ROOT / "outputs/benchmarks/ime-standard-zh-v1/frozen-r1")
    parser.add_argument("--track", choices=("full", "abbrev"), default="full")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = run(args.checkpoint, args.tokenizer, args.frozen, args.out, args.track, args.device)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
