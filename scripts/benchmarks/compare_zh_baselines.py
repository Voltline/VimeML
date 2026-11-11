"""Score a Rime configuration or an external LM on the Standard zh benchmark, for vimeml-tiny-zh-v1.

usage:
  python -I -X utf8 scripts/benchmarks/compare_zh_baselines.py engine {ice,ice-gram,luna} OUT --track full [--split blind --blind-plan PLAN]
  python -I -X utf8 scripts/benchmarks/compare_zh_baselines.py lm {qwen2.5-0.5b,qwen3-0.6b-base} OUT --track full [--split ...] [--device cuda]
  python -I -X utf8 scripts/benchmarks/score_standard_ime_zh.py compare A/result B/result OUT.json

Engines need `rime_candidates.py deploy --variant V` once. See docs/zh-CN/benchmarks/baselines-zh-v1-plan.md.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.zh_baselines import ENGINES, EXTERNAL, engine, lm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, choices in (("engine", ENGINES), ("lm", sorted(EXTERNAL))):
        p = commands.add_parser(name)
        p.add_argument("system", choices=choices)
        p.add_argument("out")
        p.add_argument("--track", choices=("full", "abbrev"), required=True)
        p.add_argument("--split", choices=("development", "blind"), default="development")
        p.add_argument("--blind-plan")
        if name == "lm":
            p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "engine":
        summary = engine(ROOT, args.system, args.split, args.track, args.out, args.blind_plan)
    else:
        summary = lm(ROOT, args.system, args.split, args.track, args.out, args.blind_plan, args.device)
    a = summary["all"]
    print(f'{summary["system"]} {args.track}/{args.split}: top1 {a["top1"]}/{a["n"]} ({a["top1_rate"]:.3f}), '
          f'top3 {a["top3_rate"]:.3f}, fallbacks {summary["fallbacks"]}')


if __name__ == "__main__":
    main()
