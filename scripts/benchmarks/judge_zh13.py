"""Judge the eight vimeml-tiny-zh-v1 arms by the preregistered rules (development only).

RUNS/<arm>/full.jsonl and full.meta.json come from score_zh13_lm.py on each arm's final.pt;
abbrev.jsonl/abbrev.meta.json are optional and only described. Writes OUT/judgement.json,
OUT/report.md and the per-arm score_standard_ime_zh.py outputs under OUT/scores/. OUT must not exist.

usage:
  python -I -X utf8 scripts/benchmarks/judge_zh13.py RUNS OUT [--frozen DIR] [--models DIR] [--smoke]

--models DIR reads each arm's training directory from DIR/<arm> instead of the config's output_dir.
--smoke accepts training status smoke_complete and marks the report as not a result.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.zh13_judge import judge  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs")
    parser.add_argument("out")
    parser.add_argument("--frozen", default=str(ROOT / "outputs/benchmarks/ime-standard-zh-v1/frozen-r1"))
    parser.add_argument("--models")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = judge(args.runs, args.frozen, args.out, models=args.models, smoke=args.smoke, root=ROOT)
    n = result["noise_band"]["top1"]
    print(f"seed range {n['range']} (Top-1 {n['values']}), Rime {result['rime_baseline_top1']}")
    for c in result["contrasts"]:
        print(f"{c['name']:<12} {c['a']} {c['top1_a']} -> {c['b']} {c['top1_b']} ({c['difference']:+d}), "
              f"McNemar p {c['mcnemar_p']:.3g}, Holm p {c['holm_p']:.3g}, effective {c['effective']}")
    d = result["decision"]
    print(f"data: {d['data']['choice']} ({d['data']['basis']})")
    print(f"vocabulary: {d['vocabulary']['choice']} ({d['vocabulary']['basis']})")
    print(f"wrote {args.out}/judgement.json and report.md")


if __name__ == "__main__":
    main()
