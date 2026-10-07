"""Pair frozen V1/V2 case outcomes and compute exact McNemar statistics."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.data import write_json
from vimeml.benchmarks.comparison import paired, rows


def main():
    out = ROOT / "outputs/ime-eval/tiny-ja-v2.0-comparison"
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "artifacts/models/tiny-ja-v2.0-e16k-d320-l6"
    full = json.loads((source / "full-validation.json").read_text(encoding="utf-8"))
    v1_bpc = 3.4736559
    result = {
        "model_selection": "best.pt step 375000; minimum full BPC among last/subset-best and strongest dev result",
        "v1_bpc": v1_bpc,
        "v2_best_bpc": full["best"]["bpc"],
        "v2_last_bpc": full["last"]["bpc"],
        "relative_bpc_reduction": 1 - full["best"]["bpc"] / v1_bpc,
        "benchmarks": {},
        "test_split_used": False,
        "note": "Two small regression sets; larger frozen dev/blind sets are needed for robust selection.",
    }
    for name, v1_directory in [
        ("ajimee", "tiny-ja-v1-ajimee"),
        ("development", "tiny-ja-v1-dev-v2-scores"),
    ]:
        before = rows(ROOT / "outputs/ime-eval" / v1_directory / "scores.jsonl")
        best = rows(ROOT / "outputs/ime-eval" / f"tiny-ja-v2.0-best-{name}" / "scores.jsonl")
        last = rows(source / "epoch-evaluation/epoch-4" / name / "scores.jsonl")
        result["benchmarks"][name] = {"best": paired(before, best), "last": paired(before, last)}
    write_json(out / "comparison.json", result)
    print(
        json.dumps(
            {k: {m: x["paired"] for m, x in v.items()} for k, v in result["benchmarks"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
