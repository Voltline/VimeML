"""Compare frozen development orders without opening the blind LM split."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.comparison import paired, rows
from vimeml.benchmarks.evaluate_ajimee import metrics
from vimeml.training.data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1", type=Path, default=ROOT / "outputs/ime-eval/expanded-v21-dev-draft-v1"
    )
    parser.add_argument(
        "--v2", type=Path, default=ROOT / "outputs/ime-eval/expanded-v21-dev-draft-v2"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/ime-eval/expanded-v21-dev-draft-comparison"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve earlier comparisons; choose a fresh output.")
    reports = {
        name: json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        for name, path in [("v1", args.v1), ("v2", args.v2)]
    }
    for report in reports.values():
        if (
            report.get("result_role") != "provisional_expanded_development"
            or report["benchmark_manifest"]["split"] != "development"
        ):
            raise ValueError("Only expanded development diagnostic scores allowed here.")
    before, after = rows(args.v1 / "scores.jsonl"), rows(args.v2 / "scores.jsonl")
    if len(before) != 2000 or len(after) != 2000:
        raise ValueError("Expected all 2000 development cases.")
    result = paired(before, after)
    result.update(
        {
            "format": "expanded_ime_development_comparison_v1",
            "labels_formal_gold": False,
            "blind_lm_scored": False,
            "result_role": "provisional_label_diagnostic",
            "candidate_label_version": "ime-expanded-v21-candidates-v1",
            "models": {name: report["model"] for name, report in reports.items()},
            "methods": {name: report["metrics"] for name, report in reports.items()},
            "elapsed_seconds": {
                name: report["elapsed_seconds"] for name, report in reports.items()
            },
            "strata": {},
            "policy": "Same actual candidate pool, full denominator, fixed suffix logP sum, FP32 CUDA, no EOS/truncation.",
            "limitations": [
                "Provisional references; spelling gaps and corpus noise still need audit",
                "Development diagnostics, not final blind inference or production selection",
                "Near-duplicate overlap is not fully audited",
            ],
        }
    )
    dimensions = {
        "reading_length": lambda r: "<=16"
        if len(r["query"]) <= 16
        else "17-32"
        if len(r["query"]) <= 32
        else ">32",
        "candidate_count": lambda r: "1-5"
        if len(r["candidates"]) <= 5
        else "6-10"
        if len(r["candidates"]) <= 10
        else "11-20",
    }
    for dim, group in dimensions.items():
        result["strata"][dim] = {}
        for key in sorted({group(r) for r in before}):
            a = [r for r in before if group(r) == key]
            b = [r for r in after if group(r) == key]
            result["strata"][dim][key] = {
                "azookey": metrics(a, "azookey"),
                "v1": metrics(a, "lm_context_sum"),
                "v2": metrics(b, "lm_context_sum"),
            }
    records = []
    for a, b in zip(before, after):
        assert a["id"] == b["id"]
        records.append(
            {
                "id": a["id"],
                "query": a["query"],
                "context": a["left_context"],
                "answers": a["answers"],
                "covered_exact": bool(set(a["answers"]).intersection(a["orders"]["azookey"])),
                "azookey_top1": a["orders"]["azookey"][0] if a["orders"]["azookey"] else "",
                "v1_top1": a["orders"]["lm_context_sum"][0]
                if a["orders"]["lm_context_sum"]
                else "",
                "v2_top1": b["orders"]["lm_context_sum"][0]
                if b["orders"]["lm_context_sum"]
                else "",
            }
        )
    args.output.mkdir(parents=True)
    write_json(args.output / "comparison.json", result)
    write_json(args.output / "case-outcomes.json", records)
    print(
        json.dumps(
            {
                "v1": result["v1"],
                "v2": result["v2"],
                "paired": result["paired"],
                "context": {
                    n: {
                        g: r["metrics"][g]["lm_context_sum"]["top1_correct"]
                        for g in ("with_context", "without_context")
                    }
                    for n, r in reports.items()
                },
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
