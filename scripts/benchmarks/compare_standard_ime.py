"""Compare completed standard IME scores without rerunning model inference."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.comparison import paired, rows
from vimeml.benchmarks.standard_ime import read
from vimeml.training.data import write_json


def comparison(before, after):
    result = paired(before, after)
    counts = result.pop("paired")
    counts["before_wrong_after_correct"] = counts.pop("v1_wrong_v2_correct")
    counts["before_correct_after_wrong"] = counts.pop("v1_correct_v2_wrong")
    # Cases share authors/users; the unadjusted case-level p is exploratory.
    counts.pop("significant_at_0_05")
    result["before"] = result.pop("v1")
    result["after"] = result.pop("v2")
    result["paired"] = counts
    result["top1_delta_percentage_points"] = 100 * (
        result["after"]["top1_accuracy"] - result["before"]["top1_accuracy"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior comparisons; choose a new output.")
    before_report = read(args.before / "metrics.json")
    after_report = read(args.after / "metrics.json")
    # Historical regression caches predate explicit label-quality metadata.
    for report in (before_report, after_report):
        if report.get("split") == "regression":
            report.setdefault("label_quality", "legacy_historical_draft")
    for key in ("format", "split", "label_quality", "labels_formal_gold", "policy", "result_role"):
        if before_report[key] != after_report[key]:
            raise ValueError(f"Evaluation protocol differs: {key}")
    if before_report["format"] != "vimeml_standard_ime_metrics_v1":
        raise ValueError("Expected completed standard IME reports.")
    if before_report["status"] != "complete" or after_report["status"] != "complete":
        raise ValueError("Both evaluations must be complete.")
    if before_report["benchmark_manifest"] != after_report["benchmark_manifest"]:
        raise ValueError("Frozen benchmark versions differ.")
    if before_report["export_provenance"] != after_report["export_provenance"]:
        raise ValueError("Converter/dictionary exports differ.")
    if before_report["split"] == "blind":
        if before_report["evaluation_plan"] != after_report["evaluation_plan"]:
            raise ValueError("Blind comparisons must share the registered plan.")
    before = rows(args.before / "scores.jsonl")
    after = rows(args.after / "scores.jsonl")
    after_by_id = {row["id"]: row for row in after}
    for row in before:
        other = after_by_id.get(row["id"])
        if other is None or any(row[k] != other[k] for k in ("source", "provenance", "candidates")):
            raise ValueError("Frozen source, provenance or candidate contents differ.")
    sources = sorted({row["source"] for row in before})
    report = {
        "format": "vimeml_standard_ime_paired_v1", "status": "complete",
        "split": before_report["split"], "result_role": before_report["result_role"],
        "benchmark_release_id": before_report["benchmark_manifest"]["release_id"],
        "label_quality": before_report["label_quality"],
        "labels_formal_gold": before_report["labels_formal_gold"],
        "before_model": before_report["model"], "after_model": after_report["model"],
        "before_scores": str(args.before.resolve()), "after_scores": str(args.after.resolve()),
        "policy": before_report["policy"],
        "statistical_note": "Case-level exact McNemar p is exploratory: shared author/user groups and multiple comparisons are not adjusted; no equivalence claim.",
        "by_source": {source: comparison([r for r in before if r["source"] == source],
                                         [r for r in after if r["source"] == source]) for source in sources},
        "micro": comparison(before, after),
    }
    if set(sources) == {"wrime", "jmultiwoz"}:
        report["equal_source_macro_top1_delta_percentage_points"] = sum(
            result["top1_delta_percentage_points"] for result in report["by_source"].values()) / 2
    write_json(args.output, report)
    print({"split": report["split"], "paired": report["micro"]["paired"],
           "delta_pp": report["micro"]["top1_delta_percentage_points"]})


if __name__ == "__main__":
    main()
