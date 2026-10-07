"""Classify frozen V1/V2 orders; preserve gold and candidate pools."""

import collections
import csv
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ime-eval/v2-error-audit-20261007"
LABELS = {
    "C": "correct",
    "H": "homophone_semantic",
    "T": "technical_term",
    "B": "compound_segmentation",
    "I": "inflection_or_structure",
    "P": "proper_name",
    "S": "surface_reference_gap_pending",
    "U": "context_underspecified",
}


def read_scores(relative):
    return [
        json.loads(line)
        for line in (ROOT / "outputs/ime-eval" / relative / "scores.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def length_band(query):
    n = len(query)
    return "<=16" if n <= 16 else "17-32" if n <= 32 else "33-64" if n <= 64 else ">64"


def main():
    decisions = list(
        csv.DictReader(
            (ROOT / "scripts/benchmarks/v2-error-decisions.tsv").open(encoding="utf-8"),
            delimiter="\t",
        )
    )
    assert len({r["audit_id"] for r in decisions}) == len(decisions)
    decisions = {r["audit_id"]: r for r in decisions}
    all_rows = []
    summary = {}
    error_ids = set()
    sources = [
        ("ajimee", "tiny-ja-v1-ajimee", "tiny-ja-v2.0-best-ajimee"),
        ("development_reviewed", "dev-label-audit-20261007/v1", "dev-label-audit-20261007/v2"),
    ]
    for dataset, left, right in sources:
        before = read_scores(left)
        after = {r["id"]: r for r in read_scores(right)}
        stats = {
            model: {
                "total": len(before),
                "top1": 0,
                "mean_top1_secondary": 0,
                "stage": collections.Counter(),
                "error_category": collections.Counter(),
                "covered_error_category": collections.Counter(),
                "reading_length": {},
                "context": {},
                "candidate_count": {},
            }
            for model in ("v1", "v2")
        }
        covered_total = 0
        for i, a in enumerate(before, 1):
            b = after[a["id"]]
            gold = set(a["answers"])
            assert a["answers"] == b["answers"] and a["orders"]["azookey"] == b["orders"]["azookey"]
            covered = bool(gold.intersection(a["orders"]["azookey"]))
            covered_total += covered
            aid = ("A" if dataset == "ajimee" else "D") + f"{i:03}"
            note = decisions.get(aid)
            if any(row["orders"]["lm_context_sum"][0] not in gold for row in (a, b)):
                error_ids.add(aid)
            record = {
                "audit_id": aid,
                "dataset": dataset,
                "id": a["id"],
                "context": a["left_context"],
                "query": a["query"],
                "answers": a["answers"],
                "candidate_coverage_exact": covered,
                "candidate_count": len(a["orders"]["azookey"]),
                "reading_length_band": length_band(a["query"]),
                "note_zh": note["note_zh"] if note else "",
                "models": {},
            }
            for name, row in (("v1", a), ("v2", b)):
                top = row["orders"]["lm_context_sum"][0]
                correct = top in gold
                category = LABELS[note[name + "_category"]] if note else "correct"
                assert (category == "correct") == correct, (aid, name, top)
                stage = (
                    "correct"
                    if correct
                    else "candidate_missing_exact"
                    if not covered
                    else "covered_top1_miss"
                )
                model = stats[name]
                model["top1"] += correct
                model["mean_top1_secondary"] += (
                    row["orders"]["lm_context_mean_secondary"][0] in gold
                )
                model["stage"][stage] += 1
                if not correct:
                    model["error_category"][category] += 1
                    if covered:
                        model["covered_error_category"][category] += 1
                for dimension, key in [
                    ("reading_length", length_band(row["query"])),
                    ("context", "with_context" if row["left_context"] else "without_context"),
                    (
                        "candidate_count",
                        "1-5"
                        if len(row["orders"]["azookey"]) <= 5
                        else "6-10"
                        if len(row["orders"]["azookey"]) <= 10
                        else ">10",
                    ),
                ]:
                    group = model[dimension].setdefault(key, {"total": 0, "top1": 0, "covered": 0})
                    group["total"] += 1
                    group["top1"] += correct
                    group["covered"] += covered
                record["models"][name] = {
                    "top1": top,
                    "correct_exact": correct,
                    "stage": stage,
                    "category": category,
                    "annotation_status": "diagnostic_pending_adjudication"
                    if category in ("surface_reference_gap_pending", "context_underspecified")
                    else "reviewed_diagnostic",
                }
            all_rows.append(record)
        summary[dataset] = {"cases": len(before), "covered_exact": covered_total, "models": stats}
    assert set(decisions) == error_ids, "All error cases must be reviewed exactly once."
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "case-classification.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "format": "v2_error_audit_v1",
        "reviewed_error_union_cases": len(decisions),
        "classified_total_cases": len(all_rows),
        "summary": summary,
        "label_changes_in_this_analysis": False,
        "inference_rerun": False,
        "limitations": [
            "AI diagnosis; no external native adjudication",
            "Candidate missing means exact frozen references absent; some are possible label gaps",
            "Pending surface/ambiguity annotations do not alter primary scores",
            "Length/context strata are descriptive, not causal",
        ],
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "reviewed_error_union_cases": len(decisions),
                "classified_total_cases": len(all_rows),
                "top1": {
                    dataset: {name: s["top1"] for name, s in data["models"].items()}
                    for dataset, data in summary.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
