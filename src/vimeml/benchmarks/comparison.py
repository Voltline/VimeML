"""Frozen-case alignment and exact paired IME statistics."""

import json
import math
from pathlib import Path

from vimeml.benchmarks.evaluate_ajimee import metrics


def rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def paired(before, after):
    left = {r["id"]: r for r in before}
    right = {r["id"]: r for r in after}
    if left.keys() != right.keys() or len(left) != len(before) or len(right) != len(after):
        raise ValueError("Expected identical unique frozen case IDs.")
    wins = losses = both_correct = both_wrong = 0
    changes = []
    for ident, a in left.items():
        b = right[ident]
        for field in ("answers", "left_context", "query"):
            if a[field] != b[field]:
                raise ValueError(f"Frozen case differs: {ident}/{field}")
        if a["orders"]["azookey"] != b["orders"]["azookey"]:
            raise ValueError("Candidate pool/order differs.")
        av = (
            a["orders"]["lm_context_sum"][0] in a["answers"]
            if a["orders"]["lm_context_sum"]
            else False
        )
        bv = (
            b["orders"]["lm_context_sum"][0] in b["answers"]
            if b["orders"]["lm_context_sum"]
            else False
        )
        wins += bv and not av
        losses += av and not bv
        both_correct += av and bv
        both_wrong += not av and not bv
        if av != bv:
            changes.append(
                {
                    "id": ident,
                    "change": "improved" if bv else "regressed",
                    "context": a["left_context"],
                    "reading": a["query"],
                    "answers": a["answers"],
                    "before": a["orders"]["lm_context_sum"][0],
                    "after": b["orders"]["lm_context_sum"][0],
                }
            )
    n = wins + losses
    p = min(1, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.0
    return {
        "cases": len(left),
        "v1": metrics(before, "lm_context_sum"),
        "v2": metrics(after, "lm_context_sum"),
        "paired": {
            "v1_wrong_v2_correct": wins,
            "v1_correct_v2_wrong": losses,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "net_correct": wins - losses,
            "exact_mcnemar_p_two_sided": p,
            "significant_at_0_05": p < 0.05,
        },
        "changed_cases": changes,
    }
