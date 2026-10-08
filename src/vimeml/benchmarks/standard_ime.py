"""Versioned IME benchmark protocol; labels and blind access are explicit."""
import json
from pathlib import Path

from vimeml.benchmarks.expanded_ime import FORMAT as LEGACY_FORMAT, load_expanded_export

FORMAT = "vimeml_standard_ime_candidates_v1"


def reviewed_and_frozen(manifest):
    """AI expert references are usable, with their quality class kept explicit."""
    return bool(manifest.get("labels_formal_gold") or (
        manifest.get("labels_frozen") and manifest.get("review_completed") and
        manifest.get("label_quality") in {"ai_expert_reviewed", "human_native_reviewed"}))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_standard_export(directory):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    if manifest.get("format") != FORMAT or manifest.get("status") != "complete":
        raise ValueError("Expected a completed standard IME candidate import.")
    # Reuse the existing real-candidate identity/version/rank checks.
    adapted = {**manifest, "format": LEGACY_FORMAT}
    _, provenance, rows = load_expanded_export(directory, adapted)
    originals = read(directory / "evaluation_items.json")
    by_id = {f"ajimee:{r['index']}": r for r in originals}
    for row in rows:
        row["source"] = by_id[row["id"]]["provenance"]["source"]
        row["provenance"] = by_id[row["id"]]["provenance"]
    return manifest, provenance, rows


def allow_evaluation(manifest, *, allow_draft=False, plan=None, checkpoint=None):
    role = manifest["split"]
    if role not in {"development", "blind", "regression"}:
        raise ValueError("Unknown benchmark role.")
    if role == "regression":
        return  # Historical draft references remain a separately named diagnostic.
    if not reviewed_and_frozen(manifest) and not (role == "development" and allow_draft):
        raise ValueError("External labels need review and freezing; only development permits explicit draft diagnostics.")
    if role == "blind":
        if plan is None or checkpoint is None:
            raise ValueError("Blind evaluation requires a precommitted model plan and checkpoint.")
        if plan.get("benchmark_release_id") != manifest["release_id"]:
            raise ValueError("Blind plan belongs to a different benchmark release.")
        identity = str(Path(checkpoint).resolve())
        matching = [m for m in plan["models"] if m["checkpoint"] == identity]
        if (len(matching) != 1 or Path(identity).stat().st_size != matching[0]["bytes"] or
                Path(identity).stat().st_mtime_ns != matching[0]["mtime_ns"]):
            raise ValueError("Checkpoint is not one of the precommitted model snapshots.")


def source_metrics(rows, metric_function):
    groups = {source: [r for r in rows if r["source"] == source] for source in sorted({r["source"] for r in rows})}
    metrics = {source: metric_function(items, "lm_context_sum") for source, items in groups.items()}
    result = {"by_source": metrics}
    if all(source in metrics for source in ("wrime", "jmultiwoz")):
        independent = [r for r in rows if r["source"] in ("wrime", "jmultiwoz")]
        result["independent_micro"] = metric_function(independent, "lm_context_sum")
        result["independent_equal_source_macro_top1"] = sum(metrics[s]["top1_accuracy"] for s in ("wrime", "jmultiwoz")) / 2
        if len(independent) != len(rows):
            result["all_source_micro_secondary"] = metric_function(rows, "lm_context_sum")
    else:
        result["role"] = "legacy_regression_diagnostic"
    return result
