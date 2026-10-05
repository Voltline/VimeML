"""Export explicitly selected audit decisions to the local approval ledger.

This never calls an API or automatically approves model recommendations.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.annotations import load_annotations


def approve(results_path, output, ids, action, reviewer, reason):
    raw = results_path.read_bytes()
    results = json.loads(raw.decode("utf-8-sig"))
    by_id = {item["id"]: item for item in results}
    if len(by_id) != len(results):
        raise ValueError("Audit contains duplicated case IDs.")
    if not ids or len(set(ids)) != len(ids) or not reason.strip():
        raise ValueError("Select unique case IDs and provide your approval reason.")
    records = []
    for case_id in ids:
        case = by_id[case_id]
        kind = {"boundary_candidate": "boundary", "fragment_candidate": "fragment"}.get(case["kind"])
        if kind is None or action not in ({"join", "separate"} if kind == "boundary" else {"keep", "drop", "review"}):
            raise ValueError(f"Action {action} is incompatible with {case_id}.")
        record = {
            "schema_version": 1, "id": f"{results_path.parent.parent.name}:{case_id}",
            "status": "approved", "approved_by": reviewer,
            "approved_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason_zh": reason, "kind": kind, "action": action,
            "doc_id": case["doc_id"], "doc_hash": case["doc_hash"],
            "evidence": {"case_id": case_id, "results_sha256": hashlib.sha256(raw).hexdigest(),
                         "model_assessment": case["assessment"], "model_reason_zh": case["reason_zh"]},
        }
        fields = ("left_block", "right_block", "left_text", "right_text") if kind == "boundary" else ("text", "cleaned_block_spans")
        record.update({field: case[field] for field in fields})
        if kind == "fragment":
            record["text_hash"] = hashlib.sha256(case["text"].encode("utf-8")).hexdigest()
        records.append(record)
    existing = output.read_text(encoding="utf-8-sig") if output.exists() else ""
    combined = existing.rstrip() + ("\n" if existing.strip() else "")
    combined += "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate the complete ledger before atomically replacing it.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, suffix=".jsonl", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(combined)
        with closing(sqlite3.connect(":memory:")) as db:
            total = load_annotations(db, temporary)
        temporary.replace(output)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    return {"added": len(records), "total": total, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True, help="Cases you reviewed in their original context.")
    parser.add_argument("--action", choices=("join", "separate", "keep", "drop", "review"), required=True)
    parser.add_argument("--approved-by", choices=("human", "assistant"), default="human")
    parser.add_argument("--reason", required=True, help="Your review rationale; model suggestions are evidence only.")
    args = parser.parse_args()
    print(json.dumps(approve(args.results.resolve(), args.output.resolve(), args.ids, args.action, args.approved_by, args.reason), ensure_ascii=False))


if __name__ == "__main__":
    main()
