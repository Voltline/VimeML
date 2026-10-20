"""Build a frozen, model-output-free Chinese human annotation packet."""
import hashlib
import json
from pathlib import Path
import random


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_packet(legacy_path: Path, new_path: Path, template: Path, output: Path, private: Path):
    legacy = [json.loads(line) for line in legacy_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    legacy = [row for row in legacy if row["suite"] == "upstream_sentences"]
    fresh = [json.loads(line) for line in new_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(legacy) != 60 or len(fresh) != 40:
        raise ValueError("Expected 60 legacy audit cases and 40 new diagnostics")
    combined = [("legacy_audit", row) for row in legacy] + [("new_diagnostic", row) for row in fresh]
    if len({(r.get("context", ""), r["input"]) for _, r in combined}) != 100:
        raise ValueError("Duplicate context/input pair")
    random.Random(2026101701).shuffle(combined)
    cases, mapping = [], []
    for index, (suite, row) in enumerate(combined, 1):
        ident = f"H{index:03d}"
        cases.append({"id": ident, "context": row.get("context", ""), "input": row["input"]})
        mapping.append({"blind_id": ident, "suite": suite, "source_id": row["id"],
                        "tags": row.get("tags", ""), "original_reference_only": row.get("gold"),
                        "authored_seed_not_human_gold": row.get("seed_text")})
    dataset_id = "zh-human-100-20261017-" + digest(cases)[:16]
    payload = {"schema_version": 1, "dataset_id": dataset_id, "cases": cases}
    serialized = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    page = template.read_text(encoding="utf-8")
    if page.count("__PAYLOAD__") != 1:
        raise ValueError("Expected exactly one template insertion point")
    files = {
        output / "index.html": page.replace("__PAYLOAD__", serialized),
        output / "blind-inputs.json": json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        private / "mapping-private.json": json.dumps({"dataset_id": dataset_id, "mapping": mapping}, ensure_ascii=False, indent=2) + "\n",
        private / "manifest.json": json.dumps({"dataset_id": dataset_id, "n": 100, "legacy_n": 60,
            "new_n": 40, "cases_sha256": digest(cases), "selection": "All 60 original sentence inputs plus 40 authored diagnostics, frozen before native labels. No model outputs supplied or used to select the new cases.",
            "source_files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (legacy_path, new_path)}}, ensure_ascii=False, indent=2) + "\n",
    }
    for path, content in files.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Frozen packet differs: {path}; choose a new directory")
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return payload
