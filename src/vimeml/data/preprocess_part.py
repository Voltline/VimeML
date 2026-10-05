"""Preprocess assigned source files; final deduplication and splits happen at merge."""

import argparse
import json
import sys
import unicodedata
from collections import Counter
from contextlib import ExitStack, closing
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.corpus_parts import (AUDITS, PART_SCHEMA_VERSION, load_plan, parse_part,
                                    create_database, process_content, aggregate_metrics)
from vimeml.data.build import sha, file_sha, display_path, dump_json
from vimeml.data.readers import iter_fineweb, iter_tatoeba


def run(config_path, part, output=None, list_only=False):
    index, total = parse_part(part)
    config, inventory, policies, _, policy_hashes, code_hashes, signature = load_plan(config_path)
    selected = [item for number, item in enumerate(inventory) if number % total == index]
    if not selected:
        raise ValueError("This part has no inputs; use fewer parts.")
    if list_only:
        print(json.dumps({"part": part, "files": [p for p, _, _ in selected]}, ensure_ascii=False, indent=2))
        return
    paths = [ROOT / p for p, _, _ in selected]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing assigned input: {path}")
    snapshots = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    settings = config["build"]
    output = (ROOT / (output or f"{settings['output_dir']}-part-{index:03d}-of-{total:03d}")).resolve()
    selected_documents = Counter({p: 0 for p, _, _ in selected})
    raw_characters = 0
    with ExitStack() as stack:
        db, annotation_count = create_database(output, policies.get("annotations"))
        stack.callback(db.close)
        audits = {name: stack.enter_context((output / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n")) for name in AUDITS}
        for relative, source, limit in selected:
            path = ROOT / relative
            reader = iter_fineweb(path, settings["batch_size"]) if source == "fineweb" else iter_tatoeba(path)
            with closing(reader):
                for document in islice(reader, limit) if limit else reader:
                    selected_documents[relative] += 1
                    raw_characters += len(document.text)
                    normalized = unicodedata.normalize("NFC", document.text.replace("\r\n", "\n").replace("\r", "\n")).lstrip("\ufeff")
                    doc_hash = sha(normalized)
                    first = db.execute("INSERT OR IGNORE INTO contents(doc_hash) VALUES (?)", (doc_hash,)).rowcount == 1
                    identity = {"source": source, "doc_id": document.doc_id, "source_id": document.source_id,
                                "source_file": document.source_file, "row_index": document.row_index,
                                "doc_hash": doc_hash, "metadata": document.metadata}
                    db.execute("INSERT INTO origins VALUES (?,?,?,?,?,?)", (doc_hash, document.doc_id, source,
                               document.source_file, document.row_index, json.dumps(identity, ensure_ascii=False, allow_nan=False)))
                    if first:
                        metrics = process_content(db, document, normalized, doc_hash, settings, audits)
                        db.execute("INSERT INTO document_metrics VALUES (?,?)", (doc_hash, json.dumps(metrics, ensure_ascii=False)))
                    count = sum(selected_documents.values())
                    if count % settings.get("commit_every", 5000) == 0:
                        db.commit()
                    if count % settings.get("progress_every", 5000) == 0:
                        print(f"Part {part}: read {count} document origins", flush=True)
        db.commit()
        unmatched_here = [row[0] for row in db.execute("SELECT a.id FROM annotations a JOIN contents c ON c.doc_hash=a.doc_hash WHERE a.matched=0")]
        if config.get("review", {}).get("require_all_annotations", False) and unmatched_here:
            raise ValueError(f"Annotations in assigned documents did not match: {unmatched_here}")
        metrics = aggregate_metrics(db)
        stats = {**metrics, "document_origins": sum(selected_documents.values()), "raw_characters": raw_characters,
                 "annotation_records": annotation_count,
                 "matched_annotations": db.execute("SELECT COUNT(*) FROM annotations WHERE matched=1").fetchone()[0]}
        dump_json(output / "part-stats.json", stats)
    for path, before in snapshots.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != before:
            raise ValueError(f"Input changed during processing: {path}")
    if load_plan(config_path)[-1] != signature:
        raise ValueError("Code or policy changed during processing; part is incomplete.")
    print(f"Part {part}: hashing completed files", flush=True)
    artifact_hashes = {name: file_sha(output / name) for name in ("index.sqlite", "part-stats.json", *(f"{name}.jsonl" for name in AUDITS))}
    dump_json(output / "manifest.json", {
        "status": "complete", "stage": "preprocessed_part", "part_schema_version": PART_SCHEMA_VERSION,
        "partition": {"index": index, "count": total}, "pipeline_signature": signature,
        "config": config, "config_sha256": file_sha(config_path), "python_version": sys.version,
        "planned_inputs": [p for p, _, _ in inventory], "selected_documents": dict(selected_documents),
        "input_sha256": {display_path(p): file_sha(p) for p in paths},
        "policy_sha256": policy_hashes, "code_sha256": code_hashes, "artifact_sha256": artifact_hashes,
        "fingerprint_normalization": "Code and policy: UTF-8 text with LF line endings, no BOM; input/artifact hashes: exact bytes.",
        "ready_for_lm_training": False,
    })
    print(json.dumps({"part": part, "document_origins": stats["document_origins"], "output": str(output)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/corpus-sharded.toml")
    parser.add_argument("--part", required=True, help="Zero-based INDEX/COUNT; 0/2 and 1/2 for two machines.")
    parser.add_argument("--output")
    parser.add_argument("--list-only", action="store_true", help="Show assigned files; no raw data needed.")
    args = parser.parse_args()
    run(args.config.resolve(), args.part, args.output, args.list_only)


if __name__ == "__main__":
    main()
