"""Merge completed preprocessing parts and perform global corpus dedup/splitting."""

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.corpus_parts import (PART_SCHEMA_VERSION, AUDITS, load_plan, create_database, aggregate_metrics)
from vimeml.data.corpus_export import export_corpus
from vimeml.data.build import attach_alias, canonical_url, file_sha, display_path, dump_json, dump_line


def validate_parts(config_path, parts):
    plan = load_plan(config_path)
    _, inventory, _, _, _, _, signature = plan
    loaded = []
    for directory in parts:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("status") != "complete" or manifest.get("stage") != "preprocessed_part"
                or manifest.get("part_schema_version") != PART_SCHEMA_VERSION):
            raise ValueError(f"Not a completed preprocessing part: {directory}")
        if manifest["pipeline_signature"] != signature:
            raise ValueError(f"Code/config/approval versions differ: {directory}")
        loaded.append((directory, manifest))
    if not loaded:
        raise ValueError("No parts supplied.")
    total = loaded[0][1]["partition"]["count"]
    if len(loaded) != total or any(m["partition"]["count"] != total for _, m in loaded):
        raise ValueError("All planned parts must be supplied exactly once.")
    if {m["partition"]["index"] for _, m in loaded} != set(range(total)):
        raise ValueError("Missing or duplicated part index.")
    for directory, manifest in loaded:
        index = manifest["partition"]["index"]
        expected = {p for number, (p, _, _) in enumerate(inventory) if number % total == index}
        if manifest["planned_inputs"] != [p for p, _, _ in inventory] or set(manifest["input_sha256"]) != expected or set(manifest["selected_documents"]) != expected:
            raise ValueError(f"Part input assignment differs from plan: {directory}")
        print(f"Validating part {index + 1}/{total}: {directory}", flush=True)
        for name in ("index.sqlite", "part-stats.json", *(f"{name}.jsonl" for name in AUDITS)):
            if file_sha(directory / name) != manifest["artifact_sha256"].get(name):
                raise ValueError(f"Copied part file is damaged or changed: {directory / name}")
    return plan, sorted(loaded, key=lambda item: item[1]["partition"]["index"])


def run(config_path, parts, output):
    # Raw parquet/TSV files are not needed on the merging machine.
    plan, parts = validate_parts(config_path, parts)
    config, _, policies, calibration_ids, policy_hashes, code_hashes, signature = plan
    selected_documents, input_hashes = {}, {}
    raw_characters = 0
    with ExitStack() as stack:
        db, annotation_count = create_database(output, policies.get("annotations"))
        stack.callback(db.close)
        audits = {name: stack.enter_context((output / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n")) for name in AUDITS}
        for directory, manifest in parts:
            print(f"Importing part {manifest['partition']['index']}: {directory}", flush=True)
            # Read-only ATTACH works for both Windows and POSIX file paths.
            db.execute("ATTACH DATABASE ? AS part", ((directory / "index.sqlite").resolve().as_uri() + "?mode=ro",))
            try:
                db.execute("CREATE TEMP TABLE new_documents AS SELECT doc_hash FROM part.contents WHERE doc_hash NOT IN (SELECT doc_hash FROM contents)")
                db.execute("CREATE UNIQUE INDEX temp.new_documents_hash ON new_documents(doc_hash)")
                db.execute("INSERT INTO contents(doc_hash) SELECT doc_hash FROM new_documents")
                db.execute("INSERT INTO parents SELECT doc_hash,doc_hash FROM new_documents")
                # An identical document is cleaned once globally, even if it
                # appeared on both workers. All of its origins are retained.
                db.execute("INSERT INTO units SELECT * FROM part.units WHERE doc_hash IN (SELECT doc_hash FROM new_documents)")
                db.execute("INSERT INTO document_metrics SELECT * FROM part.document_metrics WHERE doc_hash IN (SELECT doc_hash FROM new_documents)")
                db.execute("INSERT INTO origins SELECT * FROM part.origins")
                db.execute("UPDATE annotations SET matched=1 WHERE id IN (SELECT id FROM part.annotations WHERE matched=1)")
                for name in AUDITS:
                    with (directory / f"{name}.jsonl").open(encoding="utf-8") as stream:
                        for line in stream:
                            item = json.loads(line)
                            if db.execute("SELECT 1 FROM new_documents WHERE doc_hash=?", (item["doc_hash"],)).fetchone():
                                dump_line(audits[name], item)
                db.execute("DROP TABLE new_documents")
                db.commit()
            finally:
                db.rollback()
                db.execute("DETACH DATABASE part")
            selected_documents.update(manifest["selected_documents"])
            input_hashes.update(manifest["input_sha256"])
            raw_characters += json.loads((directory / "part-stats.json").read_text(encoding="utf-8"))["raw_characters"]
        origin_count = db.execute("SELECT COUNT(*) FROM origins").fetchone()[0]
        if origin_count != sum(selected_documents.values()):
            raise ValueError("Imported source record counts disagree with part manifests.")
        print(f"Grouping {origin_count} document origins globally", flush=True)
        for number, row in enumerate(db.execute("SELECT doc_hash,doc_id,info FROM origins ORDER BY source_file,row_index"), 1):
            attach_alias(db, row["doc_hash"], "id:" + row["doc_id"])
            url = canonical_url(json.loads(row["info"])["metadata"].get("url"))
            if url:
                attach_alias(db, row["doc_hash"], "url:" + url)
            if number % config["build"].get("commit_every", 5000) == 0:
                db.commit()
            if number % config["build"].get("progress_every", 5000) == 0:
                print(f"Grouped {number} document origins", flush=True)
        db.commit()
        metrics = aggregate_metrics(db)
        metrics["counts"]["document_origins"] = origin_count
        metrics["counts"]["raw_characters"] = raw_characters
        print("Deduplicating sentences, assigning splits and exporting corpus", flush=True)
        stats = export_corpus(db, output, config, calibration_ids, metrics["counts"], metrics["rule_reasons"],
                              metrics["fragment_flags"], metrics["effective_blocks"], annotation_count)
    if load_plan(config_path)[-1] != signature:
        raise ValueError("Code or policy changed during merge; output is incomplete.")
    dump_json(output / "manifest.json", {
        "status": "complete", "stage": "merged_corpus_staging", "ready_for_lm_training": False,
        "quality_mode": config["build"]["quality_mode"], "config": config,
        "config_sha256": file_sha(config_path), "pipeline_signature": signature, "python_version": sys.version,
        "input_sha256": input_hashes, "selected_documents": selected_documents,
        "policy_sha256": policy_hashes, "code_sha256": code_hashes,
        "fingerprint_normalization": "Code and policy: UTF-8 text with LF line endings, no BOM; input/artifact hashes: exact bytes.",
        "part_manifest_sha256": {display_path(p / "manifest.json"): file_sha(p / "manifest.json") for p, _ in parts},
        "grouping": "transitive groups of identical NFC documents, source IDs and canonical URLs",
        "sentence_dedup": "exact NFC text; calibration text forced to train, otherwise test > validation > train",
        "near_duplicate_detection": False, "span_offsets": "Unicode code points within cleaned blocks",
        "calibration_policy": "known calibration documents and their groups and sentences are forced into train",
        "annotation_policy": "Only approved snapshot-matched local labels apply; model replies alone are advisory.",
    })
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Corpus: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/corpus-sharded.toml")
    parser.add_argument("--parts", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), [p.resolve() for p in args.parts], args.output.resolve())


if __name__ == "__main__":
    main()
