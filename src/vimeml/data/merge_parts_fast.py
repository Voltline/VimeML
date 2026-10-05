"""Merge existing cleaning parts using global document groups and parallel hash buckets.

Only completed worker outputs are read. No raw inputs or network calls are needed.
index.sqlite is a document/review index; sentence storage lives in the exported files.
"""

import argparse
import json
import math
import os
import shutil
import sqlite3
import sys
import time
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack, closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.annotations import load_annotations
from vimeml.data.build import SPLITS, canonical_url, display_path, dump_json, dump_line, file_sha, sha
from vimeml.data.corpus_parts import AUDITS, load_plan, text_sha
from vimeml.data.merge_parts import validate_parts


def log(message):
    print(f"[fast-merge {time.strftime('%H:%M:%S')}] {message}", flush=True)


def readonly(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def tune(db, cache_mb):
    # Budgets are per process, not a hard upper bound on Python/Arrow memory.
    db.execute(f"PRAGMA cache_size=-{cache_mb * 1024}")
    db.execute("PRAGMA temp_store=FILE")


def parallel(function, jobs, workers, stage):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(function, job) for job in jobs}
        total = len(pending)
        while pending:
            finished, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in finished:
                results.append(future.result())
            log(f"{stage}: {len(results)}/{total} completed, {len(pending)} pending")
    return results


def root_of(parents, node):
    root = node
    while parents[root] != root:
        root = parents[root]
    while parents[node] != node:
        following = parents[node]
        parents[node] = root
        node = following
    return root


def make_metadata(output, work, parts, config, policies, calibration_ids, cache_mb):
    """Keep wide sentence data out of the global database entirely."""
    db = sqlite3.connect(output / "index.sqlite", uri=True)
    db.row_factory = sqlite3.Row
    tune(db, cache_mb)
    db.executescript("""
        CREATE TABLE contents (
            doc_hash TEXT PRIMARY KEY, group_id TEXT, split_rank INTEGER,
            owner INTEGER NOT NULL, forced INTEGER NOT NULL DEFAULT 0,
            origin_priority INTEGER, primary_origin INTEGER, source_mask INTEGER
        ) WITHOUT ROWID;
        CREATE TABLE origins (
            doc_hash TEXT NOT NULL, doc_id TEXT NOT NULL, source TEXT NOT NULL,
            source_file TEXT NOT NULL, row_index INTEGER NOT NULL, info TEXT NOT NULL,
            PRIMARY KEY(source_file,row_index)
        );
    """)
    annotation_count = load_annotations(db, policies.get("annotations"))
    excluded = {}
    try:
        for directory, manifest in parts:
            index = manifest["partition"]["index"]
            log(f"Import document metadata: part {index}")
            db.execute("ATTACH DATABASE ? AS part", ((directory / "index.sqlite").resolve().as_uri() + "?mode=ro",))
            try:
                excluded[index] = [r[0] for r in db.execute(
                    "SELECT p.doc_hash FROM part.contents p JOIN contents c ON c.doc_hash=p.doc_hash")]
                db.execute("INSERT OR IGNORE INTO contents(doc_hash,owner) SELECT doc_hash,? FROM part.contents", (index,))
                db.execute("INSERT INTO origins SELECT * FROM part.origins")
                db.execute("UPDATE annotations SET matched=1 WHERE id IN (SELECT id FROM part.annotations WHERE matched=1)")
                db.commit()
            finally:
                db.rollback()
                db.execute("DETACH DATABASE part")
        log("Build document lookup indexes")
        db.executescript("CREATE INDEX origins_content ON origins(doc_hash); CREATE INDEX origins_identity ON origins(doc_id);")
        hashes = [r[0] for r in db.execute("SELECT doc_hash FROM contents ORDER BY doc_hash")]
        nodes = {key: i for i, key in enumerate(hashes)}
        parents = array("I", range(len(hashes)))
        aliases = {}
        inspected = {nodes[r[0]] for r in db.execute("SELECT DISTINCT doc_hash FROM annotations") if r[0] in nodes}
        for number, row in enumerate(db.execute("SELECT doc_hash,doc_id,info FROM origins"), 1):
            node = nodes[row["doc_hash"]]
            if row["doc_id"] in calibration_ids:
                inspected.add(node)
            url = canonical_url(json.loads(row["info"])["metadata"].get("url"))
            for alias in ("id:" + row["doc_id"], "url:" + url if url else None):
                if alias is None:
                    continue
                previous = aliases.get(alias)
                if previous is None:
                    aliases[alias] = node
                else:
                    left, right = root_of(parents, node), root_of(parents, previous)
                    # Nodes follow lexicographic hash order, exactly as the old union-find.
                    if left != right:
                        parents[max(left, right)] = min(left, right)
            if number % 100000 == 0:
                log(f"Group documents: {number:,} origins")
        del aliases
        forced_roots = {root_of(parents, node) for node in inspected}
        settings = config["build"]
        updates = []
        for node, key in enumerate(hashes):
            root = root_of(parents, node)
            value = int(sha(f"{settings['seed']}:{hashes[root]}")[:16], 16) / 2**64
            forced = root in forced_roots
            rank = (0 if forced or value < config["split"]["train"]
                    else 1 if value < config["split"]["train"] + config["split"]["validation"] else 2)
            updates.append((hashes[root], rank, int(forced), key))
            if len(updates) == 10000:
                db.executemany("UPDATE contents SET group_id=?,split_rank=?,forced=? WHERE doc_hash=?", updates)
                updates.clear()
        db.executemany("UPDATE contents SET group_id=?,split_rank=?,forced=? WHERE doc_hash=?", updates)
        db.commit()
        # Compute one preferred origin and source mask per content before sentence joins.
        # Fixed-size arrays avoid a second dictionary of 2 million wide origin records.
        primary = array("Q", [0]) * len(hashes)
        priority = array("Q", [0]) * len(hashes)
        masks = bytearray(len(hashes))
        log("Choose primary document origins")
        for order, row in enumerate(db.execute("""
            SELECT rowid AS origin_id,doc_hash,source FROM origins
            ORDER BY CASE source WHEN 'tatoeba' THEN 0 ELSE 1 END,doc_id,source_file,row_index
        """), 1):
            node = nodes[row["doc_hash"]]
            if not primary[node]:
                primary[node], priority[node] = row["origin_id"], order
            if row["source"] not in {"fineweb", "tatoeba"}:
                raise ValueError(f"Unsupported source: {row['source']}")
            masks[node] |= 1 if row["source"] == "fineweb" else 2
        updates = []
        for node, key in enumerate(hashes):
            updates.append((priority[node], primary[node], masks[node], key))
            if len(updates) == 10000:
                db.executemany("UPDATE contents SET origin_priority=?,primary_origin=?,source_mask=? WHERE doc_hash=?", updates)
                updates.clear()
        db.executemany("UPDATE contents SET origin_priority=?,primary_origin=?,source_mask=? WHERE doc_hash=?", updates)
        db.commit()
        with (output / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in db.execute("""
                SELECT o.info,c.group_id,c.split_rank FROM origins o JOIN contents c ON c.doc_hash=o.doc_hash
                ORDER BY o.source_file,o.row_index
            """):
                dump_line(stream, {**json.loads(row["info"]), "group_id": row["group_id"],
                                   "assigned_split": SPLITS[row["split_rank"]]})
        unmatched = [r[0] for r in db.execute("SELECT id FROM annotations WHERE matched=0 ORDER BY id")]
        dump_json(output / "unmatched_annotations.json", unmatched)
        if config.get("review", {}).get("require_all_annotations", False) and unmatched:
            raise ValueError("Approved annotations did not match; inspect unmatched_annotations.json.")
        origin_count = db.execute("SELECT COUNT(*) FROM origins").fetchone()[0]
        if origin_count != sum(sum(m["selected_documents"].values()) for _, m in parts):
            raise ValueError("Imported origin counts disagree with manifests.")
        result = {"origin_count": origin_count, "annotation_count": annotation_count,
                  "matched_annotations": annotation_count - len(unmatched),
                  "unmatched_annotations": len(unmatched), "forced_train_groups": len(forced_roots)}
    finally:
        db.close()
    # Workers receive only small exclusion lists, not all document identities.
    for index, values in excluded.items():
        dump_json(work / f"excluded-{index:03d}.json", values)
    return result


def repartition(job):
    directory, work, index, buckets, cache_mb = job
    import pyarrow as pa
    import pyarrow.parquet as pq

    start = time.monotonic()
    destination = work / f"part-{index:03d}"
    destination.mkdir()
    excluded = set(json.loads((work / f"excluded-{index:03d}.json").read_text(encoding="utf-8")))
    fields = ("doc_hash", "text_hash", "text", "paragraph_index", "sentence_index", "spans", "annotation_ids")
    schema = pa.schema([(name, pa.int64() if name.endswith("_index") else pa.string()) for name in fields])
    metrics = {name: Counter() for name in ("counts", "rule_reasons", "fragment_flags", "effective_blocks")}
    sentences = 0
    with ExitStack() as stack:
        db = stack.enter_context(closing(readonly(directory / "index.sqlite")))
        tune(db, cache_mb)
        writers = {}
        cursor = db.execute("SELECT * FROM units")
        while rows := cursor.fetchmany(65536):
            groups = {}
            for row in rows:
                if row["doc_hash"] in excluded:
                    continue
                bucket = int(row["text_hash"][:2], 16) * buckets // 256
                groups.setdefault(bucket, []).append(dict(row))
                sentences += 1
            for bucket, values in groups.items():
                if bucket not in writers:
                    writers[bucket] = stack.enter_context(pq.ParquetWriter(
                        destination / f"bucket-{bucket:03d}.parquet", schema, compression="zstd", use_dictionary=False))
                writers[bucket].write_table(pa.Table.from_pylist(values, schema=schema))
        for row in db.execute("SELECT doc_hash,payload FROM document_metrics"):
            if row["doc_hash"] not in excluded:
                payload = json.loads(row["payload"])
                for name, total in metrics.items():
                    total.update(payload[name])
        for name in AUDITS:
            with (directory / f"{name}.jsonl").open(encoding="utf-8") as source, \
                    (destination / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as target:
                for line in source:
                    # Preserve original audit formatting/order when no global content duplicate exists.
                    if not excluded or json.loads(line)["doc_hash"] not in excluded:
                        target.write(line)
    log(f"Repartition part {index}: {sentences:,} sentences in {time.monotonic() - start:.1f}s")
    return {"index": index, "sentences": sentences, "metrics": {k: dict(v) for k, v in metrics.items()}}


def export_bucket(job):
    bucket, work, metadata, part_indices, quality_mode, cache_mb = job
    import pyarrow.parquet as pq

    start = time.monotonic()
    destination = work / f"export-{bucket:03d}"
    destination.mkdir()
    scratch = destination / "bucket.sqlite"
    split_stats = {name: {"sentences": 0, "characters": 0, "primary_sources": Counter()} for name in SPLITS}
    with ExitStack() as stack:
        db = stack.enter_context(closing(sqlite3.connect(scratch, uri=True)))
        db.row_factory = sqlite3.Row
        tune(db, cache_mb)
        db.executescript("""
            CREATE TABLE units(doc_hash TEXT,text_hash TEXT,text TEXT,paragraph_index INTEGER,
                               sentence_index INTEGER,spans TEXT,annotation_ids TEXT);
        """)
        # Independent scratch DBs have no shared writer and indexes are built after bulk loading.
        fields = ("doc_hash", "text_hash", "text", "paragraph_index", "sentence_index", "spans", "annotation_ids")
        for index in part_indices:
            path = work / f"part-{index:03d}" / f"bucket-{bucket:03d}.parquet"
            if path.exists():
                for batch in pq.ParquetFile(path).iter_batches(batch_size=16384, use_threads=False):
                    columns = [batch.column(name).to_pylist() for name in fields]
                    db.executemany("INSERT INTO units VALUES(?,?,?,?,?,?,?)", zip(*columns))
        db.commit()
        db.execute("ATTACH DATABASE ? AS meta", (metadata.resolve().as_uri() + "?mode=ro",))
        db.execute(f"PRAGMA meta.cache_size=-{cache_mb * 1024}")
        # Copy only this bucket's narrow per-document lookup, keeping hot joins in its own DB.
        db.executescript("""
            CREATE TABLE docs AS SELECT c.doc_hash,c.group_id,c.split_rank,c.forced,c.origin_priority,
                                        c.primary_origin,c.source_mask
                FROM meta.contents c JOIN (SELECT DISTINCT doc_hash FROM units) u ON u.doc_hash=c.doc_hash;
            CREATE UNIQUE INDEX docs_hash ON docs(doc_hash);
            CREATE TABLE winners AS
                SELECT u.text_hash, CASE WHEN MAX(d.forced) THEN 0 ELSE MAX(d.split_rank) END split_rank,
                       MAX(d.source_mask & 1) fineweb,MAX(d.source_mask & 2) tatoeba
                FROM units u JOIN docs d ON d.doc_hash=u.doc_hash GROUP BY u.text_hash;
            CREATE UNIQUE INDEX winners_hash ON winners(text_hash);
            CREATE TABLE chosen AS
                SELECT unit_id,text_hash FROM (
                    SELECT u.rowid unit_id,u.text_hash,
                           ROW_NUMBER() OVER(PARTITION BY u.text_hash
                               ORDER BY d.origin_priority,u.paragraph_index,u.sentence_index) preference
                    FROM units u JOIN docs d ON d.doc_hash=u.doc_hash
                    JOIN winners w ON w.text_hash=u.text_hash WHERE d.split_rank=w.split_rank
                ) WHERE preference=1;
            CREATE UNIQUE INDEX chosen_hash ON chosen(text_hash);
        """)
        outputs = {name: (stack.enter_context((destination / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n")),
                          stack.enter_context((destination / f"{name}.txt").open("w", encoding="utf-8", newline="\n")))
                   for name in SPLITS}
        for row in db.execute("""
            SELECT u.*,d.group_id,o.doc_id,o.source,o.source_file,o.row_index,w.split_rank,w.fineweb,w.tatoeba
            FROM chosen x JOIN units u ON u.rowid=x.unit_id JOIN docs d ON d.doc_hash=u.doc_hash
            JOIN meta.origins o ON o.rowid=d.primary_origin JOIN winners w ON w.text_hash=u.text_hash
            ORDER BY x.text_hash
        """):
            name = SPLITS[row["split_rank"]]
            record = {
                "text": row["text"], "text_hash": row["text_hash"], "source": row["source"],
                "sources": [source for source, present in (("fineweb", row["fineweb"]), ("tatoeba", row["tatoeba"])) if present],
                "doc_id": row["doc_id"], "doc_hash": row["doc_hash"], "group_id": row["group_id"],
                "source_file": row["source_file"], "row_index": row["row_index"],
                "paragraph_index": row["paragraph_index"], "sentence_index": row["sentence_index"],
                "cleaned_block_spans": json.loads(row["spans"]), "quality_mode": quality_mode,
                "lm_reviewed": False, "approved_annotation_ids": json.loads(row["annotation_ids"]),
            }
            dump_line(outputs[name][0], record)
            outputs[name][1].write(row["text"] + "\n")
            split_stats[name]["sentences"] += 1
            split_stats[name]["characters"] += len(row["text"])
            split_stats[name]["primary_sources"][row["source"]] += 1
        with (destination / "provenance.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in db.execute("""
                SELECT u.text_hash,u.doc_hash,u.paragraph_index,u.sentence_index,
                       o.doc_id,o.source,o.source_file,o.row_index,d.group_id,u.annotation_ids,
                       d.split_rank assigned,w.split_rank exported
                FROM units u JOIN docs d ON d.doc_hash=u.doc_hash
                JOIN meta.origins o ON o.doc_hash=u.doc_hash JOIN winners w ON w.text_hash=u.text_hash
                ORDER BY u.text_hash,o.source_file,o.row_index,u.paragraph_index,u.sentence_index
            """):
                dump_line(stream, {key: row[key] for key in row.keys() if key not in {"assigned", "exported", "annotation_ids"}} | {
                    "approved_annotation_ids": json.loads(row["annotation_ids"]),
                    "assigned_split": SPLITS[row["assigned"]], "exported_split": SPLITS[row["exported"]],
                    "retained_in_assigned_split": row["assigned"] == row["exported"],
                })
        unique = db.execute("SELECT COUNT(*) FROM winners").fetchone()[0]
        removed = db.execute("""SELECT COUNT(*) FROM units u JOIN docs d ON d.doc_hash=u.doc_hash
            JOIN winners w ON w.text_hash=u.text_hash WHERE d.split_rank!=w.split_rank""").fetchone()[0]
    # Only the exact, locally created scratch database is removed; all exports/parts remain.
    scratch.unlink()
    log(f"Bucket {bucket}: {unique:,} unique sentences in {time.monotonic() - start:.1f}s")
    return {"bucket": bucket, "unique_sentences": unique, "cross_split_occurrences_removed": removed,
            "splits": {name: {**s, "primary_sources": dict(s["primary_sources"])} for name, s in split_stats.items()}}


def concatenate(job):
    destination, sources = job
    with destination.open("wb") as target:
        for source in sources:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, target, length=4 * 1024 * 1024)
    return destination.name


def run(config_path, parts, output, *, workers=4, buckets=64, cache_mb=128, work=None):
    if type(workers) is not int or not 1 <= workers <= 32 or buckets not in {1, 4, 16, 64, 256} or not 16 <= cache_mb <= 2048:
        raise ValueError("workers: 1..32; buckets: 1/4/16/64/256; cache-mb: 16..2048")
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    parts = [Path(p).resolve() for p in parts]
    work = Path(work).resolve() if work else output.with_name(output.name + "-work")
    # Refuse overlapping paths, including accidentally pointing at a completed worker.
    for left, right in [(output, work), *((path, target) for path in parts for target in (output, work))]:
        if left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError("Inputs, output and work directories must not overlap.")
    for path in (output, work):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"Use a new, empty output/work directory: {path}")
    started = time.monotonic()
    log("Validate completed worker artifacts (read only)")
    plan, parts = validate_parts(config_path, parts)
    config, _, policies, calibration_ids, policy_hashes, code_hashes, signature = plan
    implementation_sha = text_sha(Path(__file__))
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    stage_times = {}
    start = time.monotonic()
    metadata = make_metadata(output, work, parts, config, policies, calibration_ids, cache_mb)
    stage_times["metadata_seconds"] = time.monotonic() - start
    indices = [m["partition"]["index"] for _, m in parts]
    start = time.monotonic()
    log(f"Repartition existing sentences: {workers} processes, {buckets} buckets")
    part_results = parallel(repartition, [(p, work, m["partition"]["index"], buckets, cache_mb) for p, m in parts], workers, "repartition")
    stage_times["repartition_seconds"] = time.monotonic() - start
    start = time.monotonic()
    bucket_results = parallel(export_bucket, [(b, work, output / "index.sqlite", indices,
                                              config["build"]["quality_mode"], cache_mb) for b in range(buckets)], workers, "dedup/export")
    stage_times["dedup_export_seconds"] = time.monotonic() - start
    start = time.monotonic()
    files = [f"{name}.{suffix}" for name in SPLITS for suffix in ("txt", "jsonl")] + ["provenance.jsonl"]
    jobs = [(output / name, [work / f"export-{b:03d}" / name for b in range(buckets)]) for name in files]
    jobs.extend((output / f"{name}.jsonl", [work / f"part-{i:03d}" / f"{name}.jsonl" for i in indices]) for name in AUDITS)
    parallel(concatenate, jobs, min(workers, len(jobs)), "assemble")
    stage_times["assemble_seconds"] = time.monotonic() - start
    totals = {name: Counter() for name in ("counts", "rule_reasons", "fragment_flags", "effective_blocks")}
    for result in sorted(part_results, key=lambda r: r["index"]):
        for name in totals:
            totals[name].update(result["metrics"][name])
    counts = totals["counts"]
    counts["document_origins"] = metadata["origin_count"]
    counts["raw_characters"] = sum(json.loads((p / "part-stats.json").read_text(encoding="utf-8"))["raw_characters"] for p, _ in parts)
    split_stats = {name: {"sentences": 0, "characters": 0, "primary_sources": Counter()} for name in SPLITS}
    for result in bucket_results:
        for name, values in result["splits"].items():
            for key in ("sentences", "characters"):
                split_stats[name][key] += values[key]
            split_stats[name]["primary_sources"].update(values["primary_sources"])
    unique = sum(r["unique_sentences"] for r in bucket_results)
    characters = sum(s["characters"] for s in split_stats.values())
    if sum(r["sentences"] for r in part_results) != counts["sentence_occurrences"] or sum(s["sentences"] for s in split_stats.values()) != unique:
        raise ValueError("Repartition/export counts disagree; output is incomplete.")
    stats = {
        **dict(counts), "duplicate_documents": counts["document_origins"] - counts["unique_documents"],
        "unique_sentences": unique, "duplicate_sentence_occurrences": counts["sentence_occurrences"] - unique,
        "cross_split_occurrences_removed": sum(r["cross_split_occurrences_removed"] for r in bucket_results),
        "final_characters": characters,
        "splits": {name: {**s, "primary_sources": dict(s["primary_sources"])} for name, s in split_stats.items()},
        "rule_reasons": dict(totals["rule_reasons"]),
        "block_count_note": "keep/drop/review_blocks count rule-stage decisions before annotations.",
        "effective_block_actions": dict(totals["effective_blocks"]), "fragment_flags": dict(totals["fragment_flags"]),
        "forced_train_groups": metadata["forced_train_groups"], "annotation_records": metadata["annotation_count"],
        "matched_annotations": metadata["matched_annotations"], "unmatched_annotations": metadata["unmatched_annotations"],
        "token_estimate_scenarios": {f"{n}_characters_per_token": math.ceil(characters / n) for n in (1, 2, 3)},
        "token_estimate_note": "Heuristic scenarios, not measured SentencePiece counts.",
    }
    dump_json(output / "stats.json", stats)
    if load_plan(config_path)[-1] != signature or text_sha(Path(__file__)) != implementation_sha:
        raise ValueError("Code/config/policy changed during merge; output is incomplete.")
    dump_json(output / "manifest.json", {
        "status": "complete", "stage": "merged_corpus_staging", "ready_for_lm_training": False,
        "quality_mode": config["build"]["quality_mode"], "config": config, "config_sha256": file_sha(config_path),
        "pipeline_signature": signature, "python_version": sys.version,
        "input_sha256": {key: value for _, m in parts for key, value in m["input_sha256"].items()},
        "selected_documents": {key: value for _, m in parts for key, value in m["selected_documents"].items()},
        "policy_sha256": policy_hashes, "code_sha256": code_hashes,
        "fingerprint_normalization": "Code and policy: UTF-8 text with LF line endings, no BOM; input/artifact hashes: exact bytes.",
        "merge_code_sha256": {display_path(Path(__file__)): implementation_sha},
        "part_manifest_sha256": {display_path(p / "manifest.json"): file_sha(p / "manifest.json") for p, _ in parts},
        "grouping": "transitive groups of identical NFC documents, source IDs and canonical URLs",
        "sentence_dedup": "exact NFC text; calibration text forced to train, otherwise test > validation > train",
        "near_duplicate_detection": False, "span_offsets": "Unicode code points within cleaned blocks",
        "calibration_policy": "known calibration documents and their groups and sentences are forced into train",
        "annotation_policy": "Only approved snapshot-matched local labels apply; model replies alone are advisory.",
        "index_format": "document-review-index-v1; origins, contents, annotations; no units/winners tables",
        "merge_backend": "parallel SQLite hash buckets via Parquet; global in-memory document union-find",
        "workers": workers, "buckets": buckets, "cache_mb_per_connection": cache_mb,
        "work_directory": display_path(work),
    })
    stage_times["total_seconds"] = time.monotonic() - started
    dump_json(output / "merge-timing.json", stage_times)
    log(f"Complete: {unique:,} sentences; {stage_times['total_seconds']:.1f}s; corpus: {output}")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/corpus-parallel.toml")
    parser.add_argument("--parts-dir", type=Path, default=ROOT / "outputs/corpus-parallel-v1-work/parts")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/corpus-fast-v1")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--buckets", type=int, default=64, choices=(1, 4, 16, 64, 256))
    parser.add_argument("--cache-mb", type=int, default=128)
    args = parser.parse_args()
    parts = sorted(p.parent for p in args.parts_dir.resolve().glob("*/manifest.json"))
    if not parts:
        parser.error("No completed worker manifests under --parts-dir")
    run(args.config, parts, args.output, workers=args.workers, buckets=args.buckets, cache_mb=args.cache_mb, work=args.work)


if __name__ == "__main__":
    main()
