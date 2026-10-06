"""Read-only integrity checks before deleting completed merge intermediates."""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.data.build import file_sha, dump_json
from vimeml.data.corpus_parts import load_plan, text_sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_files(paths):
    digest, lines, size, last = hashlib.sha256(), 0, 0, b""
    for path in paths:
        with path.open("rb") as stream:
            while data := stream.read(4 * 1024 * 1024):
                digest.update(data)
                lines += data.count(b"\n")
                size += len(data)
                last = data[-1:]
    require(not size or last == b"\n", f"Missing final newline: {paths[-1]}")
    return {"sha256": digest.hexdigest(), "lines": lines, "bytes": size}


def compare(job):
    target, sources, expected = job
    reference = digest_files(sources)
    actual = digest_files([target])
    require(actual == reference, f"Export differs from ordered intermediates: {target}")
    require(actual["lines"] == expected, f"Export row count differs: {target}")
    return target.name, actual


def verify_text(path, expected, blocked=None, collect=False):
    count, characters, previous = 0, 0, None
    hashes = set() if collect else None
    with path.open("rb") as stream:
        for raw in stream:
            require(raw.endswith(b"\n"), f"Incomplete line: {path}")
            raw = raw[:-1]
            text = raw.decode("utf-8", errors="strict")
            digest = hashlib.sha256(raw).digest()
            require(previous is None or previous < digest, f"Duplicate or unsorted text: {path}, line {count + 1}")
            require(blocked is None or digest not in blocked, f"Exact sentence leakage: {path}, line {count + 1}")
            require(bool(text) and "\r" not in text, f"Empty/CR sentence: {path}, line {count + 1}")
            if hashes is not None:
                hashes.add(digest)
            previous = digest
            count += 1
            characters += len(text)
            if count % 5000000 == 0:
                print(f"TXT integrity: {path.name}: {count:,} sentences checked", flush=True)
    require(count == expected["sentences"] and characters == expected["characters"], f"TXT statistics differ: {path}")
    print(f"TXT passed: {path.name}: {count:,} sentences, {characters:,} characters", flush=True)
    return hashes


def verify(corpus, report):
    start = time.monotonic()
    manifest = json.loads((corpus / "manifest.json").read_text("utf-8"))
    stats = json.loads((corpus / "stats.json").read_text("utf-8"))
    require(manifest["status"] == "complete", "Merge is incomplete")
    config_path = ROOT / "configs/corpus-parallel.toml"
    require(load_plan(config_path)[-1] == manifest["pipeline_signature"], "Current code/config/policy differs")
    for relative, expected in manifest["merge_code_sha256"].items():
        require(text_sha(ROOT / relative) == expected, "Merge implementation differs")
    part_indices = []
    part_dirs = {}
    for relative, expected in manifest["part_manifest_sha256"].items():
        path = ROOT / relative
        require(file_sha(path) == expected, f"Part manifest differs: {relative}")
        index = json.loads(path.read_text("utf-8"))["partition"]["index"]
        part_indices.append(index)
        part_dirs[index] = path.parent
    part_indices.sort()
    require(stats["document_origins"] == sum(manifest["selected_documents"].values()), "Origin counts differ")
    require(stats["unique_sentences"] == sum(s["sentences"] for s in stats["splits"].values()), "Sentence counts differ")
    require(stats["final_characters"] == sum(s["characters"] for s in stats["splits"].values()), "Character counts differ")
    require(json.loads((corpus / "unmatched_annotations.json").read_text("utf-8")) == [], "Unmatched annotations")
    print("Manifest, policy and part identities passed", flush=True)
    with sqlite3.connect((corpus / "index.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        require(db.execute("PRAGMA quick_check").fetchall() == [("ok",)], "SQLite quick_check failed")
        require(db.execute("SELECT count(*) FROM origins").fetchone()[0] == stats["document_origins"], "Origin index differs")
        require(db.execute("SELECT count(*) FROM contents").fetchone()[0] == stats["unique_documents"], "Content index differs")
        require(db.execute("SELECT count(*) FROM annotations WHERE matched=1").fetchone()[0] == stats["matched_annotations"], "Annotation index differs")
        by_file = dict(db.execute("SELECT source_file,count(*) FROM origins GROUP BY source_file"))
        require(by_file == manifest["selected_documents"], "Input file coverage differs")
        ids = json.loads((ROOT / "annotations/calibration-documents-v1.json").read_text("utf-8"))
        for doc_id in ids:
            ranks = db.execute("SELECT DISTINCT c.split_rank FROM origins o JOIN contents c ON c.doc_hash=o.doc_hash WHERE o.doc_id=?", (doc_id,)).fetchall()
            require(ranks == [(0,)], f"Calibration document is missing or held out: {doc_id}")
        duplicate_contents = list(db.execute("""SELECT o.doc_hash,c.owner,count(*)-1 copies FROM origins o
            JOIN contents c ON c.doc_hash=o.doc_hash GROUP BY o.doc_hash HAVING count(*)>1"""))
    provenance_count = stats["sentence_occurrences"]
    for doc_hash, owner, copies in duplicate_contents:
        with sqlite3.connect((part_dirs[owner] / "index.sqlite").as_uri() + "?mode=ro", uri=True) as db:
            provenance_count += copies * db.execute("SELECT count(*) FROM units WHERE doc_hash=?", (doc_hash,)).fetchone()[0]
    print("SQLite integrity, input coverage, calibration groups and annotations passed", flush=True)
    work = ROOT / manifest["work_directory"]
    jobs = []
    for name in ("train", "validation", "test"):
        for suffix in ("txt", "jsonl"):
            filename = f"{name}.{suffix}"
            jobs.append((corpus / filename, [work / f"export-{b:03d}" / filename for b in range(manifest["buckets"])], stats["splits"][name]["sentences"]))
    jobs.append((corpus / "provenance.jsonl", [work / f"export-{b:03d}" / "provenance.jsonl" for b in range(manifest["buckets"])], provenance_count))
    # This build has no approved block labels adding extra audit records.
    require(stats["approved_block_labels"] == 0, "Audit count policy needs adjustment for block annotations")
    for name, expected in (("review_blocks", stats["review_blocks"]), ("dropped_blocks", stats["drop_blocks"]), ("fragments", stats["fragments"])):
        jobs.append((corpus / f"{name}.jsonl", [work / f"part-{i:03d}" / f"{name}.jsonl" for i in part_indices], expected))
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(compare, job) for job in jobs]
        # TXT hashes are checked on all sentences; only held-out hashes stay in memory.
        validation = verify_text(corpus / "validation.txt", stats["splits"]["validation"], collect=True)
        test = verify_text(corpus / "test.txt", stats["splits"]["test"], blocked=validation, collect=True)
        validation.update(test)
        del test
        verify_text(corpus / "train.txt", stats["splits"]["train"], blocked=validation)
        for future in as_completed(futures):
            name, details = future.result()
            results[name] = details
            print(f"SHA256 and row counts passed: {name}", flush=True)
    docs = digest_files([corpus / "documents.jsonl"])
    require(docs["lines"] == stats["document_origins"], "Document export count differs")
    results["documents.jsonl"] = docs
    # Check actual raw input bytes too, rather than relying only on earlier worker manifests.
    for relative, expected in manifest["input_sha256"].items():
        require(file_sha(ROOT / relative) == expected, f"Original input differs: {relative}")
    dump_json(report, {"status": "passed", "corpus": str(corpus), "manifest_sha256": file_sha(corpus / "manifest.json"),
        "document_origins": stats["document_origins"], "sentences": stats["unique_sentences"],
        "characters": stats["final_characters"], "provenance_occurrences": provenance_count,
        "exact_cross_split_overlaps": 0, "sqlite_quick_check": "ok", "calibration_ids_checked": len(ids),
        "input_sha256_checked": True, "exports": results, "seconds": time.monotonic() - start,
        "quality_note": "Storage/export integrity only. Japanese quality review and near-duplicate detection remain separate."})
    print(f"PASS. Report: {report}; {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "outputs/corpus-fast-v1")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/corpus-fast-v1/integrity-check.json")
    args = parser.parse_args()
    verify(args.corpus.resolve(), args.report.resolve())
