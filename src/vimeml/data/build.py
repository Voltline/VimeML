"""Build a reproducible Japanese corpus, optionally applying approved local labels."""

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import tomllib
import unicodedata
from collections import Counter
from contextlib import ExitStack, closing
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.clean import clean_block
from vimeml.data.readers import iter_fineweb, iter_tatoeba
from vimeml.data.segment import restore_linebreaks, split_sentences
from vimeml.data.annotations import load_annotations, document_annotations, match_fragment

SPLITS = ("train", "validation", "test")

SCHEMA = """
CREATE TABLE parents (key TEXT PRIMARY KEY, parent TEXT NOT NULL);
CREATE TABLE aliases (key TEXT PRIMARY KEY, doc_hash TEXT NOT NULL);
CREATE TABLE contents (
    doc_hash TEXT PRIMARY KEY, group_id TEXT, split_rank INTEGER
);
CREATE TABLE origins (
    doc_hash TEXT NOT NULL, doc_id TEXT NOT NULL, source TEXT NOT NULL,
    source_file TEXT NOT NULL, row_index INTEGER NOT NULL, info TEXT NOT NULL,
    PRIMARY KEY (source_file, row_index)
);
CREATE INDEX origins_content ON origins(doc_hash);
CREATE INDEX origins_identity ON origins(doc_id);
CREATE TABLE units (
    doc_hash TEXT NOT NULL, text_hash TEXT NOT NULL, text TEXT NOT NULL,
    paragraph_index INTEGER NOT NULL, sentence_index INTEGER NOT NULL,
    spans TEXT NOT NULL, annotation_ids TEXT NOT NULL,
    PRIMARY KEY (doc_hash, paragraph_index, sentence_index)
);
CREATE INDEX units_text ON units(text_hash);
"""

WINNERS = """
CREATE TABLE winners AS
SELECT u.text_hash, MAX(c.split_rank) AS split_rank,
       GROUP_CONCAT(DISTINCT o.source) AS sources
FROM units u
JOIN contents c ON c.doc_hash = u.doc_hash
JOIN origins o ON o.doc_hash = u.doc_hash
GROUP BY u.text_hash;
CREATE UNIQUE INDEX winners_text ON winners(text_hash);
"""

PRIMARY_ROWS = """
WITH ranked AS (
    SELECT u.*, o.doc_id, o.source, o.source_file, o.row_index,
           c.group_id, w.split_rank, w.sources,
           ROW_NUMBER() OVER (
               PARTITION BY u.text_hash
               ORDER BY CASE o.source WHEN 'tatoeba' THEN 0 ELSE 1 END,
                        o.doc_id, o.source_file, o.row_index,
                        u.paragraph_index, u.sentence_index
           ) AS preference
    FROM units u
    JOIN contents c ON c.doc_hash = u.doc_hash
    JOIN origins o ON o.doc_hash = u.doc_hash
    JOIN winners w ON w.text_hash = u.text_hash
    WHERE c.split_rank = w.split_rank
)
SELECT * FROM ranked WHERE preference = 1 ORDER BY text_hash;
"""


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path):
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def dump_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def dump_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical_url(value):
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value.strip())
        if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
            return None
        # Keep the query and path; remove only the fragment, lowercase host.
        return urlunsplit((
            parts.scheme.lower(), parts.netloc.lower(),
            parts.path or "/", parts.query, "",
        ))
    except ValueError:
        return None


def find(db, key):
    trail = []
    while True:
        parent = db.execute(
            "SELECT parent FROM parents WHERE key = ?", (key,)
        ).fetchone()[0]
        if parent == key:
            break
        trail.append(key)
        key = parent
    for item in trail:
        db.execute("UPDATE parents SET parent = ? WHERE key = ?", (key, item))
    return key


def attach_alias(db, doc_hash, alias):
    previous = db.execute(
        "SELECT doc_hash FROM aliases WHERE key = ?", (alias,)
    ).fetchone()
    if previous is None:
        db.execute("INSERT INTO aliases VALUES (?, ?)", (alias, doc_hash))
        return
    left, right = find(db, doc_hash), find(db, previous[0])
    if left != right:
        root, child = sorted((left, right))
        db.execute("UPDATE parents SET parent = ? WHERE key = ?", (root, child))


def validate(config):
    if config["build"]["quality_mode"] not in {"rules_only", "rules_with_approved_annotations"}:
        raise ValueError("Unknown quality_mode.")
    if config["build"]["quality_mode"] == "rules_with_approved_annotations" and not config.get("review", {}).get("annotations"):
        raise ValueError("Approved-annotation mode requires a local annotations JSONL.")
    ratios = [config["split"][name] for name in SPLITS]
    if (
        any(isinstance(x, bool) or not isinstance(x, (int, float))
            or not math.isfinite(x) or x <= 0 for x in ratios)
        or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9)
    ):
        raise ValueError("Split ratios must be positive and sum to 1.")
    for key in ("fineweb_documents_per_shard", "tatoeba_documents"):
        value = config["build"][key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer; 0 means all.")
    for key in ("batch_size", "max_join_chars"):
        value = config["build"][key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    seed = config["build"]["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    for key in ("commit_every", "progress_every"):
        value = config["build"].get(key, 200)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer.")


def run(config_path, output_override=None):
    config_path = config_path.resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    validate(config)
    settings = config["build"]
    files = sorted(ROOT.glob(config["inputs"]["fineweb_glob"]))
    if not files:
        raise FileNotFoundError("No FineWeb2 parquet inputs found.")
    tatoeba = (ROOT / config["inputs"]["tatoeba_path"]).resolve()
    if not tatoeba.is_file():
        raise FileNotFoundError(tatoeba)

    calibration_ids = set()
    calibration_path = config["inputs"].get("calibration_review")
    calibration_file = (ROOT / calibration_path).resolve() if calibration_path else None
    if calibration_file:
        reviews = json.loads(calibration_file.read_text(encoding="utf-8"))
        calibration_ids = {
            f"{item['source']}:{item['id']}" for item in reviews
        }
    calibration_files = [(ROOT / value).resolve() for value in config["inputs"].get("calibration_cases", [])]
    for path in calibration_files:
        cases = json.loads(path.read_text(encoding="utf-8"))
        calibration_ids.update(item["doc_id"] for item in cases)
    annotation_file = None
    if settings["quality_mode"] == "rules_with_approved_annotations":
        annotation_file = (ROOT / config["review"]["annotations"]).resolve()
        if not annotation_file.is_file():
            raise FileNotFoundError(annotation_file)

    tracked_files = files + [tatoeba, config_path] + ([calibration_file] if calibration_file else []) + calibration_files + ([annotation_file] if annotation_file else [])
    code_files = [Path(__file__).resolve()] + [ROOT / f"src/vimeml/data/{name}.py" for name in ("readers", "clean", "segment", "annotations")]
    tracked_files += code_files
    snapshots = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in tracked_files}
    # Small policy/code files are fingerprinted before processing as well.
    initial_hashes = {path: file_sha(path) for path in tracked_files if path not in files + [tatoeba]}

    output = (ROOT / (output_override or settings["output_dir"])).resolve()
    # Refuse to overwrite any earlier build, including an interrupted one.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    rule_reasons = Counter()
    fragment_flags = Counter()
    selected_documents = Counter()

    with ExitStack() as stack:
        db = sqlite3.connect(output / "index.sqlite")
        stack.callback(db.close)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA temp_store = FILE")
        db.executescript(SCHEMA)
        annotation_count = load_annotations(db, annotation_file)
        effective_blocks = Counter()
        audits = {
            name: stack.enter_context(
                (output / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n")
            )
            for name in ("review_blocks", "dropped_blocks", "fragments")
        }
        inputs = [
            (path, "fineweb", settings["fineweb_documents_per_shard"])
            for path in files
        ]
        inputs.append((tatoeba, "tatoeba", settings["tatoeba_documents"]))

        for path, source, limit in inputs:
            reader = (
                iter_fineweb(path, settings["batch_size"])
                if source == "fineweb" else iter_tatoeba(path)
            )
            with closing(reader):
                documents = islice(reader, limit) if limit else reader
                for document in documents:
                    counts["document_origins"] += 1
                    counts["raw_characters"] += len(document.text)
                    selected_documents[document.source_file] += 1
                    normalized = unicodedata.normalize(
                        "NFC", document.text.replace("\r\n", "\n").replace("\r", "\n")
                    ).lstrip("\ufeff")
                    doc_hash = sha(normalized)
                    first = db.execute(
                        "INSERT OR IGNORE INTO contents(doc_hash) VALUES (?)",
                        (doc_hash,),
                    ).rowcount == 1
                    db.execute(
                        "INSERT OR IGNORE INTO parents VALUES (?, ?)",
                        (doc_hash, doc_hash),
                    )
                    attach_alias(db, doc_hash, "id:" + document.doc_id)
                    url = canonical_url(document.metadata.get("url"))
                    if url:
                        attach_alias(db, doc_hash, "url:" + url)
                    identity = {
                        "source": document.source, "doc_id": document.doc_id,
                        "source_id": document.source_id,
                        "source_file": document.source_file,
                        "row_index": document.row_index, "doc_hash": doc_hash,
                        "metadata": document.metadata,
                    }
                    db.execute(
                        "INSERT INTO origins VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            doc_hash, document.doc_id, document.source,
                            document.source_file, document.row_index,
                            json.dumps(identity, ensure_ascii=False, allow_nan=False),
                        ),
                    )

                    if first:
                        counts["unique_documents"] += 1
                        blocks = []
                        for index, original in enumerate(normalized.split("\n")):
                            result = clean_block(original)
                            block_id = f"b{index:03d}"
                            counts["input_blocks"] += 1
                            counts[result["action"] + "_blocks"] += 1
                            rule_reasons[result["reason"]] += 1
                            blocks.append({
                                "id": block_id, "line_index": index,
                                "text": result["text"], "action": result["action"],
                            })
                            if result["action"] != "keep":
                                destination = (
                                    "dropped_blocks" if result["action"] == "drop"
                                    else "review_blocks"
                                )
                                dump_line(audits[destination], {
                                    "source": document.source,
                                    "doc_id": document.doc_id, "doc_hash": doc_hash,
                                    "block_id": block_id, "original_text": original,
                                    **result,
                                })

                        boundaries, fragment_labels, block_labels = document_annotations(db, doc_hash, blocks)
                        counts["approved_block_labels"] += len(block_labels)
                        effective_blocks.update(block["action"] for block in blocks)
                        for block_id, label_id in block_labels.items():
                            block = next(item for item in blocks if item["id"] == block_id)
                            destination = "dropped_blocks" if block["action"] == "drop" else "review_blocks"
                            dump_line(audits[destination], {
                                "source": document.source, "doc_id": document.doc_id,
                                "doc_hash": doc_hash, "block_id": block_id,
                                "text": block["text"], "action": block["action"],
                                "reason": "approved_local_annotation", "annotation_id": label_id,
                            })

                        paragraphs = restore_linebreaks(
                            blocks, max_chars=settings["max_join_chars"],
                            boundary_decisions={pair: label["action"] for pair, label in boundaries.items()},
                        )
                        counts["paragraphs"] += len(paragraphs)
                        for pi, paragraph in enumerate(paragraphs):
                            counts["join_events"] += len(paragraph["joins"])
                            counts["approved_join_events"] += sum(join["reason"] == "approved_annotation" for join in paragraph["joins"])
                            paragraph_label_ids = {
                                block_labels[part["block_id"]] for part in paragraph["parts"]
                                if part["block_id"] in block_labels
                            }
                            paragraph_label_ids.update(
                                boundaries[(join["left_block"], join["right_block"])]["id"]
                                for join in paragraph["joins"] if join["reason"] == "approved_annotation"
                            )
                            for si, sentence in enumerate(split_sentences(paragraph["text"])):
                                flags = sorted(set(
                                    paragraph["flags"] + sentence["flags"]
                                ))
                                spans = []
                                for part in paragraph["parts"]:
                                    left = max(sentence["start"], part["start"])
                                    right = min(sentence["end"], part["end"])
                                    if left < right:
                                        spans.append({
                                            "block_id": part["block_id"],
                                            "start": left - part["start"],
                                            "end": right - part["start"],
                                        })
                                fragment_label = match_fragment(db, fragment_labels, sentence["text"], spans)
                                explicitly_kept = fragment_label is not None and fragment_label["action"] == "keep"
                                explicitly_excluded = fragment_label is not None and fragment_label["action"] in {"drop", "review"}
                                if explicitly_kept and flags:
                                    raise ValueError(f"Annotation {fragment_label['id']}: flagged fragments cannot be rescued.")
                                if (
                                    (not sentence["terminated"] and not explicitly_kept) or flags or explicitly_excluded
                                    or not any(c.isalnum() for c in sentence["text"])
                                ):
                                    counts["fragments"] += 1
                                    fragment_flags.update(flags)
                                    dump_line(audits["fragments"], {
                                        "source": document.source,
                                        "doc_id": document.doc_id, "doc_hash": doc_hash,
                                        "paragraph_index": pi, "sentence_index": si,
                                        "text": sentence["text"],
                                        "terminated": sentence["terminated"],
                                        "flags": flags, "cleaned_block_spans": spans,
                                        "paragraph_joins": paragraph["joins"],
                                        "approved_annotation_id": fragment_label["id"] if fragment_label else None,
                                    })
                                    continue
                                text = sentence["text"]
                                label_ids = set(paragraph_label_ids)
                                if fragment_label:
                                    label_ids.add(fragment_label["id"])
                                    counts["approved_fragment_recoveries"] += int(not sentence["terminated"])
                                db.execute(
                                    "INSERT INTO units VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (
                                        doc_hash, sha(text), text, pi, si,
                                        json.dumps(spans, ensure_ascii=False),
                                        json.dumps(sorted(label_ids)),
                                    ),
                                )
                                counts["sentence_occurrences"] += 1
                    if counts["document_origins"] % settings.get("commit_every", 200) == 0:
                        db.commit()
                    if counts["document_origins"] % settings.get("progress_every", 200) == 0:
                        print(f"Read {counts['document_origins']} document origins", flush=True)

        db.commit()
        forced_roots = set()
        for row in db.execute("SELECT DISTINCT doc_hash FROM annotations"):
            if db.execute("SELECT 1 FROM parents WHERE key=?", (row[0],)).fetchone():
                forced_roots.add(find(db, row[0]))
        for doc_id in sorted(calibration_ids):
            for row in db.execute(
                "SELECT DISTINCT doc_hash FROM origins WHERE doc_id = ?", (doc_id,)
            ):
                forced_roots.add(find(db, row["doc_hash"]))

        for row in db.execute("SELECT doc_hash FROM contents ORDER BY doc_hash"):
            root = find(db, row["doc_hash"])
            value = int(sha(f"{settings['seed']}:{root}")[:16], 16) / 2**64
            rank = (
                0 if root in forced_roots or value < config["split"]["train"]
                else 1 if value < config["split"]["train"] + config["split"]["validation"]
                else 2
            )
            db.execute(
                "UPDATE contents SET group_id = ?, split_rank = ? WHERE doc_hash = ?",
                (root, rank, row["doc_hash"]),
            )
        db.commit()
        db.executescript(WINNERS)
        # Text inspected while tuning the pipeline must not reappear in held-out
        # splits via an unreviewed document with the same sentence.
        db.execute("""
            UPDATE winners SET split_rank=0 WHERE text_hash IN (
                SELECT u.text_hash FROM units u JOIN contents c ON c.doc_hash=u.doc_hash
                WHERE c.group_id IN (SELECT value FROM json_each(?))
            )
        """, (json.dumps(sorted(forced_roots)),))

        split_stats = {
            name: {"sentences": 0, "characters": 0, "primary_sources": Counter()}
            for name in SPLITS
        }
        outputs = {
            name: (
                stack.enter_context((output / f"{name}.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                )),
                stack.enter_context((output / f"{name}.txt").open(
                    "w", encoding="utf-8", newline="\n"
                )),
            )
            for name in SPLITS
        }
        for row in db.execute(PRIMARY_ROWS):
            name = SPLITS[row["split_rank"]]
            record = {
                "text": row["text"], "text_hash": row["text_hash"],
                "source": row["source"], "sources": sorted(row["sources"].split(",")),
                "doc_id": row["doc_id"], "doc_hash": row["doc_hash"],
                "group_id": row["group_id"],
                "source_file": row["source_file"], "row_index": row["row_index"],
                "paragraph_index": row["paragraph_index"],
                "sentence_index": row["sentence_index"],
                "cleaned_block_spans": json.loads(row["spans"]),
                "quality_mode": settings["quality_mode"], "lm_reviewed": False,
                "approved_annotation_ids": json.loads(row["annotation_ids"]),
            }
            dump_line(outputs[name][0], record)
            outputs[name][1].write(row["text"] + "\n")
            split_stats[name]["sentences"] += 1
            split_stats[name]["characters"] += len(row["text"])
            split_stats[name]["primary_sources"][row["source"]] += 1

        with (output / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in db.execute("""
                SELECT o.info, c.group_id, c.split_rank FROM origins o
                JOIN contents c ON c.doc_hash = o.doc_hash
                ORDER BY o.source_file, o.row_index
            """):
                dump_line(stream, {
                    **json.loads(row["info"]), "group_id": row["group_id"],
                    "assigned_split": SPLITS[row["split_rank"]],
                })

        # Full sentence provenance includes occurrences removed from another split.
        with (output / "provenance.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in db.execute("""
                SELECT u.text_hash, u.doc_hash, u.paragraph_index, u.sentence_index,
                       o.doc_id, o.source, o.source_file, o.row_index,
                       c.group_id, u.annotation_ids, c.split_rank AS assigned, w.split_rank AS exported
                FROM units u JOIN contents c ON c.doc_hash = u.doc_hash
                JOIN origins o ON o.doc_hash = u.doc_hash
                JOIN winners w ON w.text_hash = u.text_hash
                ORDER BY u.text_hash, o.source_file, o.row_index,
                         u.paragraph_index, u.sentence_index
            """):
                dump_line(stream, {
                    key: row[key] for key in row.keys()
                    if key not in {"assigned", "exported", "annotation_ids"}
                } | {
                    "approved_annotation_ids": json.loads(row["annotation_ids"]),
                    "assigned_split": SPLITS[row["assigned"]],
                    "exported_split": SPLITS[row["exported"]],
                    "retained_in_assigned_split": row["assigned"] == row["exported"],
                })

        unique_sentences = db.execute("SELECT COUNT(*) FROM winners").fetchone()[0]
        cross_split_removed = db.execute("""
            SELECT COUNT(*) FROM units u
            JOIN contents c ON c.doc_hash = u.doc_hash
            JOIN winners w ON w.text_hash = u.text_hash
            WHERE c.split_rank != w.split_rank
        """).fetchone()[0]
        characters = sum(item["characters"] for item in split_stats.values())
        matched_annotations = db.execute("SELECT COUNT(*) FROM annotations WHERE matched=1").fetchone()[0]
        unmatched_annotations = [row[0] for row in db.execute("SELECT id FROM annotations WHERE matched=0 ORDER BY id")]
        dump_json(output / "unmatched_annotations.json", unmatched_annotations)
        if config.get("review", {}).get("require_all_annotations", False) and unmatched_annotations:
            raise ValueError("Approved annotations did not match this build; inspect unmatched_annotations.json.")
        stats = {
            **dict(counts),
            "duplicate_documents": counts["document_origins"] - counts["unique_documents"],
            "unique_sentences": unique_sentences,
            "duplicate_sentence_occurrences": counts["sentence_occurrences"] - unique_sentences,
            "cross_split_occurrences_removed": cross_split_removed,
            "final_characters": characters,
            "splits": {
                name: {**item, "primary_sources": dict(item["primary_sources"])}
                for name, item in split_stats.items()
            },
            "rule_reasons": dict(rule_reasons),
            "block_count_note": "keep/drop/review_blocks count rule-stage decisions before annotations.",
            "effective_block_actions": dict(effective_blocks),
            "fragment_flags": dict(fragment_flags),
            "forced_train_groups": len(forced_roots),
            "annotation_records": annotation_count, "matched_annotations": matched_annotations,
            "unmatched_annotations": len(unmatched_annotations),
            "token_estimate_scenarios": {
                f"{n}_characters_per_token": math.ceil(characters / n) for n in (1, 2, 3)
            },
            "token_estimate_note": "Heuristic scenarios, not measured SentencePiece counts.",
        }
        dump_json(output / "stats.json", stats)
        db.commit()

    input_files = files + [tatoeba] + ([calibration_file] if calibration_file else []) + calibration_files + ([annotation_file] if annotation_file else [])
    for path, before in snapshots.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != before:
            raise ValueError(f"Input or code changed during build: {display_path(path)}")
    for path, before in initial_hashes.items():
        if file_sha(path) != before:
            raise ValueError(f"Policy or code changed during build: {display_path(path)}")
    dump_json(output / "manifest.json", {
        "status": "complete",
        "quality_mode": settings["quality_mode"],
        "stage": "full_corpus_staging" if not any(limit for _, _, limit in inputs) else "bounded_integration_check",
        "ready_for_lm_training": False,
        "purpose": "pipeline smoke test" if any(
            limit for _, _, limit in inputs
        ) else "full local corpus staging; quality audit and near dedup pending",
        "config": config,
        "python_version": sys.version,
        "selection": "first N documents per input file; 0 means all",
        "selected_documents": dict(selected_documents),
        "grouping": "transitive groups of identical NFC documents, source IDs and canonical URLs",
        "sentence_dedup": "exact NFC text; calibration text forced to train, otherwise test > validation > train",
        "near_duplicate_detection": False,
        "span_offsets": "Unicode code points within cleaned blocks",
        "calibration_policy": "known calibration documents and their groups are forced into train",
        "annotation_policy": "Only approved snapshot-matched local annotations apply; annotation document groups are forced into train; model replies alone are advisory.",
        "input_sha256": {display_path(path): file_sha(path) for path in input_files},
        "code_sha256": {display_path(path): file_sha(path) for path in code_files},
        "config_sha256": file_sha(config_path),
        "effective_output_dir": display_path(output),
    })
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Corpus: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/corpus.toml")
    parser.add_argument("--output", help="Override output directory; must be absent or empty.")
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
