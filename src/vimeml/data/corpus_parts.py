"""Portable input plans and shared operations for independent corpus parts."""

import json
import sqlite3
import sys
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.build import validate, file_sha, sha, display_path, SCHEMA, dump_line, dump_json
from vimeml.data.clean import clean_block
from vimeml.data.segment import restore_linebreaks, split_sentences
from vimeml.data.annotations import load_annotations, document_annotations, match_fragment

PART_SCHEMA_VERSION = 1
AUDITS = ("review_blocks", "dropped_blocks", "fragments")


def text_sha(path):
    # Git may check out CRLF on Windows and LF on Mac. These text files have
    # identical semantics; hash normalized UTF-8 text rather than platform bytes.
    return sha(path.read_text(encoding="utf-8-sig"))


def load_plan(config_path):
    config = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    validate(config)
    paths = config["inputs"]["fineweb_paths"]
    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
        raise ValueError("Partition builds require an explicit fineweb_paths list.")
    inventory = [(p, "fineweb", config["build"]["fineweb_documents_per_shard"]) for p in sorted(paths)]
    inventory.append((config["inputs"]["tatoeba_path"], "tatoeba", config["build"]["tatoeba_documents"]))
    if len({p for p, _, _ in inventory}) != len(inventory):
        raise ValueError("Input paths must be unique.")
    for p, _, _ in inventory:
        if Path(p).is_absolute() or not (ROOT / p).resolve().is_relative_to(ROOT):
            raise ValueError("Input paths must be portable paths inside the repository.")
    policy_paths = {}
    if config["build"]["quality_mode"] == "rules_with_approved_annotations":
        policy_paths["annotations"] = ROOT / config["review"]["annotations"]
    policy_paths["calibration_ids"] = ROOT / config["inputs"]["calibration_ids"]
    calibration_ids = json.loads(policy_paths["calibration_ids"].read_text(encoding="utf-8-sig"))
    if not isinstance(calibration_ids, list) or any(not isinstance(p, str) for p in calibration_ids):
        raise ValueError("calibration_ids must contain a JSON list of document IDs.")
    policy_hashes = {key: text_sha(path) for key, path in policy_paths.items()}
    names = ("build", "readers", "clean", "segment", "annotations", "corpus_parts", "corpus_export", "preprocess_part", "merge_parts")
    code_hashes = {f"src/vimeml/data/{name}.py": text_sha(ROOT / f"src/vimeml/data/{name}.py") for name in names}
    relevant = {key: value for key, value in config["build"].items() if key not in {"output_dir", "commit_every", "progress_every", "batch_size"}}
    signature = sha(json.dumps({"schema": PART_SCHEMA_VERSION, "inventory": inventory,
                               "build": relevant, "split": config["split"], "review": config.get("review", {}),
                               "policy_sha256": policy_hashes, "code_sha256": code_hashes}, sort_keys=True))
    return config, inventory, policy_paths, set(calibration_ids), policy_hashes, code_hashes, signature


def parse_part(value):
    try:
        index, total = map(int, value.split("/"))
    except (ValueError, AttributeError):
        raise ValueError("Use --part INDEX/COUNT, e.g. 0/2 or 1/2.") from None
    if not 1 <= total <= 128 or not 0 <= index < total:
        raise ValueError("Part index is zero-based and must be less than the part count.")
    return index, total


def create_database(output, annotation_path):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / "index.sqlite", uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA temp_store=FILE")
        db.executescript(SCHEMA)
        db.execute("CREATE TABLE document_metrics(doc_hash TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        annotation_count = load_annotations(db, annotation_path)
        return db, annotation_count
    except BaseException:
        db.close()
        raise


def aggregate_metrics(db):
    totals = {key: Counter() for key in ("counts", "rule_reasons", "fragment_flags", "effective_blocks")}
    for row in db.execute("SELECT payload FROM document_metrics ORDER BY doc_hash"):
        metrics = json.loads(row[0])
        for key in totals:
            totals[key].update(metrics[key])
    return totals


def process_content(db, document, normalized, doc_hash, settings, audits):
    counts, rule_reasons, fragment_flags, effective_blocks = (Counter() for _ in range(4))
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
    return {'counts': dict(counts), 'rule_reasons': dict(rule_reasons),
            'fragment_flags': dict(fragment_flags), 'effective_blocks': dict(effective_blocks)}
