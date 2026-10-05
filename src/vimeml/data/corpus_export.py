"""Shared corpus export contract for partitioned preprocessing results."""

import json
import math
from collections import Counter
from contextlib import ExitStack

from vimeml.data.build import SPLITS, WINNERS, PRIMARY_ROWS, find, sha, dump_line, dump_json


def export_corpus(db, output, config, calibration_ids, counts, rule_reasons,
                  fragment_flags, effective_blocks, annotation_count):
    settings = config['build']
    with ExitStack() as stack:
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

    return stats
