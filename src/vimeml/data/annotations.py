"""Validated, locally approved corpus decisions; model replies alone are advisory."""

import hashlib
import json
import re
import sqlite3

from vimeml.data.segment import LIST_ITEM, split_sentences


def load_annotations(db, path):
    db.executescript("""
        CREATE TABLE annotations (
            id TEXT PRIMARY KEY, doc_hash TEXT NOT NULL, target TEXT NOT NULL UNIQUE, payload TEXT NOT NULL,
            matched INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX annotations_document ON annotations(doc_hash);
    """)
    if path is None:
        return 0
    count = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Annotation line {line_number}: expected an object.")
            if item.get("schema_version") != 1 or item.get("status") != "approved":
                raise ValueError(f"Annotation line {line_number}: only approved v1 records are accepted.")
            if item.get("approved_by") not in {"human", "assistant"}:
                raise ValueError("Annotations require an explicit human/assistant review.")
            if not isinstance(item.get("id"), str) or not item["id"] or not isinstance(item.get("doc_id"), str):
                raise ValueError("Annotation identity is missing.")
            if not isinstance(item.get("doc_hash"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["doc_hash"]):
                raise ValueError("Invalid annotation document hash.")
            if not isinstance(item.get("reason_zh"), str) or not item["reason_zh"].strip():
                raise ValueError("Annotation approval reason is missing.")
            kind, action = item.get("kind"), item.get("action")
            if kind == "boundary" and action in {"join", "separate"}:
                fields = ("left_block", "right_block", "left_text", "right_text")
                target = (item["doc_hash"], kind, item.get("left_block"), item.get("right_block"))
            elif kind == "block" and action in {"keep", "drop", "review"}:
                fields = ("block_id", "text")
                target = (item["doc_hash"], kind, item.get("block_id"))
            elif kind == "fragment" and action in {"keep", "drop", "review"}:
                fields = ("text", "text_hash")
                spans = item.get("cleaned_block_spans")
                if not isinstance(spans, list) or not spans:
                    raise ValueError("Fragment annotation requires cleaned-block spans.")
                for span in spans:
                    if not isinstance(span, dict) or not re.fullmatch(r"b[0-9]+", str(span.get("block_id", ""))) or type(span.get("start")) is not int or type(span.get("end")) is not int or not 0 <= span["start"] < span["end"]:
                        raise ValueError("Invalid fragment span.")
                if any(not isinstance(item.get(field), str) or not item[field] for field in fields):
                    raise ValueError("Fragment annotation snapshot is missing.")
                if hashlib.sha256(item["text"].encode("utf-8")).hexdigest() != item["text_hash"]:
                    raise ValueError("Fragment annotation text hash mismatch.")
                target = (item["doc_hash"], kind, item["text_hash"], json.dumps(spans, sort_keys=True))
            else:
                raise ValueError("Unknown annotation kind/action.")
            if any(not isinstance(item.get(field), str) or not item[field] for field in fields):
                raise ValueError("Annotation snapshot is missing.")
            try:
                db.execute("INSERT INTO annotations(id,doc_hash,target,payload) VALUES (?,?,?,?)", (
                    item["id"], item["doc_hash"], json.dumps(target, sort_keys=True), json.dumps(item, ensure_ascii=False),
                ))
            except sqlite3.IntegrityError:
                raise ValueError("Conflicting or duplicated annotation target/ID.") from None
            count += 1
    return count


def document_annotations(db, doc_hash, blocks):
    items = [json.loads(row[0]) for row in db.execute(
        "SELECT payload FROM annotations WHERE doc_hash=? ORDER BY id", (doc_hash,)
    )]
    by_id = {block["id"]: block for block in blocks}
    boundaries, fragments, block_ids = {}, [], {}
    for item in items:
        if item["kind"] == "fragment":
            fragments.append(item)
            continue
        fields = (("left_block", "left_text"), ("right_block", "right_text")) if item["kind"] == "boundary" else (("block_id", "text"),)
        for id_field, text_field in fields:
            block = by_id.get(item[id_field])
            if block is None or block["text"] != item[text_field]:
                raise ValueError(f"Annotation {item['id']}: cleaned block snapshot mismatch.")
        if item["kind"] == "block":
            block = by_id[item["block_id"]]
            if item["action"] == "keep" and block["action"] != "keep":
                raise ValueError("Rule-quarantined or dropped blocks cannot be rescued by a keep label.")
            block["action"] = item["action"]
            block_ids[item["block_id"]] = item["id"]
    # Check boundaries after all block labels have been applied.
    for item in items:
        if item["kind"] != "boundary":
            continue
        left, right = by_id[item["left_block"]], by_id[item["right_block"]]
        if right["line_index"] != left["line_index"] + 1:
            raise ValueError("Reviewed boundaries must be between adjacent original lines.")
        if item["action"] == "join" and (
            any(block["action"] != "keep" or not block["text"] or LIST_ITEM.match(block["text"]) for block in (left, right))
            or split_sentences(left["text"])[-1]["terminated"]
        ):
            raise ValueError(f"Annotation {item['id']}: join crosses a protected boundary.")
        boundaries[(item["left_block"], item["right_block"])] = item
    for item in items:
        if item["kind"] != "fragment":
            db.execute("UPDATE annotations SET matched=1 WHERE id=?", (item["id"],))
    return boundaries, fragments, block_ids


def match_fragment(db, labels, text, spans):
    matched = [item for item in labels if item["text"] == text and item["cleaned_block_spans"] == spans]
    if len(matched) > 1:
        raise ValueError("Multiple fragment annotations matched one output.")
    if not matched:
        return None
    item = matched[0]
    db.execute("UPDATE annotations SET matched=1 WHERE id=?", (item["id"],))
    return item
