"""Build a corpus preview from locally reviewed samples."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.clean import clean_block
from vimeml.data.segment import restore_linebreaks, split_sentences

INPUT = ROOT / "outputs/history/review/review.json"
OUTPUT = ROOT / "outputs/history/preview"


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def fenced(text):
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}text\n{text}\n{fence}\n"


def main():
    documents = json.loads(INPUT.read_text(encoding="utf-8"))
    if any(doc["status"] != "ok" for doc in documents):
        raise ValueError("审核结果尚未全部成功。")

    buckets = {
        "sentence_candidates": [],
        "fragments": [],
        "review_blocks": [],
        "dropped_blocks": [],
    }
    stats = {
        "documents": len(documents),
        "input_blocks": 0,
        "kept_blocks": 0,
        "paragraphs": 0,
        "join_events": 0,
    }
    preview = ["# 换行恢复预览", "", "展示所有发生拼接的段落。", ""]

    for document in documents:
        identity = {
            "source": document["source"],
            "doc_id": document["id"],
            "source_file": document["file"],
            "sample_index": document["sample_index"],
        }
        blocks = []
        original_blocks = {}

        for block in document["blocks"]:
            stats["input_blocks"] += 1
            original_blocks[block["id"]] = block["text"]
            cleaned = clean_block(block["text"])

            action = cleaned["action"]
            if action == "keep" and block["action"] != "keep":
                action = "review"

            blocks.append({
                "id": block["id"],
                "line_index": int(block["id"][1:]),
                "text": cleaned["text"],
                "action": action,
            })

            if action == "keep":
                stats["kept_blocks"] += 1
            else:
                destination = (
                    "dropped_blocks" if action == "drop"
                    else "review_blocks"
                )
                buckets[destination].append({
                    **identity,
                    "block_id": block["id"],
                    "original_text": block["text"],
                    "text": cleaned["text"],
                    "rule_action": cleaned["action"],
                    "rule_reason": cleaned["reason"],
                    "changes": cleaned["changes"],
                    "flags": cleaned["flags"],
                    "llm_action": block["action"],
                    "llm_reason": block["why_zh"],
                })

        paragraphs = restore_linebreaks(blocks)
        stats["paragraphs"] += len(paragraphs)

        for paragraph_index, paragraph in enumerate(paragraphs):
            stats["join_events"] += len(paragraph["joins"])
            sentences = split_sentences(paragraph["text"])

            if paragraph["joins"]:
                preview.extend([
                    f"## {document['source']} / "
                    f"{document['id']} / 段落 {paragraph_index}",
                    "",
                    "原始文本块：",
                    "",
                ])
                for part in paragraph["parts"]:
                    block_id = part["block_id"]
                    preview.extend([
                        f"**{block_id}**", "",
                        fenced(original_blocks[block_id]),
                    ])
                preview.extend([
                    "恢复后：", "", fenced(paragraph["text"]),
                    "拼接依据：", "",
                    fenced(json.dumps(
                        paragraph["joins"],
                        ensure_ascii=False,
                        indent=2,
                    )),
                    "分句结果：", "",
                    fenced(json.dumps(
                        sentences, ensure_ascii=False, indent=2
                    )),
                ])

            for sentence_index, sentence in enumerate(sentences):
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

                flags = sorted(set(
                    paragraph["flags"] + sentence["flags"]
                ))
                record = {
                    **identity,
                    "paragraph_index": paragraph_index,
                    "sentence_index": sentence_index,
                    "text": sentence["text"],
                    "text_hash": hashlib.sha256(
                        sentence["text"].encode("utf-8")
                    ).hexdigest(),
                    "terminated": sentence["terminated"],
                    "flags": flags,
                    "cleaned_block_spans": spans,
                    "paragraph_joins": paragraph["joins"],
                }

                destination = (
                    "sentence_candidates"
                    if sentence["terminated"] and not flags
                    else "fragments"
                )
                buckets[destination].append(record)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, records in buckets.items():
        write_jsonl(OUTPUT / f"{name}.jsonl", records)
        stats[name] = len(records)

    stats["candidate_characters"] = sum(
        len(record["text"])
        for record in buckets["sentence_candidates"]
    )
    write_json(OUTPUT / "stats.json", stats)
    (OUTPUT / "joins.md").write_text(
        "\n".join(preview), encoding="utf-8"
    )

    files = [
        INPUT,
        ROOT / "src/vimeml/data/clean.py",
        ROOT / "src/vimeml/data/segment.py",
        Path(__file__).resolve(),
    ]
    write_json(OUTPUT / "manifest.json", {
        "policy": "rules_and_llm_keep_v1",
        "python_version": sys.version,
        "span_offsets": "Unicode code points within cleaned blocks",
        "file_hashes": {
            path.relative_to(ROOT).as_posix():
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    })

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"拼接预览：{OUTPUT / 'joins.md'}")


if __name__ == "__main__":
    main()