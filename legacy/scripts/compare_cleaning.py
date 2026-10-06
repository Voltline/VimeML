"""Compare local cleaning rules with saved LLM judgments."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.clean import clean_block

INPUT = ROOT / "outputs/history/review/review.json"
OUTPUT = ROOT / "outputs/history/cleaning"


def fenced(text):
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}text\n{text}\n{fence}\n"


def main():
    documents = json.loads(INPUT.read_text(encoding="utf-8"))
    if any(document["status"] != "ok" for document in documents):
        raise ValueError("审核结果尚未全部成功。")

    records = []
    rule_counts = Counter()
    llm_counts = Counter()

    for document in documents:
        for block in document["blocks"]:
            result = clean_block(block["text"])
            llm_action = block["action"]
            comparable_action = (
                "review" if llm_action == "uncertain" else llm_action
            )

            records.append({
                "source": document["source"],
                "file": document["file"],
                "document_id": document["id"],
                "sample_index": document["sample_index"],
                "block_id": block["id"],
                "original_text": block["text"],
                "cleaned_text": result["text"],
                "rule_action": result["action"],
                "rule_reason": result["reason"],
                "changes": result["changes"],
                "flags": result["flags"],
                "llm_action": llm_action,
                "llm_reason": block["why_zh"],
                "disagreement": result["action"] != comparable_action,
            })
            rule_counts[result["action"]] += 1
            llm_counts[llm_action] += 1

    summary = {
        "documents": len(documents),
        "blocks": len(records),
        "rule_actions": dict(rule_counts),
        "llm_actions": dict(llm_counts),
        "changed_blocks": sum(bool(r["changes"]) for r in records),
        "disagreements": sum(r["disagreement"] for r in records),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "comparison.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    (OUTPUT / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 清洗对照报告",
        "",
        "展示有分歧、文本变更或疑点标记的块。计数单位为文本块。",
        "",
    ]
    for record in records:
        if not (
            record["disagreement"]
            or record["changes"]
            or record["flags"]
        ):
            continue

        lines.extend([
            f"## {record['source']} / "
            f"{record['document_id']} / {record['block_id']}",
            "",
            f"规则：{record['rule_action']} · {record['rule_reason']}",
            "",
            f"LLM：{record['llm_action']} · {record['llm_reason']}",
            "",
            "原文：",
            "",
            fenced(record["original_text"]),
        ])
        if record["original_text"] != record["cleaned_text"]:
            lines.extend([
                "清洗后：", "", fenced(record["cleaned_text"]),
            ])
        if record["changes"]:
            lines.extend([
                f"变更：{', '.join(record['changes'])}", "",
            ])
        if record["flags"]:
            lines.extend([
                f"疑点：{', '.join(record['flags'])}", "",
            ])

    (OUTPUT / "comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"报告：{OUTPUT / 'comparison.md'}")


if __name__ == "__main__":
    main()