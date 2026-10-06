"""Review local corpus audit cases through the SJTU API, with resumable batches."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://models.sjtu.edu.cn/api/v1/chat/completions"
MODEL = "deepseek-chat"
ASSESSMENTS = {"ok", "issue", "uncertain"}
ISSUE_TYPES = {
    "none", "web_noise", "incomplete_text", "natural_without_terminator",
    "bracket_convention", "wrong_join", "missed_join", "other",
}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_decisions(body, expected_ids):
    choice = body["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("输出被截断；可减小 --batch-size 后重跑。")
    content = choice["message"]["content"].strip()
    if content.startswith("```") and content.endswith("```"):
        content = "\n".join(content.splitlines()[1:-1]).strip()
    rows = json.loads(content)
    if isinstance(rows, dict):
        rows = rows.get("results")
    if not isinstance(rows, list):
        raise ValueError("回复必须是 JSON 数组，或包含 results 数组的对象。")
    decisions = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("判断条目不是对象。")
        case_id = row.get("id")
        if not isinstance(case_id, str) or case_id in decisions:
            raise ValueError("样本 ID 无效或重复。")
        if row.get("assessment") not in ASSESSMENTS:
            raise ValueError("未知 assessment。")
        if row.get("issue_type") not in ISSUE_TYPES:
            raise ValueError("未知 issue_type。")
        reason = row.get("reason_zh")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("缺少中文理由。")
        decisions[case_id] = {
            "id": case_id, "assessment": row["assessment"],
            "issue_type": row["issue_type"], "reason_zh": reason.strip(),
        }
    if set(decisions) != set(expected_ids):
        raise ValueError("样本 ID 有遗漏或新增。")
    return decisions


def request_review(payload, api_key, response_dir, signature, timeout):
    request = Request(
        ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    filename = response_dir / f"{time.time_ns()}-{signature[:12]}.json"
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        write_json(filename, {
            "request": payload, "http_status": error.code,
            "raw_response": raw.replace(api_key, "[REDACTED]"),
        })
        raise
    # Save before JSON parsing, including malformed or truncated responses.
    raw = raw.replace(api_key, "[REDACTED]")
    write_json(filename, {"request": payload, "http_status": status, "raw_response": raw})
    return json.loads(raw)


def write_reports(output, cases, decisions, failures, usage):
    reviewed = [{**case, **decisions[case["id"]]} for case in cases if case["id"] in decisions]
    issues = [row for row in reviewed if row["assessment"] != "ok"]
    write_json(output / "results.json", reviewed)
    write_json(output / "issues.json", issues)
    write_json(output / "failures.json", failures)
    stats = {
        "total_cases": len(cases), "completed_cases": len(reviewed),
        "pending_cases": len(cases) - len(reviewed),
        "assessments": {
            label: sum(row["assessment"] == label for row in reviewed)
            for label in sorted(ASSESSMENTS)
        },
        "successful_request_usage_this_run": usage,
    }
    write_json(output / "stats.json", stats)
    lines = ["# Corpus 抽查中需要检查的条目", ""]
    for row in issues:
        lines.extend([
            f"## {row['id']} · {row['kind']} · {row['assessment']}", "",
            f"问题类型：{row['issue_type']}", "", f"理由：{row['reason_zh']}", "",
        ])
        fence = "```"
        while fence in row["text"]:
            fence += "`"
        lines.extend([fence + "text", row["text"], fence, "", "原始行：", ""])
        for original in row["original_lines"]:
            lines.append(f"- {original['block_id']}：{original['text']}")
        lines.append("")
    if not issues:
        lines.extend(["已完成的回复中没有 issue / uncertain；请同时检查完成数量。", ""])
    (output / "issues.md").write_text("\n".join(lines), encoding="utf-8")
    return stats


def run(input_dir, output, api_key, batch_size=10, timeout=180):
    cases = json.loads((input_dir / "cases.json").read_text(encoding="utf-8"))
    packet = (input_dir / "deepseek-packet.md").read_text(encoding="utf-8")
    prompt, separator, data = packet.partition("\n```json\n")
    if not separator or json.loads(data.rsplit("\n```", 1)[0]) != cases:
        raise ValueError("cases.json 与抽查材料不一致，请重新生成材料。")
    if not isinstance(cases, list) or not cases:
        raise ValueError("没有抽查样本。")
    ids = [case["id"] for case in cases]
    if any(not isinstance(key, str) for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("输入样本 ID 无效或重复。")
    for directory in (output, output / "responses", output / "cache"):
        directory.mkdir(parents=True, exist_ok=True)

    decisions, failures, usage = {}, [], []
    last_request = None
    batches = [cases[start:start + batch_size] for start in range(0, len(cases), batch_size)]
    for index, batch in enumerate(batches, 1):
        expected_ids = [case["id"] for case in batch]
        payload = {
            "model": MODEL, "temperature": 0, "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": prompt.strip()},
                {"role": "user", "content": "请只审核以下样本，每个 ID 恰好返回一次。\n"
                 + json.dumps(batch, ensure_ascii=False)},
            ],
        }
        signature = hashlib.sha256(json.dumps(
            {"endpoint": ENDPOINT, "payload": payload}, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")).hexdigest()
        cache_path = output / "cache" / f"{signature}.json"
        stop = False
        try:
            if cache_path.exists():
                body = json.loads(cache_path.read_text(encoding="utf-8"))
                parsed = parse_decisions(body, expected_ids)
                status = "cached"
            else:
                if last_request is not None:
                    time.sleep(max(0, last_request + 7.5 - time.monotonic()))
                print(f"[{index}/{len(batches)}] 请求 {', '.join(expected_ids)}", flush=True)
                last_request = time.monotonic()
                body = request_review(payload, api_key, output / "responses", signature, timeout)
                parsed = parse_decisions(body, expected_ids)
                write_json(cache_path, body)
                usage.append(body.get("usage"))
                status = "ok"
            decisions.update(parsed)
            print(f"[{index}/{len(batches)}] {status}", flush=True)
        except HTTPError as error:
            failures.append({"ids": expected_ids, "error": f"HTTP {error.code}"})
            print(f"[{index}/{len(batches)}] HTTP {error.code}", flush=True)
            if error.code in {401, 403, 429}:
                stop = True
                print("鉴权或限流错误，停止发送后续请求；解决后重跑即可。", flush=True)
        except Exception as error:
            message = str(error).replace(api_key, "[REDACTED]")
            failures.append({"ids": expected_ids, "error": f"{type(error).__name__}: {message}"})
            print(f"[{index}/{len(batches)}] 失败，详情已保存到 failures.json", flush=True)
        stats = write_reports(output, cases, decisions, failures, usage)
        if stop:
            break
    print(f"完成：{stats['completed_cases']} 条，待完成：{stats['pending_cases']} 条")
    print(f"报告：{output / 'issues.md'}")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/history/corpus-smoke-audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/history/corpus-smoke-audit/review")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.batch_size < 1 or args.timeout < 1:
        parser.error("batch-size 和 timeout 必须大于 0。")
    api_key = os.environ.get("SJTU_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("请先在当前终端设置 SJTU_API_KEY 环境变量。")
    stats = run(args.input_dir.resolve(), args.output_dir.resolve(), api_key, args.batch_size, args.timeout)
    if stats["pending_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
