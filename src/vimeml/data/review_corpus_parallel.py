"""Review prepared cases with bounded concurrency, shared quotas and resumable caches."""

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.build import file_sha
from vimeml.data.prepare_corpus_review import write_json

ENDPOINT = "https://models.sjtu.edu.cn/api/v1/chat/completions"
MODEL = "deepseek-chat"
ISSUE_TYPES = {"none", "web_noise", "incomplete_text", "natural_without_terminator",
               "bracket_convention", "wrong_join", "missed_join", "other"}
ACTIONS = {
    "review_block": {"keep", "drop", "review", "no_change"},
    "dropped_block": {"keep", "drop", "review", "no_change"},
    "retained_sentence": {"keep", "drop", "review", "no_change"},
    "fragment_candidate": {"keep", "drop", "review", "no_change"},
    "joined_boundary": {"join", "separate", "review", "no_change"},
    "boundary_candidate": {"join", "separate", "review", "no_change"},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {key: redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    return value


def parse_decisions(body, cases):
    choice = body["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Truncated output; rerun with a smaller --batch-size or larger --max-output-tokens.")
    content = choice["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Response content must be text.")
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = "\n".join(content.splitlines()[1:-1]).strip()
    rows = json.loads(content)
    if isinstance(rows, dict):
        rows = rows.get("results")
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON array or results array.")
    expected = {case["id"]: case for case in cases}
    decisions = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Invalid result ID.")
        key = row["id"]
        if key not in expected or key in decisions:
            raise ValueError("Unexpected or duplicate result ID.")
        if row.get("assessment") not in {"ok", "issue", "uncertain"} or row.get("issue_type") not in ISSUE_TYPES:
            raise ValueError("Invalid assessment or issue type.")
        if row.get("suggested_action") not in ACTIONS[expected[key]["kind"]]:
            raise ValueError("Suggested action is incompatible with case kind.")
        reason = row.get("reason_zh")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise ValueError("Missing or excessive reason.")
        decisions[key] = {name: row[name] for name in ("id", "assessment", "issue_type", "suggested_action")}
        decisions[key]["reason_zh"] = reason.strip()
    if set(decisions) != set(expected):
        raise ValueError("Response omitted some case IDs.")
    return decisions


def model_case(case):
    # Provenance and full source text remain local. Send only decision evidence.
    keys = ("id", "kind", "text", "block_id", "cleaned_block_spans", "flags", "terminated",
            "reason", "audit_reasons", "left_block", "right_block", "left_text", "right_text",
            "processing", "original_lines", "context_truncated")
    return {key: case[key] for key in keys if key in case}


def payload_for(cases, prompt, model, output_tokens):
    return {"model": model, "temperature": 0, "max_tokens": output_tokens,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": "逐个审核以下数据，每个 ID 恰好返回一次。\n"
                          + json.dumps([model_case(case) for case in cases], ensure_ascii=False)}]}


def token_reservation(payload):
    # Offline estimate, not the provider's tokenizer. Japanese often uses fewer
    # tokens than this UTF-8-byte/2 estimate; reserve the complete output allowance.
    return 256 + sum(math.ceil(len(message["content"].encode("utf-8")) / 2)
                     for message in payload["messages"]) + payload["max_tokens"]


def make_batches(cases, prompt, model, batch_size, output_tokens, max_tokens):
    batches, oversized, current = [], [], []
    for case in cases:
        if token_reservation(payload_for([case], prompt, model, output_tokens)) > max_tokens:
            oversized.append(case)
            continue
        proposed = current + [case]
        if current and (len(proposed) > batch_size or token_reservation(payload_for(proposed, prompt, model, output_tokens)) > max_tokens):
            batches.append(current)
            current = []
        current.append(case)
    if current:
        batches.append(current)
    return batches, oversized


class QuotaLimiter:
    """One rolling request/token quota shared by every worker and retry."""

    def __init__(self, rpm=8, tpm=80000, window=60.0):
        self.rpm, self.tpm, self.window = rpm, tpm, window
        self.lock, self.events = threading.Lock(), deque()
        self.next_start, self.cooldown_until = 0.0, 0.0

    def acquire(self, tokens, stop):
        if tokens > self.tpm:
            raise ValueError("Single request exceeds token quota.")
        while not stop.is_set():
            with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0]["start"] >= self.window:
                    self.events.popleft()
                wait = max(0.0, self.next_start - now, self.cooldown_until - now)
                if len(self.events) >= self.rpm or sum(event["tokens"] for event in self.events) + tokens > self.tpm:
                    wait = max(wait, self.events[0]["start"] + self.window - now)
                if wait <= 0:
                    ticket = {"start": now, "tokens": tokens}
                    self.events.append(ticket)
                    self.next_start = now + self.window / self.rpm
                    return ticket
            stop.wait(min(max(wait, 0.001), 0.5))
        return None

    def finish(self, ticket, usage):
        if not isinstance(usage, dict):
            return
        actual = usage.get("total_tokens")
        if type(actual) is int and actual >= 0:
            with self.lock:
                ticket["tokens"] = actual
                if actual > self.tpm:
                    self.cooldown_until = max(self.cooldown_until, time.monotonic() + self.window)

    def pause(self, seconds):
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)


def retry_after(headers, default):
    value = headers.get("Retry-After") if headers else None
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                pass
    return default


def send_request(endpoint, payload, api_key, timeout):
    request = Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read(8 * 1024 * 1024 + 1).decode("utf-8", errors="replace")
    except HTTPError as error:
        return error.code, dict(error.headers), error.read(8 * 1024 * 1024 + 1).decode("utf-8", errors="replace")


def load_materials(input_dir):
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("schema_version") != 1:
        raise ValueError("Review materials are not complete/compatible.")
    for filename, field in (("cases.json", "cases_sha256"), ("prompt.txt", "prompt_sha256")):
        if file_sha(input_dir / filename) != manifest[field]:
            raise ValueError(f"Review material changed: {filename}")
    cases = json.loads((input_dir / "cases.json").read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Cases must be an array.")
    ids = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"] or case.get("kind") not in ACTIONS:
            raise ValueError("Invalid case identity or kind.")
        ids.append(case["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs.")
    return manifest, cases, (input_dir / "prompt.txt").read_text(encoding="utf-8").strip()


def report(output, cases, decisions, failures, usage, preparation, request_counts):
    results = [{**case, **decisions[case["id"]]} for case in cases if case["id"] in decisions]
    issues = [case for case in results if case["assessment"] != "ok"]
    pending = [case["id"] for case in cases if case["id"] not in decisions]
    strata = defaultdict(Counter)
    for case in results:
        for layer in case["sampling_strata"]:
            strata[layer][case["assessment"]] += 1
    stats = {"total_cases": len(cases), "completed_cases": len(results), "pending_cases": len(pending),
             "assessments": dict(Counter(case["assessment"] for case in results)),
             "by_stratum": {key: dict(value) for key, value in sorted(strata.items())},
             "requests_this_run": dict(request_counts), "usage_this_run": usage,
             "quality_rate_note": preparation["quality_rate_note"],
             "advisory_only": True, "all_issues_human_confirmed": False}
    for filename, value in (("results.json", results), ("issues.json", issues), ("pending.json", pending),
                            ("failures.json", failures), ("stats.json", stats)):
        write_json(output / filename, value)
    lines = ["# Corpus 审核结果", "", f"完成 {len(results)}/{len(cases)}；待完成 {len(pending)}；需核对 {len(issues)}。", "",
             "模型判断为建议，未修改 corpus 或批准标注。分层样本不能用来直接推算全库坏数据率。", ""]
    for case in issues:
        lines.extend([f"## {case['id']} · {case['kind']} · {case['assessment']}", "",
                      f"来源：{case['source_file']}，row {case['row_index']}，{case['assigned_split']}", "",
                      f"类型：{case['issue_type']}；建议：{case['suggested_action']}", "", case["reason_zh"], ""])
        fence = "```"
        while fence in case["text"]:
            fence += "`"
        lines.extend([fence + "text", case["text"], fence, "", "原文上下文：", ""])
        for original in case["original_lines"]:
            lines.append(f"- {original['block_id']}：{original['text']}" + (" [截断]" if original["truncated"] else ""))
        lines.append("")
    if pending:
        lines.extend(["## 待完成", "", "查看 pending.json / failures.json；修复后以同一输出目录续跑。", ""])
    temporary = output / "issues.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output / "issues.md")
    return stats


def run(input_dir, output, api_key="", workers=3, batch_size=10, rpm=8, tpm=80000,
        max_batch_tokens=10000, max_output_tokens=2048, timeout=180, retries=2,
        dry_run=False, endpoint=ENDPOINT, model=MODEL, transport=send_request, limiter=None):
    input_dir, output = Path(input_dir).resolve(), Path(output).resolve()
    if output == input_dir or input_dir.is_relative_to(output):
        raise ValueError("Output must not replace or contain the input directory.")
    preparation, cases, prompt = load_materials(input_dir)
    base = {"schema_version": 1, "materials_manifest_sha256": file_sha(input_dir / "manifest.json"),
            "cases_sha256": preparation["cases_sha256"], "prompt_sha256": preparation["prompt_sha256"],
            "endpoint": endpoint, "model": model, "temperature": 0}
    case_keys = {case["id"]: digest({"base": base, "case": case}) for case in cases}
    decisions, failures, usage, counts = {}, [], [], Counter()
    if (output / "run-manifest.json").exists():
        previous = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
        if previous.get("identity") != base:
            raise ValueError("Output belongs to different materials, prompt, model or endpoint; use a new directory.")
        for case in cases:
            cache = output / "decisions" / f"{case_keys[case['id']]}.json"
            if cache.exists():
                saved = json.loads(cache.read_text(encoding="utf-8"))
                if saved.get("case_signature") != case_keys[case["id"]]:
                    raise ValueError("Corrupted decision cache identity.")
                body = {"choices": [{"message": {"content": json.dumps([saved["decision"]], ensure_ascii=False)}}]}
                decisions.update(parse_decisions(body, [case]))
                counts["cached_cases"] += 1
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is nonempty without a compatible run manifest.")
    remaining = [case for case in cases if case["id"] not in decisions]
    batches, oversized = make_batches(remaining, prompt, model, batch_size, max_output_tokens, max_batch_tokens)
    estimates = [token_reservation(payload_for(batch, prompt, model, max_output_tokens)) for batch in batches]
    preview = {"total_cases": len(cases), "cached_cases": len(decisions), "remaining_cases": len(remaining),
               "requests_planned": len(batches), "oversized_cases": [case["id"] for case in oversized],
               "batch_case_counts": dict(Counter(len(batch) for batch in batches)),
               "estimated_reserved_tokens": sum(estimates), "quota_lower_bound_minutes": max(len(batches) / rpm, sum(estimates) / tpm),
               "estimate_note": "Offline UTF-8-based token estimate; actual provider usage reconciled after each response.",
               "network_requests_sent": 0 if dry_run else None}
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return preview
    if batches and not api_key:
        raise ValueError("Set SJTU_API_KEY before starting review; --dry-run needs no key.")
    for directory in (output, output / "responses", output / "cache", output / "decisions"):
        directory.mkdir(parents=True, exist_ok=True)
    write_json(output / "run-manifest.json", {"identity": base,
               "runtime": {"workers": workers, "batch_size": batch_size, "rpm": rpm, "tpm": tpm,
                           "max_batch_tokens": max_batch_tokens, "max_output_tokens": max_output_tokens, "retries": retries},
               "review_code_sha256": file_sha(Path(__file__)), "advisory_only": True})
    for case in oversized:
        failures.append({"ids": [case["id"]], "error": "Case exceeds --max-batch-tokens; increase budget or prepare a bounded context; no request sent."})
    limiter = limiter or QuotaLimiter(rpm, tpm)
    stop = threading.Event()

    def review_batch(batch):
        payload = payload_for(batch, prompt, model, max_output_tokens)
        signature = digest({"endpoint": endpoint, "payload": payload})
        cache = output / "cache" / f"{signature}.json"
        if cache.exists():
            saved = json.loads(cache.read_text(encoding="utf-8"))
            return parse_decisions(saved["body"], batch), [], [], Counter({"cached_batches": 1})
        local_failures, local_usage, local_counts = [], [], Counter()
        for attempt in range(retries + 1):
            ticket = limiter.acquire(token_reservation(payload), stop)
            if ticket is None:
                return {}, local_failures, local_usage, local_counts
            local_counts["sent"] += 1
            record = {"request": payload, "request_signature": signature, "attempt": attempt + 1}
            filename = output / "responses" / f"{time.time_ns()}-{signature[:12]}-{attempt + 1}.json"
            status, headers, raw = None, {}, ""
            try:
                status, headers, raw = transport(endpoint, payload, api_key, timeout)
                record.update(http_status=status, raw_response=raw)
                write_json(filename, redact(record, api_key))
                if len(raw.encode("utf-8")) > 8 * 1024 * 1024:
                    raise ValueError("Response exceeds local size limit.")
                if status in {401, 403}:
                    stop.set()
                    raise ValueError(f"HTTP {status}: authentication failed; remaining requests stopped.")
                if status == 429:
                    limiter.pause(retry_after(headers, min(60 * (attempt + 1), 180)))
                if status < 200 or status >= 300:
                    raise ValueError(f"HTTP {status}")
                body = redact(json.loads(raw), api_key)
                limiter.finish(ticket, body.get("usage"))
                local_usage.append({"response_file": filename.name, "usage": body.get("usage")})
                parsed = parse_decisions(body, batch)
                write_json(cache, redact({"body": body, "response_file": filename.name}, api_key))
                local_counts["successful"] += 1
                return parsed, local_failures, local_usage, local_counts
            except Exception as error:
                message = redact(f"{type(error).__name__}: {error}", api_key)
                if not filename.exists():
                    write_json(filename, redact({**record, "error": message, "http_status": status}, api_key))
                local_failures.append({"ids": [case["id"] for case in batch], "attempt": attempt + 1,
                                       "error": message, "response_file": filename.name})
                local_counts["failed_attempts"] += 1
                if stop.is_set() or (status is not None and 400 <= status < 500 and status not in {408, 429}):
                    break
                if attempt < retries:
                    stop.wait(min(2 ** (attempt + 1), 30))
        return {}, local_failures, local_usage, local_counts

    stats = report(output, cases, decisions, failures, usage, preparation, counts)
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    try:
        for batch in batches:
            futures[executor.submit(review_batch, batch)] = batch
        for future in as_completed(futures):
            parsed, batch_failures, batch_usage, batch_counts = future.result()
            failures.extend(batch_failures)
            usage.extend(batch_usage)
            counts.update(batch_counts)
            decisions.update(parsed)
            for case_id, decision in parsed.items():
                write_json(output / "decisions" / f"{case_keys[case_id]}.json",
                           {"case_signature": case_keys[case_id], "decision": redact(decision, api_key)})
            stats = report(output, cases, decisions, failures, usage, preparation, counts)
            print(f"Completed {stats['completed_cases']}/{stats['total_cases']}; pending {stats['pending_cases']}", flush=True)
    except BaseException:
        stop.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
    print(f"Report: {output / 'issues.md'}", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/corpus-review-v1")
    parser.add_argument("--output-dir", type=Path, help="Default: INPUT/review")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--rpm", type=int, default=8)
    parser.add_argument("--tpm", type=int, default=80000)
    parser.add_argument("--max-batch-tokens", type=int, default=10000)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dry-run", action="store_true", help="Validate/count batches without writing or making network requests.")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.rpm <= 10 or not 1 <= args.tpm <= 100000:
        parser.error("workers must be 1..8; rpm 1..10; tpm 1..100000.")
    if min(args.batch_size, args.timeout, args.max_output_tokens) < 1 or not args.max_output_tokens < args.max_batch_tokens <= args.tpm or not 0 <= args.retries <= 5:
        parser.error("Invalid batch, output budget, timeout or retries.")
    api_key = "" if args.dry_run else os.environ.get("SJTU_API_KEY", "").strip()
    result = run(args.input_dir, args.output_dir or args.input_dir / "review", api_key,
                 args.workers, args.batch_size, args.rpm, args.tpm, args.max_batch_tokens, args.max_output_tokens,
                 args.timeout, args.retries, args.dry_run, model=args.model)
    if not args.dry_run and result["pending_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
