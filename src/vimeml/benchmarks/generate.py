"""Resumable DeepSeek generation and label-blind verification of IME cases."""
import argparse
import json
import math
import os
import random
import re
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vimeml.data.build import file_sha
from vimeml.review.prepare import write_json
from vimeml.review.run import ENDPOINT, MODEL, digest, redact, retry_after, send_request, token_reservation
from vimeml.review.multi_key import AccountPool, output_lock

ROOT = Path(__file__).resolve().parents[3]
VERSION = "synthetic_ime_v1"
CATEGORIES = {
    "日常": "家族や友人との日常会話", "ビジネス": "メール、会議、職場",
    "テクノロジー": "アプリ、デバイス、サービス", "感情": "気持ちの表現",
    "天気": "季節、気温、予定", "食事": "料理、レストラン、食材",
    "旅行": "交通、観光、宿泊", "挨拶": "お礼、お詫び、依頼"}
JA = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
PROMPT = """你为日语输入法制作小型合成评测数据，只输出JSON数组，不要Markdown。
每条是当前一句中尚未输入完的前缀 context，加上后续词/短语。上下文必须提供充分信息，
使正确候选明确自然，其他候选在这个具体上下文中明显不合适。不要依赖外部事实、
新闻、品牌特性、地理距离、隐含场景或常识问答；不要仅用生僻字/乱码/语法错误充当负例。
避免只靠情绪词、天气词和常见套话匹配；同时包含搭配、动作对象、时间、因果、肯否定。
context为5~50个字符的一句内前缀，不含换行，不以句末标点结束；不重复样例。
不同条目尽量改变句式、主题和词汇。提供中文reason_zh解释为何其他候选不合上下文。
数据用于评测而非训练。不要输出指令文字，不要把目标答案或其他候选复制到context中。
"""
VERIFY_PROMPT = """你独立审核日语输入法评测样本。只输出JSON数组，每个id恰好一次。
输入只有当前句内context和随机排序的候选。你不知道预设答案，不要假设一定有唯一正确项。
判断 context+候选 的语法、搭配、语义和时态。只用给定上下文，不要借外部事实裁定。
候选可以是可继续的短词/短语；未加句号不等于错误。若多个候选合理，请全部列出。
phrase_contrast任务的两个候选单独看都须语法通顺、表达自然；若某项仅因语法错误、
乱码或完全不自然的表达而失败，则status=invalid，不作为有挑战的语义负例。
若上下文不足以区分、存在合理的其他解释，status=ambiguous；没有自然候选或样本本身
损坏则status=invalid；只有唯一明确自然的候选且其余明显不合适才status=clear。
homophone任务额外检查所有候选在此处能否读成给定reading。不一致时status=invalid。
每条返回 {"id":"原ID","status":"clear|ambiguous|invalid",
"acceptable_ids":["A"],"confidence":"high|medium|low","reason_zh":"简短中文理由"}。
即使是clear也要如实评估confidence。不要改写输入，不要参考任何所谓预设答案。
"""


def json_content(body):
    choice = body["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Truncated model response; lower batch size in a new run configuration.")
    text = choice["message"]["content"]
    if not isinstance(text, str):
        raise ValueError("Expected text response.")
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    rows = json.loads(text)
    if isinstance(rows, dict):
        rows = rows.get("items", rows.get("results"))
    if not isinstance(rows, list):
        raise ValueError("Expected JSON array.")
    return rows


def clean_text(value, minimum, maximum):
    if not isinstance(value, str) or value != value.strip() or not minimum <= len(value) <= maximum:
        raise ValueError("Invalid text type, whitespace or length.")
    if not JA.search(value) or any(unicodedata.category(char).startswith("C") for char in value):
        raise ValueError("Non-Japanese or control characters in text.")
    if "\ufffd" in value or "http" in value.lower():
        raise ValueError("Replacement character or URL in text.")
    return value


def make_jobs(config, pools):
    jobs = []
    for category, description in CATEGORIES.items():
        for start in range(0, config["per_category"], config["batch_size"]):
            jobs.append({"task": "phrase_contrast", "category": category, "description": description,
                "count": min(config["batch_size"], config["per_category"] - start), "start": start})
    for group in pools["groups"]:
        for target in group["candidates"]:
            for start in range(0, config["contexts_per_target"], config["batch_size"]):
                jobs.append({"task": "homophone", "group": group["id"], "category": "同音語",
                    "reading": group["reading"], "candidates": group["candidates"], "target": target,
                    "count": min(config["batch_size"], config["contexts_per_target"] - start), "start": start})
    return [{**job, "id": digest({"version": VERSION, "seed": config["seed"], "job": job})} for job in jobs]


def generation_payload(job, config):
    if job["task"] == "phrase_contrast":
        instruction = (f"类别：{job['category']}（{job['description']}），生成恰好{job['count']}条。\n"
            '每条返回 {"context":"...","positive":"...","negative":"...","reason_zh":"..."}。\n'
            "positive/negative均为2~30字符，相差不超过3字符且长度比在0.8~1.25；"
            "两者单独看都语法通顺、表达自然、句式类似，只有positive符合显式上下文。"
            "negative在另一个合理上下文应可成立，是有挑战的近似负例。")
    else:
        instruction = (f"读音：{job['reading']}；候选：{json.dumps(job['candidates'], ensure_ascii=False)}；"
            f"目标词：{job['target']}。生成恰好{job['count']}个不同context，使目标词唯一合适。\n"
            '每条仅返回 {"context":"...","reason_zh":"..."}。目标词由本地程序加入，'
            "不要改写候选。context中不得出现任何候选原文，其他候选在此上下文必须不合适。")
    return {"model": config["model"], "temperature": .9, "max_tokens": config["output_tokens"],
        "messages": [{"role": "system", "content": PROMPT},
            {"role": "user", "content": instruction + f"\n本批独立编号：{job['id'][:16]}，避免沿用之前的套话。"}]}


def parse_generated(body, job):
    rows = json_content(body)
    if len(rows) != job["count"]:
        raise ValueError("Response count differs from fixed job count; retry this slot.")
    items, seen = [], set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("Generated item must be an object.")
        context = clean_text(raw.get("context"), 5, 50)
        if context.endswith(tuple("。！？!?")) or normalized(context) in seen:
            raise ValueError("Completed or duplicate context within batch.")
        reason = raw.get("reason_zh")
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 500:
            raise ValueError("Missing generation rationale.")
        seen.add(normalized(context))
        if job["task"] == "phrase_contrast":
            positive = clean_text(raw.get("positive"), 2, 30)
            negative = clean_text(raw.get("negative"), 2, 30)
            if normalized(positive) == normalized(negative):
                raise ValueError("Equal positive and negative.")
            if abs(len(positive) - len(negative)) > 3 or not .8 <= len(negative) / len(positive) <= 1.25:
                raise ValueError("Continuation lengths do not match closely enough.")
            candidates, gold = [positive, negative], positive
        else:
            candidates, gold = job["candidates"], job["target"]
        if any(candidate in context for candidate in candidates):
            raise ValueError("Candidate text leaked into context.")
        items.append({"context": context, "candidates": candidates, "acceptable": [gold],
            "reason_zh": reason.strip()})
    return items


def normalized(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def canonical_cases(jobs, generated, config):
    result, duplicate = [], []
    seen = set()
    for job in jobs:
        for index, item in enumerate(generated.get(job["id"], [])):
            case_id = digest({"job": job["id"], "index": index, "item": item})
            # Catch duplicate prefixes even when labels/candidates were changed.
            key = normalized(item["context"])
            if key in seen:
                duplicate.append({"id": case_id, "context": item["context"], "reason": "duplicate_context_nfkc"})
                continue
            seen.add(key)
            candidates = list(item["candidates"])
            group = job.get("group", case_id)
            # Shared order within a homophone contrast group; independent of
            # correct target. Pair positions are likewise independent of gold.
            random.Random(digest({"seed": config["seed"], "group": group})).shuffle(candidates)
            result.append({"id": case_id, "task": job["task"], "category": job["category"],
                "context": item["context"], "candidates": candidates, "acceptable": item["acceptable"],
                "group": group, "reading": job.get("reading", ""), "generation_reason_zh": item["reason_zh"],
                "provenance": {"kind": "llm_synthetic", "job_id": job["id"], "generator": config["model"]}})
    return result, duplicate


def verification_payload(cases, config):
    evidence = []
    for index, case in enumerate(cases):
        # No acceptable/gold/positive/negative or generation rationale on wire.
        evidence.append({"id": str(index), "task": case["task"], "context": case["context"],
            "reading": case["reading"], "candidates": [{"id": chr(65 + i), "text": text}
                for i, text in enumerate(case["candidates"])]})
    return {"model": config["judge_model"], "temperature": 0, "max_tokens": config["output_tokens"],
        "messages": [{"role": "system", "content": VERIFY_PROMPT},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}]}


def parse_verified(body, cases):
    rows = json_content(body)
    if len(rows) != len(cases):
        raise ValueError("Missing verification IDs.")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("id")) is not str or not row["id"].isdigit():
            raise ValueError("Invalid verification ID.")
        index = int(row["id"])
        if str(index) != row["id"] or not 0 <= index < len(cases) or index in result:
            raise ValueError("Unexpected or duplicated verification ID.")
        status, confidence, accepted = row.get("status"), row.get("confidence"), row.get("acceptable_ids")
        if status not in {"clear", "ambiguous", "invalid"} or confidence not in {"high", "medium", "low"}:
            raise ValueError("Invalid verification assessment.")
        allowed = {chr(65 + i): candidate for i, candidate in enumerate(cases[index]["candidates"])}
        if not isinstance(accepted, list) or any(not isinstance(v, str) or v not in allowed for v in accepted) or len(set(accepted)) != len(accepted):
            raise ValueError("Invalid verified candidate IDs.")
        if status == "clear" and len(accepted) != 1:
            raise ValueError("Clear assessment requires exactly one accepted candidate.")
        reason = row.get("reason_zh")
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 1000:
            raise ValueError("Missing verification reason.")
        result[index] = {"status": status, "confidence": confidence,
            "acceptable": [allowed[value] for value in accepted], "reason_zh": reason.strip()}
    return [result[index] for index in range(len(cases))]


class Requests:
    """One quota pool for generation, verification and retries; cache by request."""
    def __init__(self, output, config, pool, timeout, retries, transport=send_request):
        self.output, self.config, self.pool = output, config, pool
        self.timeout, self.retries, self.transport = timeout, retries, transport
        self.stop = threading.Event()
        self.started = 0
        self.lock = threading.Lock()

    def signature(self, payload):
        return digest({"version": VERSION, "endpoint": self.config["endpoint"], "payload": payload})

    def cached(self, payload, parser):
        path = self.output / "cache" / f"{self.signature(payload)}.json"
        if not path.exists():
            return None
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("signature") != self.signature(payload) or saved.get("request") != payload:
            raise ValueError("Cache signature or payload mismatch.")
        return parser(saved["body"])

    def call(self, payload, parser):
        cached = self.cached(payload, parser)
        if cached is not None:
            return cached
        if self.pool is None:
            raise ValueError("Set SJTU_API_KEY in your local shell; do not paste the key into chat or files.")
        signature = self.signature(payload)
        last = None
        for attempt in range(self.retries + 1):
            ticket = self.pool.acquire(token_reservation(payload), self.stop)
            if ticket is None:
                raise RuntimeError("Requests stopped or all accounts disabled.")
            status = None
            raw = ""
            try:
                with self.lock:
                    self.started += 1
                status, headers, raw = self.transport(self.config["endpoint"], payload, self.pool.key(ticket), self.timeout)
                # Every reply, including malformed JSON/429, is retained. No
                # authorization headers; redact all configured keys everywhere.
                write_json(self.output / "responses" / f"{signature}-{time.time_ns()}.json", redact({
                    "signature": signature, "attempt": attempt + 1, "request": payload,
                    "status": status, "raw_response": raw, "account": self.pool.label(ticket)}, self.pool.secrets))
                if status == 429:
                    self.pool.pause(ticket, retry_after(headers, min(60 * (attempt + 1), 180)))
                if status in {401, 403}:
                    self.pool.disable(ticket, status)
                if not 200 <= status < 300:
                    raise ValueError(f"HTTP {status}")
                body = json.loads(raw)
                self.pool.finish(ticket, body.get("usage"))
                # Live derived records must use the same redacted content as
                # cache replay, including an accidentally echoed secret.
                body = redact(body, self.pool.secrets)
                parsed = parser(body)
                write_json(self.output / "cache" / f"{signature}.json", redact({
                    "signature": signature, "request": payload, "body": body}, self.pool.secrets))
                self.pool.success(ticket)
                return parsed
            except Exception as error:
                last = redact(f"{type(error).__name__}: {error}", self.pool.secrets)
                write_json(self.output / "failures" / f"{signature}-{time.time_ns()}.json", {
                    "signature": signature, "attempt": attempt + 1, "error": last})
                if status is not None and 400 <= status < 500 and status not in {401, 403, 408, 429}:
                    break
            finally:
                self.pool.release(ticket)
            if attempt < self.retries:
                self.stop.wait(min(2 ** (attempt + 1), 15))
        raise RuntimeError(last or "Request failed.")


def execute(requests, work, workers):
    results, failures = {}, []
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(requests.call, payload, parser): key for key, payload, parser in work}
        done = 0
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as error:
                failures.append({"id": key, "error": str(error)})
            done += 1
            print(f"Requests completed {done}/{len(work)}; failed {len(failures)}", flush=True)
    except KeyboardInterrupt:
        requests.stop.set()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if failures:
        write_json(requests.output / "pending.json", failures)
        raise RuntimeError(f"{len(failures)} requests pending; successful caches retained. Rerun the same command.")
    write_json(requests.output / "pending.json", [])
    return results


def jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def finalize(cases, decisions):
    if len(cases) != len(decisions):
        raise ValueError("Verification count differs from draft count.")
    accepted, rejected = [], []
    for case, decision in zip(cases, decisions):
        if decision["status"] == "clear" and decision["confidence"] == "high" and set(decision["acceptable"]) == set(case["acceptable"]):
            accepted.append({**case, "quality": decision})
        else:
            rejected.append({**case, "quality": decision, "quarantine_reason": "ambiguous_invalid_low_confidence_or_label_disagreement"})
    return accepted, rejected


def audit_sample(accepted, rejected):
    strata = {}
    for case in accepted:
        key = case["task"] + ":" + (case["group"] if case["task"] == "homophone" else case["category"])
        strata.setdefault(key, []).append(case)
    chosen = []
    for key, cases in sorted(strata.items()):
        count = 2 if key.startswith("homophone:") else 5
        chosen.extend(sorted(cases, key=lambda case: digest({"audit": case["id"]}))[:count])
    return {"selection": "Fixed hash per task/category or homophone group, independent of local LM scores",
        "accepted": chosen, "quarantined": sorted(rejected, key=lambda case: digest({"audit": case["id"]}))[:20]}


def run(output, config, pools, stage="all", api_keys=(), workers_per_key=3, rpm=8, tpm=80000,
        timeout=180, retries=2, dry_run=False, transport=send_request, window=60, account_count=None):
    output = Path(output).resolve()
    jobs = make_jobs(config, pools)
    identity = digest({"version": VERSION, "config": config, "generation_prompt": PROMPT, "verification_prompt": VERIFY_PROMPT})
    manifest_path = output / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if old and old["identity"] != identity:
        raise ValueError("Benchmark configuration/prompts changed. Use a new output directory.")
    probe = Requests(output, config, None, timeout, retries, transport)
    generated, missing = {}, []
    for job in jobs:
        payload = generation_payload(job, config)
        cached = probe.cached(payload, lambda body, job=job: parse_generated(body, job))
        if cached is None:
            missing.append(job)
        else:
            generated[job["id"]] = cached
    target = sum(job["count"] for job in jobs)
    verification_upper = math.ceil(target / config["verify_batch_size"])
    # Dry-run does not touch filesystem or need credentials.
    accounts = account_count or max(1, len(api_keys))
    preview = {"target_generated_cases": target, "phrase_pairs": config["per_category"] * len(CATEGORIES),
        "homophone_cases": target - config["per_category"] * len(CATEGORIES), "generation_requests_total": len(jobs),
        "generation_requests_pending": len(missing), "verification_requests_upper_bound": verification_upper,
        "generator": config["model"], "verifier": config["judge_model"], "accounts": accounts,
        "rpm_per_account": rpm, "network_requests_sent": 0,
        "request_quota_estimate_minutes_full_run": max(0, len(jobs) + verification_upper - 1) / (rpm * accounts),
        "note": "Quota-based estimate excludes latency/retries; accepted count can shrink after dedup/verification."}
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return preview
    if missing and stage == "verify":
        raise ValueError("Generation incomplete; run --stage generate or --stage all first.")
    if missing and not api_keys:
        raise ValueError("SJTU_API_KEY unavailable in this process. Set it in your local shell and rerun.")
    pool = AccountPool(api_keys, workers_per_key=workers_per_key, rpm=rpm, tpm=tpm, window=window) if api_keys else None
    request_runner = Requests(output, config, pool, timeout, retries, transport)
    workers = max(1, len(api_keys)) * workers_per_key
    for folder in ("cache", "responses", "failures"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    manifest = {"format": VERSION, "identity": identity, "config": config, "status": "generating",
        "generation_prompt_sha256": digest(PROMPT), "verification_prompt_sha256": digest(VERIFY_PROMPT),
        "note": "Synthetic AI-generated/verified evaluation only; training overlap unknown; not AzooKey outputs or open-ended suggestion accuracy."}
    write_json(output / "manifest.json", manifest)
    if stage in {"all", "generate"}:
        work = [(job["id"], generation_payload(job, config), lambda body, job=job: parse_generated(body, job)) for job in missing]
        generated.update(execute(request_runner, work, workers))
    cases, duplicates = canonical_cases(jobs, generated, config)
    jsonl(output / "draft.jsonl", cases)
    write_json(output / "duplicates.json", duplicates)
    if stage == "generate":
        # Do not downgrade an already completed benchmark on a cached rerun.
        if old and old.get("status") == "complete":
            write_json(manifest_path, old)
        else:
            write_json(manifest_path, {**manifest, "status": "awaiting_verification", "draft_cases": len(cases)})
        return {"draft_cases": len(cases), "status": "awaiting_verification"}
    batches = [cases[i:i + config["verify_batch_size"]] for i in range(0, len(cases), config["verify_batch_size"])]
    write_json(manifest_path, {**manifest, "status": "verifying", "draft_cases": len(cases)})
    checked = execute(request_runner, [(str(i), verification_payload(batch, config),
        lambda body, batch=batch: parse_verified(body, batch)) for i, batch in enumerate(batches)], workers)
    decisions = [decision for i in range(len(batches)) for decision in checked[str(i)]]
    accepted, rejected = finalize(cases, decisions)
    jsonl(output / "quarantine.jsonl", rejected)
    jsonl(output / "benchmark.jsonl", accepted)
    write_json(output / "verification.json", decisions)
    write_json(output / "manual-audit.json", audit_sample(accepted, rejected))
    stats = {"target_generated_cases": target, "unique_draft_cases": len(cases), "duplicate_contexts": len(duplicates),
        "accepted_cases": len(accepted), "quarantined_cases": len(rejected),
        "accepted_by_task": dict(Counter(case["task"] for case in accepted)),
        "accepted_by_category": dict(Counter(case["category"] for case in accepted)),
        "gold_position_counts": dict(Counter(str(case["candidates"].index(case["acceptable"][0])) for case in accepted)),
        "network_requests_sent_this_run": request_runner.started,
        "accounts_this_run": pool.snapshot() if pool else [],
        "generator_equals_verifier": config["model"] == config["judge_model"]}
    write_json(output / "stats.json", stats)
    write_json(manifest_path, {**manifest, "status": "complete", "benchmark_sha256": file_sha(output / "benchmark.jsonl"),
        "stats": stats, "manual_review_done": False})
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return stats


def load_pools(path):
    pools = json.loads(Path(path).read_text(encoding="utf-8"))
    seen = set()
    for group in pools["groups"]:
        if group["id"] in seen or not re.fullmatch(r"[ぁ-ゖー]+", group["reading"]):
            raise ValueError("Invalid homophone group or reading.")
        seen.add(group["id"])
        if not 2 <= len(group["candidates"]) <= 8 or len(set(group["candidates"])) != len(group["candidates"]):
            raise ValueError("Invalid homophone pool.")
        for text in group["candidates"]:
            clean_text(text, 1, 15)
    return pools


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/ime-synthetic-v1")
    parser.add_argument("--pools", type=Path, default=ROOT / "configs/ime-homophones.json")
    parser.add_argument("--per-category", type=int, default=50)
    parser.add_argument("--contexts-per-target", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--verify-batch-size", type=int, default=20)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--judge-model", default="qwen", help="Blind verifier; separate model by default.")
    parser.add_argument("--key-envs", default="SJTU_API_KEY", help="Comma-separated env names for independent accounts.")
    parser.add_argument("--workers-per-key", type=int, default=3)
    parser.add_argument("--rpm", type=int, default=8)
    parser.add_argument("--tpm", type=int, default=80000)
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=("all", "generate", "verify"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.per_category, args.contexts_per_target) < 0 or args.per_category + args.contexts_per_target == 0:
        parser.error("Enable at least one task with a positive target.")
    if not 1 <= args.batch_size <= 20 or not 1 <= args.verify_batch_size <= 30 or not 1 <= args.workers_per_key <= 8:
        parser.error("Invalid batch/concurrency settings.")
    if not 1 <= args.rpm <= 10 or not 10000 <= args.tpm <= 100000 or not 1 <= args.output_tokens <= 16000 or args.timeout < 1 or not 0 <= args.retries <= 5:
        parser.error("Invalid quota/output/timeout/retry settings.")
    envs = [name.strip() for name in args.key_envs.split(",")]
    if not 1 <= len(envs) <= 8 or len(set(envs)) != len(envs) or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in envs):
        parser.error("Invalid key environment variable names.")
    # Dry-run accepts account labels without reading/printing secret values.
    keys = tuple(os.environ.get(name, "").strip() for name in envs) if not args.dry_run else ()
    if keys and any(not key for key in keys):
        keys = ()
    if keys and len(set(keys)) != len(keys):
        parser.error("Duplicate keys do not increase quota.")
    config = {"per_category": args.per_category, "contexts_per_target": args.contexts_per_target,
        "batch_size": args.batch_size, "verify_batch_size": args.verify_batch_size,
        "model": args.model, "judge_model": args.judge_model, "output_tokens": args.output_tokens,
        "seed": args.seed, "endpoint": ENDPOINT, "pools_sha256": file_sha(args.pools)}
    pools = load_pools(args.pools)
    if args.dry_run:
        run(args.output, config, pools, dry_run=True, rpm=args.rpm, stage=args.stage, account_count=len(envs))
        return
    with output_lock(args.output):
        run(args.output, config, pools, args.stage, keys, args.workers_per_key, args.rpm,
            args.tpm, args.timeout, args.retries)


if __name__ == "__main__":
    main()
