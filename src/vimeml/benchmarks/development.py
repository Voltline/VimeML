"""Generate an independent kana-conversion development set and verify it blind."""
import argparse
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from vimeml.benchmarks.ajimee import ROOT, convert_items, sha
from vimeml.benchmarks.generate import Requests, json_content, normalized
from vimeml.benchmarks.hybrid import overlap_keys
from vimeml.review.multi_key import AccountPool, output_lock
from vimeml.review.prepare import write_json
from vimeml.review.run import ENDPOINT, digest, send_request, token_reservation

VERSION = "ime_development_generation_v1"
CATEGORIES = {
    "daily": "日常の用事・生活・自然な会話",
    "work": "職場・メール・会議・依頼",
    "technology": "アプリ・機器・サービス・技術用語",
    "food_travel": "食事・交通・旅行・買い物",
    "arts_media": "作品・音楽・番組・メディア",
    "clothing": "衣服・身につける動作・日常の物の扱い",
    "dates_numbers": "予定・日付・数量・時間の表現",
    "names": "読みが明確な人名・地名・店名を含む文。外部知識で正解を決めない",
    "inflection": "動詞の活用・助詞・文中の自然な変換区間",
    "homophones": "同音語の選択。文脈や同じ変換区間内の語句で意味を特定する",
}
GENERATION_PROMPT = """日本語IMEのかな漢字変換用に、独立した開発データを作成してください。
応答はJSON配列だけ。各要素のキーはcontext_text, input, expected_output, reason_zhです。
context_textとexpected_outputは日本語です。中国語への翻訳や中国語の文章を絶対に入れないでください。
reason_zhだけは簡潔な中国語で、80文字以内の結論を1文。思考過程、独り言、反復は禁止です。
形式の例（例文を生成データにコピーしないこと）：
[{"context_text":"駅に着いたら、","input":"キップヲカウ","expected_output":["切符を買う","きっぷを買う"],"reason_zh":"在车站买票，读音与语义一致。"}]
context_textは既に確定した同じ文内の左側、expected_outputは次の変換区間です。
両者は隣接し、重複せず、連結すると自然な日本語になります。右文は与えません。
左文は空文字または5〜50文字、非空なら文末の。！？で終えないこと。
各表記は2〜60文字、inputは2〜150文字の全角カタカナと元の位置の句読点のみ。
数字・日付・略語もinputでは実際の読みをカタカナで書き、漢字・数字・英字・ひらがなを混ぜないこと。
inputはIMEに入力するかな表記であり、発音記号ではありません。助詞は/へ/をはハ/ヘ/ヲ。
長音、促音、拗音、活用を一字ずつ確認し、読みが違う表記を絶対に混ぜないこと。
expected_outputは同じ意味・同じ読みの表記ゆれだけを1〜8個、重複なしで列挙します。
例：頑張る/がんばるは表記ゆれ。期間/機関/帰還は意味が違い、等価な正解ではありません。
同音異義語では左文や変換区間の他の語から意味が特定できる例を作ってください。
左文が空なら単独の曖昧な同音語を出さず、意味が明確な短い文・語句にします。
自然に続けられる文中の語句も可能ですが、単語・活用の途中で切らないこと。
外部の事実や未提示の背景を正解の根拠にせず、読みの不確かな人名や架空の作品名を避けてください。
日付と時刻と数量を無関係に並べず、日常で実際に書く自然な表現にしてください。
語彙と文型を変え、負例・候補リスト・AJIMEE/JWTDの模倣・本地モデルの失敗例の改作を生成しないこと。
キー、URL、Markdown、追加説明、本文の改行を出力しないでください。
"""
VERIFICATION_PROMPT = """日本語IMEのかな漢字変換例を独立に検査してください。
入力にはid, context_text, inputだけがあります。作者の想定解は知らされていません。
応答はJSON配列だけ。各idは入力と同じ文字列で、必ず1回ずつ返します。
形式の例（架空の形式例であり、検査対象の正解ではありません）：
[{"id":"0","status":"clear","confidence":"high","expected_output":["荷物を持つ"],"reason_zh":"语义明确，读音一致。"}]
expected_outputは日本語のみ。中国語に翻訳しないこと。reason_zhだけは中国語で80文字以内の結論1文。
思考過程、自己問答、反復、長い説明を出力しないこと。不確かならlowで短く報告してください。
inputの各文字を確認し、左文に自然に続く、完全に同じ読みの日本語表記を再構成してください。
inputを訂正したり、存在しない右文を補ったり、外部の事実を推測したりしてはいけません。
IMEのかな表記なので助詞は/へ/をはハ/ヘ/ヲです。発音のワ/エとは区別します。
長音・促音・拗音・句読点も一致させ、読みの似ている別の単語を正解にしないこと。
clear: 与えられた情報だけで意味が明確。同じ意味の漢字/かな・数字の表記ゆれは1〜8個、重複なし。
ambiguous: 意味が異なる複数の解釈が自然。例：左文なしのキカンは期間/機関/帰還が区別できません。
invalid: 自然な日本語に戻せない、不自然な構文、誤った読み、単語・活用の途中で切れている。
statusはclear/ambiguous/invalid、confidenceはhigh/medium/lowのみ。
ambiguous/invalidではexpected_outputを空配列にして構いません。
意味と全ての読みを確実に検査できた場合だけhighにしてください。
文中で続けられる語句には句点は不要ですが、最後の語・活用は完結している必要があります。
"""
KANA = re.compile(r"[ァ-ヺー。、！？「」『』（）()・：:，,.!?\s]+")
JA = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def text(value, minimum, maximum):
    if not isinstance(value, str) or value != value.strip() or not minimum <= len(value) <= maximum:
        raise ValueError("Invalid text type/length/outer whitespace.")
    if any(unicodedata.category(char).startswith("C") for char in value) or "\ufffd" in value or "http" in value.lower():
        raise ValueError("Control character, replacement character or URL in text.")
    if value and not JA.search(value):
        raise ValueError("Expected Japanese text.")
    return value


def answers(values, allow_empty=False):
    if not isinstance(values, list) or not (0 if allow_empty else 1) <= len(values) <= 8:
        raise ValueError("Expected 1..8 acceptable forms (or an empty invalid assessment).")
    result = [text(value, 2, 60) for value in values]
    if len(set(result)) != len(result):
        raise ValueError("Duplicate acceptable forms.")
    return result


def make_jobs(config):
    strata = [(category, with_context) for category in CATEGORIES for with_context in (True, False)]
    jobs = []
    for position, (category, with_context) in enumerate(strata):
        total = config["count"] // len(strata) + (position < config["count"] % len(strata))
        for start in range(0, total, config["batch_size"]):
            job = {"category": category, "with_context": with_context, "start": start,
                   "count": min(config["batch_size"], total - start)}
            jobs.append({**job, "id": digest({"version": VERSION, "seed": config["seed"], "job": job})})
    return jobs


def generation_instruction(job, legacy=False):
    if legacy:
        mode = "每条必须有5~50字的句内左文" if job["with_context"] else "每条context_text必须为空，待转换区间自身要足够明确"
        return (f"主题：{CATEGORIES[job['category']]}。恰好生成{job['count']}条。{mode}。\n"
                f"独立批次编号：{job['id'][:16]}；此编号仅用于请求区分，不要写进语料。")
    mode = "各例に5〜50文字の句内左文が必要" if job["with_context"] else "全例context_textは空文字。変換区間自体で意味が明確な語句を生成"
    return (f"テーマ：{CATEGORIES[job['category']]}。ちょうど{job['count']}例。{mode}。\n"
            f"バッチID：{job['id'][:16]}（データに含めない）。")


def generation_payload(job, config):
    instruction = generation_instruction(job)
    return {"model": config["model"], "temperature": .9, "max_tokens": config["output_tokens"],
            "messages": [{"role": "system", "content": GENERATION_PROMPT}, {"role": "user", "content": instruction}]}


def parse_generated(body, job):
    rows = json_content(body)
    if len(rows) != job["count"]:
        raise ValueError("Response count differs from fixed job count.")
    result, seen = [], set()
    for position, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError("Expected generated object.")
        context = text(raw.get("context_text"), 5 if job["with_context"] else 0, 50 if job["with_context"] else 0)
        if context.endswith(tuple("。！？!?")):
            raise ValueError("Context must remain within the current sentence.")
        query = text(raw.get("input"), 2, 150)
        if not KANA.fullmatch(query) or not re.search(r"[ァ-ヺ]", query):
            raise ValueError("Reading must contain katakana and punctuation only.")
        expected = answers(raw.get("expected_output"))
        reason = raw.get("reason_zh")
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 800:
            raise ValueError("Missing generation rationale.")
        key = (normalized(query), normalized(context))
        if key in seen:
            raise ValueError("Duplicate query/context in generation batch.")
        seen.add(key)
        result.append({"index": f"dev-{job['id'][:16]}-{position:02d}", "context_text": context, "input": query,
                       "expected_output": expected, "reason_zh": reason.strip(), "category": job["category"],
                       "provenance": {"job_id": job["id"], "kind": "llm_synthetic"}})
    return result


def verification_payload(cases, config):
    # No generator answers/rationale/category/local-model scores on the wire.
    evidence = [{"id": str(i), "context_text": case["context_text"], "input": case["input"]} for i, case in enumerate(cases)]
    return {"model": config["judge_model"], "temperature": 0, "max_tokens": config["output_tokens"],
            "messages": [{"role": "system", "content": VERIFICATION_PROMPT},
                         {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}]}


def parse_verified(body, cases):
    rows = json_content(body)
    if len(rows) != len(cases):
        raise ValueError("Verification response count differs.")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("id")) is not str or not row["id"].isdigit():
            raise ValueError("Invalid verification ID.")
        index = int(row["id"])
        if str(index) != row["id"] or not 0 <= index < len(cases) or index in result:
            raise ValueError("Unexpected/duplicate verification ID.")
        status, confidence = row.get("status"), row.get("confidence")
        if status not in {"clear", "ambiguous", "invalid"} or confidence not in {"high", "medium", "low"}:
            raise ValueError("Invalid verification status/confidence.")
        # A bad label belongs in quarantine, not in a retry of nine good rows.
        # Broken JSON/counts/IDs still fail the request because alignment is lost.
        issues, repairs = [], []
        values = row.get("expected_output")
        if isinstance(values, list) and all(isinstance(value, str) for value in values):
            deduped = list(dict.fromkeys(values))
            if deduped != values:
                repairs.append("duplicate_forms_removed")
            values = deduped
        try:
            expected = answers(values, allow_empty=status != "clear")
        except ValueError as error:
            expected = []
            issues.append(f"invalid_acceptable_forms: {error}")
        # Kana-only forms have an exact spelling check; kanji readings still
        # require the blind verifier and the subsequent manual review.
        for form in expected:
            katakana = ''.join(chr(ord(c)+0x60) if 'ぁ' <= c <= 'ゖ' else c for c in form)
            if KANA.fullmatch(katakana) and katakana != cases[index]["input"]:
                issues.append("kana_form_reading_mismatch")
        reason = row.get("reason_zh")
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 1000:
            issues.append("missing_or_excessive_verification_reason")
            reason = reason[:1000] if isinstance(reason, str) else ""
        if issues:
            status, confidence = "invalid", "low"
        decision = {"status": status, "confidence": confidence, "expected_output": expected, "reason_zh": reason.strip()}
        if issues:
            decision["validation_errors"] = issues
        if repairs:
            decision["repairs"] = repairs
        result[index] = decision
    return [result[i] for i in range(len(cases))]


def unique_cases(jobs, generated, excluded):
    _, excluded_rows, _ = convert_items(excluded)
    exclude_queries, exclude_texts = overlap_keys(excluded_rows)
    seen_queries, seen_texts, kept, removed = set(), set(), [], []
    for job in jobs:
        for case in generated.get(job["id"], []):
            query = (normalized(case["input"]), normalized(case["context_text"]))
            forms = {normalized(case["context_text"] + answer) for answer in case["expected_output"]}
            reason = None
            if query in exclude_queries or forms & exclude_texts:
                reason = "overlap_with_ajimee"
            elif query in seen_queries or forms & seen_texts:
                reason = "duplicate_development_example"
            if reason:
                removed.append({**case, "excluded_reason": reason})
            else:
                seen_queries.add(query)
                seen_texts.update(forms)
                kept.append(case)
    return kept, removed


def finalize(cases, decisions, excluded):
    if len(cases) != len(decisions):
        raise ValueError("Verification decisions missing.")
    _, excluded_rows, _ = convert_items(excluded)
    exclude_queries, exclude_texts = overlap_keys(excluded_rows)
    accepted, quarantine, seen = [], [], set()
    for case, decision in zip(cases, decisions):
        generated = set(case["expected_output"])
        independently_reconstructed = set(decision["expected_output"])
        reason = None
        if decision["status"] != "clear" or decision["confidence"] != "high":
            reason = "ambiguous_invalid_or_low_confidence"
        elif not generated <= independently_reconstructed:
            reason = "independent_reconstruction_disagrees_with_generated_forms"
        # Only verifier-supported generated forms survive; extra blind forms
        # are retained as explicitly verified alternative spellings.
        forms = list(case["expected_output"]) + [s for s in decision["expected_output"] if s not in generated]
        keys = {normalized(case["context_text"] + form) for form in forms}
        if reason is None and (keys & exclude_texts or (normalized(case["input"]), normalized(case["context_text"])) in exclude_queries):
            reason = "verified_form_overlaps_ajimee"
        if reason is None and keys & seen:
            reason = "verified_form_duplicates_development_example"
        if reason:
            quarantine.append({**case, "quality": decision, "quarantine_reason": reason})
        else:
            seen.update(keys)
            accepted.append({**case, "expected_output": forms, "generator_expected_output": case["expected_output"], "quality": decision})
    return accepted, quarantine


class DevelopmentRequests(Requests):
    def signature(self, payload):
        return digest({"version": VERSION, "endpoint": self.config["endpoint"], "payload": payload})


def reuse_generation(source, config, jobs, excluded_sha, require_complete=True):
    """Import complete generation caches, keeping their actual prompt provenance.

    Never reuse old verification decisions after changing the verifier prompt.
    This reads the source only; in-progress verification cannot alter these caches.
    """
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != VERSION or manifest.get("excluded_source_sha256") != excluded_sha:
        raise ValueError("Generation reuse format/exclusion source differs.")
    original = manifest["config"]
    for key in ("count", "batch_size", "seed", "model", "endpoint"):
        if original.get(key) != config.get(key):
            raise ValueError(f"Generation reuse configuration differs: {key}.")
    found, hashes = {}, {}
    by_instruction = {generation_instruction(j, legacy=legacy): j for j in jobs for legacy in (False, True)}
    for path in sorted((source / "cache").glob('*.json')):
        raw = path.read_bytes()
        cached = json.loads(raw.decode("utf-8"))
        payload = cached.get("request", {})
        messages = payload.get("messages", [])
        if payload.get("model") != original["model"] or len(messages) != 2:
            continue
        instruction = messages[1].get("content")
        job = by_instruction.get(instruction)
        if job is None:
            continue  # Verification cache, including same-model verification.
        actual_prompt = messages[0].get("content")
        if digest(actual_prompt) != manifest["generation_prompt_sha256"]:
            continue
        signature = digest({"version": VERSION, "endpoint": original["endpoint"], "payload": payload})
        if cached.get("signature") != signature or path.stem != signature:
            raise ValueError("Generation reuse cache signature differs.")
        expected_payload = {"model": original["model"], "temperature": .9, "max_tokens": original["output_tokens"],
                            "messages": [{"role": "system", "content": actual_prompt},
                                         {"role": "user", "content": instruction}]}
        if payload != expected_payload or job["id"] in found:
            raise ValueError("Unexpected or duplicate generation reuse payload.")
        found[job["id"]] = parse_generated(cached["body"], job)
        hashes[path.name] = sha(raw)
    if require_complete and len(found) != len(jobs):
        raise ValueError(f"Generation reuse requires all {len(jobs)} successful batches; found {len(found)}.")
    origin = {"source": str(source.resolve()), "source_identity": manifest["identity"],
              "generation_prompt_sha256": manifest["generation_prompt_sha256"], "cache_sha256": hashes}
    return found, origin


def export_cached_drafts(source, output, excluded_path):
    """Export a frozen partial snapshot for review, without credentials/network.

    Generation caches are drafts, not verification decisions. Incomplete jobs
    remain explicitly missing; failed raw responses are never silently promoted.
    """
    if source.resolve() == output.resolve():
        raise ValueError("Use a separate snapshot output directory.")
    if output.exists() and any(p.name != '.review.lock' for p in output.iterdir()):
        raise ValueError("Snapshot output is not empty; use a new directory.")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    excluded_raw = excluded_path.read_bytes()
    excluded = json.loads(excluded_raw.decode("utf-8-sig"))
    config = manifest["config"]
    jobs = make_jobs(config)
    generated, origin = reuse_generation(source, config, jobs, sha(excluded_raw), require_complete=False)
    cases, removed = unique_cases(jobs, generated, excluded)
    if not cases:
        raise ValueError("No successful unique generation drafts to export.")
    missing = [job for job in jobs if job["id"] not in generated]
    stats = {"requested_draft_cases": config["count"], "cached_generation_batches": len(generated),
             "missing_generation_batches": len(missing), "unique_draft_cases": len(cases),
             "excluded_cases": len(removed), "with_context": sum(bool(c["context_text"]) for c in cases),
             "without_context": sum(not c["context_text"] for c in cases),
             "by_category": dict(Counter(c["category"] for c in cases)),
             "automatically_verified_cases": 0, "manual_review_done": False, "network_requests_sent": 0}
    output.mkdir(parents=True, exist_ok=True)
    data = {"draft.json": cases, "excluded.json": removed, "missing-jobs.json": missing, "stats.json": stats,
            "review-pack.json": {"unverified": cases, "manual_review_done": False}}
    for name, value in data.items():
        write_json(output / name, value)
    write_json(output / "manifest.json", {"format": "ime_development_draft_snapshot_v1",
               "status": "drafts_exported", "config": config, "generation_origin": origin,
               "excluded_source_sha256": sha(excluded_raw), "stats": stats,
               "files_sha256": {name: sha((output / name).read_bytes()) for name in data},
               "note": "Offline generation drafts only; neither API verification nor label review is implied."})
    return stats


def execute_stage(requests, work, workers, stage, total):
    results, failures = {}, []
    cached_count = total - len(work)
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(requests.call, payload, parser): key for key, payload, parser in work}
    pending = set(futures)
    print(f"[{stage}] requests {cached_count}/{total}; cached {cached_count}; pending {len(pending)}", flush=True)
    try:
        while pending:
            done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as error:
                    failures.append({"stage": stage, "id": key, "error": str(error)})
            print(f"[{stage}] requests {cached_count + len(results)}/{total}; pending {len(pending)}; failed {len(failures)}; "
                  f"network requests this run {requests.started}", flush=True)
    except KeyboardInterrupt:
        requests.stop.set()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    write_json(requests.output / "pending.json", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} {stage} requests pending. Successful caches retained; rerun the same command.")
    return results


def run(output, config, excluded_path, api_keys=(), workers_per_key=4, rpm=10, tpm=80000,
        timeout=180, retries=2, stage="all", dry_run=False, account_count=1, transport=send_request, window=60.,
        reuse_generation_from=None):
    exclude_raw = excluded_path.read_bytes()
    excluded = json.loads(exclude_raw.decode("utf-8-sig"))
    convert_items(excluded)
    jobs = make_jobs(config)
    imported, origin = {}, None
    if reuse_generation_from is not None:
        if reuse_generation_from.resolve() == output.resolve():
            raise ValueError("Generation reuse needs a separate new --output directory.")
        imported, origin = reuse_generation(reuse_generation_from, config, jobs, sha(exclude_raw))
    identity_data = {"version": VERSION, "config": config, "generation_prompt": GENERATION_PROMPT,
                     "verification_prompt": VERIFICATION_PROMPT, "excluded_source_sha256": sha(exclude_raw)}
    if origin:
        identity_data["generation_reuse"] = origin
    identity = digest(identity_data)
    manifest_path = output / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if old and (old.get("format") != VERSION or old.get("identity") != identity):
        raise ValueError("Configuration/prompts/exclusion set changed; use a new --output.")
    if old and old.get("status") == "complete":
        for name, checksum in old["files_sha256"].items():
            if sha((output / name).read_bytes()) != checksum:
                raise ValueError(f"Frozen generated file changed: {name}. Keep reviewed edits as a new version.")
        if not dry_run:
            stats = {**old["stats"], "network_requests_sent_this_run": 0, "accounts_this_run": []}
            print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
            print(f"Reused complete generation: {output.resolve() / 'ime-dev-draft.json'}")
            if not stats["automatically_accepted_cases"]:
                raise ValueError("All drafts quarantined; inspect quarantine.json.")
            return stats
    if not old and output.exists() and any(p.name != '.review.lock' for p in output.iterdir()):
        raise ValueError("Output contains unrelated data; use a new --output.")
    generated, missing = dict(imported), []
    probe = DevelopmentRequests(output, config, None, timeout, retries, transport)
    for job in jobs:
        if job["id"] in generated:
            continue
        payload = generation_payload(job, config)
        parsed = probe.cached(payload, lambda body, job=job: parse_generated(body, job))
        if parsed is None:
            missing.append(job)
        else:
            generated[job["id"]] = parsed
    accounts = max(account_count, len(api_keys), 1)
    verification_upper = math.ceil(config["count"] / config["verify_batch_size"])
    preview = {"target_draft_cases": config["count"], "with_context_target": sum(j["count"] for j in jobs if j["with_context"]),
               "without_context_target": sum(j["count"] for j in jobs if not j["with_context"]),
               "generation_requests": len(jobs), "generation_requests_pending": len(missing),
               "reused_generation_cases": sum(len(rows) for rows in imported.values()),
               "verification_requests_upper_bound": verification_upper, "accounts": accounts,
               "generator": config["model"], "blind_verifier": config["judge_model"],
               "request_quota_lower_bound_minutes_full_run": max(0, len(jobs)+verification_upper-1)/(rpm*accounts),
               "request_quota_lower_bound_minutes_remaining_upper_plan": max(0, len(missing)+verification_upper-1)/(rpm*accounts),
               "network_requests_sent": 0, "note": "No count guarantee after dedup/verification; quota estimate excludes latency, token throttling and retries."}
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return preview
    if missing and stage == "verify":
        raise ValueError("Generation incomplete; run --stage all or generate first.")
    if missing and not api_keys:
        raise ValueError("Set SJTU_API_KEY in the local terminal before starting (never put the key in source files).")
    pool = AccountPool(api_keys, workers_per_key=workers_per_key, rpm=rpm, tpm=tpm, window=window) if api_keys else None
    requests = DevelopmentRequests(output, config, pool, timeout, retries, transport)
    workers = max(1, len(api_keys))*workers_per_key
    # Fail before dispatch if any request cannot fit even once in the quota.
    if any(token_reservation(generation_payload(job, config)) > tpm for job in jobs):
        raise ValueError("A generation request exceeds per-account TPM; reduce output tokens/batch size in a new output.")
    for folder in ("cache", "responses", "failures"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    manifest = {"format": VERSION, "identity": identity, "config": config, "status": "generating",
                "excluded_source_sha256": sha(exclude_raw), "generation_prompt_sha256": digest(GENERATION_PROMPT),
                "verification_prompt_sha256": digest(VERIFICATION_PROMPT), "manual_review_done": False,
                "generator_equals_verifier": config["model"] == config["judge_model"],
                "note": "Independent synthetic development draft; blind AI reconstruction is provisional; no local LM filtering, candidate generation or training."}
    if origin:
        manifest.update(generation_reuse=origin, configured_generation_prompt_sha256=digest(GENERATION_PROMPT),
                        generation_prompt_sha256=origin["generation_prompt_sha256"])
    write_json(manifest_path, manifest)
    work = [(j["id"], generation_payload(j, config), lambda body, job=j: parse_generated(body, job)) for j in missing]
    if stage in {"all", "generate"}:
        generated.update(execute_stage(requests, work, workers, "generate", len(jobs)))
    cases, removed = unique_cases(jobs, generated, excluded)
    write_json(output / "draft.json", cases)
    write_json(output / "excluded.json", removed)
    if stage == "generate":
        write_json(manifest_path, old if old and old.get("status") == "complete" else {**manifest, "status": "awaiting_verification"})
        return {"status": "awaiting_verification", "unique_draft_cases": len(cases)}
    if not cases:
        raise ValueError("No unique development drafts remain; inspect excluded.json.")
    batches = [cases[i:i+config["verify_batch_size"]] for i in range(0,len(cases),config["verify_batch_size"])]
    verified, work = {}, []
    for index, batch in enumerate(batches):
        payload = verification_payload(batch, config)
        if token_reservation(payload) > tpm:
            raise ValueError("A verification batch exceeds per-account TPM; use a new output with smaller --verify-batch-size.")
        parser = lambda body, batch=batch: parse_verified(body,batch)
        cached = requests.cached(payload, parser)
        if cached is None:
            work.append((str(index), payload, parser))
        else:
            verified[str(index)] = cached
    if work and not api_keys:
        raise ValueError("Verification is pending; set SJTU_API_KEY in your local terminal and rerun.")
    write_json(manifest_path, {**manifest, "status": "verifying", "unique_draft_cases": len(cases)})
    verified.update(execute_stage(requests, work, workers, "verify", len(batches)))
    decisions = [decision for i in range(len(batches)) for decision in verified[str(i)]]
    accepted, quarantine = finalize(cases, decisions, excluded)
    write_json(output / "ime-dev-draft.json", accepted)
    write_json(output / "quarantine.json", quarantine)
    write_json(output / "verification.json", [{"index": case["index"], **decision} for case, decision in zip(cases,decisions)])
    write_json(output / "review-pack.json", {"accepted": accepted, "quarantined": quarantine, "manual_review_done": False})
    stats = {"requested_draft_cases": config["count"], "unique_draft_cases": len(cases), "excluded_cases": len(removed),
             "automatically_accepted_cases": len(accepted), "quarantined_cases": len(quarantine),
             "accepted_with_context": sum(bool(c["context_text"]) for c in accepted),
             "accepted_without_context": sum(not c["context_text"] for c in accepted),
             "accepted_by_category": dict(Counter(c["category"] for c in accepted)),
             "network_requests_sent_this_run": requests.started, "accounts_this_run": pool.snapshot() if pool else [],
             "reused_generation_cases": sum(len(rows) for rows in imported.values()),
             "manual_review_done": False}
    write_json(output / "stats.json", stats)
    outputs = ["draft.json", "excluded.json", "ime-dev-draft.json", "quarantine.json", "verification.json", "review-pack.json", "stats.json"]
    write_json(manifest_path, {**manifest, "status": "complete", "stats": stats,
                              "files_sha256": {name: sha((output/name).read_bytes()) for name in outputs}})
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print(f"Draft for review: {output.resolve() / 'ime-dev-draft.json'}")
    print(f"Review pack: {output.resolve() / 'review-pack.json'}")
    if not accepted:
        raise ValueError("All drafts quarantined; inspect replies and quarantine.json. Do not calibrate on them.")
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/ime-dev-generation-v2")
    parser.add_argument("--reuse-generation-from", type=Path, help="Reuse all successful generation caches from another directory; verify again with the current prompt.")
    parser.add_argument("--exclude", type=Path, default=ROOT / "artifacts/benchmarks/ajimee-jwtd-v2-v1/evaluation_items.json")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--verify-batch-size", type=int, default=10)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--key-envs", default="SJTU_API_KEY")
    parser.add_argument("--workers-per-key", type=int, default=4)
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--tpm", type=int, default=80000)
    parser.add_argument("--output-tokens", type=int, default=6144)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--stage", choices=("all","generate","verify"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 20 <= args.count <= 10000 or not 1 <= args.batch_size <= 20 or not 1 <= args.verify_batch_size <= 20:
        parser.error("count must be 20..10000; batch sizes 1..20.")
    if not 1 <= args.workers_per_key <= 8 or not 1 <= args.rpm <= 10 or not 10000 <= args.tpm <= 100000:
        parser.error("Invalid concurrency/quotas.")
    if not 512 <= args.output_tokens <= 16000 or args.timeout < 1 or not 0 <= args.retries <= 5:
        parser.error("Invalid output token budget, timeout or retries.")
    envs = [name.strip() for name in args.key_envs.split(',')]
    if not 1 <= len(envs) <= 8 or len(set(envs)) != len(envs) or any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) for name in envs):
        parser.error("--key-envs requires distinct environment variable names, never actual keys.")
    keys = tuple(os.environ.get(name,'').strip() for name in envs) if not args.dry_run else ()
    if keys and all(not key for key in keys):
        keys = ()  # A fully cached run can resume without credentials.
    if keys and any(not key for key in keys):
        parser.error("Missing environment variable(s): " + ', '.join(name for name,key in zip(envs,keys) if not key))
    if keys and len(set(keys)) != len(keys):
        parser.error("Duplicate keys do not provide independent account quota.")
    config = {"count": args.count, "batch_size": args.batch_size, "verify_batch_size": args.verify_batch_size,
              "model": args.model, "judge_model": args.judge_model, "output_tokens": args.output_tokens,
              "seed": args.seed, "endpoint": ENDPOINT}
    kwargs = dict(api_keys=keys, workers_per_key=args.workers_per_key, rpm=args.rpm, tpm=args.tpm,
                  timeout=args.timeout, retries=args.retries, stage=args.stage, account_count=len(envs),
                  reuse_generation_from=args.reuse_generation_from)
    try:
        if args.dry_run:
            run(args.output, config, args.exclude, dry_run=True, **kwargs)
        else:
            with output_lock(args.output):
                run(args.output, config, args.exclude, **kwargs)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; successful caches retained. Rerun the same command.\n")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == '__main__':
    main()
