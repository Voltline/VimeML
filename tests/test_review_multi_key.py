import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.build import file_sha
from vimeml.review.prepare import PROMPT, write_json
from vimeml.review.run import QuotaLimiter, run, payload_for, parse_model_reply, request_signature
from vimeml.review.multi_key import AccountPool, main, output_lock


class MultiKeyReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="multi-key-review-test-", dir=ROOT / "outputs")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.materials, self.output = self.directory / "materials", self.directory / "review"
        self.materials.mkdir()
        self.cases = [{"id": f"C{i}", "kind": "retained_sentence", "text": f"自然な文章です。例{i}",
                       "source_file": "fixture.parquet", "row_index": i, "assigned_split": "train",
                       "sampling_strata": ["keep:fixture"], "original_lines": []} for i in range(12)]
        write_json(self.materials / "cases.json", self.cases)
        (self.materials / "prompt.txt").write_text(PROMPT, encoding="utf-8")
        write_json(self.materials / "manifest.json", {
            "status": "complete", "schema_version": 1,
            "cases_sha256": file_sha(self.materials / "cases.json"),
            "prompt_sha256": file_sha(self.materials / "prompt.txt"), "quality_rate_note": "fixture"})

    @staticmethod
    def response(payload, reason="自然な文章"):
        data = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
        rows = [{"id": case["id"], "assessment": "ok", "issue_type": "none",
                 "suggested_action": "keep", "reason_zh": reason} for case in data]
        return 200, {}, json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(rows, ensure_ascii=False)}}], "usage": {"total_tokens": 40}}, ensure_ascii=False)

    @staticmethod
    def pool(**kwargs):
        return AccountPool(["fixture-secret-A", "fixture-secret-B"], labels=["KEY_A", "KEY_B"],
                           rpm=100, tpm=1000000, window=0.001, **kwargs)

    def review(self, pool, transport, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(self.materials, self.output, workers=pool.size * pool.workers_per_key,
                       request_pool=pool, transport=transport, **kwargs)

    def test_independent_parallel_accounts_complete_once_and_redact_both_secrets(self):
        active, peak, seen, lock = Counter(), Counter(), [], threading.Lock()

        def fake(endpoint, payload, key, timeout):
            rows = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
            with lock:
                active[key] += 1
                peak[key] = max(peak[key], active[key])
                peak["combined"] = max(peak["combined"], sum(active.values()))
                by_text = {case["text"]: case["id"] for case in self.cases}
                seen.extend(by_text[case["text"]] for case in rows)
            time.sleep(0.04)
            with lock:
                active[key] -= 1
            return self.response(payload, "fixture-secret-A fixture-secret-B")

        stats = self.review(self.pool(), fake, batch_size=1)
        self.assertEqual(stats["pending_cases"], 0)
        self.assertEqual(Counter(seen), Counter(case["id"] for case in self.cases))
        self.assertGreaterEqual(peak["combined"], 4)
        for key in ("fixture-secret-A", "fixture-secret-B"):
            self.assertGreaterEqual(peak[key], 2)
            self.assertLessEqual(peak[key], 3)
        self.assertEqual(sum(row["total_tokens_reported"] for row in stats["accounts_this_run"]), 480)
        self.assertTrue(all(row["in_flight"] == 0 for row in stats["accounts_this_run"]))
        for path in self.output.rglob("*.json"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("fixture-secret-A", text)
            self.assertNotIn("fixture-secret-B", text)

    def test_resume_single_key_partial_cache_and_switch_back_without_repeating_cases(self):
        def partial(endpoint, payload, key, timeout):
            rows = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
            return self.response(payload) if rows[0]["id"] in {"C0", "C1", "C2"} else (400, {}, "invalid fixture")

        # Produce old full-ID requests to verify existing per-case caches work
        # unchanged after switching to the compact-ID protocol.
        def legacy_payload(*args):
            return payload_for(*args, compact_ids=False)

        with contextlib.redirect_stdout(io.StringIO()), patch("vimeml.review.run.payload_for", side_effect=legacy_payload):
            original = run(self.materials, self.output, "fixture-single", workers=1, batch_size=1,
                           retries=0, transport=partial, limiter=QuotaLimiter(100, 1000000, window=0.001))
        self.assertEqual(original["completed_cases"], 3)
        seen = []

        def fake(endpoint, payload, key, timeout):
            by_text = {case["text"]: case["id"] for case in self.cases}
            seen.extend(by_text[row["text"]] for row in json.loads(payload["messages"][1]["content"].split("\n", 1)[1]))
            return self.response(payload)

        stats = self.review(self.pool(), fake, batch_size=2)
        self.assertEqual(stats["requests_this_run"]["cached_cases"], 3)
        self.assertEqual(set(seen), {case["id"] for case in self.cases} - {"C0", "C1", "C2"})
        self.assertEqual(len(seen), 9)
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = run(self.materials, self.output, transport=lambda *args: self.fail("Cache sent requests"))
        self.assertEqual(resumed["pending_cases"], 0)
        self.assertEqual(resumed["requests_this_run"]["cached_cases"], 12)

    def test_auth_failure_disables_only_bad_account_and_reassigns_batch_without_retry_budget(self):
        calls = Counter()

        def fake(endpoint, payload, key, timeout):
            calls[key] += 1
            return (401, {}, "fixture-secret-A") if key == "fixture-secret-A" else self.response(payload)

        stats = self.review(self.pool(workers_per_key=1), fake, batch_size=2, retries=0)
        self.assertEqual(stats["pending_cases"], 0)
        self.assertEqual(calls["fixture-secret-A"], 1)
        accounts = {row["account"]: row for row in stats["accounts_this_run"]}
        self.assertFalse(accounts["KEY_A"]["enabled"])
        self.assertEqual(accounts["KEY_A"]["disabled_http_status"], 401)
        self.assertTrue(accounts["KEY_B"]["enabled"])
        self.assertEqual(accounts["KEY_B"]["successful_requests"], 6)

    def test_all_auth_failures_stop_and_keep_all_cases_pending(self):
        stats = self.review(self.pool(workers_per_key=1), lambda *args: (403, {}, "forbidden"),
                            batch_size=1, retries=0)
        self.assertEqual(stats["pending_cases"], 12)
        self.assertEqual(stats["requests_this_run"]["sent"], 2)
        self.assertTrue(all(not row["enabled"] for row in stats["accounts_this_run"]))

    def test_429_cools_only_affected_account_and_other_account_finishes_retry(self):
        calls = Counter()

        def fake(endpoint, payload, key, timeout):
            calls[key] += 1
            return (429, {"Retry-After": "60"}, "limited") if key == "fixture-secret-A" else self.response(payload)

        stats = self.review(self.pool(workers_per_key=1), fake, batch_size=20, retries=1)
        self.assertEqual(stats["pending_cases"], 0)
        self.assertEqual(calls, {"fixture-secret-A": 1, "fixture-secret-B": 1})
        self.assertEqual(stats["accounts_this_run"][0]["http_429"], 1)

    def test_per_account_rolling_token_quota_does_not_block_other_account(self):
        pool = AccountPool(["a", "b"], workers_per_key=2, rpm=2, tpm=100, window=0.12)
        stop, tickets, lock = threading.Event(), [], threading.Lock()

        def acquire():
            ticket = pool.acquire(60, stop)
            with lock:
                tickets.append(ticket)
            pool.release(ticket)

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: acquire(), range(4)))
        starts = sorted((ticket["quota_ticket"]["start"], ticket["index"]) for ticket in tickets)
        self.assertNotEqual(starts[0][1], starts[1][1])
        self.assertLess(starts[1][0] - starts[0][0], 0.08)
        for index in (0, 1):
            times = [start for start, account in starts if account == index]
            self.assertEqual(len(times), 2)
            self.assertGreaterEqual(times[1] - times[0], 0.115)

    def test_cli_dry_run_never_reads_keys_or_writes_and_accounts_halve_quota_bound(self):
        argv = ["run_multi_key.py", "--input-dir", str(self.materials), "--output-dir", str(self.output), "--dry-run"]
        stream = io.StringIO()
        environment_get = os.environ.get

        def check_environment(name, default=None):
            if name in {"SJTU_API_KEY", "SJTU_API_KEY_2"}:
                raise AssertionError("Read a key")
            return environment_get(name, default)

        with patch.object(sys, "argv", argv), patch("vimeml.review.multi_key.os.environ.get", side_effect=check_environment):
            with contextlib.redirect_stdout(stream):
                main()
        preview = json.loads(stream.getvalue())
        self.assertEqual(preview["accounts"]["aggregate_rpm"], 16)
        self.assertEqual(preview["quota_lower_bound_minutes"], preview["requests_planned"] / 16)
        self.assertEqual(preview["network_requests_sent"], 0)
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "distinct"):
            AccountPool(["same-secret", "same-secret"])

    def test_output_lock_rejects_second_runner_before_transport_and_releases(self):
        with output_lock(self.output):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                self.review(self.pool(), lambda *args: self.fail("Locked output sent requests"))
        stats = self.review(self.pool(), lambda endpoint, payload, key, timeout: self.response(payload))
        self.assertEqual(stats["pending_cases"], 0)

    def test_short_ids_map_reordered_replies_and_reject_wrong_ids_or_actions(self):
        cases = [{**self.cases[0], "id": "C" + "a" * 24, "kind": "boundary_candidate"},
                 {**self.cases[1], "id": "C" + "b" * 24, "kind": "review_block"}]
        payload = payload_for(cases, PROMPT, "fixture", 2048)
        sent = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
        self.assertEqual([row["id"] for row in sent], ["R1", "R2"])
        self.assertNotIn("drop", sent[0]["allowed_actions"])
        self.assertIn("drop", sent[1]["allowed_actions"])
        rows = [{"id": "R2", "assessment": "ok", "issue_type": "none", "suggested_action": "keep", "reason_zh": "正文"},
                {"id": "R1", "assessment": "ok", "issue_type": "none", "suggested_action": "separate", "reason_zh": "保持边界"}]
        body = {"choices": [{"message": {"content": json.dumps(rows)}}]}
        parsed, canonical = parse_model_reply(body, cases, payload)
        self.assertEqual(parsed[cases[0]["id"]]["suggested_action"], "separate")
        self.assertEqual(parsed[cases[1]["id"]]["suggested_action"], "keep")
        self.assertEqual({row["id"] for row in json.loads(canonical["choices"][0]["message"]["content"])},
                         {case["id"] for case in cases})
        rows[1]["suggested_action"] = "drop"
        body["choices"][0]["message"]["content"] = json.dumps(rows)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            parse_model_reply(body, cases, payload)
        rows[1]["suggested_action"], rows[1]["id"] = "separate", "R9"
        body["choices"][0]["message"]["content"] = json.dumps(rows)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            parse_model_reply(body, cases, payload)

    def test_identical_evidence_with_distinct_local_ids_cannot_share_batch_cache(self):
        first, second = [self.cases[0]], [{**self.cases[0], "id": "different-local-id"}]
        payload1, payload2 = (payload_for(cases, PROMPT, "fixture", 2048) for cases in (first, second))
        self.assertEqual(payload1, payload2)
        self.assertNotEqual(request_signature("fixture", payload1, first), request_signature("fixture", payload2, second))


if __name__ == "__main__":
    unittest.main()
