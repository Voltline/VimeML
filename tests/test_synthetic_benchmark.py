"""Mock API coverage: resumability, blind judging, ambiguity and frozen metrics."""
import contextlib
import io
import json
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.benchmarks.generate import (PROMPT, canonical_cases, digest, finalize,
    generation_payload, make_jobs, parse_generated, parse_verified, run, verification_payload)
from vimeml.benchmarks.evaluate import accuracy, evaluate, load_benchmark
from vimeml.training.infer import JapaneseLM


POOLS = {"groups": [{"id": "hashi", "reading": "はし", "candidates": ["箸", "橋"]}]}
CONFIG = {"per_category": 1, "contexts_per_target": 1, "batch_size": 1, "verify_batch_size": 5,
    "model": "deepseek-chat", "judge_model": "qwen", "output_tokens": 2000,
    "seed": 42, "endpoint": "https://fixture.invalid/api", "pools_sha256": digest(POOLS)}


def body(rows, **kwargs):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(rows, ensure_ascii=False)}}],
        "usage": {"total_tokens": 20}, **kwargs}


class Transport:
    def __init__(self):
        self.gold, self.calls, self.lock = {}, [], threading.Lock()
        self.fail_first = False
        self.fail_wire_id = None

    def __call__(self, endpoint, payload, key, timeout):
        with self.lock:
            self.calls.append(payload)
        message = payload["messages"][1]["content"]
        if payload["model"] == "deepseek-chat":
            nonce = re.search(r"本批独立编号：([a-f0-9]+)", message)[1]
            if self.fail_first and nonce == self.fail_wire_id:
                return 500, {}, "bad fixture reply"
            target = re.search(r"目标词：([^。]+)", message)
            context = f"今日の作業では{nonce}まず"
            gold = target[1] if target else "手順を確認します"
            self.gold[context] = gold
            row = {"context": context, "reason_zh": "测试说明 fixture-secret"}
            if not target:
                row.update(positive=gold, negative="手順を忘却します")
            return 200, {}, json.dumps(body([row]), ensure_ascii=False)
        evidence = json.loads(message)
        rows = [{"id": item["id"], "status": "clear", "confidence": "high",
            "acceptable_ids": [next(candidate["id"] for candidate in item["candidates"]
                if candidate["text"] == self.gold[item["context"]])], "reason_zh": "独立测试核验"} for item in evidence]
        return 200, {}, json.dumps(body(rows), ensure_ascii=False)


class SyntheticBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="benchmark-test-", dir=ROOT / "outputs")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "benchmark"

    def run_fixture(self, transport, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(self.directory, CONFIG, POOLS, api_keys=("fixture-secret",),
                transport=transport, window=.0001, rpm=100, tpm=1000000, retries=0, **kwargs)

    def test_dry_run_writes_nothing_and_never_calls_transport(self):
        def unexpected(*args):
            self.fail("Dry-run sent a request.")
        with contextlib.redirect_stdout(io.StringIO()):
            preview = run(self.directory, CONFIG, POOLS, dry_run=True, transport=unexpected, account_count=2)
        self.assertEqual(preview["target_generated_cases"], 10)
        self.assertEqual(preview["accounts"], 2)
        self.assertFalse(self.directory.exists())

    def test_complete_run_saves_raw_bodies_redacts_key_and_resume_sends_zero(self):
        transport = Transport()
        result = self.run_fixture(transport)
        self.assertEqual(result["accepted_cases"], 10)
        manifest, cases = load_benchmark(self.directory)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(transport.calls), 12)
        self.assertTrue((self.directory / "manual-audit.json").exists())
        for path in self.directory.rglob("*.json"):
            self.assertNotIn("fixture-secret", path.read_text(encoding="utf-8"))
        for path in (self.directory / "responses").glob("*.json"):
            self.assertIn("raw_response", json.loads(path.read_text(encoding="utf-8")))
        calls = len(transport.calls)
        resumed = self.run_fixture(transport)
        self.assertEqual(len(transport.calls), calls)
        self.assertEqual(resumed["network_requests_sent_this_run"], 0)
        self.assertEqual(load_benchmark(self.directory)[1], cases)

    def test_partial_failure_resume_only_missing_fixed_slot(self):
        transport = Transport()
        jobs = make_jobs(CONFIG, POOLS)
        transport.fail_wire_id = jobs[0]["id"][:16]
        transport.fail_first = True
        with self.assertRaisesRegex(RuntimeError, "pending"):
            self.run_fixture(transport)
        self.assertEqual(len(list((self.directory / "cache").glob("*.json"))), 9)
        failed_replies = [json.loads(path.read_text(encoding="utf-8")) for path in (self.directory / "responses").glob("*.json")]
        self.assertTrue(any(row["raw_response"] == "bad fixture reply" for row in failed_replies))
        calls = len(transport.calls)
        transport.fail_first = False
        result = self.run_fixture(transport)
        self.assertEqual(result["accepted_cases"], 10)
        self.assertEqual(len(transport.calls) - calls, 3)  # one generation + two verification batches
        self.assertEqual(json.loads((self.directory / "pending.json").read_text(encoding="utf-8")), [])

    def test_configuration_changes_cannot_mix_old_caches(self):
        self.run_fixture(Transport())
        with self.assertRaisesRegex(ValueError, "changed"):
            run(self.directory, {**CONFIG, "model": "other"}, POOLS, dry_run=True)

    def test_frozen_benchmark_tampering_rejected(self):
        self.run_fixture(Transport())
        path = self.directory / "benchmark.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_benchmark(self.directory)

    def test_gold_not_sent_to_blind_verifier(self):
        jobs = make_jobs(CONFIG, POOLS)
        generated = {jobs[0]["id"]: [{"context": "今日の会議ではまず", "candidates": ["手順を確認します", "手順を忘却します"],
            "acceptable": ["手順を確認します"], "reason_zh": "hidden explanation"}]}
        cases, _ = canonical_cases(jobs, generated, CONFIG)
        evidence = json.loads(verification_payload(cases, CONFIG)["messages"][1]["content"])
        self.assertNotIn("acceptable", evidence[0])
        self.assertNotIn("positive", evidence[0])
        self.assertNotIn("hidden explanation", json.dumps(evidence))
        self.assertEqual({candidate["text"] for candidate in evidence[0]["candidates"]}, set(cases[0]["candidates"]))

    def test_ambiguous_and_disagreed_labels_quarantined_without_relabeling(self):
        cases = [{"acceptable": ["箸"]}] * 4
        decisions = [{"status": status, "confidence": confidence, "acceptable": gold} for status, confidence, gold in
            (("clear", "high", ["箸"]), ("ambiguous", "high", ["箸", "橋"]),
             ("clear", "high", ["橋"]), ("clear", "medium", ["箸"]))]
        accepted, rejected = finalize(cases, decisions)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 3)
        self.assertTrue(all(case["acceptable"] == ["箸"] for case in rejected))

    def test_exact_job_count_and_length_matching_enforced(self):
        job = make_jobs(CONFIG, POOLS)[0]
        row = {"context": "今日の会議ではまず", "positive": "手順を確認します", "negative": "手順を忘却します", "reason_zh": "说明"}
        self.assertEqual(len(parse_generated(body([row]), job)), 1)
        with self.assertRaisesRegex(ValueError, "count"):
            parse_generated(body([]), job)
        with self.assertRaisesRegex(ValueError, "length"):
            parse_generated(body([{**row, "negative": "忘れます"}]), job)

    def test_homophone_candidate_leak_is_rejected(self):
        job = next(job for job in make_jobs(CONFIG, POOLS) if job["task"] == "homophone")
        with self.assertRaisesRegex(ValueError, "leaked"):
            parse_generated(body([{"context": "食べるときには箸を", "reason_zh": "说明"}]), job)

    def test_nfkc_context_dedup_even_if_candidate_order_or_label_changes(self):
        jobs = make_jobs(CONFIG, POOLS)[:2]
        item = {"context": "今日の会議は１回", "candidates": ["手順を確認します", "手順を忘却します"], "acceptable": ["手順を確認します"], "reason_zh": "说明"}
        generated = {jobs[0]["id"]: [item], jobs[1]["id"]: [{**item, "context": "今日の会議は1回", "acceptable": ["手順を忘却します"]}]}
        cases, duplicate = canonical_cases(jobs, generated, CONFIG)
        self.assertEqual(len(cases), 1)
        self.assertEqual(len(duplicate), 1)

    def test_verifier_ids_and_clear_singleton_are_strict(self):
        cases = [{"candidates": ["箸", "橋"]}]
        row = {"id": "0", "status": "clear", "confidence": "high", "acceptable_ids": ["A"], "reason_zh": "说明"}
        self.assertEqual(parse_verified(body([row]), cases)[0]["acceptable"], ["箸"])
        for changed in ({"id": "00"}, {"acceptable_ids": ["A", "B"]}, {"acceptable_ids": ["Z"]}):
            with self.assertRaises(ValueError):
                parse_verified(body([{**row, **changed}]), cases)

    def test_tie_accuracy_is_independent_of_gold_position(self):
        row = {"acceptable": ["箸"], "contextual": {"candidates": [{"text": word, "log_probability_sum": -1.} for word in ["箸", "橋"]]}}
        result = accuracy([row])
        self.assertEqual(result["top1_accuracy"], .5)
        self.assertEqual(result["top2_accuracy"], 1.)
        self.assertEqual(result["mrr"], .75)
        row["contextual"]["candidates"].reverse()
        self.assertEqual(accuracy([row]), result)

    def test_homophone_group_candidate_order_stable_across_targets(self):
        jobs = [job for job in make_jobs(CONFIG, POOLS) if job["task"] == "homophone"]
        generated = {job["id"]: [{"context": f"今日の作業は{i}番目", "candidates": job["candidates"],
            "acceptable": [job["target"]], "reason_zh": "说明"}] for i, job in enumerate(jobs)}
        cases, _ = canonical_cases(jobs, generated, CONFIG)
        self.assertEqual(cases[0]["candidates"], cases[1]["candidates"])

    def test_conditional_model_beats_prior_and_tie_aware_length_baseline(self):
        class Processor:
            def encode(self, text, out_type=int):
                return [{"a": 4, "b": 5, "c": 6}[char] for char in text]
            def decode(self, ids):
                return "".join({4: "a", 5: "b", 6: "c"}.get(token, "") for token in ids)
            def id_to_piece(self, token):
                return str(token)
        class ConditionalModel:
            config = SimpleNamespace(context_length=16)
            def __call__(self, inputs):
                logits = torch.full((*inputs.shape, 8), -5.)
                logits[:, :, 5] = torch.where(inputs == 5, 6., 4.)
                logits[:, :, 6] = torch.where(inputs == 4, 6., 3.)
                return logits
        lm = JapaneseLM.__new__(JapaneseLM)
        lm.device, lm.model, lm.processor = torch.device("cpu"), ConditionalModel(), Processor()
        lm.special = dict(pad=0, unk=1, bos=2, eos=3)
        cases = [{"id": str(i), "task": "phrase_contrast", "category": "fixture", "context": context,
            "candidates": ["b", "c"], "acceptable": [gold]} for i, (context, gold) in enumerate((("a", "c"), ("b", "b")))]
        with contextlib.redirect_stdout(io.StringIO()):
            rows, metrics = evaluate(lm, cases)
        summary = metrics["by_task"]["phrase_contrast"]
        self.assertEqual(summary["contextual_sum_primary"]["top1_accuracy"], 1.)
        self.assertEqual(summary["context_free_sum"]["top1_accuracy"], .5)
        self.assertEqual(summary["shorter_token_baseline"]["top1_accuracy"], .5)
        self.assertEqual(summary["equal_scored_token_length"]["cases"], 2)

    def test_nonfinite_scores_cannot_be_reported_as_accuracy(self):
        row = {"acceptable": ["箸"], "contextual": {"candidates": [{"text": "箸", "log_probability_sum": float("nan")},
            {"text": "橋", "log_probability_sum": -1.}]}}
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            accuracy([row])


if __name__ == "__main__":
    unittest.main()
