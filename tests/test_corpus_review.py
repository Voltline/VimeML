import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.build import file_sha, run as build, sha
from vimeml.data.prepare_corpus_review import prepare, lines_in_range, ranges, write_json, PROMPT
from vimeml.data.review_corpus_parallel import (run as review, QuotaLimiter, parse_decisions,
                                              retry_after, make_batches, payload_for, token_reservation)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "outputs"
        parent.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="review-test-", dir=parent)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        raw = self.directory / "inputs"
        raw.mkdir()
        texts = [
            "美容鍼灸って、お顔を綺麗にするのが\n目的だけだと思っていませんか?",
            "人付き合\nいのコツは人それぞれです。",
            "今日は買い物に行きます。\n\nhttps://example.com\n★★★\n",
            "料理が冷めにくいという点も長所です。Next",
            "文字が�壊れています。",
            "明日は学校へ行きます",
            "「閉じない表現",
            "突然閉じる表現」",
            "商品 A123 の価格は 1500 円です。",
            "これは長い文章であり、" + "日本語の説明を続けます、" * 20 + "最後まで読めます。",
            "これは自然な表現です",
            "彼が\n彼女は先に帰りました。",
        ]
        paths = []
        for i in range(2):
            path = raw / f"{i}.parquet"
            rows = [{"id": f"{i}-{j}", "text": text} for j, text in enumerate(texts)]
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
            paths.append(path)
        tatoeba = raw / "tatoeba.tsv"
        tatoeba.write_text("1\tjpn\t猫が好きです。\n2\tjpn\t私は学生です。\n", encoding="utf-8")
        paths.append(tatoeba)
        ledger = self.directory / "labels.jsonl"
        labels = [
            {"schema_version": 1, "id": "keep", "status": "approved", "approved_by": "human",
             "doc_id": "fineweb:0-10", "doc_hash": sha(texts[10]), "kind": "fragment", "action": "keep",
             "reason_zh": "人工确认完整表达", "text": texts[10], "text_hash": sha(texts[10]),
             "cleaned_block_spans": [{"block_id": "b000", "start": 0, "end": len(texts[10])}]},
            {"schema_version": 1, "id": "separate", "status": "approved", "approved_by": "human",
             "doc_id": "fineweb:0-11", "doc_hash": sha(texts[11]), "kind": "boundary", "action": "separate",
             "reason_zh": "保持已确认的原始边界", "left_block": "b000", "right_block": "b001",
             "left_text": "彼が", "right_text": "彼女は先に帰りました。"},
        ]
        ledger.write_text("".join(json.dumps(label, ensure_ascii=False) + "\n" for label in labels), encoding="utf-8")
        calibration = self.directory / "calibration.json"
        self.calibration = calibration
        calibration.write_text(json.dumps([f"fineweb:{i}-{j}" for i in range(2) for j in range(len(texts))]
                                           + ["tatoeba:1", "tatoeba:2"]), encoding="utf-8")
        rel = lambda path: path.relative_to(ROOT).as_posix()
        config = self.directory / "config.toml"
        self.config = config
        self.corpus = self.directory / "corpus"
        config.write_text(f'''[inputs]
fineweb_glob = "{rel(raw)}/*.parquet"
tatoeba_path = "{rel(tatoeba)}"
calibration_ids = "{rel(calibration)}"
[build]
quality_mode = "rules_with_approved_annotations"
output_dir = "{rel(self.corpus)}"
seed = 42
batch_size = 4
max_join_chars = 2048
fineweb_documents_per_shard = 0
tatoeba_documents = 0
commit_every = 20
progress_every = 100
[review]
annotations = "{rel(ledger)}"
require_all_annotations = true
[split]
train = 0.98
validation = 0.01
test = 0.01
''', encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            build(config)

    def test_byte_ranges_cover_unicode_and_line_boundaries_exactly(self):
        path = self.directory / "unicode.jsonl"
        rows = [{"text": "日本語🙂", "i": i} for i in range(12)]
        path.write_bytes("\n".join(json.dumps(row, ensure_ascii=False) for row in rows).encode("utf-8"))
        for workers in range(1, 20):
            actual = [row for start, end in ranges(path, workers) for row in lines_in_range(path, start, end)]
            self.assertEqual(actual, rows)

    def test_preparation_reproducible_parallel_and_replays_labels_read_only(self):
        baseline = {name: file_sha(self.corpus / name) for name in ("manifest.json", "index.sqlite", "train.jsonl")}
        outputs = [self.directory / "one", self.directory / "three"]
        with contextlib.redirect_stdout(io.StringIO()):
            for workers, output in zip((1, 3), outputs):
                prepare(self.corpus, output, workers=workers)
        self.assertEqual((outputs[0] / "cases.json").read_bytes(), (outputs[1] / "cases.json").read_bytes())
        cases = json.loads((outputs[0] / "cases.json").read_text(encoding="utf-8"))
        reviews = [case for case in cases if case["kind"] == "review_block"]
        self.assertEqual(len(reviews), 2)  # Duplicate documents on two shards coalesce.
        self.assertTrue(all(len(case["origin_references"]) == 2 for case in reviews))
        self.assertTrue(any(case["kind"] == "retained_sentence" and case["text"] == "これは自然な表現です" for case in cases))
        self.assertTrue(any(case["kind"] == "joined_boundary" and case["processing"]["target_joins"] for case in cases))
        self.assertTrue(any(case["kind"] == "boundary_candidate" and case["left_text"] == "人付き合" for case in cases))
        self.assertTrue(any(case["kind"] == "boundary_candidate" and case["left_text"] == "彼が"
                            and case["processing"]["paragraph"]["text"] == "彼が" for case in cases))
        self.assertTrue(all(case["assigned_split"] == "train" for case in cases if case["kind"] != "review_block"))
        self.assertEqual(baseline, {name: file_sha(self.corpus / name) for name in baseline})
        with self.assertRaises(FileExistsError):
            prepare(self.corpus, outputs[0])

    def test_diagnostic_samples_train_only_but_review_all_is_exhaustive(self):
        self.calibration.write_text("[]", encoding="utf-8")
        heldout = self.directory / "heldout-corpus"
        config = self.config.read_text(encoding="utf-8").replace(
            self.corpus.relative_to(ROOT).as_posix(), heldout.relative_to(ROOT).as_posix())
        config = config.replace("train = 0.98", "train = 0.4").replace("validation = 0.01", "validation = 0.3").replace("test = 0.01", "test = 0.3")
        path = self.directory / "heldout.toml"
        path.write_text(config, encoding="utf-8")
        output = self.directory / "heldout-audit"
        with contextlib.redirect_stdout(io.StringIO()):
            build(path)
            prepare(heldout, output, workers=2)
        with contextlib.closing(sqlite3.connect(heldout / "index.sqlite")) as db:
            self.assertGreater(db.execute("SELECT COUNT(*) FROM contents WHERE split_rank!=0").fetchone()[0], 0)
        cases = json.loads((output / "cases.json").read_text(encoding="utf-8"))
        self.assertTrue(all(case["assigned_split"] == "train" for case in cases if case["kind"] != "review_block"))
        self.assertEqual(sum(case["kind"] == "review_block" for case in cases), 2)

    def test_incomplete_or_changed_code_manifest_rejected_before_sampling(self):
        path = self.corpus / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["status"] = "working"
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "not complete"):
            prepare(self.corpus, self.directory / "incomplete")
        manifest["status"] = "complete"
        manifest["code_sha256"]["src/vimeml/data/segment.py"] = "changed"
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "versions differ"):
            prepare(self.corpus, self.directory / "changed")


class ReviewerTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "outputs"
        parent.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="api-review-test-", dir=parent)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.materials = self.directory / "materials"
        self.materials.mkdir()
        self.cases = [{"id": f"C{i}", "kind": "retained_sentence", "text": "自然な文章です。",
                       "source_file": "fixture.parquet", "row_index": i, "assigned_split": "train",
                       "sampling_strata": ["keep:fixture.parquet"],
                       "original_lines": [{"block_id": "b000", "text": "自然な文章です。", "truncated": False}]}
                      for i in range(8)]
        write_json(self.materials / "cases.json", self.cases)
        (self.materials / "prompt.txt").write_text(PROMPT, encoding="utf-8")
        self.manifest = {"status": "complete", "schema_version": 1,
                         "cases_sha256": file_sha(self.materials / "cases.json"),
                         "prompt_sha256": file_sha(self.materials / "prompt.txt"),
                         "quality_rate_note": "fixture, not a corpus quality rate"}
        write_json(self.materials / "manifest.json", self.manifest)
        self.output = self.directory / "review"

    @staticmethod
    def response(payload, reason="没有明确问题"):
        data = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
        rows = [{"id": case["id"], "assessment": "ok", "issue_type": "none", "suggested_action": "keep", "reason_zh": reason}
                for case in data]
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(rows, ensure_ascii=False)}}],
                "usage": {"total_tokens": 50}}

    def test_dry_run_no_key_no_files_no_transport_and_changed_material_rejected(self):
        def forbidden(*args):
            self.fail("Dry run sent a request")
        with contextlib.redirect_stdout(io.StringIO()):
            result = review(self.materials, self.output, dry_run=True, transport=forbidden)
        self.assertEqual(result["network_requests_sent"], 0)
        self.assertFalse(self.output.exists())
        (self.materials / "cases.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "material changed"):
            review(self.materials, self.output, dry_run=True)

    def test_parallel_requests_and_case_cache_survive_batch_size_changes(self):
        state, lock = {"active": 0, "peak": 0, "calls": 0}, threading.Lock()
        def fake(endpoint, payload, key, timeout):
            with lock:
                state["active"] += 1
                state["calls"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.06)
            with lock:
                state["active"] -= 1
            return 200, {}, json.dumps(self.response(payload), ensure_ascii=False)
        with contextlib.redirect_stdout(io.StringIO()):
            first = review(self.materials, self.output, "test-only-secret", workers=3, batch_size=2,
                           transport=fake, limiter=QuotaLimiter(100, 1000000, window=0.01))
            before = state["calls"]
            second = review(self.materials, self.output, "", workers=1, batch_size=1, transport=fake)
        self.assertGreaterEqual(state["peak"], 2)
        self.assertEqual(first["pending_cases"], 0)
        self.assertEqual(second["pending_cases"], 0)
        self.assertEqual(state["calls"], before)
        self.assertEqual(second["requests_this_run"]["cached_cases"], 8)

    def test_429_retries_and_raw_responses_redact_secret(self):
        calls = []
        def fake(endpoint, payload, key, timeout):
            calls.append(1)
            if len(calls) == 1:
                return 429, {"Retry-After": "0.01"}, "error test-only-secret"
            return 200, {}, json.dumps(self.response(payload, reason="test-only-secret"), ensure_ascii=False)
        with contextlib.redirect_stdout(io.StringIO()):
            stats = review(self.materials, self.output, "test-only-secret", workers=1, batch_size=10,
                           retries=1, transport=fake, limiter=QuotaLimiter(100, 1000000, window=0.01))
        self.assertEqual(stats["pending_cases"], 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(stats["requests_this_run"]["failed_attempts"], 1)
        for path in self.output.rglob("*.json"):
            self.assertNotIn("test-only-secret", path.read_text(encoding="utf-8"))
        self.assertEqual(len(list((self.output / "responses").glob("*.json"))), 2)

    def test_auth_failure_stops_remaining_batches_and_preserves_pending(self):
        calls = []
        def fake(*args):
            calls.append(1)
            return 401, {}, "Unauthorized"
        with contextlib.redirect_stdout(io.StringIO()):
            stats = review(self.materials, self.output, "test-only-secret", workers=1, batch_size=1,
                           transport=fake, limiter=QuotaLimiter(100, 1000000, window=0.01))
        self.assertEqual(len(calls), 1)
        self.assertEqual(stats["pending_cases"], 8)
        self.assertEqual(json.loads((self.output / "pending.json").read_text()), [case["id"] for case in self.cases])

    def test_malformed_reply_saved_and_no_partial_results_accepted(self):
        def fake(endpoint, payload, key, timeout):
            body = self.response(payload)
            rows = json.loads(body["choices"][0]["message"]["content"])
            body["choices"][0]["message"]["content"] = json.dumps(rows[:-1])
            return 200, {}, json.dumps(body)
        with contextlib.redirect_stdout(io.StringIO()):
            stats = review(self.materials, self.output, "test-only-secret", retries=0, transport=fake)
        self.assertEqual(stats["pending_cases"], 8)
        self.assertFalse(list((self.output / "decisions").glob("*.json")))
        self.assertEqual(len(list((self.output / "responses").glob("*.json"))), 1)
        calls = []
        def valid(endpoint, payload, key, timeout):
            calls.append(1)
            return 200, {}, json.dumps(self.response(payload))
        with contextlib.redirect_stdout(io.StringIO()):
            recovered = review(self.materials, self.output, "test-only-secret", batch_size=2, transport=valid,
                               limiter=QuotaLimiter(100, 1000000, window=0.01))
        self.assertEqual(recovered["pending_cases"], 0)
        self.assertEqual(len(calls), 4)

    def test_model_change_requires_new_output_and_oversized_case_not_sent(self):
        def fake(endpoint, payload, key, timeout):
            return 200, {}, json.dumps(self.response(payload))
        with contextlib.redirect_stdout(io.StringIO()):
            review(self.materials, self.output, "test-only-secret", transport=fake)
        with self.assertRaisesRegex(ValueError, "different materials"):
            review(self.materials, self.output, "", model="another-model", dry_run=True)
        big = {**self.cases[0], "text": "文字" * 50000}
        batches, oversized = make_batches([big, self.cases[1]], PROMPT, "model", 10, 2048, 10000)
        self.assertEqual(oversized, [big])
        self.assertEqual(batches, [[self.cases[1]]])

    def test_shared_rolling_quota_limits_concurrent_threads(self):
        limiter, stop = QuotaLimiter(rpm=2, tpm=100, window=0.10), threading.Event()
        starts, lock = [], threading.Lock()
        def acquire():
            ticket = limiter.acquire(60, stop)
            with lock:
                starts.append(ticket["start"])
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(lambda _: acquire(), range(3)))
        starts.sort()
        self.assertGreaterEqual(starts[1] - starts[0], 0.095)
        self.assertGreaterEqual(starts[2] - starts[1], 0.095)
        self.assertEqual(retry_after({"Retry-After": "12"}, 60), 12)


if __name__ == "__main__":
    unittest.main()
