import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.annotations import load_annotations, document_annotations
from vimeml.data.approve_audit import approve
from vimeml.data.build import run, sha
from vimeml.data.segment import restore_linebreaks


def label(identity, doc_text, kind, action, **snapshot):
    return {"schema_version": 1, "id": identity, "status": "approved", "approved_by": "human",
            "doc_id": "fineweb:" + identity, "doc_hash": sha(doc_text), "reason_zh": "Reviewed original context.",
            "kind": kind, "action": action, **snapshot}


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        parent = (ROOT / "outputs").resolve()
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="annotation-test-", dir=parent)
        self.directory = Path(self.temporary.name).resolve()
        assert self.directory.is_relative_to(parent)
        self.addCleanup(self.temporary.cleanup)

    def write_labels(self, records):
        path = self.directory / "labels.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
        return path

    def test_decisions_and_protected_boundaries(self):
        blocks = [{"id": f"b{i:03d}", "line_index": i, "text": text, "action": "keep"}
                  for i, text in enumerate(["土を掘", "り起こしました。"])]
        self.assertEqual(len(restore_linebreaks(blocks)), 2)
        result = restore_linebreaks(blocks, boundary_decisions={("b000", "b001"): "join"})
        self.assertEqual(result[0]["text"], "土を掘り起こしました。")
        with self.assertRaises(ValueError):
            restore_linebreaks(blocks, max_chars=4, boundary_decisions={("b000", "b001"): "join"})
        blocks[0]["text"] = "説明の"
        self.assertEqual(len(restore_linebreaks(blocks)), 1)
        self.assertEqual(len(restore_linebreaks(blocks, boundary_decisions={("b000", "b001"): "separate"})), 2)
        records = [label("protected", "x", "boundary", "join", left_block="b000", right_block="b001",
                         left_text="説明の", right_text="り起こしました。")]
        with contextlib.closing(sqlite3.connect(":memory:")) as db:
            load_annotations(db, self.write_labels(records))
            blocks[1]["action"] = "drop"
            with self.assertRaises(ValueError):
                document_annotations(db, sha("x"), blocks)
        # Explicit separation also stops bracket lookahead across three lines.
        quoted = [{"id": f"b{i:03d}", "line_index": i, "text": text, "action": "keep"}
                  for i, text in enumerate(["「始まり", "途中", "終わり」。"])]
        self.assertEqual(len(restore_linebreaks(quoted)), 1)
        self.assertEqual(len(restore_linebreaks(quoted, boundary_decisions={("b001", "b002"): "separate"})), 3)

    def test_ledger_rejects_unsafe_records(self):
        record = label("a", "text", "block", "drop", block_id="b000", text="text")
        for records in ([record, record], [{**record, "status": "pending"}],
                        [label("f", "text", "fragment", "keep", text=3, text_hash="x", cleaned_block_spans=[{"block_id": "b000", "start": 0, "end": 1}])]):
            with contextlib.closing(sqlite3.connect(":memory:")) as db, self.assertRaises(ValueError):
                load_annotations(db, self.write_labels(records))
        with contextlib.closing(sqlite3.connect(":memory:")) as db:
            load_annotations(db, self.write_labels([record]))
            with self.assertRaises(ValueError):
                document_annotations(db, sha("text"), [{"id": "b000", "line_index": 0, "text": "changed", "action": "keep"}])

    def test_builder_recovery_drop_provenance_and_leakage(self):
        inputs = self.directory / "inputs"
        inputs.mkdir()
        originals = {"recover": "ニュースです。\n大阪には製薬会社が多い", "join": "土を掘\nり起こしました。",
                     "drop": "広告です。\n本文です。", "separate": "説明の\n補足です。"}
        # Ensure a held-out document duplicates a sentence seen during calibration.
        for number in range(10000):
            text = f"別の文書{number}です。\nニュースです。"
            if int(sha(f"42:{sha(text)}")[:16], 16) / 2**64 >= .99:
                originals["heldout"] = text
                break
        self.assertIn("heldout", originals)
        pq.write_table(pa.Table.from_pylist([{"id": key, "text": value} for key, value in originals.items()]), inputs / "input.parquet")
        (inputs / "tatoeba.tsv").write_text("1\tjpn\t本文です。\n", encoding="utf-8")
        records = [
            label("recover", originals["recover"], "fragment", "keep", text="大阪には製薬会社が多い",
                  text_hash=sha("大阪には製薬会社が多い"), cleaned_block_spans=[{"block_id": "b001", "start": 0, "end": 11}]),
            label("join", originals["join"], "boundary", "join", left_block="b000", right_block="b001", left_text="土を掘", right_text="り起こしました。"),
            label("drop", originals["drop"], "block", "drop", block_id="b000", text="広告です。"),
            label("separate", originals["separate"], "boundary", "separate", left_block="b000", right_block="b001", left_text="説明の", right_text="補足です。"),
        ]
        ledger = self.write_labels(records)
        prefix = inputs.relative_to(ROOT).as_posix()
        config = self.directory / "config.toml"
        config.write_text(f'''[inputs]
fineweb_glob = "{prefix}/*.parquet"
tatoeba_path = "{prefix}/tatoeba.tsv"
[build]
quality_mode = "rules_with_approved_annotations"
output_dir = "{self.directory.relative_to(ROOT).as_posix()}/corpus"
seed = 42
batch_size = 16
max_join_chars = 2048
fineweb_documents_per_shard = 0
tatoeba_documents = 0
[review]
annotations = "{ledger.relative_to(ROOT).as_posix()}"
require_all_annotations = true
[split]
train = 0.98
validation = 0.01
test = 0.01
''', encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            run(config)
        output = self.directory / "corpus"
        sets = {}
        for name in ("train", "validation", "test"):
            rows = [json.loads(line) for line in (output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["text"] for row in rows], (output / f"{name}.txt").read_text(encoding="utf-8").splitlines())
            sets[name] = {row["text"] for row in rows}
            for row in rows:
                if row["text"] == "大阪には製薬会社が多い":
                    self.assertEqual(row["approved_annotation_ids"], ["recover"])
        self.assertTrue({"大阪には製薬会社が多い", "ニュースです。", "土を掘り起こしました。", "補足です。"} <= sets["train"])
        self.assertNotIn("広告です。", set.union(*sets.values()))
        self.assertNotIn("説明の補足です。", set.union(*sets.values()))
        self.assertFalse(sets["train"] & sets["test"])
        stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["matched_annotations"], 4)
        self.assertEqual(stats["approved_fragment_recoveries"], 1)
        self.assertEqual(stats["approved_join_events"], 1)
        self.assertGreaterEqual(stats["cross_split_occurrences_removed"], 1)
        # A changed document must fail rather than silently discard an approval.
        records[0]["doc_hash"] = "0" * 64
        self.write_labels(records)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "did not match"):
            run(config, str(self.directory / "mismatch"))

    def test_approval_cli_does_not_auto_accept_model_decisions(self):
        results = self.directory / "review" / "results.json"
        results.parent.mkdir()
        cases = [{"id": "F01", "kind": "fragment_candidate", "doc_id": "fineweb:1", "doc_hash": sha("a"),
                  "text": "大阪には製薬会社が多い", "cleaned_block_spans": [{"block_id": "b000", "start": 0, "end": 11}],
                  "assessment": "issue", "reason_zh": "Model evidence only."}]
        results.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
        output = self.directory / "approved.jsonl"
        approve(results, output, ["F01"], "keep", "human", "Original context reviewed.")
        before = output.read_bytes()
        with self.assertRaises(ValueError):
            approve(results, output, ["F01"], "keep", "human", "Duplicate.")
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
