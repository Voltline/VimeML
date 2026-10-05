import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.build import run as build, sha
from vimeml.data.preprocess_part import run as preprocess
from vimeml.data.merge_parts import run as merge, validate_parts
from vimeml.data.corpus_parts import text_sha


class PartTests(unittest.TestCase):
    def setUp(self):
        parent = (ROOT / "outputs").resolve()
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="parts-test-", dir=parent)
        self.directory = Path(self.temporary.name).resolve()
        assert self.directory.is_relative_to(parent)
        self.addCleanup(self.temporary.cleanup)
        inputs = self.directory / "inputs"
        inputs.mkdir()
        recovered = "自然な表現です"
        repeated = "重複した文章です。\n同じ続きです。"
        rows = [
            [{"id": "recover", "text": recovered}, {"id": "d1", "text": repeated},
             {"id": "shared", "text": "一つ目の内容です。", "url": "https://example.org/a"}],
            [{"id": "recover-copy", "text": recovered}, {"id": "d2", "text": repeated},
             {"id": "shared", "text": "二つ目の内容です。", "url": "https://example.org/b"}],
            [{"id": "third", "text": "三つ目の内容です。", "url": "https://example.org/b#fragment"},
             {"id": "other", "text": "別の文章です。\n\n補足の文章です。"}],
        ]
        self.raw_paths = []
        for number, table_rows in enumerate(rows):
            # Explicit schema retains URLs even when first rows have none.
            table = pa.Table.from_pylist(table_rows, schema=pa.schema([("id", pa.string()), ("text", pa.string()), ("url", pa.string())]))
            path = inputs / f"{number}.parquet"
            pq.write_table(table, path)
            self.raw_paths.append(path)
        tatoeba = inputs / "tatoeba.tsv"
        tatoeba.write_text("1\tjpn\t重複した文章です。\n2\tjpn\t短い文です。\n", encoding="utf-8")
        self.raw_paths.append(tatoeba)
        ledger = self.directory / "approved.jsonl"
        record = {"schema_version": 1, "id": "recover", "status": "approved", "approved_by": "human",
                  "doc_id": "fineweb:recover", "doc_hash": sha(recovered), "kind": "fragment", "action": "keep",
                  "reason_zh": "Reviewed complete expression.", "text": recovered, "text_hash": sha(recovered),
                  "cleaned_block_spans": [{"block_id": "b000", "start": 0, "end": len(recovered)}]}
        ledger.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        ids = self.directory / "calibration.json"
        ids.write_text(json.dumps(["fineweb:shared"]), encoding="utf-8")
        cases = self.directory / "cases.json"
        cases.write_text(json.dumps([{"doc_id": "fineweb:shared"}]), encoding="utf-8")
        rel = lambda p: p.relative_to(ROOT).as_posix()
        self.config = self.directory / "config.toml"
        self.config.write_text(f'''[inputs]
fineweb_glob = "{rel(inputs)}/*.parquet"
fineweb_paths = {json.dumps([rel(p) for p in self.raw_paths[:-1]])}
tatoeba_path = "{rel(tatoeba)}"
calibration_ids = "{rel(ids)}"
calibration_cases = ["{rel(cases)}"]
[build]
quality_mode = "rules_with_approved_annotations"
output_dir = "{rel(self.directory)}/single"
seed = 42
batch_size = 16
max_join_chars = 2048
fineweb_documents_per_shard = 0
tatoeba_documents = 0
commit_every = 2
progress_every = 100
[review]
annotations = "{rel(ledger)}"
require_all_annotations = true
[split]
train = 0.98
validation = 0.01
test = 0.01
''', encoding="utf-8")

    def test_code_fingerprints_ignore_git_line_endings(self):
        first, second = self.directory / "lf.txt", self.directory / "crlf.txt"
        first.write_bytes(b"text\nsecond line\n")
        second.write_bytes(b"\xef\xbb\xbftext\r\nsecond line\r\n")
        self.assertEqual(text_sha(first), text_sha(second))
        second.write_bytes(b"text\r\nchanged line\r\n")
        self.assertNotEqual(text_sha(first), text_sha(second))

    def test_two_parts_match_single_machine_without_remote_raw_files(self):
        first, second = self.directory / "windows", self.directory / "mac"
        with contextlib.redirect_stdout(io.StringIO()):
            build(self.config)
            # Simulate a Mac with only its assigned raw inputs.
            absent = [self.raw_paths[0], self.raw_paths[2]]
            for path in absent:
                path.rename(path.with_suffix(".hidden"))
            try:
                preprocess(self.config, "1/2", str(second))
            finally:
                for path in absent:
                    path.with_suffix(".hidden").rename(path)
            preprocess(self.config, "0/2", str(first))
            # Merge requires no raw files: temporarily move all inputs aside.
            for path in self.raw_paths:
                path.rename(path.with_suffix(".hidden"))
            try:
                merged = self.directory / "merged"
                merge(self.config, [second, first], merged)
            finally:
                for path in self.raw_paths:
                    path.with_suffix(".hidden").rename(path)
        single = self.directory / "single"
        for name in ("train.txt", "validation.txt", "test.txt", "train.jsonl", "validation.jsonl", "test.jsonl", "documents.jsonl", "provenance.jsonl"):
            self.assertEqual((merged / name).read_bytes(), (single / name).read_bytes(), name)
        self.assertEqual(json.loads((merged / "stats.json").read_text(encoding="utf-8")),
                         json.loads((single / "stats.json").read_text(encoding="utf-8")))
        manifest = json.loads((merged / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["stage"], "merged_corpus_staging")
        self.assertEqual(sum(manifest["selected_documents"].values()), 10)
        self.assertEqual(len(manifest["input_sha256"]), 4)
        documents = [json.loads(line) for line in (merged / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
        chain = [d for d in documents if d["doc_id"] in {"fineweb:shared", "fineweb:third"}]
        self.assertEqual(len({d["group_id"] for d in chain}), 1)
        self.assertEqual({d["assigned_split"] for d in chain}, {"train"})
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_parts(self.config, [first])
        with self.assertRaisesRegex(ValueError, "duplicated part"):
            validate_parts(self.config, [first, first])
        changed_config = self.directory / "changed.toml"
        changed_config.write_text(self.config.read_text(encoding="utf-8").replace("max_join_chars = 2048", "max_join_chars = 2049"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "versions differ"):
            validate_parts(changed_config, [first, second])
        with (second / "index.sqlite").open("ab") as stream:
            stream.write(b"corrupted copy")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "damaged or changed"):
            validate_parts(self.config, [first, second])


if __name__ == "__main__":
    unittest.main()
