import contextlib
import io
import json
import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import pyarrow as pa
import pyarrow.parquet as pq

import test_corpus_parts as part_fixtures
from vimeml.data.build import run as build, sha
from vimeml.data.corpus_parts import text_sha
from vimeml.data.merge_parts_fast import run as merge_fast
from vimeml.data.preprocess_part import run as preprocess
from vimeml.data.prepare_corpus_review import prepare


class FastMergeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = part_fixtures.PartTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_parallel_merge_preserves_exact_exports_and_global_leakage_rules(self):
        fixture = self.fixture
        rows = pq.read_table(fixture.raw_paths[2]).to_pylist()
        # Different documents with a shared sentence: one train and one held-out.
        # Their document hashes determine their groups in the absence of shared aliases.
        common = "共通する文章です。"
        forced = "一つ目の内容です。"
        found = {}
        for number in range(5000):
            text = common + "\n" + forced + f"\n番号{number}の文章です。"
            value = int(sha(f"42:{sha(text)}")[:16], 16) / 2**64
            rank = "train" if value < 0.98 else "validation" if value < 0.99 else "test"
            found.setdefault(rank, {"id": f"independent-{number}", "text": text, "url": None})
            if len(found) == 3:
                break
        self.assertEqual(len(found), 3)
        rows.extend(found.values())
        # Include rule audit rows and a bracket fragment, not only valid sentences.
        rows.append({"id": "audit", "text": "https://example.org\n★★★\n閉じない（文章。\n正常な文です。", "url": None})
        pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(fixture.raw_paths[2])), fixture.raw_paths[2])
        before = text_sha(ROOT / "src/vimeml/data/merge_parts.py")
        parts = [fixture.directory / "part-0", fixture.directory / "part-1"]
        with contextlib.redirect_stdout(io.StringIO()):
            build(fixture.config)
            for index, path in enumerate(parts):
                preprocess(fixture.config, f"{index}/2", str(path))
            artifact_before = [(path / "index.sqlite").stat().st_size for path in parts]
            # All raw files may be absent on the merge machine.
            for path in fixture.raw_paths:
                path.rename(path.with_suffix(".hidden"))
            try:
                fast = fixture.directory / "fast"
                merge_fast(fixture.config, list(reversed(parts)), fast, workers=2, buckets=4, cache_mb=16)
                one = fixture.directory / "one"
                merge_fast(fixture.config, parts, one, workers=1, buckets=16, cache_mb=16)
            finally:
                for path in fixture.raw_paths:
                    path.with_suffix(".hidden").rename(path)
        self.assertEqual(text_sha(ROOT / "src/vimeml/data/merge_parts.py"), before)
        self.assertEqual(artifact_before, [(path / "index.sqlite").stat().st_size for path in parts])
        single = fixture.directory / "single"
        names = [f"{split}.{suffix}" for split in ("train", "validation", "test") for suffix in ("txt", "jsonl")]
        names += ["documents.jsonl", "provenance.jsonl", "review_blocks.jsonl", "dropped_blocks.jsonl", "fragments.jsonl"]
        for name in names:
            self.assertEqual((fast / name).read_bytes(), (one / name).read_bytes(), name)
            # The serial builder processes content in a different order than parts for audit logs.
            if name.endswith("blocks.jsonl") or name == "fragments.jsonl":
                canonical = lambda p: sorted(json.dumps(json.loads(line), sort_keys=True, ensure_ascii=False) for line in p.read_text(encoding="utf-8").splitlines())
                self.assertEqual(canonical(fast / name), canonical(single / name), name)
            else:
                self.assertEqual((fast / name).read_bytes(), (single / name).read_bytes(), name)
        stats = json.loads((fast / "stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats, json.loads((single / "stats.json").read_text(encoding="utf-8")))
        self.assertGreater(stats["cross_split_occurrences_removed"], 0)
        train = (fast / "train.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn(forced, train)
        self.assertIn(common, (fast / "test.txt").read_text(encoding="utf-8").splitlines())
        with sqlite3.connect(fast / "index.sqlite") as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertEqual(tables, {"origins", "contents", "annotations"})
            self.assertEqual(db.execute("SELECT COUNT(*) FROM origins").fetchone()[0], stats["document_origins"])
            self.assertEqual(db.execute("SELECT COUNT(*) FROM annotations WHERE matched=1").fetchone()[0], stats["matched_annotations"])
        # The compact index and normalized code fingerprint must work with the real audit tool.
        with contextlib.redirect_stdout(io.StringIO()):
            review = fixture.directory / "review"
            prepare(fast, review, workers=1, sample_scale=0.1)
        self.assertEqual(json.loads((review / "manifest.json").read_text(encoding="utf-8"))["status"], "complete")
        with self.assertRaises(FileExistsError):
            merge_fast(fixture.config, parts, fast)
        with self.assertRaisesRegex(ValueError, "overlap"):
            merge_fast(fixture.config, parts, parts[0] / "bad-output")


if __name__ == "__main__":
    unittest.main()
