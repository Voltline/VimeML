import contextlib
import io
import json
import unittest
from pathlib import Path

import test_corpus_parts as fixtures
from vimeml.data.build import run as build
from vimeml.data.build_parallel import run as parallel


class ParallelTests(unittest.TestCase):
    def setUp(self):
        # Reuse the cross-file duplicate/URL/approval fixture, not its tests.
        fixtures.PartTests.setUp(self)

    def test_real_processes_match_single_machine_and_resume_completed_workers(self):
        output = self.directory / "parallel"
        with contextlib.redirect_stdout(io.StringIO()):
            build(self.config)
            parallel(self.config, str(output), workers=2, progress_seconds=.1)
        single = self.directory / "single"
        names = ("train.txt", "validation.txt", "test.txt", "train.jsonl", "validation.jsonl", "test.jsonl", "documents.jsonl", "provenance.jsonl")
        for name in names:
            self.assertEqual((output / name).read_bytes(), (single / name).read_bytes(), name)
        self.assertEqual(json.loads((output / "stats.json").read_text(encoding="utf-8")),
                         json.loads((single / "stats.json").read_text(encoding="utf-8")))
        work = output.with_name(output.name + "-work")
        summary = json.loads((work / "run-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(set(summary["worker_pids"].values())), 2)
        self.assertEqual(summary["document_origins"], 10)
        manifests = [work / "parts" / f"worker-{index:03d}" / "manifest.json" for index in range(2)]
        before = [path.stat().st_mtime_ns for path in manifests]
        backup = self.directory / "finished-backup"
        assert output.resolve().is_relative_to(self.directory) and backup.resolve().is_relative_to(self.directory)
        output.rename(backup)
        output.mkdir()
        (output / "interrupted-marker").write_text("partial export", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            parallel(self.config, str(output), workers=2, resume=True, progress_seconds=.1)
        self.assertEqual(before, [path.stat().st_mtime_ns for path in manifests])
        summary = json.loads((work / "run-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["reused_workers"], [0, 1])
        self.assertEqual(summary["worker_pids"], {})
        preserved = list(self.directory.glob("parallel-interrupted-*"))
        self.assertEqual(len(preserved), 1)
        self.assertEqual((preserved[0] / "interrupted-marker").read_text(encoding="utf-8"), "partial export")
        for name in names:
            self.assertEqual((output / name).read_bytes(), (backup / name).read_bytes(), name)
        with self.assertRaisesRegex(FileExistsError, "--resume"):
            parallel(self.config, str(output), workers=2)
        with self.assertRaisesRegex(ValueError, "same output"):
            parallel(self.config, str(output), workers=3, resume=True)

    def test_worker_failure_is_reported_without_final_corpus(self):
        self.raw_paths[-1].write_text("malformed TSV\n", encoding="utf-8")
        output = self.directory / "failed"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Worker .* failed"):
            parallel(self.config, str(output), workers=2, progress_seconds=.1)
        self.assertFalse((output / "manifest.json").exists())
        work = output.with_name(output.name + "-work")
        log = (work / "logs" / "worker-001.log").read_text(encoding="utf-8")
        self.assertIn("TSV", log)


if __name__ == "__main__":
    unittest.main()
