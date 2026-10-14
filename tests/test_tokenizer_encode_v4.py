"""V4 TXT export: independent sequences, provenance and loss-preserving windows."""
import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.tokenizer import encode_v4 as v4
from vimeml.tokenizer.store import TokenStore
from vimeml.training.data import SentenceWindowDataset


class V4EncodingTests(unittest.TestCase):
    def fixture(self, root):
        import sentencepiece as spm
        corpus, tokenizer, source_store, experiment = [root / p for p in ("corpus", "tokenizer", "source", "experiment")]
        for p in (corpus, tokenizer, source_store, experiment):
            p.mkdir()
        original = ROOT / "artifacts/tokenizers/v4-a/ja-unigram-8192"
        model = tokenizer / "tokenizer.model"
        model.write_bytes((original / "tokenizer.model").read_bytes())
        v4.dump_json(tokenizer / "manifest.json", v4.read_json(original / "manifest.json"))
        processor = spm.SentencePieceProcessor(model_file=str(model))
        texts = {"train": ["日本語🙂。", "  空白  ", "か\u3099、ｶﾞ。", "日本語" * 400, "tab\there", "最後の行"],
                 "validation": ["𠮷野家。", "短い文", "検証" * 300]}
        corpus_stats, tokenizer_stats = {}, {}
        for split, rows in texts.items():
            (corpus / f"{split}.txt").write_text("\n".join(rows), encoding="utf-8", newline="\n")
            source = bytes(i % 2 for i in range(len(rows)))
            (source_store / f"{split}.sources.bin").write_bytes(source)
            corpus_stats[split] = {"sentences": len(rows), "characters": sum(map(len, rows)),
                                  "primary_sources": dict(Counter("fineweb" if i == 0 else "tatoeba" for i in source))}
            encoded = processor.encode(rows)
            tokenizer_stats[split] = {"tokens_without_special_tokens": sum(map(len, encoded)),
                "tokens_with_bos_eos_per_sentence": sum(map(len, encoded)) + 2 * len(rows),
                "byte_fallback_tokens": sum(processor.is_byte(i) for ids in encoded for i in ids)}
        v4.dump_json(corpus / "manifest.json", {"status": "complete", "ready_for_lm_training": False})
        v4.dump_json(corpus / "stats.json", {"splits": corpus_stats})
        txt_hashes = {f"{s}.txt": v4.file_sha(corpus / f"{s}.txt") for s in v4.SPLITS}
        corpus_sha = v4.file_sha(corpus / "manifest.json")
        v4.dump_json(corpus / "integrity-check.json", {"status": "passed", "manifest_sha256": corpus_sha,
            "exports": {name: {"sha256": digest} for name, digest in txt_hashes.items()}})
        v4.dump_json(source_store / "manifest.json", {"status": "complete", "source_ids": {"fineweb": 0, "tatoeba": 1},
            "splits": corpus_stats, "input_sha256": {str(corpus / name): digest for name, digest in
                {**txt_hashes, "manifest.json": corpus_sha}.items()} | {str(tokenizer / "manifest.json"): "other-manifest"},
            "output_sha256": {f"{s}.sources.bin": v4.file_sha(source_store / f"{s}.sources.bin") for s in v4.SPLITS}})
        v4.dump_json(tokenizer / "stats.json", {"splits": tokenizer_stats})
        v4.dump_json(experiment / "manifest.json", {"status": "complete", "primary_model": str(model),
            "sample": {"train_sha256": txt_hashes["train.txt"]}, "validation_sha256": txt_hashes["validation.txt"]})
        output, index = root / "tokens", root / "indexes"
        paths = {"corpus_dir": corpus, "tokenizer_dir": tokenizer, "source_store_dir": source_store,
                 "tokenizer_experiment_dir": experiment, "output_dir": output, "index_dir": index}
        config = root / "data.toml"
        config.write_text("[paths]\n" + "\n".join(f"{k} = {json.dumps(p.as_posix())}" for k, p in paths.items()) +
            '\n[encoding]\nworkers = 2\nbatch_size = 2\nchunk_mib = 1\n[windows]\ncontext_length = 128\n' +
            '[checks]\nseed = 42\nbatch_size = 2\nnum_workers = 0\ncheck_batches = 2\n', encoding="utf-8")
        return config, corpus, output, index, texts, processor

    def test_partition_boundaries_and_source_row_positions(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temp:
            root = Path(temp)
            _, corpus, _, _, texts, processor = self.fixture(root)
            expected = v4.read_json(corpus / "stats.json")["splits"]["train"]
            for chunk in (1, 7, 100, 10000):
                jobs, stats = v4.scan_txt(corpus / "train.txt", "train", chunk, expected, lambda *_: None)
                self.assertEqual(stats["sha256"], v4.file_sha(corpus / "train.txt"))
                restored = []
                first = end = 0
                for job in jobs:
                    self.assertEqual(job["first_row"], first)
                    self.assertEqual(job["start"], end)
                    with (corpus / "train.txt").open("rb") as stream:
                        stream.seek(job["start"])
                        restored.extend(stream.read(job["end"] - job["start"]).decode("utf-8").splitlines())
                    first += job["sentences"]
                    end = job["end"]
                self.assertEqual(restored, texts["train"])
                if chunk == 100:
                    parts = root / "parts"
                    parts.mkdir()
                    encoded_bytes, actual_sources = bytearray(), bytearray()
                    for i, job in enumerate(jobs):
                        job.update(name=f"train-{i}", model=str(root / "tokenizer/tokenizer.model"),
                                   source_input=str(root / "source/train.sources.bin"),
                                   parts=str(parts), batch_size=2, signature="test",
                                   source_ids={"fineweb": 0, "tatoeba": 1})
                        v4.encode_txt_part(job)
                        files = v4.part_paths(job)
                        encoded_bytes.extend(files["tokens.bin"].read_bytes())
                        actual_sources.extend(files["sources.bin"].read_bytes())
                    expected_tokens = [i for text in texts["train"] for i in [2, *processor.encode(text), 3]]
                    self.assertEqual(encoded_bytes, struct.pack(f"<{len(expected_tokens)}H", *expected_tokens))
                    self.assertEqual(actual_sources, (root / "source/train.sources.bin").read_bytes())

    def test_full_export_window_coverage_and_completed_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temp:
            root = Path(temp)
            config, corpus, output, index, texts, processor = self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                v4.run(config)
            manifest = v4.read_json(output / "manifest.json")
            self.assertEqual(set(manifest["splits"]), {"train", "validation"})
            self.assertFalse((output / "test.tokens.bin").exists())
            for split, rows in texts.items():
                with TokenStore(output, split) as store:
                    for i, text in enumerate(rows):
                        self.assertEqual(store[i], [2, *processor.encode(text), 3])
                        self.assertEqual(processor.decode(store[i][1:-1]), text)
                positions = [p[0] for p in struct.iter_unpack("<Q", (output / f"{split}.rows.bin").read_bytes())]
                with (corpus / f"{split}.txt").open("rb") as stream:
                    for position, text in zip(positions, rows, strict=True):
                        stream.seek(position)
                        self.assertEqual(stream.readline().decode("utf-8").removesuffix("\n"), text)
                dataset = SentenceWindowDataset(output, index, split)
                labels = {i: [] for i in range(len(rows))}
                try:
                    for i in range(len(dataset)):
                        sample = dataset[i]
                        self.assertLessEqual(len(sample["input_ids"]), 128)
                        labels[sample["sentence_index"]].extend(sample["labels"])
                    for i, text in enumerate(rows):
                        self.assertEqual(labels[i], [*processor.encode(text), 3])
                finally:
                    dataset.close()
            before = (output / "manifest.json").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), patch.object(v4, "encode_txt_part", side_effect=AssertionError("re-encoded")):
                v4.run(config, resume=True)
            self.assertEqual((output / "manifest.json").read_bytes(), before)
            self.assertFalse((output / ".parts").exists())


if __name__ == "__main__":
    unittest.main()
