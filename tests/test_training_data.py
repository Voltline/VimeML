"""No lost/duplicated targets, cross-sentence leakage or unbounded shuffle lists."""
import importlib.util
import json
import pickle
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.data import (
    BlockShuffleSampler, LengthBucketBatchSampler, SentenceWindowDataset, collate_arrays,
    file_sha, make_loader, prepare_indexes, write_json,
)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "outputs")
        self.root = Path(self.temporary.name)
        self.tokens, self.indexes = self.root / "tokens", self.root / "indexes"
        self.tokens.mkdir()
        # Exact window boundaries, immediate overflows and multiple adjacent long rows.
        self.sequences = [[2, *[4 + (i % 200) for i in range(n - 2)], 3]
                          for n in (3, 128, 129, 130, 257, 258, 590, 4)]
        splits, hashes = {}, {}
        for split in ("train", "validation", "test"):
            offsets = [0]
            for sequence in self.sequences:
                offsets.append(offsets[-1] + len(sequence))
            files = {"tokens.bin": struct.pack("<" + "H" * offsets[-1], *sum(self.sequences, [])),
                     "offsets.bin": struct.pack("<" + "Q" * len(offsets), *offsets),
                     "sources.bin": bytes(i % 2 for i in range(len(self.sequences)))}
            for name, content in files.items():
                path = self.tokens / f"{split}.{name}"
                path.write_bytes(content)
                hashes[path.name] = file_sha(path)
            splits[split] = {"sentences": len(self.sequences), "stored_tokens": offsets[-1],
                             "prediction_pairs": offsets[-1] - len(self.sequences),
                             "max_sequence_tokens": max(map(len, self.sequences))}
        write_json(self.tokens / "manifest.json", {"status": "complete",
                   "format": "vimeml_sentence_tokens_v1", "token_dtype": "uint16_le",
                   "offset_dtype": "uint64_le", "offset_unit": "tokens", "vocab_size": 256,
                   "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
                   "source_ids": {"fineweb": 0, "tatoeba": 1},
                   "splits": splits, "output_sha256": hashes})
        self.manifest = prepare_indexes(self.tokens, self.indexes, 128)

    def tearDown(self):
        self.temporary.cleanup()

    def test_every_original_prediction_pair_occurs_once_within_its_sentence(self):
        dataset = SentenceWindowDataset(self.tokens, self.indexes)
        actual = [[] for _ in self.sequences]
        try:
            for i in range(len(dataset)):
                sample = dataset[i]
                sentence = sample["sentence_index"]
                self.assertEqual(sample["source_id"], sentence % 2)
                self.assertLessEqual(len(sample["labels"]), 128)
                actual[sentence].extend(zip(sample["input_ids"], sample["labels"]))
            for sequence, pairs in zip(self.sequences, actual):
                self.assertEqual(pairs, list(zip(sequence[:-1], sequence[1:])))
                self.assertEqual(sum(y == 3 for x, y in pairs), 1)
            self.assertEqual(sum(map(len, actual)), self.manifest["splits"]["train"]["prediction_pairs"])
            self.assertEqual(dataset[-1], dataset[len(dataset) - 1])
            with self.assertRaises(IndexError):
                dataset[len(dataset)]
        finally:
            dataset.close()

    def test_padding_and_spawn_serialization_preserve_labels_and_sources(self):
        dataset = SentenceWindowDataset(self.tokens, self.indexes)
        restored = None
        try:
            first, second = dataset[0], dataset[3]
            batch = collate_arrays([first, second])
            self.assertEqual(batch["input_ids"].shape, (2, 128))
            self.assertEqual(batch["input_ids"].dtype, np.int64)
            self.assertTrue(np.all(batch["labels"][~batch["attention_mask"]] == -100))
            self.assertTrue(np.all(batch["input_ids"][~batch["attention_mask"]] == 0))
            self.assertEqual(batch["attention_mask"].sum(), len(first["labels"]) + len(second["labels"]))
            restored = pickle.loads(pickle.dumps(dataset))
            self.assertIsNone(restored._store)
            self.assertEqual(restored[-1], dataset[-1])
        finally:
            dataset.close()
            if restored:
                restored.close()

    def test_shuffle_exact_coverage_reproducible_epochs_and_partial_last_block(self):
        sampler = BlockShuffleSampler(103, seed=42, block_size=16)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        self.assertEqual(sorted(first), list(range(103)))
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertNotEqual(first, second)
        self.assertEqual(sorted(second), list(range(103)))

    def test_length_buckets_cover_all_targets_and_resume_at_committed_batch(self):
        dataset = SentenceWindowDataset(self.tokens, self.indexes)
        try:
            indices = list(range(len(dataset)))
            self.assertEqual(dataset.window_lengths(indices).tolist(),
                             [len(dataset[i]["labels"]) for i in indices])
            sampler = LengthBucketBatchSampler(dataset, BlockShuffleSampler(len(dataset)), 3, multiplier=4)
            batches = list(sampler)
            self.assertEqual(sorted(sum(batches, [])), indices)
            self.assertEqual(len(batches), len(sampler))
            sampler.set_epoch(0, start_batch=2)
            self.assertEqual(list(sampler), batches[2:])
            self.assertEqual(len(sampler), len(batches) - 2)
        finally:
            dataset.close()

    def test_index_reuse_and_foreign_manifest_rejection(self):
        self.assertEqual(self.manifest, prepare_indexes(self.tokens, self.indexes, 128))
        with self.assertRaisesRegex(ValueError, "different dataset/context"):
            prepare_indexes(self.tokens, self.indexes, 64)
        manifest_path = self.tokens / "manifest.json"
        raw = json.loads(manifest_path.read_text())
        raw["vocab_size"] += 1
        write_json(manifest_path, raw)
        with self.assertRaisesRegex(ValueError, "different token store"):
            SentenceWindowDataset(self.tokens, self.indexes)

    def test_corrupt_binary_rejected_on_first_preparation(self):
        path = self.tokens / "train.tokens.bin"
        path.write_bytes(path.read_bytes() + b"xx")
        with self.assertRaises(ValueError):
            prepare_indexes(self.tokens, self.root / "corrupt-indexes", 128)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_actual_torch_spawn_loader_has_complete_nonduplicated_targets(self):
        import torch
        dataset = SentenceWindowDataset(self.tokens, self.indexes)
        loader = make_loader(dataset, batch_size=3, num_workers=2, bucket_multiplier=4)
        pairs = rows = 0
        seen = set()
        try:
            for batch in loader:
                self.assertEqual(batch["input_ids"].dtype, torch.long)
                pairs += int(batch["attention_mask"].sum())
                rows += batch["input_ids"].shape[0]
                seen.update(zip(batch["sentence_index"].tolist(), batch["window_start"].tolist()))
            self.assertEqual(rows, len(dataset))
            self.assertEqual(pairs, self.manifest["splits"]["train"]["prediction_pairs"])
            self.assertEqual(len(seen), len(dataset))
        finally:
            del loader
            dataset.close()


if __name__ == "__main__":
    unittest.main()
