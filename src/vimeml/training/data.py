"""Memory-mapped sentences, loss-preserving windows and bounded-memory shuffle.

This module can prepare/check data without importing PyTorch. Each sample has
already-shifted labels: the future model must NOT shift these labels again.
"""

import hashlib
import itertools
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from vimeml.tokenizer.store import TokenStore

SPLITS = ("train", "validation", "test")
IGNORE_INDEX = -100


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_split(token_dir, index_dir, split, context_length, manifest_sha, verification="sha256"):
    """Scan offsets in chunks; only long sentences require index entries."""
    with TokenStore(token_dir, split) as store:
        count = len(store)
        if count == 0:
            raise ValueError(f"Empty split: {split}")
        metadata = store.manifest["splits"][split]
        # Verify only files consumed by training, in parallel across splits.
        if verification == "sha256":
            for suffix in ("tokens.bin", "offsets.bin", "sources.bin"):
                name = f"{split}.{suffix}"
                if file_sha(token_dir / name) != store.manifest["output_sha256"][name]:
                    raise ValueError(f"Token data hash mismatch: {name}")
        if (token_dir / f"{split}.sources.bin").stat().st_size != count:
            raise ValueError(f"Invalid source index size: {split}")
        offsets = np.memmap(token_dir / f"{split}.offsets.bin", mode="r", dtype="<u8")
        sentence_chunks, first_chunks, end_chunks = [], [], []
        extra_total = pairs = max_length = 0
        try:
            for start in range(0, count, 1_000_000):
                end = min(count, start + 1_000_000)
                left, right = offsets[start:end], offsets[start + 1:end + 1]
                if np.any(right <= left):
                    raise ValueError(f"Non-increasing offsets: {split}")
                lengths = right - left
                if np.any(lengths < 3):
                    raise ValueError(f"Sentence shorter than BOS/text/EOS: {split}")
                pairs += int((lengths - 1).sum())
                max_length = max(max_length, int(lengths.max()))
                # ceil((L-1)/C) windows; extra windows = floor((L-2)/C).
                extras = (lengths - 2) // context_length
                local = np.flatnonzero(extras)
                long_extras = extras[local].astype(np.int64)
                sentence = local.astype(np.int64) + start
                cumulative = np.cumsum(long_extras) + extra_total
                window_end = sentence + cumulative + 1
                sentence_chunks.append(sentence)
                first_chunks.append(window_end - long_extras - 1)
                end_chunks.append(window_end)
                extra_total += int(long_extras.sum())
            if pairs != metadata["prediction_pairs"] or max_length != metadata["max_sequence_tokens"]:
                raise ValueError(f"Offsets do not agree with token manifest: {split}")
        finally:
            # Release slices before closing the underlying Windows file mapping.
            del left, right, offsets
        path = index_dir / f"{split}.windows.npz"
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, sentences=np.concatenate(sentence_chunks),
                 first=np.concatenate(first_chunks), end=np.concatenate(end_chunks))
        temporary.replace(path)
        return {"sentences": count, "windows": count + extra_total,
                "long_sentences": sum(len(x) for x in sentence_chunks),
                "extra_windows": extra_total, "prediction_pairs": pairs,
                "max_sequence_tokens": max_length, "index_bytes": path.stat().st_size,
                "index_sha256": file_sha(path), "token_manifest_sha256": manifest_sha}


def prepare_indexes(token_dir, index_dir, context_length=128, verification="sha256"):
    if not isinstance(context_length, int) or context_length < 1:
        raise ValueError("context_length must be a positive integer.")
    if verification not in {"sha256", "metadata"}:
        raise ValueError("verification must be sha256 or metadata.")
    token_dir, index_dir = Path(token_dir), Path(index_dir)
    manifest_sha = file_sha(token_dir / "manifest.json")
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (cached.get("status"), cached.get("format"), cached.get("context_length"),
                cached.get("token_manifest_sha256")) != (
                    "complete", "vimeml_sentence_windows_v1", context_length, manifest_sha):
            raise ValueError("Index directory belongs to a different dataset/context; use a new directory.")
        for split in SPLITS:
            path = index_dir / f"{split}.windows.npz"
            if (path.stat().st_size != cached["splits"][split]["index_bytes"] or
                    (verification == "sha256" and file_sha(path) != cached["splits"][split]["index_sha256"])):
                raise ValueError(f"Window index hash mismatch: {split}")
        return cached
    index_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing window indexes (3 parallel splits; {verification} verification)...", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {split: pool.submit(prepare_split, token_dir, index_dir, split,
                                  context_length, manifest_sha, verification) for split in SPLITS}
        splits = {split: future.result() for split, future in jobs.items()}
    if file_sha(token_dir / "manifest.json") != manifest_sha:
        raise ValueError("Token manifest changed during preparation.")
    result = {"status": "complete", "format": "vimeml_sentence_windows_v1",
              "context_length": context_length, "token_manifest_sha256": manifest_sha,
              "token_file_verification": verification,
              "policy": "Separate sentences; nonoverlapping target windows; one shared input token at each boundary; no prediction pair discarded.",
              "splits": splits}
    write_json(manifest_path, result)
    return result


class SentenceWindowDataset:
    """Open maps lazily in each worker, including Windows spawn workers."""

    def __init__(self, token_dir, index_dir, split="train"):
        if split not in SPLITS:
            raise ValueError("Unknown split.")
        self.token_dir = Path(token_dir).resolve()
        self.split = split
        manifest = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "vimeml_sentence_windows_v1" or manifest.get("status") != "complete":
            raise ValueError("Incomplete/unsupported window index.")
        if manifest["token_manifest_sha256"] != file_sha(self.token_dir / "manifest.json"):
            raise ValueError("Window index belongs to a different token store.")
        path = Path(index_dir) / f"{split}.windows.npz"
        if file_sha(path) != manifest["splits"][split]["index_sha256"]:
            raise ValueError("Window index checksum mismatch.")
        self.context_length = manifest["context_length"]
        self.count = manifest["splits"][split]["windows"]
        with np.load(path, allow_pickle=False) as arrays:
            self.sentences = arrays["sentences"]
            self.first = arrays["first"]
            self.end = arrays["end"]
        self._store = self._sources = None

    def __len__(self):
        return self.count

    def locate(self, index):
        index = int(index)
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        position = int(np.searchsorted(self.end, index, side="right"))
        if position < len(self.end) and index >= self.first[position]:
            return int(self.sentences[position]), int(index - self.first[position]) * self.context_length
        previous_extra = 0 if position == 0 else int(self.end[position - 1] - self.sentences[position - 1] - 1)
        return index - previous_extra, 0

    def _open(self):
        if self._store is None:
            self._store = TokenStore(self.token_dir, self.split)
            source_path = self.token_dir / f"{self.split}.sources.bin"
            if source_path.stat().st_size != len(self._store):
                self.close()
                raise ValueError("Invalid source file size.")
            self._sources = np.memmap(source_path, dtype="u1", mode="r")

    def __getitem__(self, index):
        sentence, start = self.locate(index)
        self._open()
        tokens = self._store[sentence]
        end = min(len(tokens) - 1, start + self.context_length)
        source = int(self._sources[sentence])
        if source not in self._store.manifest["source_ids"].values():
            raise ValueError("Unknown source ID.")
        return {"input_ids": tokens[start:end], "labels": tokens[start + 1:end + 1],
                "sentence_index": sentence, "window_start": start, "source_id": source}

    def window_lengths(self, indices):
        """Vectorized lengths from offsets; no token decoding or large length table."""
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.count):
            raise IndexError("Window index outside dataset.")
        position = np.searchsorted(self.end, indices, side="right")
        previous_extra = np.zeros_like(indices)
        has_previous = position > 0
        previous = position[has_previous] - 1
        previous_extra[has_previous] = self.end[previous] - self.sentences[previous] - 1
        sentences = indices - previous_extra
        starts = np.zeros_like(indices)
        candidates = np.flatnonzero(position < len(self.end))
        long = candidates[indices[candidates] >= self.first[position[candidates]]]
        sentences[long] = self.sentences[position[long]]
        starts[long] = (indices[long] - self.first[position[long]]) * self.context_length
        offsets = np.memmap(self.token_dir / f"{self.split}.offsets.bin", dtype="<u8", mode="r")
        try:
            lengths = (offsets[sentences + 1] - offsets[sentences]).astype(np.int64)
            return np.minimum(self.context_length, lengths - 1 - starts)
        finally:
            offsets._mmap.close()

    def close(self):
        if self._sources is not None:
            self._sources._mmap.close()
            self._sources = None
        if self._store is not None:
            self._store.close()
            self._store = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_store"] = state["_sources"] = None
        return state


def collate_arrays(samples, pad_id=0, pad_multiple=8):
    if not samples or pad_multiple < 1:
        raise ValueError("Empty batch or invalid padding multiple.")
    lengths = np.array([len(s["input_ids"]) for s in samples], dtype=np.int64)
    if np.any(lengths < 1) or any(len(s["labels"]) != n for s, n in zip(samples, lengths)):
        raise ValueError("Invalid input/label length.")
    width = (int(lengths.max()) + pad_multiple - 1) // pad_multiple * pad_multiple
    inputs = np.full((len(samples), width), pad_id, dtype=np.int64)
    labels = np.full_like(inputs, IGNORE_INDEX)
    mask = np.zeros_like(inputs, dtype=np.bool_)
    for row, (sample, length) in enumerate(zip(samples, lengths)):
        inputs[row, :length] = sample["input_ids"]
        labels[row, :length] = sample["labels"]
        mask[row, :length] = True
    batch = {"input_ids": inputs, "labels": labels, "attention_mask": mask, "lengths": lengths}
    for name in ("sentence_index", "window_start", "source_id"):
        batch[name] = np.array([s[name] for s in samples], dtype=np.int64)
    for name in ("original_window_start", "crop_offset", "uncropped_length", "sample_index", "sample_epoch"):
        if name in samples[0]:
            batch[name] = np.array([s[name] for s in samples], dtype=np.int64)
    return batch


class TorchCollator:
    def __init__(self, pad_id=0, pad_multiple=8):
        self.pad_id, self.pad_multiple = pad_id, pad_multiple

    def __call__(self, samples):
        import torch
        return {name: torch.from_numpy(value) for name, value in
                collate_arrays(samples, self.pad_id, self.pad_multiple).items()}


class BlockShuffleSampler:
    """Shuffle block order and each block, never materialize 25M Python ints."""
    def __init__(self, size, seed=42, block_size=65536):
        if size < 0 or block_size < 1:
            raise ValueError("Invalid sampler size.")
        self.size, self.seed, self.block_size, self.epoch = size, seed, block_size, 0

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        blocks = list(range((self.size + self.block_size - 1) // self.block_size))
        rng.shuffle(blocks)
        for block in blocks:
            start = block * self.block_size
            indices = list(range(start, min(self.size, start + self.block_size)))
            rng.shuffle(indices)
            yield from indices


class LengthBucketBatchSampler:
    """Sort a bounded random pool by length, then shuffle batch order.

    Every index occurs once, including the last partial batch. start_batch is a
    committed cursor, independent of how far worker prefetch has proceeded.
    """
    def __init__(self, dataset, sampler, batch_size, multiplier=64, seed=42, start_batch=0):
        if batch_size < 1 or multiplier < 1 or start_batch < 0:
            raise ValueError("Invalid bucket configuration.")
        self.dataset, self.sampler = dataset, sampler
        self.batch_size, self.multiplier, self.seed = batch_size, multiplier, seed
        self.epoch, self.start_batch = 0, start_batch

    def set_epoch(self, epoch, start_batch=0):
        self.epoch, self.start_batch = int(epoch), int(start_batch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)
        if isinstance(self.sampler, BlockShuffleSampler):
            self.sampler.size = len(self.dataset)

    def __len__(self):
        return max(0, (len(self.sampler) + self.batch_size - 1) // self.batch_size - self.start_batch)

    def __iter__(self):
        iterator = iter(self.sampler)
        rng = random.Random(self.seed + self.epoch + 1_000_003)
        cursor = 0
        while pool := list(itertools.islice(iterator, self.batch_size * self.multiplier)):
            order = np.argsort(self.dataset.window_lengths(pool), kind="stable")
            ordered = [pool[int(i)] for i in order]
            batches = [ordered[i:i + self.batch_size] for i in range(0, len(pool), self.batch_size)]
            rng.shuffle(batches)
            for batch in batches:
                if cursor >= self.start_batch:
                    yield [(self.epoch, index) for index in batch] if getattr(self.dataset, "epoch_indexed", False) else batch
                cursor += 1


def make_loader(dataset, batch_size=128, num_workers=4, seed=42, shuffle=None, pin_memory=False,
                bucket_multiplier=0, indices=None, start_batch=0, worker_init_fn=None):
    import torch
    from torch.utils.data import DataLoader
    if batch_size < 1 or num_workers < 0 or dataset.context_length % 8:
        raise ValueError("Positive batch size, nonnegative workers, and context multiple of 8 required.")
    if shuffle is None:
        shuffle = dataset.split == "train"
    if indices is not None and shuffle:
        raise ValueError("Use explicit indices or shuffle, not both.")
    if start_batch and not bucket_multiplier:
        raise ValueError("Resume cursor requires bucketed batching.")
    if getattr(dataset, "epoch_indexed", False) and not bucket_multiplier:
        raise ValueError("Prefix crop requires epoch-tagged bucket batching.")
    sampler = indices if indices is not None else (BlockShuffleSampler(len(dataset), seed) if shuffle else None)
    dataset._open()
    pad_id = dataset._store.manifest["special_ids"]["pad"]
    # Main process need not keep maps open after discovering special IDs.
    dataset.close()
    options = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if num_workers else {}
    if bucket_multiplier:
        batch_options = {"batch_sampler": LengthBucketBatchSampler(
            dataset, sampler if sampler is not None else range(len(dataset)), batch_size,
            bucket_multiplier, seed, start_batch)}
    else:
        batch_options = {"batch_size": batch_size, "sampler": sampler, "drop_last": False}
    return DataLoader(dataset, **batch_options,
                      num_workers=num_workers, collate_fn=TorchCollator(pad_id),
                      pin_memory=pin_memory, persistent_workers=num_workers > 0,
                      generator=torch.Generator().manual_seed(seed), worker_init_fn=worker_init_fn,
                      **options)
