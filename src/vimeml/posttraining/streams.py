"""Bucketed task streams with committed cursors, independent of worker prefetch."""
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from vimeml.training.data import BlockShuffleSampler, LengthBucketBatchSampler
from vimeml.training.train import worker_init


class ConversionWindows:
    """Keep the frozen conversion reader unchanged, including its source hash."""
    context_length = 128
    epoch_indexed = False

    def __init__(self, data):
        self.data, self.split = data, data.split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[int(index)]

    def window_lengths(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        # Normal-reading length is a cheap upper bound for the fixed10% empty
        # reading views; actual padding/denominators always use returned rows.
        return (self.data.offsets[indices + 1] - self.data.offsets[indices] - 1).astype(np.int64)


class TaskCollator:
    def __call__(self, samples):
        lengths = torch.tensor([len(s["input_ids"]) for s in samples], dtype=torch.long)
        width = math.ceil(int(lengths.max()) / 8) * 8
        if not 1 <= width <= 128:
            raise ValueError("Task sample outside the fixed context")
        ids = torch.zeros((len(samples), width), dtype=torch.long)
        labels = torch.full_like(ids, -100)
        for row, (sample, n) in enumerate(zip(samples, lengths.tolist())):
            if len(sample["labels"]) != n:
                raise ValueError("Task labels must already be shifted")
            ids[row, :n] = torch.tensor(sample["input_ids"], dtype=torch.long)
            labels[row, :n] = torch.tensor(sample["labels"], dtype=torch.long)
        return {"input_ids": ids, "labels": labels, "lengths": lengths}


class TailLastBucketBatchSampler(LengthBucketBatchSampler):
    """Keep the single partial batch last, including after cursor restoration.

    Base LM batching shuffles the tail inside its last pool. C1's exact update
    accounting requires all earlier conversion updates to contain256 examples.
    """
    def __iter__(self):
        base = LengthBucketBatchSampler(self.dataset, self.sampler, self.batch_size,
            self.multiplier, self.seed, start_batch=0)
        base.set_epoch(self.epoch, 0)
        cursor, tail = 0, None
        for batch in base:
            if len(batch) < self.batch_size:
                if tail is not None:
                    raise ValueError("More than one partial task batch")
                tail = batch
            else:
                if cursor >= self.start_batch:
                    yield batch
                cursor += 1
        if tail is not None and cursor >= self.start_batch:
            yield tail


def loader_for(data, batch_size, workers, seed, multiplier=64, indices=None, shuffle=True, pin_memory=False):
    sampler = (BlockShuffleSampler(len(data), seed) if shuffle else range(len(data))) if indices is None else indices
    batches = TailLastBucketBatchSampler(data, sampler, batch_size, multiplier, seed)
    options = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}
    return DataLoader(data, batch_sampler=batches, num_workers=workers,
                      collate_fn=TaskCollator(), pin_memory=pin_memory,
                      persistent_workers=workers > 0, worker_init_fn=worker_init,
                      generator=torch.Generator().manual_seed(seed), **options)


def slice_batch(batch, start, end):
    return {name: value[start:end] for name, value in batch.items()}


class CommittedStream:
    """The state always points to the next unconsumed row in a bucketed batch.

    Caller copies state only AFTER a successful optimizer update. A partial LM
    tail borrows rows from the next LM epoch without dropping/repeating data.
    """
    def __init__(self, loader, state=None):
        self.loader = loader
        self.epoch, self.batch, self.offset = (0, 0, 0) if state is None else (
            state["epoch"], state["batch"], state["offset"])
        if any(type(x) is not int or x < 0 for x in (self.epoch, self.batch, self.offset)):
            raise ValueError("Invalid task cursor")
        self.iterator = self.pending = None
        self.loader.batch_sampler.set_epoch(self.epoch, self.batch)

    def state_dict(self):
        return {"epoch": self.epoch, "batch": self.batch, "offset": self.offset}

    def next_batch(self):
        if self.offset:
            raise ValueError("Whole-batch consumption requires an aligned cursor")
        if self.iterator is None:
            self.iterator = iter(self.loader)
        value = next(self.iterator)
        self.batch += 1
        return value

    def next_group(self, size):
        result, remaining = [], size
        while remaining:
            if self.pending is None:
                if self.iterator is None:
                    self.iterator = iter(self.loader)
                try:
                    self.pending = next(self.iterator)
                except StopIteration:
                    if self.batch == 0:
                        raise ValueError("Empty replay stream")
                    self.epoch, self.batch, self.offset = self.epoch + 1, 0, 0
                    self.loader.batch_sampler.set_epoch(self.epoch, 0)
                    self.iterator = iter(self.loader)
                    self.pending = next(self.iterator)
            available = len(self.pending["lengths"]) - self.offset
            if available <= 0:
                raise ValueError("Replay cursor outside its saved batch")
            take = min(available, remaining)
            result.append(slice_batch(self.pending, self.offset, self.offset + take))
            remaining -= take
            self.offset += take
            if self.offset == len(self.pending["lengths"]):
                self.offset, self.pending = 0, None
                self.batch += 1
        return result


def microbatches(batches, size):
    for batch in batches:
        for start in range(0, len(batch["lengths"]), size):
            part = slice_batch(batch, start, start + size)
            width = math.ceil(int(part["lengths"].max()) / 8) * 8
            # Slicing the time dimension retains pinned storage. .to() handles
            # the strided source; never pad a short microbatch to another's max.
            part["input_ids"] = part["input_ids"][:, :width]
            part["labels"] = part["labels"][:, :width]
            yield part
