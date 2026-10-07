"""Deterministic prefix crops over the unchanged sentence window token store."""

import math

import numpy as np

from vimeml.training.data import SentenceWindowDataset

MASK64 = (1 << 64) - 1
GOLDEN64 = 0x9E3779B97F4A7C15
EPOCH64 = 0xD1B54A32D192ED03


def mix64(value):
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def mix64_array(values):
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


class PrefixCropWindowDataset(SentenceWindowDataset):
    # Bucket samplers pass (epoch, index) to workers, including prefetched work.
    epoch_indexed = True

    def __init__(
        self,
        token_dir,
        index_dir,
        split="train",
        *,
        seed=42,
        probability=0.30,
        min_remaining_tokens=8,
    ):
        if split != "train":
            raise ValueError("Prefix crops are only defined for the V2 training split.")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Crop probability must be in [0, 1].")
        if type(min_remaining_tokens) is not int or min_remaining_tokens < 1:
            raise ValueError("Minimum remaining tokens must be a positive integer.")
        if type(seed) is not int or not 0 <= seed <= MASK64:
            raise ValueError("Crop seed must be a uint64 integer.")
        super().__init__(token_dir, index_dir, split)
        self.seed = seed
        self.probability = probability
        self.threshold = int(probability * (1 << 53))
        self.min_remaining_tokens = min_remaining_tokens
        self.epoch = 0

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Crop epoch must be a nonnegative integer.")
        self.epoch = epoch

    def crop_offset(self, index, length, original_start, epoch):
        maximum = length - self.min_remaining_tokens
        if original_start != 0 or maximum < 1 or self.threshold == 0:
            return 0
        bits = mix64((self.seed + epoch * EPOCH64 + index + GOLDEN64) & MASK64)
        if (bits >> 11) >= self.threshold:
            return 0
        return 1 + mix64((bits + GOLDEN64) & MASK64) % maximum

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (self.epoch, key)
        index = int(index)
        if index < 0:
            index += len(self)
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Crop epoch must be a nonnegative integer.")
        sample = super().__getitem__(index)
        original_start = sample["window_start"]
        length = len(sample["input_ids"])
        offset = self.crop_offset(index, length, original_start, epoch)
        return {
            **sample,
            "input_ids": sample["input_ids"][offset:],
            "labels": sample["labels"][offset:],
            "window_start": original_start + offset,
            "original_window_start": original_start,
            "crop_offset": offset,
            "uncropped_length": length,
            "sample_index": index,
            "sample_epoch": epoch,
        }

    def window_lengths(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        lengths = super().window_lengths(indices)
        if not self.threshold:
            return lengths
        position = np.searchsorted(self.end, indices, side="right")
        first_window = np.ones(len(indices), dtype=np.bool_)
        candidates = np.flatnonzero(position < len(self.end))
        first_window[candidates] = indices[candidates] <= self.first[position[candidates]]
        maximum = lengths - self.min_remaining_tokens
        base = (self.seed + self.epoch * EPOCH64 + GOLDEN64) & MASK64
        bits = mix64_array(indices.astype(np.uint64) + np.uint64(base))
        apply = first_window & (maximum > 0) & ((bits >> np.uint64(11)) < np.uint64(self.threshold))
        choices = mix64_array(bits + np.uint64(GOLDEN64)) % np.maximum(maximum, 1).astype(
            np.uint64
        ) + np.uint64(1)
        return lengths - np.where(apply, choices, 0).astype(np.int64)
