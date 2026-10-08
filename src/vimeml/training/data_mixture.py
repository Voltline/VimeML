"""Epoch-specific virtual windows; chat quota counts actual cropped targets."""
import json
from pathlib import Path

import numpy as np

from vimeml.training.data import SentenceWindowDataset, file_sha
from vimeml.training.data_v2 import PrefixCropWindowDataset

ROOT = Path(__file__).resolve().parents[3]


class JoinedPrefixWindows:
    """Virtual concatenation of compatible frozen stores; no token copies."""

    epoch_indexed = True

    def __init__(self, bases, primary, source_maps):
        self.bases, self.source_maps = bases, source_maps
        self.primary = primary
        self.ends = np.cumsum([len(base) for base in bases], dtype=np.int64)
        self.starts = np.r_[0, self.ends[:-1]]
        selected = bases[primary]
        for key in ("token_dir", "context_length", "split", "seed", "probability", "min_remaining_tokens"):
            setattr(self, key, getattr(selected, key))
        self._store = None
        self.epoch = 0

    def __len__(self):
        return int(self.ends[-1])

    def set_epoch(self, epoch):
        self.epoch = epoch
        for base in self.bases:
            base.set_epoch(epoch)

    def _open(self):
        self.bases[self.primary]._open()
        self._store = self.bases[self.primary]._store

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (self.epoch, key)
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError("Joined window outside frozen stores")
        source = int(np.searchsorted(self.ends, index, side="right"))
        sample = self.bases[source][(epoch, int(index - self.starts[source]))]
        return {**sample, "source_id": self.source_maps[source][sample["source_id"]],
                "sample_index": index}

    def window_lengths(self, indices):
        return self._lengths(indices, cropped=True)

    def uncropped_window_lengths(self, indices):
        return self._lengths(indices, cropped=False)

    def _lengths(self, indices, *, cropped):
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Joined window outside frozen stores")
        lengths = np.empty(len(indices), dtype=np.int64)
        for base, start, end in zip(self.bases, self.starts, self.ends):
            positions = np.flatnonzero((indices >= start) & (indices < end))
            if len(positions):
                local = indices[positions] - start
                lengths[positions] = (base.window_lengths(local) if cropped else
                                      SentenceWindowDataset.window_lengths(base, local))
        return lengths

    def close(self):
        for base in self.bases:
            base.close()
        self._store = None

    def __getstate__(self):
        return {**self.__dict__, "_store": None}


class MixtureWindowDataset:
    epoch_indexed = True

    def __init__(self, base, directory):
        self.base = base
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete" or self.manifest.get("format") != "vimeml_token_mixture_v1":
            raise ValueError("Incomplete or unsupported mixture.")
        request = self.manifest["request"]
        if (request["token_manifest_sha256"] != file_sha(base.token_dir / "manifest.json") or
                request["window_manifest_sha256"] != file_sha(ROOT / request["index_dir"].replace("\\", "/") / "manifest.json")):
            raise ValueError("Mixture belongs to different token data/indexes.")
        if (request["seed"], request["crop_probability"], request["crop_minimum"]) != (
                base.seed, base.probability, base.min_remaining_tokens):
            raise ValueError("Mixture crop settings differ from training.")
        self.context_length = base.context_length
        self.split = "train"
        self.epoch = 0
        self._mapping = self._mapping_epoch = self._store = None

    def _map(self, epoch):
        if not 0 <= epoch < len(self.manifest["epochs"]):
            raise ValueError("Epoch is outside the frozen mixture plan.")
        if self._mapping_epoch != epoch:
            self._close_map()
            record = self.manifest["epochs"][epoch]
            path = self.directory / record["file"]
            if path.stat().st_size != record["bytes"]:
                raise ValueError("Mixture mapping size differs from manifest.")
            self._mapping = np.load(path, mmap_mode="r", allow_pickle=False)
            if self._mapping.dtype != np.dtype("uint32") or self._mapping.shape != (record["windows"],):
                self._close_map()
                raise ValueError("Invalid mixture mapping dtype/shape.")
            self._mapping_epoch = epoch
        return self._mapping

    def __len__(self):
        return self.manifest["epochs"][self.epoch]["windows"]

    def set_epoch(self, epoch):
        if not 0 <= epoch < len(self.manifest["epochs"]):
            raise ValueError("Epoch is outside the frozen mixture plan.")
        self.epoch = epoch
        self.base.set_epoch(epoch)

    def _open(self):
        self.base._open()
        self._store = self.base._store

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (self.epoch, key)
        physical = int(self._map(epoch)[index])
        return self.base[(epoch, physical)]

    def window_lengths(self, indices):
        self.base.set_epoch(self.epoch)
        return self.base.window_lengths(self._map(self.epoch)[np.asarray(indices, dtype=np.int64)])

    def _close_map(self):
        if self._mapping is not None:
            self._mapping._mmap.close()
        self._mapping = self._mapping_epoch = None

    def close(self):
        self._close_map()
        self.base.close()
        self._store = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mapping"] = state["_mapping_epoch"] = state["_store"] = None
        return state


def mixture_dataset(config, root):
    settings = config["training"]
    directory = root / config["data_mixture"]["directory"]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["request"].get("joined_stores")
    bases = []
    try:
        if records:
            maps = []
            for record in records:
                if (file_sha(root / record["token_dir"] / "manifest.json") != record["token_manifest_sha256"] or
                        file_sha(root / record["index_dir"] / "manifest.json") != record["window_manifest_sha256"]):
                    raise ValueError("Joined store identity differs from frozen mixture")
                bases.append(PrefixCropWindowDataset(root / record["token_dir"], root / record["index_dir"],
                    seed=settings["seed"], probability=settings.get("prefix_crop_probability", 0.0),
                    min_remaining_tokens=settings.get("prefix_crop_min_remaining_tokens", 8)))
                maps.append({int(k): v for k, v in record["source_map"].items()})
            base = JoinedPrefixWindows(bases, manifest["request"]["primary_store"], maps)
            if str(base.token_dir.resolve()) != str((root / config["token_dir"]).resolve()):
                raise ValueError("Joined primary store must match configuration")
        else:
            base = PrefixCropWindowDataset(root / config["token_dir"], root / config["index_dir"],
                seed=settings["seed"], probability=settings.get("prefix_crop_probability", 0.0),
                min_remaining_tokens=settings.get("prefix_crop_min_remaining_tokens", 8))
            bases.append(base)
        return MixtureWindowDataset(base, directory)
    except BaseException:
        for base in bases:
            base.close()
        raise
