"""vimeml-tiny-zh-v1 arm mixtures (docs/zh-CN/training-zh-v1-plan.md).

Web: every window of the arm's pool (train.tiers.bin <= max tier: P1 3, P5 2,
P20 1) once per epoch, prefix-cropped per epoch. Chat: one run-long sequence of
passes over LCCC train; pass k is a fresh permutation read through chat copy k,
a crop view with its own seed, so a repeated utterance gets a new crop whether
the arm has 1 epoch or 20. After epoch e the chat targets taken so far are
round(web targets so far * f / (1 - f)). Each epoch map lists joined window ids,
shuffled; maps are SHA256-pinned and must rebuild byte-identically.

This module does not import torch.
"""
import json
from pathlib import Path

import numpy as np

from vimeml.training.data import SentenceWindowDataset, file_sha
from vimeml.training.data_mixture import JoinedPrefixWindows, MixtureWindowDataset
from vimeml.training.data_v2 import MASK64, PrefixCropWindowDataset
from vimeml.training.v41_data import window_sentences

ROOT = Path(__file__).resolve().parents[3]
FORMAT = "vimeml_zh13_mixture_v1"
TAGS = {8192: "8k", 12288: "12k"}
POOLS = {"P1": 3, "P5": 2, "P20": 1}  # max train.tiers.bin value kept
PROBE_TIER = 0
CHAT_FRACTION = 0.05
CROP_PROBABILITY = 0.30
CROP_MINIMUM = 8
CHAT_COPIES = 8
CHAT_SEED_STEP = 0x9E3779B97F4A7C15
CHUNK = 1 << 20
# arm -> (vocabulary, pool, web epochs, seed). Seeds drawn once with secrets.randbelow(2**31).
ARMS = {
    "12k-fresh-s1": (12288, "P1", 1, 1591996171),
    "12k-fresh-s2": (12288, "P1", 1, 136617650),
    "12k-fresh-s3": (12288, "P1", 1, 1137767227),
    "12k-ep5": (12288, "P5", 5, 1591996171),
    "12k-ep20": (12288, "P20", 20, 1591996171),
    "8k-fresh": (8192, "P1", 1, 1591996171),
    "8k-ep5": (8192, "P5", 5, 1591996171),
    "8k-ep20": (8192, "P20", 20, 1591996171),
}


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def write_lf(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    temporary.replace(path)


def store_paths(vocab, data_root="artifacts"):
    tag = TAGS[vocab]
    return {"web": (f"{data_root}/token-data/zh13-web-{tag}", f"{data_root}/training-data/zh13-web-{tag}-c128"),
            "chat": (f"{data_root}/token-data/zh13-chat-{tag}", f"{data_root}/training-data/zh13-chat-{tag}-c128")}


def mixture_directory(arm, data_root="artifacts"):
    return f"{data_root}/training-data/zh13-{arm}-c128"


def request_for(arm, root=ROOT, data_root="artifacts"):
    vocab, pool, epochs, seed = ARMS[arm]
    paths = store_paths(vocab, data_root)
    stores = []
    for name, (tokens, index) in paths.items():
        manifest = read(Path(root) / tokens / "manifest.json")
        if manifest["vocab_size"] != vocab:
            raise ValueError(f"{tokens} is not a {vocab} store")
        record = {"name": name, "token_dir": tokens, "index_dir": index,
                  "token_manifest_sha256": file_sha(Path(root) / tokens / "manifest.json"),
                  "window_manifest_sha256": file_sha(Path(root) / index / "manifest.json"),
                  "tokenizer_model_sha256": manifest["tokenizer_model_sha256"]}
        stores.append(record)
    web, chat = stores
    if web["tokenizer_model_sha256"] != chat["tokenizer_model_sha256"]:
        raise ValueError("Web and chat stores use different tokenizers")
    web_manifest = read(Path(root) / web["token_dir"] / "manifest.json")
    return {"format": FORMAT, "arm": arm, "vocab_size": vocab, "pool": pool, "max_tier": POOLS[pool],
            "epochs": epochs, "seed": seed, "chat_fraction": CHAT_FRACTION,
            "crop_probability": CROP_PROBABILITY, "crop_minimum": CROP_MINIMUM, "chat_copies": CHAT_COPIES,
            "chat_seed_step": CHAT_SEED_STEP, "pools_sha256": web_manifest["pools_sha256"],
            "tokenizer_model_sha256": web["tokenizer_model_sha256"],
            # MixtureWindowDataset checks these against the primary (web) store.
            "token_manifest_sha256": web["token_manifest_sha256"], "index_dir": web["index_dir"],
            "window_manifest_sha256": web["window_manifest_sha256"], "web": web, "chat": chat}


def chat_seed(seed, copy):
    return (seed + (copy + 1) * CHAT_SEED_STEP) & MASK64


def joined_windows(request, root=ROOT, seed=None):
    """Web store, then chat_copies crop views of the chat store; source ids 0-1 web, 2 chat."""
    seed = request["seed"] if seed is None else seed
    root = Path(root)
    options = {"probability": request["crop_probability"], "min_remaining_tokens": request["crop_minimum"]}
    web, chat = request["web"], request["chat"]
    bases = [PrefixCropWindowDataset(root / web["token_dir"], root / web["index_dir"], seed=seed, **options)]
    try:
        for copy in range(request["chat_copies"]):
            bases.append(PrefixCropWindowDataset(root / chat["token_dir"], root / chat["index_dir"],
                                                 seed=chat_seed(seed, copy), **options))
    except BaseException:
        for base in bases:
            base.close()
        raise
    return JoinedPrefixWindows(bases, 0, [{0: 0, 1: 1}] + [{0: 2}] * request["chat_copies"])


def window_tiers(base, tiers):
    """train.tiers.bin value of every web window, uint8."""
    out = np.empty(len(base), dtype=np.uint8)
    for start in range(0, len(base), CHUNK):
        ids = np.arange(start, min(len(base), start + CHUNK), dtype=np.int64)
        out[start:start + len(ids)] = tiers[window_sentences(base, ids)]
    return out


def _lengths(joined, ids):
    total = 0
    for start in range(0, len(ids), CHUNK):
        total += int(joined.window_lengths(ids[start:start + CHUNK]).sum())
    return total


def _save(directory, name, array):
    path = Path(directory) / name
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    temporary.replace(path)
    return path


def prepare_maps(directory, root=ROOT):
    directory, root = Path(directory), Path(root)
    request = read(directory / "request.json")
    expected = read(directory / "manifest.json") if (directory / "manifest.json").exists() else None
    if expected:
        files = [r["file"] for r in expected["epochs"]] + [expected["probe"]["file"]]
        digests = [r["sha256"] for r in expected["epochs"]] + [expected["probe"]["sha256"]]
        if all((directory / f).exists() and file_sha(directory / f) == d for f, d in zip(files, digests)):
            return expected
    for store in (request["web"], request["chat"]):
        if (file_sha(root / store["token_dir"] / "manifest.json") != store["token_manifest_sha256"]
                or file_sha(root / store["index_dir"] / "manifest.json") != store["window_manifest_sha256"]):
            raise ValueError(f"{store['name']} store identity changed")
    web_manifest = read(root / request["web"]["token_dir"] / "manifest.json")
    tiers_path = root / request["web"]["token_dir"] / "train.tiers.bin"
    if file_sha(tiers_path) != web_manifest["output_sha256"]["train.tiers.bin"]:
        raise ValueError("train.tiers.bin differs from its manifest")
    tiers = np.fromfile(tiers_path, dtype=np.uint8)
    joined = joined_windows(request, root)
    try:
        if len(joined) >= 1 << 32:
            raise ValueError("Joined windows exceed uint32 map ids")
        web_base = joined.bases[0]
        if len(tiers) != web_manifest["splits"]["train"]["sentences"]:
            raise ValueError("Tier count differs from web sentences")
        per_window = window_tiers(web_base, tiers)
        web = np.flatnonzero(per_window <= request["max_tier"]).astype("<u4")
        probe = np.flatnonzero(per_window == PROBE_TIER).astype("<u4")
        del per_window
        probe_path = _save(directory, "probe-windows.npy", probe)
        chat_windows = len(joined.bases[1])
        chat_starts = joined.starts[1:]
        chat_uncropped = int(SentenceWindowDataset.window_lengths(joined.bases[1], np.arange(chat_windows)).sum())
        records = []
        web_cumulative = chat_cumulative = 0
        chat_pass, chat_position, permutation = 0, 0, None
        for epoch in range(request["epochs"]):
            joined.set_epoch(epoch)
            web_tokens = _lengths(joined, web)
            web_cumulative += web_tokens
            fraction = request["chat_fraction"]
            target = round(web_cumulative * fraction / (1 - fraction)) - chat_cumulative
            chosen, used = [], 0
            while used < target:
                if chat_pass >= request["chat_copies"]:
                    raise ValueError("Chat quota needs more crop copies than chat_copies")
                if permutation is None:
                    permutation = np.random.default_rng([request["seed"], chat_pass]).permutation(chat_windows)
                ids = permutation[chat_position:] + chat_starts[chat_pass]
                cumulative = np.cumsum(joined.window_lengths(ids), dtype=np.int64)
                take = int(np.searchsorted(cumulative, target - used, side="right"))
                if take:
                    chosen.append(ids[:take])
                    used += int(cumulative[take - 1])
                if take < len(ids):
                    chat_position += take
                    break
                chat_pass, chat_position, permutation = chat_pass + 1, 0, None
            chat_cumulative += used
            chosen = np.concatenate(chosen).astype("<u4") if chosen else np.empty(0, dtype="<u4")
            mapping = np.concatenate((web, chosen))
            np.random.default_rng([request["seed"], 1 << 20, epoch]).shuffle(mapping)
            path = _save(directory, f"epoch-{epoch}.npy", mapping)
            record = {"epoch": epoch, "file": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path),
                      "windows": len(mapping), "web_windows": len(web), "chat_windows": len(chosen),
                      "web_tokens": web_tokens, "chat_tokens": used, "effective_tokens": web_tokens + used,
                      "chat_fraction": used / (web_tokens + used),
                      "chat_cursor_after": {"pass": chat_pass, "position": chat_position}}
            if expected and record != expected["epochs"][epoch]:
                raise ValueError(f"Rebuilt map differs from frozen epoch {epoch}")
            records.append(record)
            print(json.dumps({"map_epoch": epoch + 1, "windows": len(mapping),
                              "chat_fraction": round(record["chat_fraction"], 6)}), flush=True)
            del mapping
        train = web_manifest["splits"]["train"]
        kept = [name for value, name in enumerate(("probe", "p20", "p5", "p1")) if value <= request["max_tier"]]
        web_characters = sum(train["tiers"][name]["characters"] for name in kept)
        result = {"format": FORMAT, "status": "complete", "request": request, "epochs": records,
                  "probe": {"file": probe_path.name, "sha256": file_sha(probe_path), "windows": len(probe),
                            "prediction_pairs": train["tiers"]["probe"]["prediction_pairs"],
                            "characters": train["tiers"]["probe"]["characters"],
                            "documents": train["tiers"]["probe"]["documents"]},
                  "web_tiers_kept": kept, "web_characters_per_epoch": web_characters,
                  "web_characters_total": web_characters * request["epochs"],
                  "chat_targets_total": chat_cumulative, "chat_uncropped_targets_per_pass": chat_uncropped,
                  "chat_passes": chat_cumulative / chat_uncropped,
                  "policy": ("Each pool web window once per epoch, prefix crop per epoch; chat as one run-long "
                             f"sequence of permuted passes, pass k through crop copy k, at {CHAT_FRACTION:.0%} "
                             "of cumulative cropped targets; maps SHA256-pinned")}
        if expected and result != expected:
            raise ValueError("Rebuilt mixture manifest differs")
        if expected is None:
            write_lf(directory / "manifest.json", result)
        return result
    finally:
        joined.close()


class Zh13Mixture(MixtureWindowDataset):
    """MixtureWindowDataset over a vimeml_zh13_mixture_v1 manifest."""

    def __init__(self, base, directory, root=ROOT):
        self.base = base
        self.directory = Path(directory)
        self.manifest = read(self.directory / "manifest.json")
        if self.manifest.get("status") != "complete" or self.manifest.get("format") != FORMAT:
            raise ValueError("Incomplete or unsupported vimeml-tiny-zh-v1 mixture")
        request = self.manifest["request"]
        if (request["token_manifest_sha256"] != file_sha(Path(base.token_dir) / "manifest.json")
                or request["window_manifest_sha256"] != file_sha(Path(root) / request["index_dir"] / "manifest.json")):
            raise ValueError("Mixture belongs to different token data/indexes")
        if (request["seed"], request["crop_probability"], request["crop_minimum"]) != (
                base.seed, base.probability, base.min_remaining_tokens):
            raise ValueError("Mixture crop settings differ from training")
        self.context_length = base.context_length
        self.split = "train"
        self.epoch = 0
        self._mapping = self._mapping_epoch = self._store = None


def training_dataset(directory, root=ROOT):
    manifest = read(Path(directory) / "manifest.json")
    joined = joined_windows(manifest["request"], root)
    try:
        return Zh13Mixture(joined, directory, root)
    except BaseException:
        joined.close()
        raise
