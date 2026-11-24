"""zh-800m mixture (docs/zh-CN/zh/training/training-zh-800m-plan.md, 800M Chinese desktop model).

Web: every window of three 12K stores, once: the vimeml-tiny-zh-v1 P1 store
(zh13-web-12k, all tiers), the rest of export-r1 outside P1 (zh800m-web-r1rest-12k)
and all of export-r2 (zh800m-web-r2-12k), each prefix-cropped with the run seed.
Chat: LCCC (zh13-chat-12k) as one run-long sequence of permuted passes, pass k
through crop copy k, at CHAT_FRACTION of the cumulative cropped targets; the
fraction is lower than the 20M/200M 5% so LCCC repeats about as often as a few
passes rather than ~24 (see the plan). One epoch map lists the joined window ids,
shuffled; maps are SHA256-pinned and must rebuild byte-identically. The probe is
the tier-0 windows of the P1 store, the same documents as the 20M/200M probe.
Validation is the P1 store's (export-r1) validation, comparable to 20M and 200M.

This module does not import torch.
"""
import json
from pathlib import Path

import numpy as np

from vimeml.training.data import SentenceWindowDataset, file_sha
from vimeml.training.data_mixture import JoinedPrefixWindows
from vimeml.training.data_v2 import MASK64, PrefixCropWindowDataset
from vimeml.training.zh200m_data import CHUNK, Zh200mMixture, _save, read, window_tiers, write_lf

ROOT = Path(__file__).resolve().parents[3]
FORMAT = "vimeml_zh800m_mixture_v1"
VOCAB = 12288
PROBE_TIER = 0
CHAT_FRACTION = 0.02
CROP_PROBABILITY = 0.30
CROP_MINIMUM = 8
CHAT_COPIES = 12
CHAT_SEED_STEP = 0x9E3779B97F4A7C15
# Distinct per-store crop seeds: the stores are disjoint, but equal seeds would give
# their window k the same crop draw.
STORE_SEED_STEP = 0xD1B54A32D192ED03
# (name, token store, encoder selection); the first is the primary store (validation, probe).
WEB_STORES = (("p1", "zh13-web-12k", "p1"),
              ("r1rest", "zh800m-web-r1rest-12k", "rest"),
              ("r2", "zh800m-web-r2-12k", "all"))
CHAT_STORE = "zh13-chat-12k"
# arm -> (vocabulary, context length, web epochs, seed); the seed was drawn once with
# secrets.randbelow(2**31) when the plan was written.
ARMS = {"12k-c512": (VOCAB, 512, 1, 1816189249)}


def store_paths(context, data_root="artifacts"):
    """[(name, token dir, index dir)] for the web stores, then the chat store."""
    stores = [(name, f"{data_root}/token-data/{tokens}", f"{data_root}/training-data/{tokens}-c{context}")
              for name, tokens, _ in WEB_STORES]
    stores.append(("chat", f"{data_root}/token-data/{CHAT_STORE}", f"{data_root}/training-data/{CHAT_STORE}-c{context}"))
    return stores


def mixture_directory(arm, data_root="artifacts"):
    return f"{data_root}/training-data/zh800m-{arm}"


def request_for(arm, root=ROOT, data_root="artifacts"):
    vocab, context, epochs, seed = ARMS[arm]
    records = []
    selections = {name: selection for name, _, selection in WEB_STORES}
    for name, tokens, index in store_paths(context, data_root):
        manifest = read(Path(root) / tokens / "manifest.json")
        if manifest["vocab_size"] != vocab:
            raise ValueError(f"{tokens} is not a {vocab} store")
        if name in selections and manifest.get("selection", "p1") != selections[name]:
            raise ValueError(f"{tokens} was encoded with selection {manifest.get('selection', 'p1')!r}, "
                             f"expected {selections[name]!r}")
        records.append({"name": name, "token_dir": tokens, "index_dir": index,
                        "token_manifest_sha256": file_sha(Path(root) / tokens / "manifest.json"),
                        "window_manifest_sha256": file_sha(Path(root) / index / "manifest.json"),
                        "tokenizer_model_sha256": manifest["tokenizer_model_sha256"]})
    if len({r["tokenizer_model_sha256"] for r in records}) != 1:
        raise ValueError("Stores use different tokenizers")
    web, chat = records[:-1], records[-1]
    primary = read(Path(root) / web[0]["token_dir"] / "manifest.json")
    rest = read(Path(root) / web[1]["token_dir"] / "manifest.json")
    if rest["pools_sha256"] != primary["pools_sha256"] or rest["export"] != primary["export"]:
        raise ValueError("The r1 rest store must come from the P1 store's export and pools")
    return {"format": FORMAT, "arm": arm, "vocab_size": vocab, "context_length": context,
            "epochs": epochs, "seed": seed, "chat_fraction": CHAT_FRACTION,
            "crop_probability": CROP_PROBABILITY, "crop_minimum": CROP_MINIMUM, "chat_copies": CHAT_COPIES,
            "chat_seed_step": CHAT_SEED_STEP, "store_seed_step": STORE_SEED_STEP,
            "pools_sha256": primary["pools_sha256"], "tokenizer_model_sha256": chat["tokenizer_model_sha256"],
            # MixtureWindowDataset-style checks run against the primary (P1) store.
            "token_manifest_sha256": web[0]["token_manifest_sha256"], "index_dir": web[0]["index_dir"],
            "window_manifest_sha256": web[0]["window_manifest_sha256"],
            "web": web[0], "web_stores": web, "chat": chat}


def store_seed(seed, store):
    return (seed + store * STORE_SEED_STEP) & MASK64


def chat_seed(seed, copy):
    return (seed + (copy + 1) * CHAT_SEED_STEP) & MASK64


def joined_windows(request, root=ROOT, seed=None):
    """The web stores in order, then chat_copies crop views of the chat store.
    Source ids: 0 web, 1 review (any web store), 2 chat."""
    seed = request["seed"] if seed is None else seed
    root = Path(root)
    options = {"probability": request["crop_probability"], "min_remaining_tokens": request["crop_minimum"]}
    bases = []
    try:
        for k, store in enumerate(request["web_stores"]):
            bases.append(PrefixCropWindowDataset(root / store["token_dir"], root / store["index_dir"],
                                                 seed=store_seed(seed, k), **options))
        chat = request["chat"]
        for copy in range(request["chat_copies"]):
            bases.append(PrefixCropWindowDataset(root / chat["token_dir"], root / chat["index_dir"],
                                                 seed=chat_seed(seed, copy), **options))
    except BaseException:
        for base in bases:
            base.close()
        raise
    webs = len(request["web_stores"])
    return JoinedPrefixWindows(bases, 0, [{0: 0, 1: 1}] * webs + [{0: 2}] * request["chat_copies"])


def _range_lengths(joined, start, end):
    """Cropped tokens of windows start..end-1 without materializing the id range."""
    total = 0
    for first in range(start, end, CHUNK):
        total += int(joined.window_lengths(np.arange(first, min(end, first + CHUNK), dtype=np.int64)).sum())
    return total


def prepare_maps(directory, root=ROOT):
    directory, root = Path(directory), Path(root)
    request = read(directory / "request.json")
    expected = read(directory / "manifest.json") if (directory / "manifest.json").exists() else None
    if expected:
        files = [r["file"] for r in expected["epochs"]] + [expected["probe"]["file"]]
        digests = [r["sha256"] for r in expected["epochs"]] + [expected["probe"]["sha256"]]
        if all((directory / f).exists() and file_sha(directory / f) == d for f, d in zip(files, digests)):
            return expected
    for store in request["web_stores"] + [request["chat"]]:
        if (file_sha(root / store["token_dir"] / "manifest.json") != store["token_manifest_sha256"]
                or file_sha(root / store["index_dir"] / "manifest.json") != store["window_manifest_sha256"]):
            raise ValueError(f"{store['name']} store identity changed")
    primary = read(root / request["web"]["token_dir"] / "manifest.json")
    tiers_path = root / request["web"]["token_dir"] / "train.tiers.bin"
    if file_sha(tiers_path) != primary["output_sha256"]["train.tiers.bin"]:
        raise ValueError("train.tiers.bin differs from its manifest")
    tiers = np.fromfile(tiers_path, dtype=np.uint8)
    joined = joined_windows(request, root)
    try:
        if len(joined) >= 1 << 32:
            raise ValueError("Joined windows exceed uint32 map ids")
        if len(tiers) != primary["splits"]["train"]["sentences"]:
            raise ValueError("Tier count differs from web sentences")
        webs = len(request["web_stores"])
        probe = np.flatnonzero(window_tiers(joined.bases[0], tiers) == PROBE_TIER).astype("<u4")
        del tiers
        probe_path = _save(directory, "probe-windows.npy", probe)
        web_end = int(joined.ends[webs - 1])
        chat_windows = len(joined.bases[webs])
        chat_starts = joined.starts[webs:]
        chat_uncropped = int(SentenceWindowDataset.window_lengths(joined.bases[webs], np.arange(chat_windows)).sum())
        records = []
        web_cumulative = chat_cumulative = 0
        chat_pass, chat_position, permutation = 0, 0, None
        for epoch in range(request["epochs"]):
            joined.set_epoch(epoch)
            per_store = [_range_lengths(joined, int(joined.starts[k]), int(joined.ends[k])) for k in range(webs)]
            web_tokens = sum(per_store)
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
            mapping = np.empty(web_end + len(chosen), dtype="<u4")
            mapping[:web_end] = np.arange(web_end, dtype="<u4")
            mapping[web_end:] = chosen
            np.random.default_rng([request["seed"], 1 << 20, epoch]).shuffle(mapping)
            path = _save(directory, f"epoch-{epoch}.npy", mapping)
            record = {"epoch": epoch, "file": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path),
                      "windows": len(mapping), "web_windows": web_end, "chat_windows": len(chosen),
                      "web_tokens": web_tokens,
                      "web_tokens_by_store": {s["name"]: t for s, t in zip(request["web_stores"], per_store)},
                      "chat_tokens": used, "effective_tokens": web_tokens + used,
                      "chat_fraction": used / (web_tokens + used),
                      "chat_cursor_after": {"pass": chat_pass, "position": chat_position}}
            if expected and record != expected["epochs"][epoch]:
                raise ValueError(f"Rebuilt map differs from frozen epoch {epoch}")
            records.append(record)
            print(json.dumps({"map_epoch": epoch + 1, "windows": len(mapping),
                              "chat_fraction": round(record["chat_fraction"], 6)}), flush=True)
            del mapping
        characters = {}
        for store in request["web_stores"]:
            characters[store["name"]] = read(root / store["token_dir"] / "manifest.json")["splits"]["train"]["characters"]
        train = primary["splits"]["train"]
        result = {"format": FORMAT, "status": "complete", "request": request, "epochs": records,
                  "probe": {"file": probe_path.name, "sha256": file_sha(probe_path), "windows": len(probe),
                            "prediction_pairs": train["tiers"]["probe"]["prediction_pairs"],
                            "characters": train["tiers"]["probe"]["characters"],
                            "documents": train["tiers"]["probe"]["documents"]},
                  "web_characters_by_store": characters,
                  "web_characters_per_epoch": sum(characters.values()),
                  "web_characters_total": sum(characters.values()) * request["epochs"],
                  "chat_targets_total": chat_cumulative, "chat_uncropped_targets_per_pass": chat_uncropped,
                  "chat_passes": chat_cumulative / chat_uncropped,
                  "policy": ("Every window of the three web stores once per epoch, prefix crop per epoch and store; "
                             "chat as one run-long sequence of permuted passes, pass k through crop copy k, at "
                             f"{CHAT_FRACTION:.0%} of cumulative cropped targets; maps SHA256-pinned")}
        if expected and result != expected:
            raise ValueError("Rebuilt mixture manifest differs")
        if expected is None:
            write_lf(directory / "manifest.json", result)
        return result
    finally:
        joined.close()


class Zh800mMixture(Zh200mMixture):
    """The zh-200m mixture reader over a vimeml_zh800m_mixture_v1 manifest."""

    def __init__(self, base, directory, root=ROOT):
        self.base = base
        self.directory = Path(directory)
        self.manifest = read(self.directory / "manifest.json")
        if self.manifest.get("status") != "complete" or self.manifest.get("format") != FORMAT:
            raise ValueError("Incomplete or unsupported zh-800m mixture")
        request = self.manifest["request"]
        if (request["token_manifest_sha256"] != file_sha(Path(base.token_dir) / "manifest.json")
                or request["window_manifest_sha256"] != file_sha(Path(root) / request["index_dir"] / "manifest.json")):
            raise ValueError("Mixture belongs to different token data/indexes")
        if (store_seed(request["seed"], 0), request["crop_probability"], request["crop_minimum"]) != (
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
        return Zh800mMixture(joined, directory, root)
    except BaseException:
        joined.close()
        raise
