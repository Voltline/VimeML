"""vimeml-tiny-zh-v1 token stores with nested hash pools.

measure (vocabulary-independent)
    Every web/review train document gets h = first 8 bytes of
    sha256(pool_label + "\\0" + id) as an unsigned integer, and its paragraph
    character count. q is the smallest h such that documents with h < q hold at
    least p1_target_characters; P5 is h < q // 5 and P20 is h < q // 20, so
    P20 is a subset of P5, which is a subset of P1. The memorization probe is
    the memo_documents P20 documents with the smallest memo hash.

encode (per vocabulary)
    Web store: only P1 documents, one sequence BOS + paragraph + EOS per
    paragraph, in export order, with train.tiers.bin (uint8 per paragraph):
    0 probe (in P20), 1 P20, 2 P5 only, 3 P1 only. validation holds every
    validation document. Chat store: each LCCC utterance is one sequence.

Paragraphs are the stripped nonempty lines of a document. A paragraph containing
U+2581 (SentencePiece meta space, decoded as a space) cannot roundtrip and is
dropped everywhere, in measure too, and counted. Every stored paragraph is
checked to decode to its exact text. The file layout and manifest are the
vimeml_sentence_tokens_v1 format read by TokenStore and SentenceWindowDataset.

pools.json and the store manifests hold no timing, so a rebuild from the same
export, tokenizer and sentencepiece version is byte-identical on any machine;
the wall time goes to run.json beside them, which nothing hashes.
"""
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from vimeml.tokenizer.zh13 import META_SPACE, paragraphs, sha256, write_json

SPECIAL = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
WEB_SOURCES = {"fineweb2-web": 0, "fineweb2-review": 1}
CHAT_SOURCES = {"lccc": 0}
FIRST_TEXT_ID = 7  # 0-3 special, 4-6 control (<CTX>/<READ>/<OUT>); 7 is "\n"
TIER_NAMES = ("probe", "p20", "p5", "p1")


def doc_hash(label, doc_id):
    return int.from_bytes(hashlib.sha256(f"{label}\0{doc_id}".encode("utf-8")).digest()[:8], "big")


def clean_paragraphs(text):
    kept, dropped = [], 0
    for line in paragraphs(text):
        if META_SPACE in line:
            dropped += 1
        else:
            kept.append(line)
    return kept, dropped


def web_files(export):
    train = Path(export) / "train"
    return sorted(train.glob("web-*.parquet")) + sorted(train.glob("review-*.parquet"))


def source_of(path):
    return WEB_SOURCES["fineweb2-review" if Path(path).name.startswith("review-") else "fineweb2-web"]


# ---- measure -------------------------------------------------------------------------------

def _measure_file(job):
    import pyarrow.parquet as pq
    path, label, memo_label, out, max_groups = job
    handle = pq.ParquetFile(path)
    hashes, memo, chars = [], [], []
    dropped = 0
    for group in range(min(handle.metadata.num_row_groups, max_groups or 1 << 30)):
        table = handle.read_row_group(group, columns=["id", "text"])
        for doc_id, text in zip(table["id"].to_pylist(), table["text"].to_pylist()):
            kept, lost = clean_paragraphs(text)
            dropped += lost
            hashes.append(doc_hash(label, doc_id))
            memo.append(doc_hash(memo_label, doc_id))
            chars.append(sum(map(len, kept)))
    np.savez(out, hash=np.array(hashes, dtype=np.uint64), memo=np.array(memo, dtype=np.uint64),
             chars=np.array(chars, dtype=np.uint64))
    return {"file": Path(path).name, "documents": len(hashes), "characters": int(sum(chars)),
            "dropped_meta_space_paragraphs": dropped}


def measure(config, export, pools, export_identity, max_groups=0):
    export, pools = Path(export), Path(pools)
    if (pools / "pools.json").exists():
        raise ValueError(f"{pools} already holds pools.json; use a new directory")
    (pools / "measure").mkdir(parents=True, exist_ok=True)
    files = web_files(export)
    jobs = [(str(p), config["pool_label"], config["memo_label"], str(pools / "measure" / (p.stem + ".npz")), max_groups)
            for p in files]
    began = time.time()
    with ProcessPoolExecutor(config["workers"]) as pool:
        per_file = list(pool.map(_measure_file, jobs))
    arrays = [np.load(job[3]) for job in jobs]
    h = np.concatenate([a["hash"] for a in arrays])
    m = np.concatenate([a["memo"] for a in arrays])
    c = np.concatenate([a["chars"] for a in arrays])
    order = np.argsort(h, kind="stable")
    cumulative = np.cumsum(c[order])
    target = config["p1_target_characters"]
    if cumulative[-1] < target:
        raise ValueError(f"Only {int(cumulative[-1])} characters; target {target}")
    last = int(np.searchsorted(cumulative, target))
    q = int(h[order[last]]) + 1  # documents with h < q reach the target
    thresholds = {"p1": q, "p5": q // 5, "p20": q // 20}
    in20 = h < thresholds["p20"]
    probe_pool = np.flatnonzero(in20)
    if len(probe_pool) < config["memo_documents"]:
        raise ValueError("P20 is smaller than the memorization probe")
    ranked = probe_pool[np.argsort(m[probe_pool], kind="stable")]
    memo_threshold = int(m[ranked[config["memo_documents"] - 1]]) + 1
    tiers = {}
    probe = in20 & (m < memo_threshold)
    masks = {"probe": probe, "p20": in20 & ~probe, "p5": (h < thresholds["p5"]) & ~in20,
             "p1": (h < q) & ~(h < thresholds["p5"])}
    for name, mask in masks.items():
        tiers[name] = {"documents": int(mask.sum()), "characters": int(c[mask].sum())}
    pools_by_name = {"P20": in20, "P5": h < thresholds["p5"], "P1": h < q}
    result = {"format": "vimeml_zh13_pools_v1", "status": "complete", "export": export_identity,
              "labels": {"pool": config["pool_label"], "memo": config["memo_label"]},
              "hash": "first 8 bytes of sha256(label + NUL + id), big-endian unsigned",
              "p1_target_characters": target, "thresholds": thresholds,
              "memo_threshold": memo_threshold, "memo_documents": config["memo_documents"],
              "pools": {k: {"documents": int(v.sum()), "characters": int(c[v].sum())} for k, v in pools_by_name.items()},
              "tiers": tiers, "all_train": {"documents": int(len(h)), "characters": int(c.sum())},
              "files": per_file, "max_row_groups": max_groups}
    write_json(pools / "pools.json", result)
    seconds = round(time.time() - began, 1)
    write_json(pools / "run.json", {"seconds": seconds, "pools_sha256": sha256(pools / "pools.json")})
    return {**result, "seconds": seconds}


# ---- encode --------------------------------------------------------------------------------

class PartWriter:
    # Hand the bytes to the file system in large blocks: one small write per document
    # costs more than the encoding itself on a network volume.
    FLUSH_BYTES = 8 << 20

    def __init__(self, prefix):
        self.prefix = Path(prefix)
        self.streams = {name: open(f"{self.prefix}.{name}", "wb") for name in ("tokens", "lengths", "sources", "tiers")}
        self.buffers = {name: bytearray() for name in self.streams}
        self.stats = {"sentences": 0, "stored_tokens": 0, "content_tokens": 0, "characters": 0,
                      "max_sequence_tokens": 0, "dropped_meta_space_paragraphs": 0, "documents": 0,
                      "tiers": {name: {"documents": 0, "sentences": 0, "characters": 0, "prediction_pairs": 0}
                                for name in TIER_NAMES}}

    def _write(self, name, data):
        buffer = self.buffers[name]
        buffer += data
        if len(buffer) >= self.FLUSH_BYTES:
            self.streams[name].write(buffer)
            buffer.clear()

    def _flush(self):
        for name, buffer in self.buffers.items():
            if buffer:
                self.streams[name].write(buffer)
                buffer.clear()

    def add(self, sp, texts, source, tier=None):
        ids = sp.encode(texts)
        lengths = np.fromiter((len(x) + 2 for x in ids), dtype=np.uint32, count=len(ids))
        flat = np.empty(int(lengths.sum()), dtype="<u2")
        at = 0
        for text, row in zip(texts, ids):
            if sp.decode(row) != text or min(row) < FIRST_TEXT_ID:
                raise ValueError(f"Roundtrip/reserved-ID failure in {self.prefix.name}: {text[:40]!r}")
            flat[at] = SPECIAL["bos"]
            flat[at + 1:at + 1 + len(row)] = row
            flat[at + 1 + len(row)] = SPECIAL["eos"]
            at += len(row) + 2
        self._write("tokens", flat.tobytes())
        self._write("lengths", lengths.astype("<u4").tobytes())
        self._write("sources", bytes([source]) * len(ids))
        if tier is not None:
            self._write("tiers", bytes([tier]) * len(ids))
            t = self.stats["tiers"][TIER_NAMES[tier]]
            t["sentences"] += len(ids)
            t["characters"] += sum(map(len, texts))
            t["prediction_pairs"] += int(lengths.sum()) - len(ids)
        s = self.stats
        s["sentences"] += len(ids)
        s["stored_tokens"] += int(lengths.sum())
        s["content_tokens"] += int(lengths.sum()) - 2 * len(ids)
        s["characters"] += sum(map(len, texts))
        if len(ids):
            s["max_sequence_tokens"] = max(s["max_sequence_tokens"], int(lengths.max()))

    def close(self):
        self._flush()
        for stream in self.streams.values():
            stream.close()
        return self.stats


def _tier(h, m, thresholds, memo_threshold):
    if h >= thresholds["p1"]:
        return None
    if h < thresholds["p20"]:
        return 0 if m < memo_threshold else 1
    return 2 if h < thresholds["p5"] else 3


def _encode_file(job):
    import pyarrow.parquet as pq
    import sentencepiece as spm
    path, kind, model, prefix, pools, max_groups = job
    sp = spm.SentencePieceProcessor(model_file=model)
    writer = PartWriter(prefix)
    handle = pq.ParquetFile(path)
    groups = min(handle.metadata.num_row_groups, max_groups or 1 << 30)
    try:
        for group in range(groups):
            if kind == "chat":
                texts = handle.read_row_group(group, columns=["text"])["text"].to_pylist()
                kept = [t.strip() for t in texts if t.strip() and META_SPACE not in t]
                writer.stats["dropped_meta_space_paragraphs"] += sum(1 for t in texts if META_SPACE in t)
                writer.stats["documents"] += len(kept)
                for start in range(0, len(kept), 4096):
                    writer.add(sp, kept[start:start + 4096], CHAT_SOURCES["lccc"])
                continue
            table = handle.read_row_group(group, columns=["id", "text"])
            for doc_id, text in zip(table["id"].to_pylist(), table["text"].to_pylist()):
                tier = None
                if kind == "train":
                    tier = _tier(doc_hash(pools["labels"]["pool"], doc_id), doc_hash(pools["labels"]["memo"], doc_id),
                                 pools["thresholds"], pools["memo_threshold"])
                    if tier is None:
                        continue
                kept, lost = clean_paragraphs(text)
                writer.stats["dropped_meta_space_paragraphs"] += lost
                writer.stats["documents"] += 1
                if tier is not None:
                    writer.stats["tiers"][TIER_NAMES[tier]]["documents"] += 1
                if kept:
                    writer.add(sp, kept, source_of(path), tier)
    finally:
        stats = writer.close()
    return {"file": Path(path).name, "prefix": str(prefix), "row_groups": groups, **stats}


def _merge(parts, out, split, with_tiers):
    """Concatenate part files in export order; offsets from cumulative lengths."""
    names = [f"{split}.tokens.bin", f"{split}.offsets.bin", f"{split}.sources.bin"] + ([f"{split}.tiers.bin"] if with_tiers else [])
    streams = {name: open(out / name, "wb") for name in names}
    total = 0
    try:
        streams[f"{split}.offsets.bin"].write(np.zeros(1, dtype="<u8").tobytes())
        for part in parts:
            prefix = part["prefix"]
            with open(f"{prefix}.tokens", "rb") as source:
                while block := source.read(1 << 24):
                    streams[f"{split}.tokens.bin"].write(block)
            lengths = np.fromfile(f"{prefix}.lengths", dtype="<u4").astype(np.uint64)
            streams[f"{split}.offsets.bin"].write((np.cumsum(lengths) + np.uint64(total)).astype("<u8").tobytes())
            total += int(lengths.sum())
            streams[f"{split}.sources.bin"].write(Path(f"{prefix}.sources").read_bytes())
            if with_tiers:
                streams[f"{split}.tiers.bin"].write(Path(f"{prefix}.tiers").read_bytes())
    finally:
        for stream in streams.values():
            stream.close()
    return names


def _split_stats(parts, tiers):
    keys = ("sentences", "stored_tokens", "content_tokens", "characters", "documents", "dropped_meta_space_paragraphs")
    stats = {k: sum(p[k] for p in parts) for k in keys}
    stats["max_sequence_tokens"] = max(p["max_sequence_tokens"] for p in parts)
    stats["prediction_pairs"] = stats["stored_tokens"] - stats["sentences"]
    if tiers:
        stats["tiers"] = {name: {k: sum(p["tiers"][name][k] for p in parts) for k in parts[0]["tiers"][name]}
                          for name in TIER_NAMES}
    return stats


def encode(config, export, vocab, outputs, pools_path, export_identity, max_groups=0):
    """outputs: {'web': token dir, 'chat': token dir, 'web_index': dir, 'chat_index': dir}."""
    import sentencepiece
    from vimeml.training.data import prepare_indexes
    spm_version = sentencepiece.__version__
    export = Path(export)
    tokenizers = json.loads((Path(config["tokenizer_dir"]) / "manifest.json").read_text("utf-8"))
    record = tokenizers["models"][str(vocab)]
    model = Path(config["tokenizer_dir"]) / record["directory"] / "tokenizer.model"
    if sha256(model) != record["tokenizer_model_sha256"]:
        raise ValueError("Tokenizer differs from its manifest")
    pools = json.loads(Path(pools_path).read_text("utf-8"))
    if pools.get("status") != "complete" or pools["export"] != export_identity:
        raise ValueError("Pools were measured on a different export")
    for key in ("web", "chat"):
        if Path(outputs[key]).exists() and any(Path(outputs[key]).iterdir()):
            raise ValueError(f"{outputs[key]} is not empty; use a new directory")
        (Path(outputs[key]) / "parts").mkdir(parents=True, exist_ok=True)
    began = time.time()
    jobs = {"web": {}, "chat": {}}
    for split, kind, files in (("train", "train", web_files(export)),
                               ("validation", "validation", sorted((export / "validation").glob("web-*.parquet"))
                                + sorted((export / "validation").glob("review-*.parquet")))):
        jobs["web"][split] = [(str(p), kind, str(model), str(Path(outputs["web"]) / "parts" / f"{split}-{p.stem}"),
                               pools, max_groups) for p in files]
    for split in ("train", "validation"):
        path = export / split / f"chat-lccc-{split}.parquet"
        jobs["chat"][split] = [(str(path), "chat", str(model), str(Path(outputs["chat"]) / "parts" / split), pools, max_groups)]
    results = {}
    with ProcessPoolExecutor(config["workers"]) as pool:
        futures = {(store, split): [pool.submit(_encode_file, job) for job in split_jobs]
                   for store, by_split in jobs.items() for split, split_jobs in by_split.items()}
        for key, items in futures.items():
            total = len(items)
            done = 0
            ordered = [None] * total
            # Progress per finished file: a whole split can take an hour, so waiting for the
            # group before printing anything hides every sign of progress.
            for future in as_completed(items):
                record = future.result()
                ordered[items.index(future)] = record
                done += 1
                print(json.dumps({"file": record["file"], "sentences": record["sentences"],
                                  "characters": record["characters"],
                                  "done": done, "of": total, "split": "/".join(key)}), flush=True)
            results[key] = ordered
            print(json.dumps({"encoded": list(key), "files": total,
                              "sentences": sum(r["sentences"] for r in ordered)}), flush=True)
    manifests = {}
    for store, sources in (("web", WEB_SOURCES), ("chat", CHAT_SOURCES)):
        out = Path(outputs[store])
        splits, digests = {}, {}
        for split in ("train", "validation"):
            parts = results[(store, split)]
            with_tiers = store == "web" and split == "train"
            names = _merge(parts, out, split, with_tiers)
            splits[split] = _split_stats(parts, with_tiers)
            splits[split]["file_bytes"] = {n.split(".", 1)[1]: (out / n).stat().st_size for n in names}
            splits[split]["primary_sources"] = {name: 0 for name in sources}
            source_counts = np.bincount(np.fromfile(out / f"{split}.sources.bin", dtype=np.uint8), minlength=len(sources))
            for name, index in sources.items():
                splits[split]["primary_sources"][name] = int(source_counts[index])
            for n in names:
                digests[n] = sha256(out / n)
        for part in (out / "parts").iterdir():
            part.unlink()
        (out / "parts").rmdir()
        manifest = {"format": "vimeml_sentence_tokens_v1", "status": "complete", "vocab_size": vocab,
                    "token_dtype": "uint16_le", "offset_dtype": "uint64_le", "offset_unit": "tokens",
                    "special_ids": SPECIAL, "reserved_ids": record["reserved_ids"],
                    "tokenizer_model_sha256": record["tokenizer_model_sha256"],
                    "tokenizer_manifest_sha256": sha256(Path(config["tokenizer_dir"]) / "manifest.json"),
                    "source_ids": sources, "splits": splits, "available_splits": ["train", "validation"],
                    "sequence_policy": ("BOS + stripped nonempty paragraph + EOS; P1 documents only in train, "
                                        "train.tiers.bin per paragraph (0 probe, 1 P20, 2 P5, 3 P1)"
                                        if store == "web" else "BOS + stripped LCCC utterance + EOS"),
                    "export": export_identity, "pools_sha256": sha256(pools_path), "output_sha256": digests,
                    "sentencepiece_version": spm_version,
                    "test_or_blind_read": False, "max_row_groups": max_groups}
        write_json(out / "manifest.json", manifest)
        write_json(out / "run.json", {"seconds": round(time.time() - began, 1),
                                      "manifest_sha256": sha256(out / "manifest.json")})
        index = prepare_indexes(out, Path(outputs[f"{store}_index"]), config["context_length"], splits=("train", "validation"))
        manifests[store] = {"tokens": manifest["splits"], "windows": {s: index["splits"][s]["windows"] for s in index["splits"]}}
    return manifests
