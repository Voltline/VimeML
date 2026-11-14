"""Build the pinyin-conditioned stores for the zh three-task model (stage 1).

Selection
    train: web/review documents in P1 (zh13 pools.json, label zh-v13-pool) outside the
    memorization probe, plus LCCC train utterances; validation: every export validation
    document and LCCC validation utterance. u = doc_hash(label, id) / 2**64 (chat key
    "lccc/<dialogue>/<turn>"): u < plain_fraction -> plain, the next pinyin_fraction ->
    pinyin. Disjoint documents feed the two, so no paragraph is in both. chat_pinyin_repeats
    (default 1) gives each train chat utterance that many pinyin samples, seeds "<key>#r".

Stores (one directory each under output_dir)
    plain     BOS + paragraph + EOS cut into <=128-token windows, every target supervised
              (first = 1); windows shorter than plain_minimum_tokens are dropped
    full, mixed, initials, error
              [2,4]+context+[5]+keys+[6]+output+[3], first = index of the first output
              token; one sample per simplified paragraph or utterance, kind drawn from
              kind_weights (a paragraph holding a listed misreading becomes an error
              sample around it with listed_boost_probability)

Files per store and split: {split}.tokens.bin (<u2), {split}.offsets.bin (<u8, count+1),
{split}.first.bin (u1). Every random choice comes from random.Random(doc_hash(label/rng,
id)), and parts are merged in job order, so a rebuild with the same inputs and library
versions is byte-identical.
"""
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np

from vimeml.posttraining.zh_pinyin import KINDS, Pinyin, encode
from vimeml.tokenizer.encode_zh13 import clean_paragraphs, doc_hash
from vimeml.tokenizer.zh13 import META_SPACE, sha256, write_json

FORMAT = "vimeml_zh_pinyin_v1"
STORES = ("plain",) + KINDS
SPLITS = ("train", "validation")

_state = {}


def source_of(path):
    name = Path(path).name
    return "chat" if name.startswith("chat-") else "review" if name.startswith("review-") else "web"


def jobs(export, split, per_job, max_groups=0):
    import pyarrow.parquet as pq
    directory = Path(export) / split
    files = (sorted(directory.glob("web-*.parquet")) + sorted(directory.glob("review-*.parquet"))
             + [directory / f"chat-lccc-{split}.parquet"])
    out = []
    for path in files:
        groups = pq.ParquetFile(path).metadata.num_row_groups
        groups = min(groups, max_groups) if max_groups else groups
        for start in range(0, groups, per_job):
            out.append({"path": str(path), "split": split, "groups": [start, min(groups, start + per_job)],
                        "name": f"{split}-{path.stem}-{start:04d}"})
    return out


def _init(config):
    import sentencepiece as spm
    _state["config"] = config
    _state["pinyin"] = Pinyin(config["sample"], config["readings"])
    _state["sp"] = spm.SentencePieceProcessor(model_file=config["tokenizer"])
    _state["pools"] = json.loads(Path(config["pools_path"]).read_text("utf-8"))


class Part:
    def __init__(self):
        self.tokens = {s: array("H") for s in STORES}
        self.lengths = {s: array("B") for s in STORES}
        self.first = {s: array("B") for s in STORES}

    def add(self, store, ids, first):
        if not (2 <= len(ids) <= 129 and 1 <= first < len(ids)):
            raise ValueError("Malformed sample")
        self.tokens[store].extend(ids)
        self.lengths[store].append(len(ids))
        self.first[store].append(first)

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for store in STORES:
            np.asarray(self.tokens[store], dtype="<u2").tofile(directory / f"{store}.tokens.bin")
            np.asarray(self.lengths[store], dtype="u1").tofile(directory / f"{store}.lengths.bin")
            np.asarray(self.first[store], dtype="u1").tofile(directory / f"{store}.first.bin")


def _plain(part, stats, sp, text, minimum):
    ids = [2] + sp.encode(text) + [3]
    for start in range(0, len(ids), 128):
        window = ids[start:start + 128]
        if len(window) < minimum:
            stats["plain_short_windows"] += 1
            continue
        part.add("plain", window, 1)
        stats["plain_tokens"] += len(window)


def _pinyin(part, stats, rng, text):
    config, pinyin, sp = _state["config"], _state["pinyin"], _state["sp"]
    weights = config["sample"]["kind_weights"]
    require, kind = None, None
    hit = pinyin.misread_pattern.search(text)
    if hit and rng.random() < config["listed_boost_probability"]:
        require, kind = hit.span(), "error"
    else:
        kind = rng.choices(KINDS, weights=[weights[k] for k in KINDS])[0]
    sample = pinyin.sample(rng, text, kind, require)
    if sample is None:
        stats[f"skip_{kind}"] += 1
        return
    context, keys, output, info = sample
    encoded = encode(sp, context, keys, output)
    if encoded is None:
        stats[f"skip_encode_{kind}"] += 1
        return
    ids, first, removed = encoded
    part.add(kind, ids, first)
    prefix = f"{kind}_"
    stats[prefix + "tokens"] += len(ids)
    stats[prefix + "output_chars"] += len(output)
    stats[prefix + "output_tokens"] += len(ids) - first - 1
    stats[prefix + "key_tokens"] += first - ids.index(5) - 2
    stats[prefix + "context_tokens"] += ids.index(5) - 2
    stats[prefix + "empty_context"] += ids.index(5) == 2
    stats[prefix + "apostrophes"] += "'" in keys
    stats[prefix + "context_trimmed"] += removed > 0
    for name in info.get("errors", ()):
        stats[f"error_source_{name}"] += 1
    if kind == "error":
        stats[f"error_base_{info['base']}"] += 1
        stats[f"error_count_{len(info['errors'])}"] += 1


def run_job(job):
    import pyarrow.parquet as pq
    config, sp, pools = _state["config"], _state["sp"], _state["pools"]
    split, source = job["split"], source_of(job["path"])
    label = config["label"]
    if split == "train":
        plain_fraction = config["chat_plain_fraction" if source == "chat" else "web_plain_fraction"]
        pinyin_fraction = config["chat_pinyin_fraction" if source == "chat" else "web_pinyin_fraction"]
    else:
        plain_fraction = config["validation_chat_plain_fraction" if source == "chat" else "validation_plain_fraction"]
        pinyin_fraction = config["validation_chat_pinyin_fraction" if source == "chat" else "validation_pinyin_fraction"]
    repeats = config.get("chat_pinyin_repeats", 1) if split == "train" and source == "chat" else 1
    plain_cut = int(plain_fraction * 2 ** 64)
    pinyin_cut = int((plain_fraction + pinyin_fraction) * 2 ** 64)
    thresholds, memo_threshold = pools["thresholds"], pools["memo_threshold"]
    pool_label, memo_label = pools["labels"]["pool"], pools["labels"]["memo"]
    part, stats = Part(), Counter()
    handle = pq.ParquetFile(job["path"])
    columns = ["text", "dialogue", "turn", "script"] if source == "chat" else ["id", "text", "script"]
    for group in range(*job["groups"]):
        table = handle.read_row_group(group, columns=columns).to_pydict()
        if source == "chat":
            keys = [f"lccc/{d}/{t}" for d, t in zip(table["dialogue"], table["turn"])]
        else:
            keys = table["id"]
        for key, text, script in zip(keys, table["text"], table["script"]):
            stats["documents_seen"] += 1
            if split == "train" and source != "chat":
                h = doc_hash(pool_label, key)
                if h >= thresholds["p1"]:
                    continue
                if h < thresholds["p20"] and doc_hash(memo_label, key) < memo_threshold:
                    stats["probe_excluded"] += 1
                    continue
            u = doc_hash(label, key)
            if u >= pinyin_cut:
                continue
            if source == "chat":
                text = text.strip()
                if not text or META_SPACE in text:
                    continue
                paragraphs = [text]
            else:
                paragraphs, _ = clean_paragraphs(text)
            rng = random.Random(doc_hash(label + "/rng", key))
            if u < plain_cut:
                stats["plain_documents"] += 1
                for paragraph in paragraphs:
                    _plain(part, stats, sp, paragraph, config["plain_minimum_tokens"])
            elif script == "hans":
                stats["pinyin_documents"] += 1
                for paragraph in paragraphs:
                    stats["pinyin_paragraphs"] += 1
                    _pinyin(part, stats, rng, paragraph)
                # chat_pinyin_repeats > 1 draws extra samples per train utterance, each from its own seed
                for r in range(1, repeats):
                    extra = random.Random(doc_hash(label + "/rng", f"{key}#{r}"))
                    for paragraph in paragraphs:
                        stats["pinyin_repeat_paragraphs"] += 1
                        _pinyin(part, stats, extra, paragraph)
            else:
                stats["pinyin_skipped_hant_documents"] += 1
    part.save(Path(config["parts_dir"]) / job["name"])
    counts = {s: len(part.lengths[s]) for s in STORES}
    return job["name"], source, counts, dict(stats)


def merge(parts_dir, names, output_dir, split):
    result = {}
    for store in STORES:
        directory = Path(output_dir) / store
        directory.mkdir(parents=True, exist_ok=True)
        lengths = [np.fromfile(Path(parts_dir) / n / f"{store}.lengths.bin", dtype="u1") for n in names]
        lengths = np.concatenate(lengths).astype("<u8") if lengths else np.zeros(0, "<u8")
        offsets = np.zeros(len(lengths) + 1, dtype="<u8")
        np.cumsum(lengths, out=offsets[1:])
        with open(directory / f"{split}.tokens.bin", "wb") as out:
            for n in names:
                out.write((Path(parts_dir) / n / f"{store}.tokens.bin").read_bytes())
        with open(directory / f"{split}.first.bin", "wb") as out:
            for n in names:
                out.write((Path(parts_dir) / n / f"{store}.first.bin").read_bytes())
        offsets.tofile(directory / f"{split}.offsets.bin")
        if (directory / f"{split}.tokens.bin").stat().st_size != 2 * int(offsets[-1]):
            raise ValueError(f"{store}/{split} token size mismatch")
        result[store] = {"examples": int(len(lengths)), "tokens": int(offsets[-1]),
                         "files": {f"{split}.{k}.bin": sha256(directory / f"{split}.{k}.bin")
                                   for k in ("tokens", "offsets", "first")}}
    return result


def preview(output_dir, sp, count=40):
    lines = []
    for store in STORES:
        directory = Path(output_dir) / store
        ids = np.fromfile(directory / "validation.tokens.bin", dtype="<u2")
        offsets = np.fromfile(directory / "validation.offsets.bin", dtype="<u8")
        first = np.fromfile(directory / "validation.first.bin", dtype="u1")
        for i in range(min(count, len(first))):
            seq = ids[offsets[i]:offsets[i + 1]].tolist()
            if store == "plain":
                lines.append({"store": store, "text": sp.decode(seq)})
                continue
            r, o = seq.index(5), seq.index(6)
            lines.append({"store": store, "context": sp.decode(seq[2:r]), "keys": sp.decode(seq[r + 1:o]),
                          "output": sp.decode(seq[int(first[i]):-1])})
    with open(Path(output_dir) / "preview.jsonl", "w", encoding="utf-8", newline="\n") as out:
        for line in lines:
            out.write(json.dumps(line, ensure_ascii=False) + "\n")


def build(config, export, output_dir, identity, max_groups=0):
    import sentencepiece as spm
    started = time.time()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} exists; remove it first")
    if sha256(config["tokenizer"]) != config["tokenizer_sha256"]:
        raise ValueError("tokenizer identity mismatch")
    pools = json.loads(Path(config["pools_path"]).read_text("utf-8"))
    if pools["status"] != "complete" or (identity.get("frozen") and
                                          pools["export"]["manifest_sha256"] != identity["manifest_sha256"]):
        raise ValueError("pools.json does not match the export")
    if sha256(config["readings"]) != config["readings_sha256"]:
        raise ValueError("reading table identity mismatch")
    config = dict(config, parts_dir=str(output_dir / "parts"))
    manifest = {"format": FORMAT, "status": "partial", "stores": {}, "stats": {}}
    with ProcessPoolExecutor(config["workers"], initializer=_init, initargs=(config,)) as pool:
        for split in SPLITS:
            todo = jobs(export, split, config["row_groups_per_job"], max_groups)
            names, stats, sources = [], Counter(), {}
            for done, (name, source, counts, part_stats) in enumerate(pool.map(run_job, todo), 1):
                names.append(name)
                stats.update(part_stats)
                for store, n in counts.items():
                    sources.setdefault(store, Counter())[source] += n
                if done % 20 == 0 or done == len(todo):
                    print(f"{split} {done}/{len(todo)} jobs {time.time() - started:.0f}s", flush=True)
            manifest["stores"][split] = merge(config["parts_dir"], names, output_dir, split)
            for store in STORES:
                manifest["stores"][split][store]["sources"] = dict(sources.get(store, {}))
            manifest["stats"][split] = dict(sorted(stats.items()))
    shutil.rmtree(config["parts_dir"])
    sp = spm.SentencePieceProcessor(model_file=config["tokenizer"])
    preview(output_dir, sp)
    here = Path(__file__).resolve().parent
    manifest.update({
        "status": "complete",
        "export": identity,
        "pools_sha256": sha256(config["pools_path"]),
        "tokenizer_sha256": config["tokenizer_sha256"],
        "config": {k: v for k, v in config.items() if k not in ("parts_dir",)},
        "max_row_groups": max_groups,
        "code_sha256": {name: sha256(here / name) for name in ("zh_pinyin.py", "zh_pinyin_data.py")},
        "readings_sha256": config["readings_sha256"],
        "libraries": {"sentencepiece": importlib.metadata.version("sentencepiece")},
    })
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "run.json", {"seconds": round(time.time() - started, 1)})
    return manifest
