"""Prepare frozen C1 conversion labels from explicitly allowed train/validation TXT.

CPU only. Never discovers evaluation directories or reads test/blind inputs.
Shards are resumable only under the exact same source/code/dictionary contract.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sqlite3
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from vimeml.posttraining.data import ConversionCodec, FORMAT, TOKENIZER_SHA, shifted
from vimeml.posttraining.readings import DualReading, hiragana
from vimeml.training.data import file_sha, write_json

VERSION = "c1-dual-orthographic-v1"
CORPUS = ROOT / "outputs/corpus-fast-v1"
STORE = ROOT / "artifacts/token-data/corpus-v4-8k"
TOKENIZER = ROOT / "artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model"
EXPECTED = {
    "train": "9c1b4585adaf0518dc0275e7f8086e967db26f9797fb571e1ff481aeadf2ff8f",
    "validation": "f188bea10a8b464dd385d8e0aa310225de07e2a0a170e90ce63503c150bf4cdd",
}
TRACKS = ("standard-development", "legacy-regression", "ajimee", "historical-development")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def case_key(context, reading, output):
    return digest(json.dumps([context, hiragana(reading), output], ensure_ascii=False, separators=(",", ":")))


def initialize(validation_hashes, cases, benchmark_parents):
    global ANALYZER, CODEC, VALIDATION_HASHES, CASES, BENCHMARK_PARENTS
    ANALYZER, CODEC = DualReading(), ConversionCodec(TOKENIZER)
    VALIDATION_HASHES = set(validation_hashes)
    CASES, BENCHMARK_PARENTS = set(cases), set(benchmark_parents)


def shard(task):
    split, start, stop, probability, directory, signature = task
    directory = Path(directory)
    stem = f"{split}-{start:09d}-{stop:09d}"
    done = directory / f"{stem}.json"
    data = directory / f"{stem}.jsonl"
    if done.exists():
        result = json.loads(done.read_text(encoding="utf-8"))
        if result["signature"] != signature or file_sha(data) != result["sha256"]:
            raise ValueError("Incompatible or damaged completed shard")
        return result
    positions = np.memmap(STORE / f"{split}.rows.bin", mode="r", dtype="<u8")
    sources = np.memmap(STORE / f"{split}.sources.bin", mode="r", dtype="u1")
    counts = Counter()
    temporary = data.with_suffix(".partial")
    began = time.monotonic()
    with (CORPUS / f"{split}.txt").open("rb") as source, temporary.open("w", encoding="utf-8", newline="\n") as target:
        source.seek(int(positions[start]))
        for row in range(start, stop):
            offset = source.tell()
            raw = source.readline()
            if offset != int(positions[row]) or not raw:
                raise ValueError("Frozen TXT row alignment mismatch")
            text = raw.decode("utf-8").rstrip("\r\n")
            parent = digest(text)
            # Stable content hashing: duplicates get the same inclusion decision.
            if int(parent[:16], 16) / 2**64 >= probability:
                continue
            counts["sampled_parents"] += 1
            if split == "train" and parent in VALIDATION_HASHES:
                counts["reject_validation_parent"] += 1
                continue
            if parent in BENCHMARK_PARENTS:
                counts["reject_known_nonblind_parent"] += 1
                continue
            example, reason = ANALYZER.example(text, int(parent[16:32], 16))
            if example is None:
                counts["reject_" + reason] += 1
                continue
            fingerprint = case_key(example["context"], example["reading"], example["output"])
            if fingerprint in CASES:
                counts["reject_known_nonblind_case"] += 1
                continue
            encoded = CODEC.encode(example["context"], example["reading"], example["output"])
            if encoded is None:
                counts["reject_target_reading_overflow"] += 1
                continue
            ids, first, removed = encoded
            check = shifted(ids, first)
            assert sum(x != -100 for x in check["labels"]) == len(ids) - first
            example.update(parent_sha256=parent, split=split, row=row, byte_offset=offset,
                           source_id=int(sources[row]), document_id=None, case_sha256=fingerprint,
                           context_crop_characters=removed, ids=ids, first_output=first,
                           orthographic_rule=VERSION, review="dual_dictionary_agreement_not_human_gold")
            target.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["accepted_pool"] += 1
    temporary.replace(data)
    result = {"split": split, "start": start, "stop": stop, "data": str(data), "signature": signature,
              "sha256": file_sha(data), "counts": dict(counts), "seconds": time.monotonic() - began}
    write_json(done, result)
    return result


def dictionary_contract():
    import unidic_lite
    import sudachidict_core
    binaries = sorted(Path(unidic_lite.DICDIR).glob("*.dic"))
    binaries += sorted((Path(sudachidict_core.__file__).parent / "resources").glob("*.dic"))
    if not binaries:
        raise ValueError("No dictionary binaries")
    return {"versions": {n: importlib.metadata.version(n) for n in (
                "fugashi", "unidic-lite", "SudachiPy", "SudachiDict-core", "sentencepiece")},
            "binaries": {str(p): file_sha(p) for p in binaries}}


def merge(directory, results, split, requested):
    """Deduplicate, select by stable hash within kana/kanji strata, then pack."""
    db = sqlite3.connect(directory / f"{split}.selection.sqlite")
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("DROP TABLE IF EXISTS candidates")
    db.execute("CREATE TABLE candidates (parent TEXT PRIMARY KEY, fingerprint TEXT UNIQUE, shard INTEGER, line INTEGER, kana INTEGER, priority INTEGER)")
    shards = sorted((x for x in results if x["split"] == split), key=lambda x: x["start"])
    offered = 0
    for shard_id, info in enumerate(shards):
        batch = []
        with Path(info["data"]).open(encoding="utf-8") as f:
            for line, raw in enumerate(f):
                ex = json.loads(raw)
                offered += 1
                batch.append((ex["parent_sha256"], ex["case_sha256"], shard_id, line,
                              not ex["has_kanji"], int(ex["parent_sha256"][32:48], 16) >> 1))
                if len(batch) == 10000:
                    db.executemany("INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?)", batch)
                    batch.clear()
            db.executemany("INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?)", batch)
        db.commit()
    available = dict(db.execute("SELECT kana, count(*) FROM candidates GROUP BY kana"))
    total = min(requested, available.get(0, 0) + available.get(1, 0), int(available.get(0, 0) / .8))
    kana = min(available.get(1, 0), total // 5)
    kanji = min(available.get(0, 0), total - kana)
    selected = defaultdict(set)
    db.execute("CREATE INDEX selected_priority ON candidates(kana,priority)")
    for kind, limit in ((0, kanji), (1, kana)):
        for sid, line in db.execute("SELECT shard,line FROM candidates WHERE kana=? ORDER BY priority,parent LIMIT ?", (kind, limit)):
            selected[sid].add(line)
    db.close()
    stats, histograms, examples = Counter(), defaultdict(Counter), []
    rng = random.Random(42)
    offset = 0
    with (directory / f"{split}.tokens.bin").open("wb") as tokens, \
         (directory / f"{split}.offsets.bin").open("wb") as offsets, \
         (directory / f"{split}.out-start.bin").open("wb") as starts, \
         (directory / f"{split}.reading-dropout.bin").open("wb") as dropouts, \
         (directory / f"{split}.records.jsonl").open("w", encoding="utf-8", newline="\n") as records:
        offsets.write(struct.pack("<Q", 0))
        for sid, info in enumerate(shards):
            with Path(info["data"]).open(encoding="utf-8") as source:
                for line, raw in enumerate(source):
                    if line not in selected[sid]:
                        continue
                    ex = json.loads(raw)
                    ids, first = ex.pop("ids"), ex.pop("first_output")
                    tokens.write(struct.pack(f"<{len(ids)}H", *ids))
                    offset += len(ids)
                    offsets.write(struct.pack("<Q", offset))
                    starts.write(bytes([first]))
                    ex["first_output"] = first
                    ex["reading_dropout_view"] = split == "train" and int(ex["parent_sha256"][-8:], 16) % 10 == 0
                    dropouts.write(bytes([ex["reading_dropout_view"]]))
                    records.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stats["examples"] += 1
                    stats["stored_tokens"] += len(ids)
                    stats["supervised_tokens_full_reading"] += len(ids) - first
                    stats["context_cropped"] += bool(ex["context_crop_characters"])
                    stats["kana_only"] += not ex["has_kanji"]
                    stats["reading_dropout_view"] += ex["reading_dropout_view"]
                    for name, value in (("reading_characters", len(ex["reading"])), ("output_characters", len(ex["output"])),
                                        ("context_characters", len(ex["context"])), ("source_id", ex["source_id"]),
                                        ("serialized_tokens", len(ids))):
                        histograms[name][value] += 1
                    # A bounded random review sample; machine review only until a human reviews it.
                    review = dict(ex, ids=ids)
                    if len(examples) < 200:
                        examples.append(review)
                    else:
                        k = rng.randrange(stats["examples"])
                        if k < 200:
                            examples[k] = review
    write_json(directory / f"{split}.review-sample.json", examples)
    return {**dict(stats), "requested": requested, "offered": offered,
            "deduplicated_pool": sum(available.values()), "histograms": dict(histograms),
            "selection": "deterministic parent-hash priority; kana-only <=20%; occupied length/context/kanji buckets sampled equally per parent"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/posttrain-data/v4-c1")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--train-examples", type=int, default=3_000_000)
    parser.add_argument("--validation-examples", type=int, default=20_000)
    parser.add_argument("--train-parent-fraction", type=float, default=.24)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / "manifest.json"
    if frozen.exists():
        existing = json.loads(frozen.read_text(encoding="utf-8"))
        if existing.get("status") != "complete":
            raise ValueError("Output is not complete; inspect its recorded state before resuming")
        if (existing["contract"]["requested"] != [args.train_examples, args.validation_examples]
                or existing["contract"]["train_parent_fraction"] != args.train_parent_fraction):
            raise ValueError("Frozen data has different settings; choose a fresh output directory")
        for name, expected in existing["output_sha256"].items():
            if file_sha(directory / name) != expected:
                raise ValueError(f"Frozen output damaged: {name}")
        print("Existing complete dataset preserved; see acceptance.json for its data audit.", flush=True)
        return
    parts = directory / "shards"
    parts.mkdir(exist_ok=True)
    started = time.monotonic()
    print("Verifying frozen source TXT and dictionary identities", flush=True)
    actual = {s: file_sha(CORPUS / f"{s}.txt") for s in ("train", "validation")}
    if actual != EXPECTED:
        raise ValueError("Original train/validation identity mismatch")
    token_manifest = json.loads((STORE / "manifest.json").read_text(encoding="utf-8"))
    if token_manifest["tokenizer_model_sha256"] != TOKENIZER_SHA or token_manifest["row_offset_source"] != "corpus_txt":
        raise ValueError("Token store contract mismatch")
    for split in ("train", "validation"):
        for suffix in ("rows.bin", "sources.bin"):
            name = f"{split}.{suffix}"
            if file_sha(STORE / name) != token_manifest["output_sha256"][name]:
                raise ValueError("Source provenance map hash mismatch")
    dictionaries = dictionary_contract()
    validation_hashes = [digest(raw.decode("utf-8").rstrip("\r\n")) for raw in (CORPUS / "validation.txt").open("rb")]
    cases, parents, decontamination = set(), set(), {}
    for track in TRACKS:
        path = ROOT / f"outputs/evaluation-v4-20261015/inputs/{track}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        decontamination[track] = {"sha256": file_sha(path), "examples": len(rows)}
        for row in rows:
            for answer in row["answers"]:
                cases.add(case_key(row["left_context"], row["query"], answer))
                parents.add(digest(row["left_context"] + answer))
    code = {str(p.relative_to(ROOT)): file_sha(p) for p in (
        Path(__file__), ROOT / "src/vimeml/posttraining/readings.py", ROOT / "src/vimeml/posttraining/data.py")}
    contract = {"version": VERSION, "source_sha256": actual, "tokenizer_sha256": TOKENIZER_SHA,
                "dictionary": dictionaries, "code_sha256": code, "decontamination": decontamination,
                "train_parent_fraction": args.train_parent_fraction, "seed": 42,
                "requested": [args.train_examples, args.validation_examples]}
    signature = digest(json.dumps(contract, sort_keys=True))
    contract_path = directory / "preparation-contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8"))["signature"] != signature:
        raise ValueError("Existing output uses a different preparation contract; choose a fresh output")
    write_json(contract_path, {"signature": signature, **contract})
    tasks = []
    for split in ("validation", "train"):
        count = token_manifest["splits"][split]["sentences"]
        for start in range(0, count, 250000):
            tasks.append((split, start, min(count, start + 250000),
                          1.0 if split == "validation" else args.train_parent_fraction, str(parts), signature))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(validation_hashes, cases, parents)) as pool:
        pending = {pool.submit(shard, task): task for task in tasks}
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            write_json(directory / "progress.json", {"status": "annotating", "completed_shards": len(results),
                       "total_shards": len(tasks), "accepted_pool": sum(r["counts"].get("accepted_pool", 0) for r in results),
                       "elapsed_seconds": time.monotonic() - started, "updated_at": datetime.now(timezone.utc).isoformat()})
            print(json.dumps({"done": len(results), "split": result["split"], "start": result["start"],
                              "counts": result["counts"], "seconds": round(result["seconds"], 1)}), flush=True)
    stats = {}
    for split, requested in (("validation", args.validation_examples), ("train", args.train_examples)):
        print(f"Deduplicating and packing {split}", flush=True)
        stats[split] = merge(directory, results, split, requested)
    # Source identities must still match after processing.
    if {s: file_sha(CORPUS / f"{s}.txt") for s in actual} != actual or dictionary_contract() != dictionaries:
        raise ValueError("Inputs changed during preparation")
    rejected = {s: dict(sum((Counter(r["counts"]) for r in results if r["split"] == s), Counter())) for s in stats}
    output_sha = {p.name: file_sha(p) for p in sorted(directory.iterdir())
                  if p.is_file() and (p.name.endswith(".bin") or p.name.endswith(".records.jsonl") or p.name.endswith(".review-sample.json"))}
    write_json(directory / "manifest.json", {"format": FORMAT, "status": "complete", "signature": signature,
               "contract": contract, "splits": stats, "filter_counts": rejected, "output_sha256": output_sha,
               "reading_dropout": {"fraction": .1, "selection": "train parent SHA256 last8hex modulo10 ==0", "raw_reading_retained": True},
               "sequence": "BOS CTX context READ reading OUT surface EOS; fields separately encoded; max129",
               "labels": "already shifted once; OUT and EOS only; PAD=-100", "test_or_blind_read": False,
               "isolation": "all original validation sentence hashes excluded from train; document IDs unavailable; no claim of document or semantic disjointness",
               "quality": "dual dictionary agreement; machine checks only; not human gold; native review pending",
               "elapsed_seconds": time.monotonic() - started})
    write_json(directory / "progress.json", {"status": "complete", "splits": stats, "updated_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "complete", "splits": stats, "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
