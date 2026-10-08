"""Freeze old V2 + existing eight-shard V3 pools for random-init training."""
import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import ahocorasick
import numpy as np
from vimeml.training.data import file_sha, write_json
from vimeml.training.data_mixture import JoinedPrefixWindows
from vimeml.training.data_v2 import PrefixCropWindowDataset
from vimeml.tokenizer.encode_parallel import output_lock

OLD = "artifacts/token-data/corpus-v2-16k"
NEW = "artifacts/token-data/corpus-v3-half/mixed"
OLD_INDEX = "artifacts/training-data/corpus-v2-c128"
NEW_INDEX = "artifacts/token-data/corpus-v3-half/indexes/mixed"


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def identity(text):
    # Sequence identity for content deduplication, never a full-file checksum.
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def protected_texts():
    for split in ("validation", "test"):
        with (ROOT / f"outputs/corpus-fast-v1/{split}.txt").open(encoding="utf-8") as stream:
            yield from stream
        with (ROOT / NEW / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)["text"]
    # Protect full source messages, not post-hoc answer aliases or model errors.
    benchmark = ROOT / "artifacts/benchmarks/ime-standard-ja-v1-ai-expert-r1"
    for split in ("development", "blind"):
        for item in json.loads((benchmark / split / "evaluation_items.json").read_text(encoding="utf-8")):
            text = item["provenance"]["original_message"]
            if len(canonical(text)) >= 24:
                yield text


def training_masks(output, old_manifest, new_manifest):
    path = output / "exclusions.json"
    if path.exists():
        old = np.load(output / "old.keep.npy", allow_pickle=False)
        new = np.load(output / "new.keep.npy", allow_pickle=False)
        report = json.loads(path.read_text(encoding="utf-8"))
        if old.shape != (old_manifest["splits"]["train"]["sentences"],) or new.shape != (new_manifest["splits"]["train"]["sentences"],):
            raise ValueError("Cached exclusion mask shape differs")
        return old, new, report
    exact, targets, anchors = set(), [], {}
    for raw in protected_texts():
        text = canonical(raw)
        key = identity(text)
        if not text or key in exact:
            continue
        exact.add(key)
        if len(text) >= 32:
            index = len(targets)
            targets.append(text)
            for anchor in {text[:24], text[-24:]}:
                anchors.setdefault(anchor, []).append(index)
    automaton = ahocorasick.Automaton()
    for anchor, indices in anchors.items():
        automaton.add_word(anchor, indices)
    automaton.make_automaton()
    del anchors

    def held_out(text):
        if identity(text) in exact:
            return True
        checked = set()
        for _, indices in automaton.iter(text):
            for index in indices:
                if index not in checked:
                    checked.add(index)
                    target = targets[index]
                    if target in text or (len(text) >= 48 and text in target):
                        return True
        return False

    old_keep = np.ones(old_manifest["splits"]["train"]["sentences"], dtype=np.bool_)
    new_keep = np.ones(new_manifest["splits"]["train"]["sentences"], dtype=np.bool_)
    noise_dir = ROOT / "artifacts/training-data/corpus-v3-noise-filter"
    noise = json.loads((noise_dir / "manifest.json").read_text(encoding="utf-8"))
    if noise["token_manifest_sha256"] != file_sha(ROOT / NEW / "manifest.json"):
        raise ValueError("Frozen catalogue filter belongs to different data")
    excluded = np.load(noise_dir / noise["file"], allow_pickle=False)
    if excluded.shape != new_keep.shape:
        raise ValueError("Catalogue filter shape differs")
    new_keep[excluded] = False
    counts = Counter(new_catalogue=int(excluded.sum()))
    fingerprints = {}
    print("Filtering existing V3 texts against both frozen held-out corpora and benchmark messages", flush=True)
    with (ROOT / NEW / "train.jsonl").open(encoding="utf-8") as stream:
        count = 0
        for index, line in enumerate(stream):
            row = json.loads(line)
            if new_keep[index]:
                text = canonical(row["text"])
                if held_out(text):
                    new_keep[index] = False
                    counts["new_held_out"] += 1
                elif row["source"].startswith("jpnmix:"):
                    fingerprints.setdefault(identity(text), []).append(index)
            count += 1
            if count % 250000 == 0:
                print(f"V3 rows {count:,}", flush=True)
    if count != len(new_keep):
        raise ValueError("V3 text/token row count differs")
    print("Scanning old V2 training text once for held-out overlaps and cross-pool exact duplicate blocks", flush=True)
    with (ROOT / "outputs/corpus-fast-v1/train.txt").open(encoding="utf-8") as stream:
        count = 0
        for index, line in enumerate(stream):
            text = canonical(line)
            if held_out(text):
                old_keep[index] = False
                counts["old_held_out"] += 1
            else:
                for duplicate in fingerprints.pop(identity(text), []):
                    new_keep[duplicate] = False
                    counts["new_exact_duplicate_of_old"] += 1
            count += 1
            if count % 2000000 == 0:
                print(f"Old rows {count:,}", flush=True)
                write_json(output / "progress.json", {"status": "overlap_scan", "old_rows": count})
    if count != len(old_keep):
        raise ValueError("Old text/token row count differs")
    np.save(output / "old.keep.npy", old_keep, allow_pickle=False)
    np.save(output / "new.keep.npy", new_keep, allow_pickle=False)
    report = {"status": "complete", "excluded": dict(counts),
              "old_kept_sequences": int(old_keep.sum()), "new_kept_sequences": int(new_keep.sum()),
              "protected_exact_texts": len(exact), "protected_long_texts": len(targets),
              "policy": "NFKC whitespace canonical exact held-out text; >=32-char held-out substring anchor scan; >=48-char reverse containment at anchors; catalogue mask reused; cross-pool exact web blocks removed from new pool, old retained; chat frequency preserved.",
              "limits": "Not a full document/near-duplicate audit. Different paragraph/sentence boundaries, internal unmatched anchors and paraphrases can escape. Existing source split assignments unchanged; no held-out data enters train mappings.",
              "full_file_sha256": False}
    write_json(path, report)
    return old_keep, new_keep, report


def window_sentences(dataset, ids):
    position = np.searchsorted(dataset.end, ids, side="right")
    previous = np.zeros(len(ids), dtype=np.int64)
    present = position > 0
    previous[present] = dataset.end[position[present]-1] - dataset.sentences[position[present]-1] - 1
    sentences = ids - previous
    candidates = np.flatnonzero(position < len(dataset.end))
    long = candidates[ids[candidates] >= dataset.first[position[candidates]]]
    sentences[long] = dataset.sentences[position[long]]
    return sentences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/training-data/corpus-v3-b-all8-chat05-c128")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--chat-fraction", type=float, default=.05)
    args = parser.parse_args()
    if args.epochs < 1 or not 0 < args.chat_fraction < .5:
        parser.error("Positive epochs and chat fraction in (0,.5) required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((ROOT / folder / "manifest.json").read_text(encoding="utf-8")) for folder in (OLD, NEW)]
    tokenizer_hash = manifests[1]["tokenizer_model_sha256"]
    old_hash = next(v for k, v in manifests[0]["input_sha256"].items() if k.replace("\\", "/").endswith("/tokenizer.model"))
    if old_hash != tokenizer_hash or any(m["vocab_size"] != 16384 or m["special_ids"] != manifests[0]["special_ids"] for m in manifests):
        raise ValueError("Frozen tokenizer identities differ")
    joined = []
    for folder, index, mapping in [(OLD, OLD_INDEX, {0: 6, 1: 7}), (NEW, NEW_INDEX, {i: i for i in range(6)})]:
        joined.append({"token_dir": folder, "index_dir": index,
                       "token_manifest_sha256": file_sha(ROOT / folder / "manifest.json"),
                       "window_manifest_sha256": file_sha(ROOT / index / "manifest.json"), "source_map": mapping})
    inputs = [ROOT / f"outputs/corpus-fast-v1/{split}.txt" for split in ("train", "validation", "test")]
    inputs += [ROOT / NEW / f"{split}.jsonl" for split in ("train", "validation", "test")]
    inputs += [ROOT / "artifacts/benchmarks/ime-standard-ja-v1-ai-expert-r1" / split / "evaluation_items.json" for split in ("development", "blind")]
    inputs += [ROOT / "artifacts/training-data/corpus-v3-noise-filter/manifest.json"]
    request = {"token_manifest_sha256": joined[1]["token_manifest_sha256"],
               "window_manifest_sha256": joined[1]["window_manifest_sha256"], "index_dir": NEW_INDEX,
               "joined_stores": joined, "primary_store": 1, "epochs": args.epochs,
               "chat_fraction": args.chat_fraction, "seed": 42, "crop_probability": .30, "crop_minimum": 8,
               "input_metadata": {p.relative_to(ROOT).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns] for p in inputs}}
    # JSON normalizes integer source-map keys; keep cache identity portable.
    request = json.loads(json.dumps(request))
    with output_lock(output):
        request_path = output / "request.json"
        if request_path.exists() and json.loads(request_path.read_text(encoding="utf-8")) != request:
            raise ValueError("Changed inputs/settings; choose a new output")
        write_json(request_path, request)
        if (output / "manifest.json").exists():
            cached = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            if cached["request"] != request or any((output / r["file"]).stat().st_size != r["bytes"] for r in cached["epochs"]):
                raise ValueError("Incomplete or changed frozen mappings")
            print("Completed mixture reused", flush=True)
            return
        masks = training_masks(output, *manifests)[:2]
        bases = [PrefixCropWindowDataset(ROOT / r["token_dir"], ROOT / r["index_dir"], seed=42, probability=.30) for r in joined]
        combined = JoinedPrefixWindows(bases, 1, [{int(k): v for k,v in r["source_map"].items()} for r in joined])
        web_parts, chat_parts, source_windows = [], [], Counter()
        try:
            for number, (base, mask, manifest) in enumerate(zip(bases, masks, manifests)):
                source_ids = np.memmap(base.token_dir / "train.sources.bin", dtype="u1", mode="r")
                inverse = {v:k for k,v in manifest["source_ids"].items()}
                for start in range(0, len(base), 262144):
                    local = np.arange(start, min(start+262144, len(base)), dtype=np.int64)
                    sentences = window_sentences(base, local)
                    use = mask[sentences]
                    sources = source_ids[sentences]
                    is_chat = np.isin(sources, [4,5]) if number == 1 else np.zeros(len(local), dtype=np.bool_)
                    web_parts.append(local[use & ~is_chat] + combined.starts[number])
                    chat_parts.append(local[use & is_chat] + combined.starts[number])
                    for source, name in inverse.items():
                        source_windows[("old:" if number == 0 else "")+name] += int((use & (sources == source)).sum())
                source_ids._mmap.close()
            web, chat = np.concatenate(web_parts), np.concatenate(chat_parts)
            del web_parts, chat_parts, masks
            if not len(web) or not len(chat) or len(combined) >= 2**32:
                raise ValueError("Unusable mixture pools")
            raw_web_tokens = sum(int(combined.uncropped_window_lengths(web[start:start+262144]).sum())
                                 for start in range(0, len(web), 262144))
            records = []
            for epoch in range(args.epochs):
                combined.set_epoch(epoch)
                web_tokens = 0
                for start in range(0, len(web), 262144):
                    web_tokens += int(combined.window_lengths(web[start:start+262144]).sum())
                chat_lengths = combined.window_lengths(chat)
                target = round(web_tokens * args.chat_fraction / (1-args.chat_fraction))
                rng = np.random.default_rng(42+epoch)
                selected, used = [], 0
                while used < target:
                    order = rng.permutation(len(chat))
                    cumulative = np.cumsum(chat_lengths[order], dtype=np.int64)
                    take = int(np.searchsorted(cumulative, target-used, side="right"))
                    if take:
                        selected.append(chat[order[:take]])
                        used += int(cumulative[take-1])
                    if take < len(order):
                        break
                if not selected:
                    raise ValueError("No chat quota windows")
                mapping = np.concatenate([web, *selected]).astype("<u4")
                rng.shuffle(mapping)
                temporary = output / f"epoch-{epoch}.tmp.npy"
                np.save(temporary, mapping, allow_pickle=False)
                path = output / f"epoch-{epoch}.npy"
                temporary.replace(path)
                record = {"epoch": epoch, "file": path.name, "bytes": path.stat().st_size,
                          "windows": len(mapping), "web_windows": len(web), "chat_windows": sum(len(x) for x in selected),
                          "web_tokens": web_tokens, "chat_tokens": used, "effective_tokens": web_tokens+used,
                          "uncropped_tokens": raw_web_tokens + sum(int(combined.uncropped_window_lengths(part).sum()) for part in selected),
                          "chat_fraction": used/(web_tokens+used), "chat_independent_tokens": int(chat_lengths.sum()),
                          "chat_effective_passes": used/int(chat_lengths.sum())}
                records.append(record)
                print(json.dumps(record), flush=True)
                write_json(output / "progress.json", {"status": "mapping", "completed_epochs": epoch+1})
            write_json(output / "manifest.json", {"status": "complete", "format": "vimeml_token_mixture_v1",
                "request": request, "epochs": records, "source_windows": dict(source_windows),
                "source_ids": {**manifests[1]["source_ids"], "old:fineweb": 6, "old:tatoeba": 7},
                "policy": "Joined frozen V2 and current eight-shard V3 selected pools; retained non-chat windows once/epoch; chat shuffled repeated cycles to cropped-token 5%; source-local deterministic crop; original held-out assignments unchanged; no token copies or new tokenizer.",
                "verification": "Small manifest identity and input metadata; mask/mapping shapes and sizes; one targeted text-overlap pass; no full-file SHA256"})
            write_json(output / "progress.json", {"status": "complete"})
        finally:
            combined.close()


if __name__ == "__main__":
    main()
