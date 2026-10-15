"""Subset-only lexical ambiguity quarantine after a complete C1 data audit.

Retained sequences/masks are copied unchanged from a fully checked parent.
This inherits its span/codec/isolation checks without repeating all annotations.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import fugashi
from vimeml.posttraining.data import ConversionDataset
from vimeml.posttraining.readings import KANJI
from vimeml.training.data import file_sha, write_json


def main():
    directory = ROOT / "artifacts/posttrain-data/v4-c1"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("lexical_ambiguity_refinement"):
        print("Lexical refinement already complete; preserving frozen data", flush=True)
        return
    acceptance = json.loads((directory / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["status"] == "pass" and acceptance["manifest_sha256"] == file_sha(directory / "manifest.json")
    assert acceptance["reader_code_sha256"] == file_sha(ROOT / "src/vimeml/posttraining/data.py")
    for name, expected in manifest["output_sha256"].items():
        assert file_sha(directory / name) == expected, name
    for name in ("manifest.json", "acceptance.json", "stratified-review-200.json"):
        shutil.copyfile(directory / name, directory / ("pre-lexical-" + name))
    began = time.monotonic()
    tagger = fugashi.Tagger()
    staging = directory / "lexical-staging"
    staging.mkdir(exist_ok=True)
    stats_by_split, hist_by_split = {}, {}
    buckets, seen, rng = defaultdict(list), Counter(), random.Random(42)
    for split in ("train", "validation"):
        data = ConversionDataset(directory, split)
        stats, hist = Counter(), defaultdict(Counter)
        with (directory / f"{split}.records.jsonl").open(encoding="utf-8") as source, \
             (ROOT / f"outputs/corpus-fast-v1/{split}.txt").open("rb") as original, \
             (staging / f"{split}.records.jsonl").open("w", encoding="utf-8", newline="\n") as records, \
             (staging / f"{split}.tokens.bin").open("wb") as tokens, \
             (staging / f"{split}.offsets.bin").open("wb") as offsets, \
             (staging / f"{split}.out-start.bin").open("wb") as starts, \
             (staging / f"{split}.reading-dropout.bin").open("wb") as dropout, \
             (directory / f"{split}.lexical-quarantine.jsonl").open("w", encoding="utf-8", newline="\n") as rejected:
            offsets.write(struct.pack("<Q", 0))
            for index, raw in enumerate(source):
                ex = json.loads(raw)
                issue = None
                if ex["has_kanji"]:
                    original.seek(ex["byte_offset"])
                    text = original.readline().decode("utf-8").rstrip("\r\n")
                    cursor = 0
                    for word in tagger(text):
                        start = cursor + len(word.white_space)
                        end = start + len(word.surface)
                        cursor = end
                        assert text[start:end] == word.surface
                        if (ex["span"][0] <= start and end <= ex["span"][1]
                                and len(word.surface) == 1 and KANJI.fullmatch(word.surface)
                                and word.feature.pos1 == "名詞"
                                and ((start > 0 and KANJI.fullmatch(text[start - 1]))
                                     or (end < len(text) and KANJI.fullmatch(text[end])))):
                            issue = {"reason": "single_kanji_noun_adjacent_to_kanji", "surface": word.surface,
                                     "raw_kana": word.feature.kana, "span": [start, end]}
                            break
                if issue:
                    rejected.write(json.dumps({"issue": issue, **ex}, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stats["excluded_single_kanji_compound_boundary"] += 1
                    continue
                ids = data.ids[int(data.offsets[index]):int(data.offsets[index + 1])].tobytes()
                first = int(data.first[index])
                flag = int(data.dropout[index])
                assert first == ex["first_output"] and bool(flag) == ex["reading_dropout_view"]
                # No reencoding, label edits, new samples, or reordering.
                records.write(raw)
                tokens.write(ids)
                starts.write(bytes([first]))
                dropout.write(bytes([flag]))
                length = len(ids) // 2
                stats["examples"] += 1
                stats["stored_tokens"] += length
                stats["supervised_tokens_full_reading"] += length - first
                stats["context_cropped"] += bool(ex["context_crop_characters"])
                stats["kana_only"] += not ex["has_kanji"]
                stats["reading_dropout_view"] += flag
                offsets.write(struct.pack("<Q", stats["stored_tokens"]))
                for name, value in (("reading_characters", len(ex["reading"])), ("output_characters", len(ex["output"])),
                                    ("context_characters", len(ex["context"])), ("source_id", ex["source_id"]),
                                    ("serialized_tokens", length)):
                    hist[name][value] += 1
                # Retain at most20% kana by selecting a prefix of previously
                # hash-selected kana cases after counting retained kanji below.
                if split == "train":
                    key = tuple(ex["stratum"])
                    seen[key] += 1
                    bucket = buckets[key]
                    if len(bucket) < 10:
                        bucket.append(ex)
                    else:
                        k = rng.randrange(seen[key])
                        if k < 10:
                            bucket[k] = ex
                if (index + 1) % 500000 == 0:
                    print(f"{split}: examined {index + 1}; retained {stats['examples']}", flush=True)
        data.close()
        stats_by_split[split], hist_by_split[split] = dict(stats), dict(hist)
    # A second cheap copy pass only trims excess kana; all retained encoded
    # bytes and mask flags remain unchanged from the accepted parent.
    for split, stats in stats_by_split.items():
        limit = (stats["examples"] - stats["kana_only"]) // 4
        if stats["kana_only"] <= limit:
            continue
        files = {}
        for suffix in ("tokens.bin", "offsets.bin", "out-start.bin", "reading-dropout.bin", "records.jsonl"):
            path = staging / f"{split}.{suffix}"
            renamed = path.with_name("all-" + path.name)
            path.rename(renamed)
            files[suffix] = renamed
        import numpy as np
        old_ids = np.memmap(files["tokens.bin"], dtype="<u2", mode="r")
        old_offsets = np.memmap(files["offsets.bin"], dtype="<u8", mode="r")
        old_first = np.memmap(files["out-start.bin"], dtype="u1", mode="r")
        old_drop = np.memmap(files["reading-dropout.bin"], dtype="u1", mode="r")
        extra = stats["kana_only"] - limit
        # Drop the latest hash-order kana records, preserving the first `limit`.
        kana, count, offset = 0, 0, 0
        with files["records.jsonl"].open(encoding="utf-8") as source, \
             (staging / f"{split}.records.jsonl").open("w", encoding="utf-8", newline="\n") as records, \
             (staging / f"{split}.tokens.bin").open("wb") as tokens, \
             (staging / f"{split}.offsets.bin").open("wb") as offsets, \
             (staging / f"{split}.out-start.bin").open("wb") as starts, \
             (staging / f"{split}.reading-dropout.bin").open("wb") as dropout:
            offsets.write(struct.pack("<Q", 0))
            for i, raw in enumerate(source):
                ex = json.loads(raw)
                if not ex["has_kanji"]:
                    kana += 1
                    if kana > limit:
                        length = int(old_offsets[i + 1] - old_offsets[i])
                        stats["examples"] -= 1
                        stats["stored_tokens"] -= length
                        stats["supervised_tokens_full_reading"] -= length - int(old_first[i])
                        stats["context_cropped"] -= bool(ex["context_crop_characters"])
                        stats["kana_only"] -= 1
                        stats["reading_dropout_view"] -= int(old_drop[i])
                        for name, value in (("reading_characters", len(ex["reading"])), ("output_characters", len(ex["output"])),
                                            ("context_characters", len(ex["context"])), ("source_id", ex["source_id"]), ("serialized_tokens", length)):
                            hist_by_split[split][name][value] -= 1
                        continue
                encoded = old_ids[int(old_offsets[i]):int(old_offsets[i + 1])].tobytes()
                tokens.write(encoded)
                offset += len(encoded) // 2
                offsets.write(struct.pack("<Q", offset))
                starts.write(bytes([int(old_first[i])]))
                dropout.write(bytes([int(old_drop[i])]))
                records.write(raw)
                count += 1
        assert count == stats["examples"] and offset == stats["stored_tokens"]
        stats["excluded_kana_cap_after_lexical_filter"] = extra
        for value in (old_ids, old_offsets, old_first, old_drop):
            value._mmap.close()
    # Only known final filenames are installed; intermediate copies remain audit history.
    write_json(directory / "manifest.json", {**manifest, "status": "refining"})
    for split in ("train", "validation"):
        for suffix in ("tokens.bin", "offsets.bin", "out-start.bin", "reading-dropout.bin", "records.jsonl"):
            path = staging / f"{split}.{suffix}"
            path.replace(directory / path.name)
        manifest["splits"][split].update(stats_by_split[split], histograms=hist_by_split[split])
    manifest["lexical_ambiguity_refinement"] = {"code_sha256": file_sha(Path(__file__)),
        "rule": "quarantine target morphemes that are single-kanji nouns immediately adjacent to another kanji in the original sentence; retain kana <=20%",
        "reason": "AI train-sample review: 岩石棒 shared split reading, 春 within 春也, garbled 土免状",
        "previous_manifest_sha256": acceptance["manifest_sha256"], "previous_acceptance_sha256": file_sha(directory / "pre-lexical-acceptance.json"),
        "scope": "conservative subset selection; no reading repairs, replacements or extra corpus"}
    manifest["previous_signature"] = manifest["signature"]
    manifest["signature"] = hashlib.sha256(json.dumps({"parent": manifest["previous_signature"],
        "lexical": manifest["lexical_ambiguity_refinement"]}, sort_keys=True).encode()).hexdigest()
    manifest["output_sha256"] = {p.name: file_sha(p) for p in sorted(directory.iterdir()) if p.is_file()
        and (p.name.endswith(".bin") or p.name.endswith(".records.jsonl") or p.name.endswith("quarantine.jsonl"))}
    manifest["status"] = "complete"
    write_json(directory / "manifest.json", manifest)
    # Verify final map boundaries and flags against the retained immutable bytes.
    for split in ("train", "validation"):
        data = ConversionDataset(directory, split)
        assert len(data) == stats_by_split[split]["examples"]
        assert stats_by_split[split]["kana_only"] / len(data) <= .2
        assert len(data[0]["labels"]) <= 128 and len(data[len(data) - 1]["labels"]) <= 128
        data.close()
    # Review packet is regenerated from final records, so it cannot contain a removed case.
    pool, bucket_seen = defaultdict(list), Counter()
    with (directory / "train.records.jsonl").open(encoding="utf-8") as records:
        for raw in records:
            ex = json.loads(raw)
            key = tuple(ex["stratum"])
            bucket_seen[key] += 1
            if len(pool[key]) < 10:
                pool[key].append(ex)
            else:
                k = rng.randrange(bucket_seen[key])
                if k < 10:
                    pool[key][k] = ex
    packet = [pool[key][j] for j in range(10) for key in sorted(pool) if j < len(pool[key])][:200]
    write_json(directory / "stratified-review-200.json", packet)
    write_json(directory / "acceptance.json", {"status": "pass", "manifest_sha256": file_sha(directory / "manifest.json"),
        "reader_code_sha256": acceptance["reader_code_sha256"], "checker_sha256": file_sha(Path(__file__)),
        "parent_acceptance_sha256": file_sha(directory / "pre-lexical-acceptance.json"),
        "checks": ["accepted parent manifest/reader/all data SHA256 match", "every retained sequence and mask flag copied unchanged",
                   "all selected records copied unchanged", "only row deletion; parent full span/codec/loss/split audit inherited",
                   "new final file SHA256, offsets, count and kana-cap checks"],
        "splits": stats_by_split, "review_examples": len(packet), "review_strata": len(pool),
        "linguistic_review": "AI triage and dual-dictionary silver labels, not native-human gold; remaining ambiguity is possible",
        "test_or_blind_read": False, "elapsed_seconds": time.monotonic() - began})
    print(json.dumps({"status": "pass", "splits": stats_by_split, "seconds": time.monotonic() - began}), flush=True)


if __name__ == "__main__":
    main()
