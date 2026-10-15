"""Full storage/span/split/codec audit, plus a stratified machine-review packet."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import pickle
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from vimeml.posttraining.data import ConversionCodec, ConversionDataset, shifted
from vimeml.posttraining.readings import DualReading, HIRAGANA
from vimeml.training.data import file_sha, write_json


def quarantine_known_ambiguity(directory):
    """Never repair an inferred reading. Remove known shared dictionary errors.

    Standalone 家 tagged ヤ/ケ can be a family suffix, but also mislabels the
    ordinary house noun in unpunctuated web sentences. Quarantine both rather
    than guessing from a pseudo-label. Compounds such as 家族 and readings カ
    are unaffected. No benchmark observations determine this train-review rule.
    """
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("quality_refinement", {}).get("version") == 2:
        return
    if not (directory / "annotation-manifest.json").exists():
        write_json(directory / "annotation-manifest.json", manifest)
    prior_refinement = manifest.get("quality_refinement")
    previous_manifest_sha = file_sha(directory / "manifest.json")
    if prior_refinement:
        write_json(directory / "refinement-v1-manifest.json", manifest)
        for split in ("train", "validation"):
            previous = directory / f"{split}.quality-quarantine.jsonl"
            previous.rename(directory / f"{split}.quality-v1-quarantine.jsonl")
    validation_data = ConversionDataset(directory, "validation")
    validation_sequences = {hashlib.sha256(validation_data.ids[int(validation_data.offsets[i]):int(validation_data.offsets[i + 1])].tobytes()).digest()
                            for i in range(len(validation_data))}
    validation_data.close()
    with (directory / "validation.records.jsonl").open(encoding="utf-8") as records:
        validation_cases = {json.loads(raw)["case_sha256"] for raw in records}
    reader = DualReading()
    excluded = {}
    retained_kanji = {}
    for split in ("train", "validation"):
        bad = {}
        kanji = 0
        data = ConversionDataset(directory, split)
        with (ROOT / f"outputs/corpus-fast-v1/{split}.txt").open("rb") as original, \
             (directory / f"{split}.records.jsonl").open(encoding="utf-8") as records:
            for index, raw in enumerate(records):
                ex = json.loads(raw)
                if split == "train":
                    sequence = data.ids[int(data.offsets[index]):int(data.offsets[index + 1])].tobytes()
                    if ex["case_sha256"] in validation_cases or hashlib.sha256(sequence).digest() in validation_sequences:
                        bad[index] = "exact_conversion_validation_case"
                        continue
                ambiguous = False
                if "家" in ex["output"]:
                    original.seek(ex["byte_offset"])
                    text = original.readline().decode("utf-8").rstrip("\r\n")
                    cursor = 0
                    for word in reader.mecab(text):
                        start = cursor + len(word.white_space)
                        end = start + len(word.surface)
                        cursor = end
                        if ex["span"][0] <= start and end <= ex["span"][1] and word.surface == "家" and word.feature.kana in ("ヤ", "ケ"):
                            ambiguous = True
                            break
                if ambiguous:
                    bad[index] = "standalone_house_family_suffix_ambiguity"
                else:
                    kanji += ex["has_kanji"]
        excluded[split], retained_kanji[split] = bad, kanji
        data.close()
    staging = directory / "quality-staging"
    staging.mkdir(exist_ok=True)
    new_stats = {}
    removed_counts = {}
    for split in ("train", "validation"):
        data = ConversionDataset(directory, split)
        stats, hist = Counter(), defaultdict(Counter)
        kana_limit = retained_kanji[split] // 4
        with (directory / f"{split}.records.jsonl").open(encoding="utf-8") as source, \
             (staging / f"{split}.records.jsonl").open("w", encoding="utf-8", newline="\n") as records, \
             (staging / f"{split}.tokens.bin").open("wb") as tokens, \
             (staging / f"{split}.offsets.bin").open("wb") as offsets, \
             (staging / f"{split}.out-start.bin").open("wb") as starts, \
             (staging / f"{split}.reading-dropout.bin").open("wb") as dropout, \
             (directory / f"{split}.quality-quarantine.jsonl").open("w", encoding="utf-8", newline="\n") as rejected:
            offsets.write(struct.pack("<Q", 0))
            for index, raw in enumerate(source):
                ex = json.loads(raw)
                reason = excluded[split].get(index)
                if reason is None and not ex["has_kanji"] and stats["kana_only"] >= kana_limit:
                    reason = "retain_kana_only_cap_after_quarantine"
                if reason:
                    rejected.write(json.dumps({"reason": reason, **ex}, ensure_ascii=False) + "\n")
                    stats["excluded_" + reason] += 1
                    continue
                ids = data.ids[int(data.offsets[index]):int(data.offsets[index + 1])].tolist()
                records.write(raw)
                tokens.write(struct.pack(f"<{len(ids)}H", *ids))
                starts.write(bytes([ex["first_output"]]))
                dropout.write(bytes([ex["reading_dropout_view"]]))
                stats["examples"] += 1
                stats["stored_tokens"] += len(ids)
                stats["supervised_tokens_full_reading"] += len(ids) - ex["first_output"]
                stats["context_cropped"] += bool(ex["context_crop_characters"])
                stats["kana_only"] += not ex["has_kanji"]
                stats["reading_dropout_view"] += ex["reading_dropout_view"]
                offsets.write(struct.pack("<Q", stats["stored_tokens"]))
                for name, value in (("reading_characters", len(ex["reading"])), ("output_characters", len(ex["output"])),
                                    ("context_characters", len(ex["context"])), ("source_id", ex["source_id"]),
                                    ("serialized_tokens", len(ids))):
                    hist[name][value] += 1
        data.close()
        new_stats[split] = {**manifest["splits"][split], **dict(stats), "histograms": dict(hist)}
        for key in list(new_stats[split]):
            if key.startswith("excluded_"):
                new_stats[split][key] = manifest["splits"][split].get(key, 0) + stats.get(key, 0)
        removed_counts[split] = {k: v for k, v in stats.items() if k.startswith("excluded_")}
    write_json(directory / "manifest.json", {**manifest, "status": "refining"})
    for path in staging.iterdir():
        path.replace(directory / path.name)
    manifest["splits"] = new_stats
    if prior_refinement:
        manifest.setdefault("quality_refinement_history", []).append(prior_refinement)
    manifest["quality_refinement"] = {"version": 2, "rule": "exclude exact frozen conversion-validation field tuples or serialized sequences; quarantine standalone 家 with dictionary reading ヤ/ケ; kana-only <=20%; no reading repair or replacement samples",
        "origin": "machine review of original train row14, parent000006e2bfe7819a400b082f1815355e14e445d5d7d8cec234ee855a4b3831b9",
        "counts": removed_counts, "annotation_manifest_sha256": file_sha(directory / "annotation-manifest.json"),
        "previous_manifest_sha256": previous_manifest_sha,
        "code_sha256": file_sha(Path(__file__))}
    manifest["annotation_signature"] = manifest["signature"]
    manifest["signature"] = hashlib.sha256(json.dumps({"annotation": manifest["annotation_signature"],
        "refinement": manifest["quality_refinement"]}, sort_keys=True).encode("utf-8")).hexdigest()
    # The original uniform review packet can contain a quarantined example; retain
    # it as annotation history, and use the new stratified packet for final review.
    for split in ("train", "validation"):
        old_review = directory / f"{split}.review-sample.json"
        if old_review.exists():
            old_review.rename(directory / f"{split}.annotation-review-sample.json")
    manifest["output_sha256"] = {p.name: file_sha(p) for p in sorted(directory.iterdir())
        if p.is_file() and (p.name.endswith(".bin") or p.name.endswith(".records.jsonl") or ".quality" in p.name and p.name.endswith("quarantine.jsonl"))}
    write_json(directory / "manifest.json", manifest)
    print(json.dumps({"quality_refinement": removed_counts, "retained": {s: x["examples"] for s, x in new_stats.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts/posttrain-data/v4-c1")
    args = parser.parse_args()
    directory = args.directory
    quarantine_known_ambiguity(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    began = time.monotonic()
    import unidic_lite
    import sudachipy
    dictionary_resources = list(Path(unidic_lite.DICDIR).iterdir())
    dictionary_resources += list((Path(sudachipy.__file__).parent / "resources").iterdir())
    resource_sha = {str(p): file_sha(p) for p in sorted(dictionary_resources)
                    if p.is_file() and p.suffix != ".pdf"}
    for name, expected in manifest["output_sha256"].items():
        if file_sha(directory / name) != expected:
            raise ValueError(f"Damaged conversion file: {name}")
    validation_parents = {hashlib.sha256(x.rstrip(b"\r\n")).hexdigest()
                          for x in (ROOT / "outputs/corpus-fast-v1/validation.txt").open("rb")}
    codec = ConversionCodec(ROOT / "artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model")
    validation_data = ConversionDataset(directory, "validation")
    validation_sequences = {hashlib.sha256(validation_data.ids[int(validation_data.offsets[i]):int(validation_data.offsets[i + 1])].tobytes()).digest()
                            for i in range(len(validation_data))}
    validation_data.close()
    with (directory / "validation.records.jsonl").open(encoding="utf-8") as records:
        validation_cases = {json.loads(raw)["case_sha256"] for raw in records}
    audit, review_buckets, seen = {}, defaultdict(list), Counter()
    for split in ("train", "validation"):
        data = ConversionDataset(directory, split)
        reopened = pickle.loads(pickle.dumps(data))
        assert reopened[0] == data[0]
        reopened.close()
        sources = np.memmap(ROOT / f"artifacts/token-data/corpus-v4-8k/{split}.sources.bin", mode="r", dtype="u1")
        rows = np.memmap(ROOT / f"artifacts/token-data/corpus-v4-8k/{split}.rows.bin", mode="r", dtype="<u8")
        counts = Counter()
        rng = random.Random(42)
        with (ROOT / f"outputs/corpus-fast-v1/{split}.txt").open("rb") as original, \
             (directory / f"{split}.records.jsonl").open(encoding="utf-8") as records:
            for index, raw in enumerate(records):
                ex = json.loads(raw)
                original.seek(ex["byte_offset"])
                text = original.readline().decode("utf-8").rstrip("\r\n")
                parent = hashlib.sha256(text.encode("utf-8")).hexdigest()
                start, end = ex["span"]
                assert parent == ex["parent_sha256"]
                assert text[:start] == ex["context"] and text[start:end] == ex["output"]
                assert int(rows[ex["row"]]) == ex["byte_offset"] and int(sources[ex["row"]]) == ex["source_id"]
                assert split != "train" or parent not in validation_parents
                assert ex["mecab_reading"] == ex["sudachi_reading"] == ex["reading"]
                assert HIRAGANA.fullmatch(ex["reading"])
                encoded, first, removed = codec.encode(ex["context"], ex["reading"], ex["output"])
                stored = data.ids[int(data.offsets[index]):int(data.offsets[index + 1])].tolist()
                assert stored == encoded and int(data.first[index]) == first
                if split == "train":
                    assert ex["case_sha256"] not in validation_cases
                    assert hashlib.sha256(struct.pack(f"<{len(stored)}H", *stored)).digest() not in validation_sequences
                assert removed == ex["context_crop_characters"]
                view = data[index]
                expected_ids, expected_first = encoded, first
                if ex["reading_dropout_view"]:
                    left, right = encoded.index(5) + 1, first - 1
                    expected_ids = encoded[:left] + encoded[right:]
                    expected_first = first - right + left
                assert view == shifted(expected_ids, expected_first)
                assert view["labels"][expected_first - 1:] == codec.sp.encode(ex["output"]) + [3]
                assert len(stored) <= 129 and stored.count(4) == stored.count(5) == stored.count(6) == 1
                counts["examples"] += 1
                counts["supervised_tokens"] += sum(x != -100 for x in view["labels"])
                counts["view_input_tokens"] += len(view["input_ids"])
                counts["reading_dropout"] += bool(ex["reading_dropout_view"])
                counts["no_left_context"] += not ex["context"]
                if split == "train":
                    key = tuple(ex["stratum"])
                    seen[key] += 1
                    bucket = review_buckets[key]
                    if len(bucket) < 10:
                        bucket.append(ex)
                    else:
                        k = rng.randrange(seen[key])
                        if k < 10:
                            bucket[k] = ex
                if (index + 1) % 250000 == 0:
                    print(f"{split}: audited {index + 1}", flush=True)
        assert counts["examples"] == len(data)
        assert counts["supervised_tokens"] == manifest["splits"][split]["supervised_tokens_full_reading"]
        audit[split] = dict(counts)
        data.close()
    # Round-robin occupied strata, followed by independent fresh annotation for review packet.
    packet = []
    for j in range(10):
        for key in sorted(review_buckets):
            if j < len(review_buckets[key]) and len(packet) < 200:
                packet.append(review_buckets[key][j])
    analyzer = DualReading()
    active_mecab_dictionaries = analyzer.mecab.dictionary_info
    frozen_binaries = manifest["contract"]["dictionary"]["binaries"]
    for info in active_mecab_dictionaries:
        if file_sha(info["filename"]) != frozen_binaries.get(info["filename"]):
            raise ValueError("Active MeCab dictionary differs from the frozen identity")
    with (ROOT / "outputs/corpus-fast-v1/train.txt").open("rb") as original:
        for ex in packet:
            original.seek(ex["byte_offset"])
            text = original.readline().decode("utf-8").rstrip("\r\n")
            regenerated, reason = analyzer.example(text, int(ex["parent_sha256"][16:32], 16))
            assert reason is None
            assert all(regenerated[k] == ex[k] for k in ("reading", "output", "span", "context"))
            ex["full_original_sentence"] = text
            ex["reviewer"] = "automated_reannotation; linguistic human review pending"
    write_json(directory / "stratified-review-200.json", packet)
    write_json(directory / "acceptance.json", {"status": "pass", "manifest_sha256": file_sha(directory / "manifest.json"),
               "checker_sha256": file_sha(Path(__file__)), "dictionary_supporting_resources_sha256": resource_sha,
               "reader_code_sha256": file_sha(ROOT / "src/vimeml/posttraining/data.py"),
               "active_mecab_dictionaries": active_mecab_dictionaries,
               "splits": audit, "review_examples": len(packet), "review_strata": len(review_buckets),
               "checks": ["all output file SHA256", "every original TXT row/source/span/hash", "train vs all original validation sentence hashes",
                          "no exact conversion case or serialized sequence overlap with frozen conversion validation",
                          "every structured sequence reencoded", "every shifted loss mask and dropout view", "200 stratified fresh double annotations"],
               "linguistic_review": "machine annotation only; agreement does not establish gold correctness",
               "test_or_blind_read": False, "elapsed_seconds": time.monotonic() - began})
    print(json.dumps({"status": "pass", "splits": audit, "seconds": time.monotonic() - began}), flush=True)


if __name__ == "__main__":
    main()
