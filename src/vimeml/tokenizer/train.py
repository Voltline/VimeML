"""Train SentencePiece on train only, then measure local corpus tokenization."""

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import tomllib
from collections import Counter
from itertools import islice
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("train", "validation", "test")
SPECIAL_IDS = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
PROBES = [
    "今日は良い天気ですね。", "ひらがなとカタカナ、漢字。",
    "iPhone 16は128GBです。価格は12,800円。", "ＡＢＣ１２３とABC123、ｶﾀｶﾅ。",
    "𠮷野家でコーヒー☕を飲む🙂", " leading  spaces trailing ",
    "tab\there", "<pad> <unk> <s> </s>", "",
]


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sentences(path):
    with path.open(encoding="utf-8", newline=None) as stream:
        for line in stream:
            text = line.removesuffix("\n")
            if not text:
                raise ValueError(f"Corpus contains an empty line: {path}")
            yield text


def validate(settings):
    if settings["model_type"] not in {"unigram", "bpe"}:
        raise ValueError("model_type must be unigram or bpe.")
    for name in ("vocab_size", "num_threads", "max_sentence_length"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    coverage = settings["character_coverage"]
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0.98 <= coverage <= 1:
        raise ValueError("character_coverage must be in [0.98, 1].")
    seed = settings["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer.")
    if settings["byte_fallback"] is not True or settings["normalization_rule_name"] != "identity" or settings["remove_extra_whitespaces"] is not False:
        raise ValueError("This baseline requires byte fallback, identity normalization and preserved whitespace.")


def preflight(train_path, settings, expected):
    count, characters, longest = 0, 0, 0
    for text in sentences(train_path):
        count += 1
        characters += len(text)
        longest = max(longest, len(text.encode("utf-8")))
        if "▁" in text:
            raise ValueError("Train contains literal U+2581, reserved by SentencePiece; inspect before training.")
    if not count or count != expected["sentences"] or characters != expected["characters"]:
        raise ValueError("Train text does not match the corpus statistics.")
    if longest > settings["max_sentence_length"]:
        raise ValueError(f"Longest sentence is {longest} bytes; increase max_sentence_length to avoid silent exclusion.")
    return {"sentences": count, "characters": characters, "max_utf8_bytes": longest}


def percentile(histogram, fraction):
    target = math.ceil(sum(histogram.values()) * fraction)
    count = 0
    for length, frequency in sorted(histogram.items()):
        count += frequency
        if count >= target:
            return length
    return 0


def example(processor, text):
    ids = processor.encode(text, out_type=int)
    return {"text": text, "pieces": processor.encode(text, out_type=str), "ids": ids,
            "decoded": processor.decode(ids), "roundtrip_ok": processor.decode(ids) == text}


def measure(processor, path, split, byte_ids):
    lengths = Counter()
    characters = tokens = unknown = byte_tokens = mismatches = 0
    examples, errors = [], []
    rng = random.Random(42)
    count = 0
    stream = sentences(path)
    while batch := list(islice(stream, 256)):
        encoded = processor.encode(batch, out_type=int)
        decoded = processor.decode(encoded)
        for text, ids, restored in zip(batch, encoded, decoded, strict=True):
            count += 1
            characters += len(text)
            tokens += len(ids)
            lengths[len(ids)] += 1
            unknown += ids.count(processor.unk_id())
            byte_tokens += sum(token in byte_ids for token in ids)
            if restored != text:
                mismatches += 1
                if len(errors) < 10:
                    errors.append({"line": count, "text": text, "decoded": restored})
            # Only train/validation examples are presented for inspection.
            if split != "test":
                sample = {"line": count, "text": text}
                if len(examples) < 10:
                    examples.append(sample)
                else:
                    position = rng.randrange(count)
                    if position < 10:
                        examples[position] = sample
    stats = {
        "sentences": count, "characters": characters,
        "tokens_without_special_tokens": tokens,
        "tokens_with_bos_eos_per_sentence": tokens + 2 * count,
        "characters_per_token": characters / tokens if tokens else None,
        "mean_tokens_per_sentence": tokens / count if count else None,
        "max_tokens_per_sentence": max(lengths, default=0),
        "length_percentiles_without_special_tokens": {
            f"p{percent}": percentile(lengths, percent / 100) for percent in (50, 90, 95, 99)
        },
        "fits_context_with_bos_eos": {
            str(size): {"sentences": sum(n for length, n in lengths.items() if length + 2 <= size),
                        "fraction": sum(n for length, n in lengths.items() if length + 2 <= size) / count if count else None}
            for size in (64, 128)
        },
        "unknown_tokens": unknown, "byte_fallback_tokens": byte_tokens,
        "byte_fallback_token_fraction": byte_tokens / tokens if tokens else None,
        "roundtrip_mismatches": mismatches, "roundtrip_error_examples": errors,
    }
    return stats, [{"split": split, "line": item["line"], **example(processor, item["text"])} for item in examples]


def run(config_path, output_override=None, vocab_override=None):
    try:
        import sentencepiece as spm
    except ModuleNotFoundError:
        raise SystemExit("请先运行：uv pip install --python .venv/Scripts/python.exe -r requirements-tokenizer.txt") from None
    if spm.__version__ != "0.2.1":
        raise ValueError("Use sentencepiece==0.2.1 for this recorded baseline.")
    config_path = config_path.resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    settings = dict(config["training"])
    if vocab_override is not None:
        settings["vocab_size"] = vocab_override
    validate(settings)
    corpus = (ROOT / config["paths"]["corpus_dir"]).resolve()
    corpus_manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    if corpus_manifest.get("status") != "complete":
        raise ValueError("Corpus build must be complete.")
    corpus_stats = json.loads((corpus / "stats.json").read_text(encoding="utf-8"))
    input_paths = [corpus / f"{split}.txt" for split in SPLITS]
    provenance_paths = [corpus / "manifest.json", corpus / "stats.json", config_path]
    hashes = {str(path): file_sha(path) for path in input_paths + provenance_paths}
    train_check = preflight(input_paths[0], settings, corpus_stats["splits"]["train"])
    output = (ROOT / (output_override or config["paths"]["output_dir"])).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_prefix = output / "tokenizer"
    options = {
        "input": str(input_paths[0]), "model_prefix": str(model_prefix),
        **{name: settings[name] for name in (
            "model_type", "vocab_size", "character_coverage", "byte_fallback",
            "normalization_rule_name", "remove_extra_whitespaces", "num_threads", "max_sentence_length",
        )},
        "input_sentence_size": 0, "shuffle_input_sentence": False, "hard_vocab_limit": True,
        "add_dummy_prefix": True, "escape_whitespaces": True,
        **{f"{name}_id": value for name, value in SPECIAL_IDS.items()},
    }
    spm.set_random_generator_seed(settings["seed"])
    print(f"Training on train only: {train_check['sentences']} sentences", flush=True)
    spm.SentencePieceTrainer.train(**options)
    model_path = model_prefix.with_suffix(".model")
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    if processor.get_piece_size() != settings["vocab_size"]:
        raise ValueError("Actual vocabulary size differs from requested size.")
    if any(getattr(processor, f"{name}_id")() != value for name, value in SPECIAL_IDS.items()):
        raise ValueError("Special token IDs differ from the configured IDs.")
    probe_results = [example(processor, text) for text in PROBES]
    byte_ids = {i for i in range(processor.get_piece_size()) if processor.is_byte(i)}
    split_stats, samples = {}, []
    for split, path in zip(SPLITS, input_paths, strict=True):
        print(f"Measuring {split}...", flush=True)
        stats, split_samples = measure(processor, path, split, byte_ids)
        expected = corpus_stats["splits"][split]
        if stats["sentences"] != expected["sentences"] or stats["characters"] != expected["characters"]:
            raise ValueError(f"{split} text does not match corpus statistics.")
        split_stats[split] = stats
        samples.extend(split_samples)
    stats = {"actual_vocab_size": processor.get_piece_size(), "special_ids": SPECIAL_IDS,
             "byte_piece_count": len(byte_ids), "training_preflight": train_check, "splits": split_stats}
    dump_json(output / "stats.json", stats)
    dump_json(output / "examples.json", {"probes": probe_results, "corpus_samples": samples})
    dump_json(output / "config.json", {"config": config, "effective_training": settings, "trainer_options": options})
    if any(not item["roundtrip_ok"] or processor.unk_id() in item["ids"] for item in probe_results):
        raise ValueError("Probe roundtrip or unknown-token check failed; inspect examples.json.")
    if any(item["unknown_tokens"] or item["roundtrip_mismatches"] for item in split_stats.values()):
        raise ValueError("Corpus unknown-token/roundtrip check failed; inspect stats.json.")
    if any(file_sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("Inputs changed during tokenizer training.")
    dump_json(output / "manifest.json", {
        "status": "complete", "purpose": "SentencePiece tokenizer fitted on train only",
        "corpus_ready_for_lm_training": corpus_manifest.get("ready_for_lm_training", False),
        "trained_on": "train.txt only", "evaluated_on": list(SPLITS),
        "corpus_quality_mode": corpus_manifest.get("quality_mode"),
        "sentencepiece_version": spm.__version__, "python_version": sys.version, "platform": platform.platform(),
        "input_sha256": hashes, "script_sha256": file_sha(Path(__file__).resolve()),
        "model_sha256": file_sha(model_path), "vocab_sha256": file_sha(model_prefix.with_suffix(".vocab")),
        "effective_training": settings, "special_ids": SPECIAL_IDS,
        "reproducibility": "Pinned library, fixed seed, one thread by default, all train lines without sampling; preserve the model artifact for exact IDs across environments.",
        "sequence_policy": "Statistics exclude special tokens; +2 counts hypothetical BOS/EOS per sentence. No token sequences or LM training data are packed here.",
    })
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Tokenizer: {output}")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tokenizer.toml")
    parser.add_argument("--output", help="Override output directory; must be absent or empty.")
    parser.add_argument("--vocab-size", type=int, help="Override vocabulary size for an experiment.")
    args = parser.parse_args()
    run(args.config, args.output, args.vocab_size)


if __name__ == "__main__":
    main()
