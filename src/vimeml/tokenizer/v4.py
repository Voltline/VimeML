"""V4 tokenizer experiment: shared train-only sample, strict kana checks.

Does not open test or blind text. Existing V1/V2 training stays frozen.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
import tomllib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from vimeml.tokenizer.monitor import RunMonitor
from vimeml.tokenizer.train import (
    ROOT, SPECIAL_IDS, PROBES, aggregate_measurements, dump_json, example,
    file_sha, measure_range, percentile, train_native, validate,
)

MARKERS = ["<CTX>", "<READ>", "<OUT>"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def permitted_path(path):
    """Reject held-out roles before any file is opened, including symlinks."""
    path = Path(path).resolve()
    if any("blind" in part.lower() or part.lower() in {"test", "test.txt"}
           for part in path.parts):
        raise ValueError(f"Held-out input is forbidden: {path}")
    return path


def sample_train(path, output, size, seed, expected, max_bytes, progress):
    """Explicit reservoir sampling; persist identical bytes for all vocab sizes."""
    rng = random.Random(seed)
    digest = hashlib.sha256()
    rows = []
    count = characters = longest = 0
    with permitted_path(path).open("rb") as stream:
        for count, raw in enumerate(stream, 1):
            digest.update(raw)
            text = raw.decode("utf-8").removesuffix("\n").removesuffix("\r")
            if not text or "▁" in text or any(marker in text for marker in MARKERS):
                raise ValueError(f"Empty/reserved text in train line {count}")
            characters += len(text)
            longest = max(longest, len(text.encode("utf-8")))
            if longest > max_bytes:
                raise ValueError("Train sentence exceeds max_sentence_length; no silent exclusion allowed")
            if len(rows) < size:
                rows.append(raw)
            else:
                index = rng.randrange(count)
                if index < size:
                    rows[index] = raw
            if count % 1_000_000 == 0:
                progress(count, expected["sentences"])
    if count != expected["sentences"] or characters != expected["characters"]:
        raise ValueError("Train counts differ from frozen corpus metadata")
    with output.open("wb") as stream:
        for raw in rows:
            stream.write(raw if raw.endswith(b"\n") else raw + b"\n")
    return {"sentences": count, "characters": characters, "max_utf8_bytes": longest,
            "train_sha256": digest.hexdigest(), "sample_sentences": len(rows),
            "sample_characters": sum(len(raw.decode("utf-8").rstrip("\r\n")) for raw in rows),
            "sample_sha256": file_sha(output), "seed": seed,
            "policy": "Python random.Random(seed) reservoir over every train.txt line; persisted order"}


def verify_model(model, vocab_size, kana):
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb
    processor = spm.SentencePieceProcessor(model_file=str(model))
    proto = pb.ModelProto.FromString(model.read_bytes())
    if processor.get_piece_size() != vocab_size:
        raise ValueError("Exact vocabulary size check failed")
    if any(getattr(processor, f"{name}_id")() != value for name, value in SPECIAL_IDS.items()):
        raise ValueError("Special IDs must be PAD/UNK/BOS/EOS = 0/1/2/3")
    normalizer = proto.normalizer_spec
    if (normalizer.name != "identity" or normalizer.add_dummy_prefix
            or normalizer.remove_extra_whitespaces or not normalizer.escape_whitespaces):
        raise ValueError("V4 normalization/whitespace contract failed")
    if (proto.trainer_spec.model_type != pb.TrainerSpec.UNIGRAM
            or not proto.trainer_spec.byte_fallback
            or proto.trainer_spec.required_chars != kana
            or list(proto.trainer_spec.user_defined_symbols) != MARKERS):
        raise ValueError("V4 trainer contract failed")
    for index, marker in enumerate(MARKERS, 4):
        if (processor.piece_to_id(marker) != index
                or proto.pieces[index].type != pb.ModelProto.SentencePiece.USER_DEFINED
                or processor.encode(marker) != [index]
                or processor.decode([index]) != marker):
            raise ValueError(f"Reserved marker contract failed: {marker}")
    checks = []
    for char in kana:
        token = processor.piece_to_id(char)
        ok = (token != processor.unk_id() and processor.id_to_piece(token) == char
              and proto.pieces[token].type == pb.ModelProto.SentencePiece.NORMAL
              and processor.encode(char) == [token] and processor.decode([token]) == char)
        checks.append({"character": char, "codepoint": f"U+{ord(char):04X}", "id": token, "passed": ok})
    if not all(item["passed"] for item in checks):
        raise ValueError(f"Kana single-character check failed: {[c for c in checks if not c['passed']]}")
    byte_count = sum(processor.is_byte(i) for i in range(vocab_size))
    if byte_count != 256:
        raise ValueError("Expected exactly 256 byte tokens")
    probes = [example(processor, text) for text in [*PROBES, *MARKERS, "か\u3099", "ｶﾞ", "ヴヷヸヹヺ"]]
    if any(not p["roundtrip_ok"] or processor.unk_id() in p["ids"] for p in probes):
        raise ValueError(f"Probe UNK/roundtrip check failed: {[p for p in probes if not p['roundtrip_ok'] or processor.unk_id() in p['ids']]}")
    return {"status": "passed", "vocab_size": vocab_size, "special_ids": SPECIAL_IDS,
            "marker_ids": dict(zip(MARKERS, range(4, 7))), "byte_piece_count": byte_count,
            "normal_piece_count": vocab_size - 263, "required_kana_count": len(kana),
            "kana_checks": checks, "probes": probes}


def measure_corpora(model, paths, expected, workers, batch_size, progress):
    """Exact statistics on the training sample and full validation, without test."""
    import sentencepiece as spm
    jobs = []
    collected = {split: [] for split in paths}
    for split, path in paths.items():
        path = permitted_path(path)
        size = path.stat().st_size
        partitions = min(workers * 4, max(1, math.ceil(size / (32 * 1024 * 1024))))
        for index in range(partitions):
            jobs.append((model, path, size * index // partitions,
                         size * (index + 1) // partitions, split, batch_size))
    completed = 0
    total = sum(item["sentences"] for item in expected.values())
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(measure_range, job): job[4] for job in jobs}
        for future in as_completed(pending):
            split = pending[future]
            result = future.result()
            collected[split].append(result)
            completed += result["sentences"]
            progress(completed, total, time.monotonic() - started)
    processor = spm.SentencePieceProcessor(model_file=str(model))
    stats = {}
    for split, results in collected.items():
        stats[split], _ = aggregate_measurements(processor, results, split)
        if any(stats[split][key] != expected[split][key] for key in ("sentences", "characters")):
            raise ValueError(f"Measurement count mismatch: {split}")
        if stats[split]["unknown_tokens"] or stats[split]["roundtrip_mismatches"]:
            raise ValueError(f"Measurement UNK/roundtrip failure: {split}")
        stats[split]["estimated_characters_in_128_content_tokens"] = 128 * stats[split]["characters_per_token"]
        stats[split]["estimated_characters_in_126_content_tokens_plus_bos_eos"] = 126 * stats[split]["characters_per_token"]
    return stats


def text_statistics(processor, texts):
    lengths = Counter()
    characters = tokens = byte_tokens = unknown = mismatches = 0
    coverage = []
    for text in texts:
        ids = processor.encode(text)
        lengths[len(ids)] += 1
        characters += len(text)
        tokens += len(ids)
        byte_tokens += sum(processor.is_byte(i) for i in ids)
        unknown += ids.count(processor.unk_id())
        mismatches += processor.decode(ids) != text
        # Immutable proto offsets use Unicode character positions in Python.
        proto = processor.encode(text, out_type="immutable_proto")
        if proto.pieces:
            coverage.append(proto.pieces[min(126, len(proto.pieces)) - 1].end)
        else:
            coverage.append(0)
    count = len(texts)
    if unknown or mismatches:
        raise ValueError("Development text failed UNK/roundtrip verification")
    return {"texts": count, "characters": characters, "tokens": tokens,
            "mean_tokens": tokens / count if count else None,
            "characters_per_token": characters / tokens if tokens else None,
            "byte_fallback_tokens": byte_tokens,
            "byte_fallback_token_fraction": byte_tokens / tokens if tokens else None,
            "unknown_tokens": unknown, "roundtrip_mismatches": mismatches,
            "token_length_percentiles": {f"p{n}": percentile(lengths, n / 100) for n in (50, 95, 99)},
            "fits_128_with_bos_eos_fraction": sum(n for length, n in lengths.items() if length + 2 <= 128) / count if count else None,
            "mean_characters_retained_in_128_with_bos_eos": sum(coverage) / count if count else None}


def measure_development(model, directory):
    """Read the explicit development directory only; no LM or ranking scores."""
    import sentencepiece as spm
    directory = permitted_path(directory)
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    if (directory.name != "development" or manifest.get("split") != "development"
            or manifest.get("status") != "complete"
            or manifest.get("format") != "vimeml_standard_ime_candidates_v1"):
        raise ValueError("Only completed standard development candidates are permitted")
    from vimeml.benchmarks.standard_ime import load_standard_export
    _, _, rows = load_standard_export(directory)
    if not rows or any(row.get("provenance", {}).get("role") != "development" for row in rows):
        raise ValueError("Benchmark contains a non-development row")
    processor = spm.SentencePieceProcessor(model_file=str(model))
    from vimeml.benchmarks.standard_ime import read
    originals_path = directory / "evaluation_items.json"
    originals = read(originals_path)
    candidates = [candidate["text"] for row in rows for candidate in row["candidates"]]
    readings = [row["input"] for row in originals]
    # Katakana -> hiragana for later reading tasks; originals remain immutable.
    hiragana = ["".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text) for text in readings]
    result = {"role": "development", "release_id": manifest["release_id"], "cases": len(rows),
              "purpose": "Tokenizer statistics only; no probability/ranking/reference accuracy",
              "candidates": text_statistics(processor, candidates),
              "reading_original_katakana": text_statistics(processor, readings),
              "reading_hiragana_diagnostic": text_statistics(processor, hiragana),
              "context_plus_candidate": text_statistics(processor, [row["left_context"] + c["text"] for row in rows for c in row["candidates"]])}
    result["input_sha256"] = {str(path): file_sha(path) for path in (
        manifest_path, originals_path, directory / "ajimee-results/azookey-candidates.json")}
    return result


def run(config_path, dry_run=False):
    import sentencepiece as spm
    if spm.__version__ != "0.2.1":
        raise ValueError("V4 requires sentencepiece==0.2.1")
    config_path = permitted_path(config_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    settings = config["training"]
    validate(settings)
    if (settings["model_type"] != "unigram" or settings["vocab_size"] != 8192
            or settings.get("add_dummy_prefix") is not False
            or settings.get("user_defined_symbols") != MARKERS
            or settings["input_sentence_size"] <= 100):
        raise ValueError("Fixed V4-a tokenizer contract differs from configuration")
    sizes = config["comparison"]["vocab_sizes"]
    if sizes != [8192, 12288, 16384]:
        raise ValueError("Expected 8K/12K/16K comparison")
    measurement = config["measurement"]
    if any(type(measurement[k]) is not int or measurement[k] < 1 for k in ("workers", "batch_size")):
        raise ValueError("Measurement resources must be positive integers")
    paths = config["paths"]
    corpus = permitted_path(ROOT / paths["corpus_dir"])
    output = permitted_path(ROOT / paths["output_dir"])
    kana_path = permitted_path(ROOT / paths["required_chars_file"])
    kana = "".join(kana_path.read_text(encoding="utf-8").splitlines())
    if not kana or len(set(kana)) != len(kana):
        raise ValueError("Kana list must be nonempty and unique")
    benchmark = permitted_path(ROOT / paths["benchmark_development"])
    if benchmark.name != "development":
        raise ValueError("Benchmark path must point specifically to development")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output}")
    corpus_manifest = read_json(corpus / "manifest.json")
    stats = read_json(corpus / "stats.json")["splits"]
    integrity = read_json(corpus / "integrity-check.json")
    if (corpus_manifest.get("status") != "complete" or integrity.get("status") != "passed"
            or file_sha(corpus / "manifest.json") != integrity["manifest_sha256"]):
        raise ValueError("Frozen corpus integrity metadata failed")
    inputs = [config_path, kana_path, corpus / "train.txt", corpus / "validation.txt",
              corpus / "manifest.json", corpus / "stats.json", corpus / "integrity-check.json"]
    before = {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in inputs}
    plan = {"output": str(output), "vocab_sizes": sizes, "required_kana_count": len(kana),
            "trained_on": "train.txt only; one persisted reservoir sample",
            "sample_size": settings["input_sentence_size"], "seed": settings["seed"],
            "evaluated_on": ["training sample", "full validation", "standard development"],
            "test_and_blind_access": False, "lm_training_started": False,
            "dry_run": dry_run}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return plan
    output.mkdir(parents=True)
    dump_json(output / "config.json", config)
    dump_json(output / "plan.json", plan)
    sample_path = output / "train-sample.txt"
    started = time.monotonic()
    with RunMonitor(output, ROOT / "runs/tokenizer/v4-a-sampling", config["monitoring"]["tensorboard"]) as monitor:
        monitor.update("checking_inputs")
        sample = sample_train(corpus / "train.txt", sample_path, settings["input_sentence_size"],
                              settings["seed"], stats["train"], settings["max_sentence_length"],
                              lambda done, total: monitor.update("checking_inputs", preflight_completed_sentences=done, preflight_total_sentences=total))
        if sample["train_sha256"] != integrity["exports"]["train.txt"]["sha256"]:
            raise ValueError("Actual train SHA256 differs from frozen corpus")
        validation_hash = file_sha(corpus / "validation.txt")
        if validation_hash != integrity["exports"]["validation.txt"]["sha256"]:
            raise ValueError("Actual validation SHA256 differs from frozen corpus")
        dump_json(output / "sampling.json", sample)
        monitor.update("complete")
    results = {}
    for size in sizes:
        target = output / f"ja-unigram-{size}"
        target.mkdir()
        with RunMonitor(target, ROOT / "runs/tokenizer" / f"v4-a-{size}", config["monitoring"]["tensorboard"]) as monitor:
            monitor.update("training")
            options = {k: v for k, v in settings.items() if k not in {"seed", "input_sentence_size"}}
            options.update(input=str(sample_path), model_prefix=str(target / "tokenizer"), vocab_size=size,
                           input_sentence_size=0, shuffle_input_sentence=False, hard_vocab_limit=True,
                           escape_whitespaces=True, required_chars=kana,
                           **{f"{name}_id": value for name, value in SPECIAL_IDS.items()})
            train_native(options, settings["seed"], target, monitor)
            model = target / "tokenizer.model"
            verification = verify_model(model, size, kana)
            dump_json(target / "validation.json", verification)
            monitor.update("measuring")
            split_stats = measure_corpora(model, {"train_sample": sample_path, "validation": corpus / "validation.txt"},
                {"train_sample": {"sentences": sample["sample_sentences"], "characters": sample["sample_characters"]}, "validation": stats["validation"]},
                measurement["workers"], measurement["batch_size"],
                lambda done, total, elapsed: monitor.update("measuring", completed_sentences=done, total_sentences=total, measurement_elapsed_seconds=elapsed))
            development = measure_development(model, benchmark)
            result = {"vocab_size": size, "normal_piece_count": size - 263,
                      "splits": split_stats, "development": development}
            dump_json(target / "stats.json", result)
            dump_json(target / "manifest.json", {"status": "complete", "trained_on": "shared train-only sample",
                "model_sha256": file_sha(model), "vocab_sha256": file_sha(target / "tokenizer.vocab"),
                "sample_sha256": sample["sample_sha256"], "trainer_options": options,
                "sentencepiece_version": spm.__version__, "special_ids": SPECIAL_IDS,
                "marker_ids": verification["marker_ids"], "test_and_blind_access": False})
            monitor.final_statistics(split_stats)
            monitor.update("complete")
            results[str(size)] = result
            print(f"[complete] vocab={size}; validation chars/token={split_stats['validation']['characters_per_token']:.4f}", flush=True)
    after = {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in inputs}
    if before != after or file_sha(sample_path) != sample["sample_sha256"]:
        raise ValueError("Inputs changed during training")
    dump_json(output / "comparison.json", results)
    dump_json(output / "manifest.json", {"status": "complete", "primary_vocab_size": 8192,
        "primary_model": str(output / "ja-unigram-8192/tokenizer.model"), "plan": plan,
        "sample": sample, "validation_sha256": validation_hash, "input_metadata": before,
        "input_sha256": {str(p): file_sha(p) for p in inputs if p.name not in {"train.txt", "validation.txt"}},
        "code_sha256": {str(p): file_sha(p) for p in [Path(__file__), Path(__file__).with_name("train.py"), Path(__file__).with_name("native_train.py"), Path(__file__).with_name("monitor.py")]},
        "python_version": sys.version, "elapsed_seconds": time.monotonic() - started,
        "corpus_ready_for_lm_training": corpus_manifest.get("ready_for_lm_training", False),
        "reproducibility": "Exact shared sample and artifacts are preserved; native multithreading/platform can affect learned IDs",
        "context_coverage_note": "Corpus 128-token character estimates are ratio-based, not measured packed-window coverage. Development prefix coverage is measured and includes short texts."})
    print(f"[complete] V4 tokenizer experiment: {output}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tokenizer-v4.toml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.config, args.dry_run)


if __name__ == "__main__":
    main()
