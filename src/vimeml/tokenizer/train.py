"""Train SentencePiece on train only, then measure local corpus tokenization."""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import tomllib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

from vimeml.tokenizer.monitor import RunMonitor


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
    sample_size = settings.get("input_sentence_size", 0)
    if type(sample_size) is not int or (sample_size != 0 and sample_size <= 100):
        raise ValueError("input_sentence_size must be 0 (all train) or an integer > 100.")
    coverage = settings["character_coverage"]
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0.98 <= coverage <= 1:
        raise ValueError("character_coverage must be in [0.98, 1].")
    seed = settings["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer.")
    if settings["byte_fallback"] is not True or settings["normalization_rule_name"] != "identity" or settings["remove_extra_whitespaces"] is not False:
        raise ValueError("This baseline requires byte fallback, identity normalization and preserved whitespace.")


def preflight(train_path, settings, expected, progress=None):
    count, characters, longest = 0, 0, 0
    for text in sentences(train_path):
        count += 1
        characters += len(text)
        longest = max(longest, len(text.encode("utf-8")))
        if "▁" in text:
            raise ValueError("Train contains literal U+2581, reserved by SentencePiece; inspect before training.")
        if count % 1_000_000 == 0:
            print(f"[preflight] {count:,}/{expected['sentences']:,} train sentences", flush=True)
            if progress: progress(count,expected['sentences'])
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


def range_sentences(path, start, end):
    """Disjoint byte ranges, always starting at a complete UTF-8 line boundary."""
    with path.open("rb") as stream:
        if start:
            stream.seek(start - 1)
            if stream.read(1) != b"\n":
                stream.readline()
        while stream.tell() < end:
            raw = stream.readline()
            if not raw:
                break
            text = raw.decode("utf-8").removesuffix("\n").removesuffix("\r")
            if not text:
                raise ValueError(f"Corpus contains an empty line: {path}")
            yield text


def measure_range(job):
    """Top-level entry for Windows spawn; no fitted processor crosses processes."""
    import sentencepiece as spm
    model, path, start, end, split, batch_size = job
    processor = spm.SentencePieceProcessor(model_file=str(model))
    byte_ids = {i for i in range(processor.get_piece_size()) if processor.is_byte(i)}
    lengths = Counter()
    characters = tokens = unknown = byte_tokens = mismatches = 0
    examples, errors = [], []
    rng = random.Random(42)
    count = 0
    stream = range_sentences(path, start, end)
    while batch := list(islice(stream, batch_size)):
        encoded = processor.encode(batch, out_type=int, num_threads=1)
        decoded = processor.decode(encoded, num_threads=1)
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
                    errors.append({"range_start_byte": start, "line_in_range": count, "text": text, "decoded": restored})
            # Only train/validation examples are presented for inspection.
            if split != "test":
                sample = {"range_start_byte": start, "line_in_range": count, "text": text}
                if len(examples) < 10:
                    examples.append(sample)
                else:
                    position = rng.randrange(count)
                    if position < 10:
                        examples[position] = sample
    return {"sentences":count, "characters":characters, "tokens":tokens, "unknown":unknown,
            "byte_tokens":byte_tokens, "mismatches":mismatches, "lengths":dict(lengths),
            "examples":examples, "errors":errors, "range_start_byte":start}


def aggregate_measurements(processor, results, split):
    # Stable order keeps diagnostics independent of worker completion order.
    results = sorted(results, key=lambda item: item["range_start_byte"])
    lengths = Counter()
    for item in results:
        lengths.update(item["lengths"])
    count, characters, tokens, unknown, byte_tokens, mismatches = (
        sum(item[name] for item in results)
        for name in ("sentences", "characters", "tokens", "unknown", "byte_tokens", "mismatches")
    )
    errors = [error for item in results for error in item["errors"]][:10]
    examples = [sample for item in results for sample in item["examples"]][:10]
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
    return stats, [{"split": split, "range_start_byte": item["range_start_byte"],
                    "line_in_range":item["line_in_range"], **example(processor, item["text"])} for item in examples]


def measure_parallel(model, paths, workers, batch_size, expected, progress=None):
    """Parallel full-corpus measurement; the statistics remain exact, not sampled."""
    import sentencepiece as spm
    processor = spm.SentencePieceProcessor(model_file=str(model))
    collected = {split:[] for split in SPLITS}
    jobs = []
    for split, path in zip(SPLITS, paths, strict=True):
        size = path.stat().st_size
        partitions = min(workers * 4, max(1, math.ceil(size / (32 * 1024 * 1024))))
        for index in range(partitions):
            jobs.append((model, path, size*index//partitions, size*(index+1)//partitions, split, batch_size))
    completed_rows = 0
    total_rows = sum(expected[split]["sentences"] for split in SPLITS)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(measure_range,job):job[4] for job in jobs}
        for future in as_completed(pending):
            split = pending[future]
            result = future.result()
            collected[split].append(result)
            completed_rows += result["sentences"]
            elapsed = time.monotonic() - started
            print(f"[measure] {completed_rows:,}/{total_rows:,} sentences; elapsed={elapsed:.0f}s", flush=True)
            if progress:
                progress(completed_rows,total_rows,elapsed)
    stats, samples = {}, []
    for split in SPLITS:
        stats[split], examples = aggregate_measurements(processor,collected[split],split)
        if stats[split]["sentences"] != expected[split]["sentences"] or stats[split]["characters"] != expected[split]["characters"]:
            raise ValueError(f"{split} text does not match corpus statistics.")
        samples.extend(examples)
    return stats,samples


def hash_inputs(paths):
    with ThreadPoolExecutor(max_workers=min(3,len(paths))) as pool:
        return dict(zip(map(str,paths),pool.map(file_sha,paths),strict=True))


def train_native(options, seed, output, monitor):
    request = output/'trainer-request.json'
    dump_json(request,{"seed":seed,"options":options})
    child = subprocess.Popen([sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('native_train.py')),str(request)],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        with (output/'trainer.log').open('w',encoding='utf-8',newline='\n') as log:
            for line in child.stdout:
                print(line.rstrip(),flush=True)
                log.write(line)
                log.flush()
                monitor.native_line(line)
        if child.wait() != 0:
            raise RuntimeError(f"SentencePiece training failed; inspect {output/'trainer.log'}")
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        child.stdout.close()


def run(config_path, output_override=None, vocab_override=None, dry_run=False):
    try:
        import sentencepiece as spm
    except ModuleNotFoundError:
        raise SystemExit("请先运行：uv pip install --python .venv/Scripts/python.exe -r requirements-tokenizer.txt") from None
    if spm.__version__ != "0.2.1":
        raise ValueError("Use sentencepiece==0.2.1 for this recorded baseline.")
    config_path = config_path.resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    settings = dict(config["training"])
    settings.setdefault("input_sentence_size",0)
    if vocab_override is not None:
        settings["vocab_size"] = vocab_override
    validate(settings)
    measurement = {"workers":1,"batch_size":4096,**config.get("measurement",{})}
    for name in ("workers","batch_size"):
        if type(measurement[name]) is not int or measurement[name] < 1:
            raise ValueError(f"measurement.{name} must be a positive integer.")
    corpus = (ROOT / config["paths"]["corpus_dir"]).resolve()
    corpus_manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    if corpus_manifest.get("status") != "complete":
        raise ValueError("Corpus build must be complete.")
    corpus_stats = json.loads((corpus / "stats.json").read_text(encoding="utf-8"))
    input_paths = [corpus / f"{split}.txt" for split in SPLITS]
    provenance_paths = [corpus / "manifest.json", corpus / "stats.json", config_path]
    output = (ROOT / (output_override or config["paths"]["output_dir"])).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be absent or empty: {output}")
    plan = {"corpus":str(corpus),"output":str(output),"vocab_size":settings["vocab_size"],
            "available_train_sentences":corpus_stats["splits"]["train"]["sentences"],
            "sampled_training_sentences":min(settings["input_sentence_size"] or corpus_stats["splits"]["train"]["sentences"],corpus_stats["splits"]["train"]["sentences"]),
            "sample_policy":"SentencePiece seeded random sampling across train.txt" if settings["input_sentence_size"] else "all train.txt",
            "train_threads":settings["num_threads"],"measurement":measurement,
            "measurement_scope":"all train/validation/test sentences; no sampling",
            "input_txt_bytes":sum(p.stat().st_size for p in input_paths),"logical_cpus":os.cpu_count(),
            "dry_run":dry_run,"training_started":False}
    plan["tensorboard"] = {"enabled":config.get('monitoring',{}).get('tensorboard',False),
                          "log_dir":str(ROOT/'runs/tokenizer'/output.name)}
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if dry_run:
        return plan
    output.mkdir(parents=True, exist_ok=True)
    with RunMonitor(output,Path(plan["tensorboard"]["log_dir"]),plan["tensorboard"]["enabled"]) as monitor:
        monitor.update("checking_inputs")
        print("[inputs] Hashing local corpus files...",flush=True)
        hashes = hash_inputs(input_paths + provenance_paths)
        train_check = preflight(input_paths[0], settings, corpus_stats["splits"]["train"],
            lambda done,total:monitor.update("checking_inputs",preflight_completed_sentences=done,preflight_total_sentences=total))
        progress = monitor.update
        progress("training",sampled_training_sentences=plan["sampled_training_sentences"])
        model_prefix = output / "tokenizer"
        options = {
            "input": str(input_paths[0]), "model_prefix": str(model_prefix),
            **{name: settings[name] for name in (
                "model_type", "vocab_size", "character_coverage", "byte_fallback",
                "normalization_rule_name", "remove_extra_whitespaces", "num_threads", "max_sentence_length",
            )},
            "input_sentence_size": settings["input_sentence_size"], "shuffle_input_sentence": settings["input_sentence_size"] > 0, "hard_vocab_limit": True,
            "add_dummy_prefix": True, "escape_whitespaces": True,
            **{f"{name}_id": value for name, value in SPECIAL_IDS.items()},
        }
        print(f"[train] Sample up to {plan['sampled_training_sentences']:,} of {train_check['sentences']:,} train sentences; threads={settings['num_threads']}", flush=True)
        train_native(options,settings["seed"],output,monitor)
        model_path = model_prefix.with_suffix(".model")
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        if processor.get_piece_size() != settings["vocab_size"]:
            raise ValueError("Actual vocabulary size differs from requested size.")
        if any(getattr(processor, f"{name}_id")() != value for name, value in SPECIAL_IDS.items()):
            raise ValueError("Special token IDs differ from the configured IDs.")
        probe_results = [example(processor, text) for text in PROBES]
        byte_ids = {i for i in range(processor.get_piece_size()) if processor.is_byte(i)}
        progress("measuring",workers=measurement["workers"])
        split_stats, samples = measure_parallel(model_path,input_paths,measurement["workers"],measurement["batch_size"],
            corpus_stats["splits"],lambda done,total,elapsed:progress("measuring",completed_sentences=done,total_sentences=total,measurement_elapsed_seconds=elapsed))
        stats = {"actual_vocab_size": processor.get_piece_size(), "special_ids": SPECIAL_IDS,
                 "byte_piece_count": len(byte_ids), "training_preflight": train_check,
                 "sampled_training_sentences":plan["sampled_training_sentences"],"splits": split_stats}
        dump_json(output / "stats.json", stats)
        dump_json(output / "examples.json", {"probes": probe_results, "corpus_samples": samples})
        dump_json(output / "config.json", {"config": config, "effective_training": settings, "trainer_options": options})
        if any(not item["roundtrip_ok"] or processor.unk_id() in item["ids"] for item in probe_results):
            raise ValueError("Probe roundtrip or unknown-token check failed; inspect examples.json.")
        if any(item["unknown_tokens"] or item["roundtrip_mismatches"] for item in split_stats.values()):
            raise ValueError("Corpus unknown-token/roundtrip check failed; inspect stats.json.")
        progress("verifying_inputs")
        if hash_inputs(input_paths + provenance_paths) != hashes:
            raise ValueError("Inputs changed during tokenizer training.")
        dump_json(output / "manifest.json", {
            "status": "complete", "purpose": "SentencePiece tokenizer fitted on train only",
            "corpus_ready_for_lm_training": corpus_manifest.get("ready_for_lm_training", False),
            "trained_on": "train.txt only",
            "sampling":{key:plan[key] for key in ("available_train_sentences","sampled_training_sentences","sample_policy")},
            "evaluated_on": list(SPLITS),
            "corpus_quality_mode": corpus_manifest.get("quality_mode"),
            "sentencepiece_version": spm.__version__, "python_version": sys.version, "platform": platform.platform(),
            "input_sha256": hashes, "script_sha256": file_sha(Path(__file__).resolve()),
            "code_sha256":{name:file_sha(Path(__file__).with_name(name)) for name in ("train.py","native_train.py","monitor.py")},
            "model_sha256": file_sha(model_path), "vocab_sha256": file_sha(model_prefix.with_suffix(".vocab")),
            "effective_training": settings, "special_ids": SPECIAL_IDS,
            "measurement":measurement,
            "reproducibility": "Pinned library and seed; random sampling uses the recorded full train input. Multithreading/platform changes can change learned IDs; preserve the model artifact and hashes for exact reuse.",
            "sequence_policy": "Statistics exclude special tokens; +2 counts hypothetical BOS/EOS per sentence. No token sequences or LM training data are packed here.",
        })
        monitor.final_statistics(split_stats)
        progress("complete",model_sha256=file_sha(model_path))
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"Tokenizer: {output}")
        return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tokenizer.toml")
    parser.add_argument("--output", help="Override output directory; must be absent or empty.")
    parser.add_argument("--vocab-size", type=int, help="Override vocabulary size for an experiment.")
    parser.add_argument("--dry-run",action="store_true",help="Print data and resource plan without training or writing output.")
    args = parser.parse_args()
    run(args.config, args.output, args.vocab_size,args.dry_run)


if __name__ == "__main__":
    main()
