"""V4-only TXT encoding with provenance, strict checks and resumable parts.

Original training JSONL was pruned locally. Verified V2 source IDs are reused;
all token sequences and row byte positions are regenerated from the frozen TXT.
No test/blind text is read, and V1/V2 implementations remain unchanged.
"""
import argparse
import hashlib
import json
import time
import tomllib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from vimeml.tokenizer.encode import dump_json, file_sha, write_numbers
from vimeml.tokenizer.encode_parallel import cached_part, output_lock, part_paths
from vimeml.tokenizer.monitor import RunMonitor
from vimeml.tokenizer.v4 import MARKERS, ROOT, permitted_path, read_json, verify_model

SPLITS = ("train", "validation")
FORMAT = "vimeml_sentence_tokens_v1"


def recorded_hash(manifest, name, parent=None):
    values = [v for k, v in manifest["input_sha256"].items()
              if k.replace("\\", "/").rsplit("/", 1)[-1] == name
              and (parent is None or k.replace("\\", "/").endswith(f"/{parent}/{name}"))]
    if len(values) != 1:
        raise ValueError(f"Missing unique recorded corpus hash: {name}")
    return values[0]


def scan_txt(path, split, chunk_bytes, expected, progress):
    """Hash and partition at exact line boundaries, retaining source row indices."""
    path = permitted_path(path)
    digest = hashlib.sha256()
    jobs = []
    count = characters = start = first_row = end = longest = 0
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            text = raw.decode("utf-8").removesuffix("\n").removesuffix("\r")
            if not text or "▁" in text or any(marker in text for marker in MARKERS):
                raise ValueError(f"Invalid/reserved text: {split} row {count}")
            count += 1
            characters += len(text)
            longest = max(longest, len(text.encode("utf-8")))
            end = stream.tell()
            if end - start >= chunk_bytes:
                jobs.append({"split": split, "input": str(path), "start": start, "end": end,
                             "first_row": first_row, "sentences": count - first_row})
                start, first_row = end, count
            if count % 1_000_000 == 0:
                progress(count, expected["sentences"])
        if end > start:
            jobs.append({"split": split, "input": str(path), "start": start, "end": end,
                         "first_row": first_row, "sentences": count - first_row})
    if count != expected["sentences"] or characters != expected["characters"]:
        raise ValueError(f"Corpus count mismatch: {split}")
    return jobs, {"sentences": count, "characters": characters, "max_utf8_bytes": longest,
                  "sha256": digest.hexdigest()}


def encode_txt_part(job):
    import sentencepiece as spm
    processor = spm.SentencePieceProcessor(model_file=job["model"])
    files = part_paths(job)
    files["done.json"].unlink(missing_ok=True)
    reserved = set(range(7))
    byte_ids = {i for i in range(8192) if processor.is_byte(i)}
    reverse_sources = {value: name for name, value in job["source_ids"].items()}
    count = content = stored = characters = byte_tokens = longest = 0
    lengths, sources = Counter(), Counter()
    text_digest, source_digest = hashlib.sha256(), hashlib.sha256()
    with Path(job["input"]).open("rb") as text_stream, Path(job["source_input"]).open("rb") as source_stream, \
            files["tokens.bin"].open("wb") as token_stream, files["offsets.bin"].open("wb") as offset_stream, \
            files["rows.bin"].open("wb") as row_stream, files["sources.bin"].open("wb") as out_sources:
        text_stream.seek(job["start"])
        source_stream.seek(job["first_row"])
        write_numbers(offset_stream, "Q", [0])
        while text_stream.tell() < job["end"]:
            texts, positions = [], []
            for _ in range(job["batch_size"]):
                position = text_stream.tell()
                if position >= job["end"]:
                    break
                raw = text_stream.readline()
                if not raw:
                    raise ValueError("Unexpected TXT end")
                text_digest.update(raw)
                text = raw.decode("utf-8").removesuffix("\n").removesuffix("\r")
                if not text:
                    raise ValueError("Empty sentence")
                texts.append(text)
                positions.append(position)
            encoded = processor.encode(texts, out_type=int, num_threads=1)
            decoded = processor.decode(encoded, num_threads=1)
            source_bytes = source_stream.read(len(texts))
            if len(source_bytes) != len(texts):
                raise ValueError("Source index is shorter than corpus")
            source_digest.update(source_bytes)
            batch_tokens, batch_offsets = [], []
            for text, ids, restored, source in zip(texts, encoded, decoded, source_bytes, strict=True):
                if restored != text or any(i in reserved or i >= 8192 for i in ids):
                    raise ValueError(f"Roundtrip/reserved/ID error: {job['name']} row {count}")
                if source not in reverse_sources:
                    raise ValueError("Unknown source ID")
                count += 1
                characters += len(text)
                content += len(ids)
                stored += len(ids) + 2
                longest = max(longest, len(ids) + 2)
                lengths[len(ids)] += 1
                byte_tokens += sum(i in byte_ids for i in ids)
                sources[reverse_sources[source]] += 1
                batch_tokens.extend((2, *ids, 3))
                batch_offsets.append(stored)
            write_numbers(token_stream, "H", batch_tokens)
            write_numbers(offset_stream, "Q", batch_offsets)
            write_numbers(row_stream, "Q", positions)
            out_sources.write(source_bytes)
        if text_stream.tell() != job["end"] or count != job["sentences"]:
            raise ValueError("TXT partition boundaries/counts changed")
    sizes = {suffix: path.stat().st_size for suffix, path in files.items() if suffix != "done.json"}
    if sizes != {"tokens.bin": stored * 2, "offsets.bin": (count + 1) * 8,
                 "rows.bin": count * 8, "sources.bin": count}:
        raise ValueError("Part file sizes are inconsistent")
    result = {"name": job["name"], "signature": job["signature"], "split": job["split"],
              "start": job["start"], "end": job["end"], "first_row": job["first_row"],
              "sentences": count, "characters": characters, "stored_tokens": stored,
              "content_tokens": content, "max_sequence_tokens": longest,
              "primary_sources": dict(sources), "lengths": dict(lengths),
              "byte_fallback_tokens": byte_tokens, "unknown_tokens": 0, "roundtrip_mismatches": 0,
              "txt_part_sha256": text_digest.hexdigest(), "source_part_sha256": source_digest.hexdigest(),
              "files": {suffix: {"bytes": path.stat().st_size, "sha256": file_sha(path)}
                        for suffix, path in files.items() if suffix != "done.json"}}
    dump_json(files["done.json"], result)
    return result


def collect(jobs, workers, monitor):
    results, pending = {}, []
    for job in jobs:
        cached = cached_part(job)
        if cached:
            # JSON object histogram keys need converting back to lengths.
            cached["lengths"] = {int(k): v for k, v in cached["lengths"].items()}
            results[job["name"]] = cached
        else:
            pending.append(job)
    total = sum(job["sentences"] for job in jobs)
    done = sum(r["sentences"] for r in results.values())
    initial, started = done, time.monotonic()
    print(f"[encode] cached={len(results)} pending={len(pending)} total_rows={total:,}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(encode_txt_part, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            results[job["name"]] = result
            done += result["sentences"]
            elapsed = time.monotonic() - started
            speed = (done - initial) / elapsed
            monitor.update("encoding", completed_sentences=done, total_sentences=total,
                           encoding_sentences_per_second=speed,
                           encoding_eta_seconds=(total - done) / speed)
            print(f"[encode] {done:,}/{total:,} ({100*done/total:.1f}%) {speed:,.0f} rows/s", flush=True)
    return results


def merge_txt_split(output, split, jobs, results, expected, monitor):
    from vimeml.tokenizer.encode_parallel import merge_split
    totals, histogram, sources = Counter(), Counter(), Counter()
    for job in jobs:
        result = results[job["name"]]
        for key in ("sentences", "characters", "content_tokens", "stored_tokens", "byte_fallback_tokens"):
            totals[key] += result[key]
        histogram.update(result["lengths"])
        sources.update(result["primary_sources"])
    if (totals["sentences"] != expected["sentences"] or totals["characters"] != expected["characters"]
            or dict(sources) != expected["primary_sources"]):
        raise ValueError(f"Full split counts/source counts changed: {split}")
    # This helper checks all offsets and merged sizes. Its JSONL-specific TXT
    # alignment branch is bypassed because these parts were encoded from TXT.
    exported = merge_split(None, output, split, jobs, results,
        {**dict(totals), "tokens_without_special_tokens": totals["content_tokens"],
         "tokens_with_bos_eos_per_sentence": totals["stored_tokens"]}, monitor, 0,
        totals["sentences"], verification="metadata")
    from vimeml.tokenizer.train import percentile
    exported.update(unknown_tokens=0, roundtrip_mismatches=0, byte_fallback_tokens=totals["byte_fallback_tokens"],
        byte_fallback_token_fraction=totals["byte_fallback_tokens"] / totals["content_tokens"],
        characters_per_token=totals["characters"] / totals["content_tokens"],
        mean_tokens_per_sentence=totals["content_tokens"] / totals["sentences"],
        length_percentiles_without_special_tokens={f"p{n}": percentile(histogram, n / 100) for n in (50, 95, 99)},
        fits_context_with_bos_eos={"128": {"sentences": sum(n for length, n in histogram.items() if length + 2 <= 128),
            "fraction": sum(n for length, n in histogram.items() if length + 2 <= 128) / totals["sentences"]}})
    return exported


def prepare_v4_indexes(token_dir, index_dir, checks):
    from vimeml.training.data import SentenceWindowDataset, prepare_split
    from vimeml.training.check_data import check_core, check_torch
    token_sha = file_sha(token_dir / "manifest.json")
    if (index_dir / "manifest.json").exists():
        existing = read_json(index_dir / "manifest.json")
        if existing.get("token_manifest_sha256") != token_sha or existing.get("context_length") != 128:
            raise ValueError("Index directory belongs to another token dataset")
    index_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {split: pool.submit(prepare_split, token_dir, index_dir, split, 128, token_sha, "sha256") for split in SPLITS}
        for split, future in pending.items():
            splits[split] = future.result()
    index = {"status": "complete", "format": "vimeml_sentence_windows_v1", "context_length": 128,
             "token_manifest_sha256": token_sha, "token_file_verification": "sha256",
             "available_splits": list(SPLITS), "splits": splits,
             "policy": "Separate sentences; nonoverlapping target windows with shared boundary input; all prediction pairs retained."}
    dump_json(index_dir / "manifest.json", index)
    report = {"status": "passed", "core_checks": {}, "pytorch_checks": {}, "config": checks}
    for split in SPLITS:
        report["core_checks"][split] = check_core(SentenceWindowDataset(token_dir, index_dir, split), checks["seed"])
        report["pytorch_checks"][split] = check_torch(SentenceWindowDataset(token_dir, index_dir, split), checks)
    dump_json(index_dir / "loader-check.json", report)
    return index, report


def run(config_path, resume=False, dry_run=False):
    import sentencepiece as spm
    config_path = permitted_path(config_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    paths = {key: permitted_path(ROOT / value) for key, value in config["paths"].items()}
    corpus, tokenizer, source_store = (paths[k] for k in ("corpus_dir", "tokenizer_dir", "source_store_dir"))
    output, index_dir = paths["output_dir"], paths["index_dir"]
    encoding, checks = config["encoding"], config["checks"]
    if config["windows"]["context_length"] != 128:
        raise ValueError("V4 context is fixed to 128")
    for key in ("workers", "batch_size", "chunk_mib"):
        if type(encoding[key]) is not int or encoding[key] < 1:
            raise ValueError(f"Invalid encoding resource: {key}")
    cm, cs, integrity = (read_json(corpus / name) for name in ("manifest.json", "stats.json", "integrity-check.json"))
    tm, ts = (read_json(tokenizer / name) for name in ("manifest.json", "stats.json"))
    experiment = read_json(paths["tokenizer_experiment_dir"] / "manifest.json")
    sm = read_json(source_store / "manifest.json")
    if any(m.get("status") != "complete" for m in (cm, tm, experiment, sm)) or integrity.get("status") != "passed":
        raise ValueError("Complete frozen input manifests required")
    model = tokenizer / "tokenizer.model"
    if spm.__version__ != tm["sentencepiece_version"] or file_sha(model) != tm["model_sha256"]:
        raise ValueError("Tokenizer hash/version changed")
    if Path(experiment["primary_model"]).resolve() != model.resolve():
        raise ValueError("Use the selected V4 8K model")
    kana = tm["trainer_options"]["required_chars"]
    verify_model(model, 8192, kana)
    corpus_sha = file_sha(corpus / "manifest.json")
    if corpus_sha != integrity["manifest_sha256"] or corpus_sha != recorded_hash(sm, "manifest.json", corpus.name):
        raise ValueError("Source provenance belongs to another corpus")
    expected_hashes = {"train": experiment["sample"]["train_sha256"], "validation": experiment["validation_sha256"]}
    for split in SPLITS:
        if (expected_hashes[split] != integrity["exports"][f"{split}.txt"]["sha256"]
                or expected_hashes[split] != recorded_hash(sm, f"{split}.txt")
                or any(sm["splits"][split][key] != cs["splits"][split][key] for key in ("sentences", "characters", "primary_sources"))):
            raise ValueError("Source and tokenizer corpus identity mismatch")
    plan = {"vocab_size": 8192, "context_length": 128, "available_splits": list(SPLITS),
            "sentences": {s: cs["splits"][s]["sentences"] for s in SPLITS},
            "paths": {k: str(v) for k, v in paths.items()}, "encoding": encoding,
            "test_and_blind_access": False, "model_training_started": False, "dry_run": dry_run}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return plan
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Output is nonempty; choose a new path or --resume")
    if resume and not output.exists():
        raise FileNotFoundError("Cannot resume an absent token store")
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        if (output / "manifest.json").exists():
            complete = read_json(output / "manifest.json")
            if complete.get("status") == "complete":
                if complete["tokenizer_model_sha256"] != tm["model_sha256"]:
                    raise ValueError("Completed output belongs to another tokenizer")
                for name, digest in complete["output_sha256"].items():
                    if file_sha(output / name) != digest:
                        raise ValueError(f"Completed output checksum mismatch: {name}")
                return prepare_v4_indexes(output, index_dir, checks)
        started = time.monotonic()
        with RunMonitor(output, ROOT / "runs/tokenizer/corpus-v4-8k-encode" / str(time.time_ns()), True) as monitor:
            monitor.update("checking_inputs")
            inputs = [config_path, model, tokenizer / "manifest.json", tokenizer / "stats.json",
                      paths["tokenizer_experiment_dir"] / "manifest.json", source_store / "manifest.json",
                      corpus / "manifest.json", corpus / "stats.json", corpus / "integrity-check.json"]
            inputs += [source_store / f"{split}.sources.bin" for split in SPLITS]
            inputs += [Path(__file__), Path(__file__).with_name("encode_parallel.py"), Path(__file__).with_name("encode.py"),
                       Path(__file__).with_name("monitor.py"), Path(__file__).with_name("v4.py"),
                       ROOT / "src/vimeml/training/data.py", ROOT / "src/vimeml/training/check_data.py"]
            inputs += [corpus / f"{split}.txt" for split in SPLITS]
            metadata = {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in inputs}
            hashes = {str(p): file_sha(p) for p in inputs if p.suffix != ".txt"}
            jobs, scans = [], {}
            for split in SPLITS:
                source_path = source_store / f"{split}.sources.bin"
                if (source_path.stat().st_size != cs["splits"][split]["sentences"]
                        or hashes[str(source_path)] != sm["output_sha256"][source_path.name]):
                    raise ValueError("Frozen source index changed")
                selected, scans[split] = scan_txt(corpus / f"{split}.txt", split,
                    encoding["chunk_mib"] * 1024 * 1024, cs["splits"][split],
                    lambda done, total: monitor.update("checking_inputs", preflight_completed_sentences=done, preflight_total_sentences=total))
                hashes[str(corpus / f"{split}.txt")] = scans[split]["sha256"]
                if scans[split]["sha256"] != expected_hashes[split]:
                    raise ValueError(f"Actual TXT checksum changed: {split}")
                for i, job in enumerate(selected):
                    job.update(name=f"{split}-{i:05d}", source_input=str(source_path))
                jobs.extend(selected)
            signature = hashlib.sha256(json.dumps({"hashes": hashes, "encoding": encoding,
                "splits": list(SPLITS)}, sort_keys=True).encode()).hexdigest()
            parts = output / ".parts"
            parts.mkdir(exist_ok=True)
            for job in jobs:
                job.update(model=str(model), parts=str(parts), batch_size=encoding["batch_size"],
                           signature=signature, source_ids=sm["source_ids"], verification="sha256")
            dump_json(output / "encoding-plan.json", {"signature": signature, "plan": plan, "scans": scans})
            results = collect(jobs, encoding["workers"], monitor)
            exported = {}
            for split in SPLITS:
                selected = [job for job in jobs if job["split"] == split]
                exported[split] = merge_txt_split(output, split, selected, results, cs["splits"][split], monitor)
            # Full validation statistics must match tokenizer-stage measurements.
            for key, other in (("content_tokens", "tokens_without_special_tokens"), ("stored_tokens", "tokens_with_bos_eos_per_sentence"),
                               ("byte_fallback_tokens", "byte_fallback_tokens")):
                if exported["validation"][key] != ts["splits"]["validation"][other]:
                    raise ValueError("Full validation token counts changed since tokenizer acceptance")
            monitor.update("verifying_inputs")
            if any(file_sha(p) != hashes[str(p)] for p in inputs):
                raise ValueError("Inputs changed during encoding")
            dump_json(output / "stats.json", {"actual_vocab_size": 8192, "splits": exported})
            artifacts = [output / f"{s}.{suffix}" for s in SPLITS for suffix in ("tokens.bin", "offsets.bin", "rows.bin", "sources.bin")]
            artifacts.append(output / "stats.json")
            output_hashes = {p.name: file_sha(p) for p in artifacts}
            for split in SPLITS:
                if output_hashes[f"{split}.sources.bin"] != sm["output_sha256"][f"{split}.sources.bin"]:
                    raise ValueError("Source rows were reordered during encoding")
            dump_json(output / "part-audit.json", {"signature": signature, "parts": [results[job["name"]] for job in jobs]})
            dump_json(output / "manifest.json", {"status": "complete", "format": FORMAT, "vocab_size": 8192,
                "token_dtype": "uint16_le", "offset_dtype": "uint64_le", "offset_unit": "tokens",
                "special_ids": tm["special_ids"], "reserved_marker_ids": tm["marker_ids"],
                "sentencepiece_version": spm.__version__, "tokenizer_model_sha256": tm["model_sha256"],
                "tokenizer_dir": str(tokenizer), "corpus_dir": str(corpus), "available_splits": list(SPLITS),
                "splits": exported, "source_ids": sm["source_ids"], "input_sha256": hashes,
                "output_sha256": output_hashes, "input_metadata": metadata, "signature": signature,
                "sequence_policy": "BOS + complete sentence + EOS; no truncation, concatenation or task markers.",
                "row_offset_source": "corpus_txt", "row_offset_dtype": "uint64_le",
                "provenance_policy": "rows.bin stores uint64 byte offsets into the corresponding frozen TXT, not the removed JSONL. sources.bin preserves verified original primary source IDs and row order.",
                "input_verification": "Full SHA256 before/after; every encoded text roundtrip; all sequence IDs checked; merged offsets/sizes/counts checked.",
                "test_and_blind_access": False, "corpus_ready_for_lm_training": cm.get("ready_for_lm_training", False),
                "corpus_quality_mode": cm.get("quality_mode"), "elapsed_seconds": time.monotonic() - started,
                "purpose": "V4 fixed 8K base LM data; capacity redistribution experiment, not vocabulary speed optimization."})
            # Delete only these verified temporary parts inside this output.
            if parts.resolve().parent != output.resolve():
                raise ValueError("Unexpected temporary part path")
            for job in jobs:
                for path in part_paths(job).values():
                    if path.resolve().parent != parts.resolve():
                        raise ValueError("Unexpected temporary file path")
                    path.unlink()
            parts.rmdir()
            monitor.update("complete")
    index, report = prepare_v4_indexes(output, index_dir, checks)
    print(json.dumps({"tokens": exported, "windows": index["splits"], "loader_checks": report}, ensure_ascii=False, indent=2), flush=True)
    return index, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data-v4.toml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.config, args.resume, args.dry_run)


if __name__ == "__main__":
    main()
