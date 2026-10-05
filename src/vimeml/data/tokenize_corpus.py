"""Export complete BOS/text/EOS sentences as uint16 tokens with uint64 offsets."""

import argparse
import hashlib
import json
import sys
from array import array
from collections import Counter
from itertools import islice
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("train", "validation", "test")


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


def write_numbers(stream, code, values):
    numbers = array(code, values)
    if numbers.itemsize != {"H": 2, "Q": 8}[code]:
        raise ValueError("This Python platform has unsupported array element sizes.")
    if sys.byteorder != "little":
        numbers.byteswap()
    stream.write(numbers.tobytes())


def export_split(corpus, output, split, processor, expected, batch_size):
    rows_path = corpus / f"{split}.jsonl"
    text_path = corpus / f"{split}.txt"
    count = total = content_tokens = characters = 0
    max_length = 0
    sources = Counter()
    bos, eos, unk, pad = processor.bos_id(), processor.eos_id(), processor.unk_id(), processor.pad_id()
    with rows_path.open(encoding="utf-8") as rows, text_path.open(encoding="utf-8") as lines, \
            (output / f"{split}.tokens.bin").open("wb") as tokens, \
            (output / f"{split}.offsets.bin").open("wb") as offsets, \
            (output / f"{split}.provenance.jsonl").open("w", encoding="utf-8", newline="\n") as provenance:
        write_numbers(offsets, "Q", [0])
        while batch_lines := list(islice(rows, batch_size)):
            batch = [json.loads(line) for line in batch_lines]
            texts = [row["text"] for row in batch]
            for text in texts:
                if not text or "\n" in text or "\r" in text or lines.readline().removesuffix("\n") != text:
                    raise ValueError(f"{split}: TXT/JSONL alignment or text boundary error near sentence {count}.")
            encoded = processor.encode(texts, out_type=int)
            decoded = processor.decode(encoded)
            batch_tokens, batch_ends = [], []
            for row, ids, restored in zip(batch, encoded, decoded, strict=True):
                text = row["text"]
                if restored != text or any(token in {unk, pad, bos, eos} for token in ids):
                    raise ValueError(f"{split}: roundtrip or unexpected special token at sentence {count}.")
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["text_hash"]:
                    raise ValueError(f"{split}: text hash mismatch at sentence {count}.")
                sequence = [bos, *ids, eos]
                batch_tokens.extend(sequence)
                total += len(sequence)
                batch_ends.append(total)
                content_tokens += len(ids)
                characters += len(text)
                max_length = max(max_length, len(sequence))
                sources[row["source"]] += 1
                provenance.write(json.dumps({
                    "index": count, "corpus_row": count,
                    "source": row["source"], "sources": row["sources"],
                    "doc_id": row["doc_id"], "doc_hash": row["doc_hash"],
                    "text_hash": row["text_hash"],
                }, ensure_ascii=False) + "\n")
                count += 1
            write_numbers(tokens, "H", batch_tokens)
            write_numbers(offsets, "Q", batch_ends)
        if lines.readline():
            raise ValueError(f"{split}: TXT contains more rows than JSONL.")
    if (
        count != expected["sentences"] or characters != expected["characters"]
        or content_tokens != expected["tokens_without_special_tokens"]
        or total != expected["tokens_with_bos_eos_per_sentence"]
    ):
        raise ValueError(f"{split}: exported counts differ from tokenizer statistics.")
    files = {name: (output / f"{split}.{name}").stat().st_size
             for name in ("tokens.bin", "offsets.bin", "provenance.jsonl")}
    if files["tokens.bin"] != total * 2 or files["offsets.bin"] != (count + 1) * 8:
        raise ValueError(f"{split}: binary size mismatch.")
    return {"sentences": count, "content_tokens": content_tokens, "stored_tokens": total,
            "prediction_pairs": total - count, "max_sequence_tokens": max_length,
            "primary_sources": dict(sources), "file_bytes": files}


def expected_input_hash(manifest, basename):
    matches = [digest for path, digest in manifest["input_sha256"].items() if Path(path).name == basename]
    if len(matches) != 1:
        raise ValueError(f"Cannot find a unique tokenizer input hash for {basename}.")
    return matches[0]


def run(corpus, tokenizer, output, batch_size=256):
    import sentencepiece as spm
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be absent or empty: {output}")
    corpus_manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    tokenizer_manifest = json.loads((tokenizer / "manifest.json").read_text(encoding="utf-8"))
    if corpus_manifest.get("status") != "complete" or tokenizer_manifest.get("status") != "complete":
        raise ValueError("Corpus and tokenizer must both have complete manifests.")
    if spm.__version__ != tokenizer_manifest["sentencepiece_version"]:
        raise ValueError("SentencePiece version differs from the tokenizer build.")
    model = tokenizer / "tokenizer.model"
    if file_sha(model) != tokenizer_manifest["model_sha256"]:
        raise ValueError("Tokenizer model hash mismatch.")
    processor = spm.SentencePieceProcessor(model_file=str(model))
    vocab_size = processor.get_piece_size()
    if not 0 < vocab_size <= 65536:
        raise ValueError("uint16 export supports at most 65,536 vocabulary entries.")
    special_ids = tokenizer_manifest["special_ids"]
    if any(getattr(processor, f"{name}_id")() != value for name, value in special_ids.items()) or any(value < 0 for value in special_ids.values()):
        raise ValueError("Tokenizer special token IDs are inconsistent or disabled.")
    stats = json.loads((tokenizer / "stats.json").read_text(encoding="utf-8"))
    if stats["actual_vocab_size"] != vocab_size:
        raise ValueError("Tokenizer vocabulary size differs from its statistics.")
    paths = [corpus / f"{split}.{extension}" for split in SPLITS for extension in ("txt", "jsonl")]
    paths += [corpus / "manifest.json", corpus / "stats.json", model,
              tokenizer / "manifest.json", tokenizer / "stats.json"]
    input_hashes = {str(path): file_sha(path) for path in paths}
    for split in SPLITS:
        if input_hashes[str(corpus / f"{split}.txt")] != expected_input_hash(tokenizer_manifest, f"{split}.txt"):
            raise ValueError(f"{split}: corpus text differs from the tokenizer's measured input.")
    if input_hashes[str(corpus / "manifest.json")] != expected_input_hash(tokenizer_manifest, "manifest.json"):
        raise ValueError("Corpus build manifest differs from the tokenizer's recorded corpus.")
    output.mkdir(parents=True, exist_ok=True)
    exported = {}
    for split in SPLITS:
        print(f"Exporting {split}...", flush=True)
        exported[split] = export_split(corpus, output, split, processor, stats["splits"][split], batch_size)
    if any(file_sha(Path(path)) != digest for path, digest in input_hashes.items()):
        raise ValueError("Inputs changed during token export.")
    dump_json(output / "stats.json", {"splits": exported})
    artifact_hashes = {path.name: file_sha(path) for path in sorted(output.iterdir()) if path.is_file()}
    dump_json(output / "manifest.json", {
        "status": "complete", "format": "vimeml_sentence_tokens_v1", "vocab_size": vocab_size,
        "token_dtype": "uint16_le", "offset_dtype": "uint64_le", "offset_unit": "tokens",
        "special_ids": special_ids, "sentencepiece_version": spm.__version__,
        "sequence_policy": "Each indexed sequence is BOS + encoded sentence + EOS; full sentences are retained.",
        "context_policy": "No windowing, truncation or sentence concatenation; DataLoader chooses windows later.",
        "prediction_policy": "For a sequence s, x=s[:-1], y=s[1:]; each sequence has len(s)-1 prediction pairs.",
        "provenance_policy": "Zero-based exported index equals the row in the original corpus JSONL; metadata retained there.",
        "corpus_dir": str(corpus), "tokenizer_dir": str(tokenizer),
        "corpus_quality_mode": corpus_manifest.get("quality_mode"),
        "purpose": "pilot token data; quality recovery and near dedup pending",
        "splits": exported, "input_sha256": input_hashes, "output_sha256": artifact_hashes,
        "script_sha256": file_sha(Path(__file__).resolve()),
    })
    print(json.dumps({"splits": exported}, ensure_ascii=False, indent=2))
    print(f"Token data: {output}")
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "outputs/corpus-pilot")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/tokenizers/ja-unigram-16k-pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/token-data/corpus-pilot-16k")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    run(args.corpus.resolve(), args.tokenizer.resolve(), args.output.resolve(), args.batch_size)


if __name__ == "__main__":
    main()
