"""Clean pinned V3 inputs, sample documents and write existing V2 token/window formats."""
import argparse
import hashlib
import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm
from vimeml.training.data import prepare_indexes, write_json
from vimeml.tokenizer.encode_parallel import output_lock

SPLITS = ("train", "validation", "test")
CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
NOISE = re.compile(r"^(?:https?://\S+|ログイン|ログアウト|前のページ|次のページ|広告|Cookie設定)$")
MASK = re.compile(r"(?:\[PERSON\]|\[NAME\]|\[MASK\]|<PERSON>|<NAME>)", re.I)


def normalized(text):
    return CONTROLS.sub(" ", unicodedata.normalize("NFC", text)).strip().lstrip("\ufeff")


def digest(text):
    # Content identity for deduplication, not repeated file-integrity SHA256 scans.
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def chunks(text, chat):
    text = normalized(text)
    if not text or "\ufffd" in text or MASK.search(text):
        return []
    if chat:
        return [text]  # Preserve short turns, punctuation, spaces and internal newlines.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = " ".join(line for line in lines if not NOISE.fullmatch(line))
    if len(text) < 24 or len(JAPANESE.findall(text)) / len(text) < .20:
        return []
    pieces, pending = [], ""
    for sentence in re.findall(r"[^。！？!?]+[。！？!?]*|[。！？!?]+", text):
        if pending and len(pending) + len(sentence) > 1024:
            pieces.append(pending)
            pending = ""
        while len(sentence) > 1024:
            pieces.append(sentence[:1024])
            sentence = sentence[1024:]
        pending += sentence
    if pending:
        pieces.append(pending)
    return [piece.strip() for piece in pieces if piece.strip()]


def ingest(db, sp, texts, identity, source, pool, seed, encoded=None):
    doc_id = digest(identity)
    if db.execute("SELECT 1 FROM docs WHERE id=?", (doc_id,)).fetchone():
        return
    number = int(digest(f"split:{seed}:{doc_id}"), 16) % 10000
    split = "train" if number < 9800 else "validation" if number < 9900 else "test"
    priority = int(digest(f"sample:{seed}:{doc_id}")[:15], 16)
    if encoded is None:
        encoded = sp.encode(texts, out_type=int, num_threads=1)
    pairs = 0
    reserved = {sp.bos_id(), sp.eos_id(), sp.pad_id(), sp.unk_id()}
    for ordinal, (text, ids) in enumerate(zip(texts, encoded, strict=True)):
        if not ids or any(token in reserved for token in ids):
            continue
        key = digest(text)
        seen = db.execute("SELECT split,pool FROM seen WHERE sig=?", (key,)).fetchone()
        if seen and (seen[0] != split or pool == "web"):
            continue  # No exact sequence crosses splits; preserve repeated chat within a split.
        if db.execute("SELECT 1 FROM blocked WHERE sig=?", (key,)).fetchone():
            continue
        db.execute("INSERT OR IGNORE INTO seen VALUES (?,?,?)", (key, split, pool))
        tokens = np.asarray([sp.bos_id(), *ids, sp.eos_id()], dtype="<u2").tobytes()
        n = len(ids) + 1
        db.execute("INSERT INTO seq(doc,ordinal,split,source,pool,text,tokens,pairs) VALUES (?,?,?,?,?,?,?,?)",
                   (doc_id, ordinal, split, source, pool, text, tokens, n))
        pairs += n
    db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?)", (doc_id, split, pool, source, priority, pairs))


def build_store(db, root, tokenizer, sources, selection, pool):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        return
    tm = json.loads((tokenizer / "manifest.json").read_text(encoding="utf-8"))
    stats = {}
    for split in SPLITS:
        query = "SELECT s.doc,s.ordinal,s.source,s.text,s.tokens FROM seq s JOIN selected d ON s.doc=d.id WHERE s.split=?"
        if pool == "chat":
            query += " AND s.pool='chat'"
        query += " ORDER BY s.id"
        handles = {suffix: (root / f"{split}.{suffix}").open("wb")
                   for suffix in ("tokens.bin", "offsets.bin", "sources.bin", "rows.bin", "jsonl")}
        count = total = characters = longest = 0
        source_counts = Counter()
        handles["offsets.bin"].write(np.asarray([0], dtype="<u8").tobytes())
        try:
            for doc, ordinal, source, text, tokens in db.execute(query, (split,)):
                position = handles["jsonl"].tell()
                handles["jsonl"].write((json.dumps({"doc": doc, "ordinal": ordinal, "source": source, "text": text},
                                                  ensure_ascii=False) + "\n").encode("utf-8"))
                length = len(tokens) // 2
                total += length
                count += 1
                characters += len(text)
                longest = max(longest, length)
                source_counts[source] += 1
                handles["tokens.bin"].write(tokens)
                handles["offsets.bin"].write(np.asarray([total], dtype="<u8").tobytes())
                handles["rows.bin"].write(np.asarray([position], dtype="<u8").tobytes())
                handles["sources.bin"].write(bytes([sources[source]]))
        finally:
            for handle in handles.values():
                handle.close()
        if not count:
            raise ValueError(f"Empty {pool}/{split}")
        stats[split] = {"sentences": count, "stored_tokens": total, "content_tokens": total - 2 * count,
                        "prediction_pairs": total - count, "characters": characters,
                        "max_sequence_tokens": longest, "primary_sources": dict(source_counts)}
    write_json(root / "stats.json", {"splits": stats})
    write_json(root / "manifest.json", {
        "status": "complete", "format": "vimeml_sentence_tokens_v1", "token_dtype": "uint16_le",
        "offset_dtype": "uint64_le", "offset_unit": "tokens", "vocab_size": 16384,
        "special_ids": tm["special_ids"], "tokenizer_dir": str(tokenizer),
        "tokenizer_model_sha256": tm["model_sha256"], "source_ids": sources, "splits": stats,
        "sequence_policy": "BOS + web paragraph chunk or independent chat turn + EOS; document/dialogue grouped splits",
        "corpus_dir": str(root), "selection": selection, "pool": pool,
        "input_verification": {"mode": "metadata", "full_sha256": False},
        "output_sha256": {}, "corpus_ready_for_lm_training": True,
        "limitations": "Exact chunk overlap exclusion only; no new full near-duplicate or substring contamination audit."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/token-data/corpus-v3-half")
    parser.add_argument("--web-train-tokens", type=int, default=500_000_000, help="Prediction-pair budget; chat not included")
    parser.add_argument("--parquets", type=int, choices=(8, 16), default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.web_train_tokens < 1:
        parser.error("--web-train-tokens must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = ROOT / "artifacts/tokenizers/ja-unigram-16k-v2"
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer / "tokenizer.model"))
    if sp.get_piece_size() != 16384:
        raise ValueError("Expected frozen 16K V2 tokenizer")
    tm = json.loads((tokenizer / "manifest.json").read_text(encoding="utf-8"))
    if any(getattr(sp, name + "_id")() != value for name, value in tm["special_ids"].items()):
        raise ValueError("Tokenizer special IDs differ from its manifest")
    download = json.loads((ROOT / "datasets/v3/download-manifest.json").read_text(encoding="utf-8"))
    files = download["jpnmix"]["files"][::2 if args.parquets == 8 else 1]
    inputs = [ROOT / "datasets/v3/JpnMix" / entry["path"] for entry in files]
    chats = [("real-persona-chat", ROOT / "datasets/v3/real-persona-chat/real_persona_chat/dialogues"),
             ("mrmp-chat", ROOT / "datasets/v3/mrmp-chat/multi_relational_multi_party_chat_corpus/dialogues")]
    metadata_paths = inputs + [ROOT / "datasets/v3/download-manifest.json", tokenizer / "tokenizer.model",
                              tokenizer / "manifest.json", Path(__file__).resolve()]
    metadata_paths += [path for _, folder in chats for path in sorted(folder.rglob("*.json"))]
    metadata_paths += [path for path in [ROOT / "outputs/corpus-fast-v1/validation.txt"] +
                      [ROOT / "artifacts/benchmarks" / folder / "evaluation_items.json" for folder in
                       ("ajimee-jwtd-v2-v1", "ime-dev-v2", "ime-expanded-v21-candidates-v1/development")]
                      if path.exists()]
    request = {"parquets": args.parquets, "seed": args.seed, "web_train_tokens": args.web_train_tokens,
               "files": {str(path.relative_to(ROOT)): [path.stat().st_size, path.stat().st_mtime_ns]
                         for path in metadata_paths}}
    with output_lock(output):
        request_path = output / "request.json"
        if request_path.exists() and json.loads(request_path.read_text(encoding="utf-8")) != request:
            raise ValueError("Inputs/settings changed; use a new --output")
        write_json(request_path, request)
        db = sqlite3.connect(output / "staging.sqlite")
        db.executescript("""
          PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
          CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY,split TEXT,pool TEXT,source TEXT,priority INT,pairs INT);
          CREATE TABLE IF NOT EXISTS seq(id INTEGER PRIMARY KEY,doc TEXT,ordinal INT,split TEXT,source TEXT,pool TEXT,text TEXT,tokens BLOB,pairs INT);
          CREATE TABLE IF NOT EXISTS seen(sig TEXT PRIMARY KEY,split TEXT,pool TEXT);
          CREATE TABLE IF NOT EXISTS blocked(sig TEXT PRIMARY KEY);
          CREATE TABLE IF NOT EXISTS done(input TEXT PRIMARY KEY);
          CREATE TABLE IF NOT EXISTS selected(id TEXT PRIMARY KEY);
        """)
        # Exclude exact held-out V2 sequences and sufficiently long frozen IME references.
        old_val = ROOT / "outputs/corpus-fast-v1/validation.txt"
        if not db.execute("SELECT 1 FROM done WHERE input='exclusions'").fetchone():
            if old_val.exists():
                with old_val.open(encoding="utf-8") as stream:
                    db.executemany("INSERT OR IGNORE INTO blocked VALUES (?)", ((digest(normalized(t)),) for t in stream))
            for folder in ("ajimee-jwtd-v2-v1", "ime-dev-v2", "ime-expanded-v21-candidates-v1/development"):
                path = ROOT / "artifacts/benchmarks" / folder / "evaluation_items.json"
                if not path.exists():
                    continue
                for case in json.loads(path.read_text(encoding="utf-8")):
                    for text in case.get("expected_output", []):
                        if len(text) >= 8:
                            db.execute("INSERT OR IGNORE INTO blocked VALUES (?)", (digest(normalized(text)),))
            db.execute("INSERT INTO done VALUES ('exclusions')")
            db.commit()
        jobs = [(name, folder) for name, folder in chats] + [(str(path.relative_to(ROOT)), path) for path in inputs]
        for name, path in jobs:
            if db.execute("SELECT 1 FROM done WHERE input=?", (name,)).fetchone():
                continue
            print(f"Cleaning/encoding {name}", flush=True)
            write_json(output / "progress.json", {"status": "cleaning_encoding", "input": name})
            if path.is_dir():
                for position, file in enumerate(sorted(path.rglob("*.json")), 1):
                    doc = json.loads(file.read_text(encoding="utf-8"))
                    texts = [part for item in doc["utterances"] for part in chunks(item["text"], True)]
                    ingest(db, sp, texts, f"{name}:{file.stem}", name, "chat", args.seed)
                    if position % 2000 == 0:
                        print(f"  {position:,} dialogues", flush=True)
            else:
                position = 0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text", "source"]):
                    documents = []
                    for row in batch.to_pylist():
                        texts = chunks(row["text"], False)
                        if texts:
                            documents.append((texts, row["source"]))
                    encoded = sp.encode([text for texts, _ in documents for text in texts], out_type=int, num_threads=8)
                    offset = 0
                    for texts, source in documents:
                        ingest(db, sp, texts, "\n".join(texts), "jpnmix:" + source, "web", args.seed,
                               encoded[offset:offset + len(texts)])
                        offset += len(texts)
                    position += batch.num_rows
                    if position % 10240 == 0:
                        print(f"  {position:,} documents", flush=True)
                        write_json(output / "progress.json", {"status": "cleaning_encoding", "input": name,
                                                              "processed_documents": position})
            db.execute("INSERT INTO done VALUES (?)", (name,))
            db.commit()  # Interrupted current input rolls back; completed inputs are reused.
        db.execute("DELETE FROM selected")
        db.execute("INSERT INTO selected SELECT id FROM docs WHERE pool='chat' OR split!='train'")
        used = 0
        for doc, pairs in db.execute("SELECT id,pairs FROM docs WHERE pool='web' AND split='train' ORDER BY priority,id").fetchall():
            if pairs and used + pairs <= args.web_train_tokens:
                db.execute("INSERT INTO selected VALUES (?)", (doc,))
                used += pairs
        db.commit()
        sources = {name: index for index, (name,) in enumerate(db.execute("SELECT DISTINCT source FROM seq ORDER BY source"))}
        selection = {"parquets": [entry["path"] for entry in files], "seed": args.seed,
                     "web_train_prediction_pairs": used, "budget": args.web_train_tokens,
                     "split": "98/1/1 by document or complete dialogue", "source_versions": download}
        for pool in ("mixed", "chat"):
            print(f"Exporting {pool}", flush=True)
            write_json(output / "progress.json", {"status": "exporting", "pool": pool})
            build_store(db, output / pool, tokenizer, sources, selection, pool)
            prepare_indexes(output / pool, output / "indexes" / pool, 128, verification="metadata")
        db.close()
        write_json(output / "progress.json", {"status": "complete", "web_train_prediction_pairs": used})
        print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
