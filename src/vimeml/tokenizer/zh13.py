"""vimeml-tiny-zh-v1 tokenizers: sample the frozen export, train Unigram, verify.

Paragraphs are the lines of a document after strip(); empty lines are dropped. The
sample is taken once, written to train-sample.txt and read whole by every vocabulary.
Web and review train files are grouped back by stage-a shard; every shard gets the same
quota (see sample_shard), so the accepted review documents keep their natural share.
"""
import hashlib
import json
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPECIAL = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
META_SPACE = "▁"
HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f]")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    temporary.replace(path)


def paragraphs(text, max_bytes=None):
    for line in text.split("\n"):
        line = line.strip()
        if line and (max_bytes is None or len(line.encode("utf-8")) <= max_bytes):
            yield line


def export_identity(export, frozen=True):
    """Frozen export: manifest.json and SHA256SUMS must exist; returns their hashes."""
    export = Path(export)
    if not frozen:
        return {"frozen": False}
    for name in ("manifest.json", "SHA256SUMS"):
        if not (export / name).exists():
            raise ValueError(f"{export} is not frozen: {name} missing (run zh_export.py finish)")
    sums = {}
    for line in (export / "SHA256SUMS").read_text("utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*")] = digest
    return {"frozen": True, "manifest_sha256": sha256(export / "manifest.json"),
            "sha256sums_sha256": sha256(export / "SHA256SUMS"), "sums": sums}


def web_train_files(export):
    files = sorted((Path(export) / "train").glob("web-*.parquet")) + sorted((Path(export) / "train").glob("review-*.parquet"))
    if not files:
        raise ValueError("No web/review train files")
    return files


def check_files(export, files, identity, verify):
    """Every input must be listed in SHA256SUMS; with verify, hash it too."""
    if not identity["frozen"]:
        return
    for path in files:
        name = Path(path).relative_to(export).as_posix()
        if name not in identity["sums"]:
            raise ValueError(f"{name} is not in SHA256SUMS")
        if verify and sha256(path) != identity["sums"][name]:
            raise ValueError(f"{name} differs from SHA256SUMS")


def shard_of(path):
    """web-000_00009-001.parquet and review-000_00009-000.parquet both come from shard 000_00009."""
    return Path(path).name.split("-")[1]


def sample_shard(rng, handles, quota, max_bytes):
    """Paragraphs of one stage-a shard, its web and review files pooled back together.

    Documents of the whole shard are put in one seeded random order and taken from the
    front, quota documents per round, until their paragraphs reach the quota; then quota
    paragraphs are drawn from them. Every document has the same chance however its file
    was cut into row groups (whole row groups would favor the large web groups).
    """
    groups = [(kind, handle, g, handle.metadata.row_group(g).num_rows)
              for kind, handle in handles for g in range(handle.metadata.num_row_groups)]
    starts = [0]
    for group in groups:
        starts.append(starts[-1] + group[3])
    order = list(range(starts[-1]))
    rng.shuffle(order)
    pool, taken = [], 0
    while taken < len(order) and len(pool) < quota:
        chosen = sorted(order[taken:taken + quota])
        taken += quota
        position = 0
        for index, (kind, handle, g, rows) in enumerate(groups):
            lo, hi = starts[index], starts[index] + rows
            wanted = []
            while position < len(chosen) and chosen[position] < hi:
                wanted.append(chosen[position] - lo)
                position += 1
            if not wanted:
                continue
            texts = handle.read_row_group(g, columns=["text"])["text"].to_pylist()
            pool += [(kind, line) for row in wanted for line in paragraphs(texts[row], max_bytes)]
    return rng.sample(pool, min(quota, len(pool)))


def sample(config, export, out):
    import pyarrow.parquet as pq
    rng = random.Random(config["seed"])
    files = web_train_files(export)
    shards = {}
    for path in files:
        shards.setdefault(shard_of(path), []).append(path)
    per_shard = config["web_lines"] // len(shards) + 1
    web, by_shard = [], {}
    for shard in sorted(shards):
        handles = [(p.name.split("-")[0], pq.ParquetFile(p)) for p in shards[shard]]
        picked = sample_shard(rng, handles, per_shard, config["max_line_bytes"])
        by_shard[shard] = {kind: sum(k == kind for k, _ in picked) for kind in ("web", "review")}
        web += picked
    web = [line for _, line in web[:config["web_lines"]]]
    chat_path = Path(export) / "train/chat-lccc-train.parquet"
    chat = pq.read_table(chat_path, columns=["text"])["text"].to_pylist()
    chat = [c for c in rng.sample(chat, min(config["chat_lines"], len(chat))) if c.strip() and "\n" not in c]
    lines = web + chat
    rng.shuffle(lines)
    path = out / "train-sample.txt"
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path, {"web_lines": len(web), "chat_lines": len(chat), "per_shard_quota": per_shard,
                  "review_lines": sum(v["review"] for v in by_shard.values()), "by_shard": by_shard,
                  "characters": sum(map(len, lines)),
                  "han": sum(len(HAN.findall(x)) for x in lines), "seed": config["seed"],
                  "files": [p.relative_to(export).as_posix() for p in files] + [chat_path.relative_to(export).as_posix()],
                  "sha256": sha256(path)}


def train_one(config, sample_path, prefix, vocab):
    import sentencepiece as spm
    spm.SentencePieceTrainer.train(
        input=str(sample_path), model_prefix=str(prefix), model_type="unigram", vocab_size=vocab,
        character_coverage=config["character_coverage"], byte_fallback=True,
        normalization_rule_name="identity", remove_extra_whitespaces=False, add_dummy_prefix=False,
        split_digits=False, hard_vocab_limit=True, pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        control_symbols=config["control_symbols"], user_defined_symbols=config["user_defined_symbols"],
        input_sentence_size=0, shuffle_input_sentence=False, max_sentence_length=config["max_line_bytes"],
        seed_sentencepiece_size=1_000_000, num_threads=config["num_threads"], minloglevel=2)


def verify(model_path, config, validation_texts):
    """Special/reserved IDs, newline and marker behavior, exact roundtrip of validation."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    found = {name: getattr(sp, f"{name}_id")() for name in SPECIAL}
    if found != SPECIAL:
        raise ValueError(f"Special IDs {found}")
    reserved = {}
    next_id = 4
    for symbol in config["control_symbols"]:
        if sp.piece_to_id(symbol) != next_id or not sp.is_control(next_id):
            raise ValueError(f"Control symbol {symbol!r} is not ID {next_id}")
        reserved[symbol] = next_id
        next_id += 1
    for symbol in config["user_defined_symbols"]:
        if sp.piece_to_id(symbol) != next_id or sp.encode(symbol) != [next_id]:
            raise ValueError(f"User symbol {symbol!r} is not the single ID {next_id}")
        reserved[symbol] = next_id
        next_id += 1
    for symbol in config["control_symbols"]:
        ids = sp.encode(symbol)
        if any(i < next_id for i in ids) or sp.decode(ids) != symbol:
            raise ValueError(f"Literal {symbol!r} must stay ordinary text")
    bytes_ = sum(sp.is_byte(i) for i in range(sp.get_piece_size()))
    if sp.get_piece_size() != config["vocab"] or bytes_ != 256:
        raise ValueError("Vocabulary size or byte pieces differ")
    checked = mismatched = meta_space = 0
    for text in validation_texts:
        for piece in (text, *paragraphs(text)):
            if META_SPACE in piece:
                meta_space += 1  # SentencePiece decodes U+2581 as a space; the encoder drops these
                continue
            checked += 1
            ids = sp.encode(piece)
            # IDs below the newline symbol are special or control: never produced from text.
            if sp.decode(ids) != piece or any(i < 4 + len(config["control_symbols"]) for i in ids):
                mismatched += 1
    if mismatched:
        raise ValueError(f"{mismatched}/{checked} validation texts fail roundtrip or contain reserved IDs")
    return {"special_ids": SPECIAL, "reserved_ids": reserved, "byte_pieces": bytes_,
            "vocab_size": sp.get_piece_size(), "roundtrip_checked": checked,
            "roundtrip_skipped_meta_space": meta_space}


def run(config, export, out, frozen=True, verify_inputs=True):
    import pyarrow.parquet as pq
    export, out = Path(export), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; a new tokenizer needs a new output directory")
    identity = export_identity(export, frozen)
    files = web_train_files(export) + [export / "train/chat-lccc-train.parquet"]
    validation = sorted((export / "validation").glob("*.parquet"))
    check_files(export, files + validation, identity, verify_inputs)
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    sample_path, sample_info = sample(config, export, out)
    print(json.dumps({"sample": {k: v for k, v in sample_info.items() if k != "files"}}), flush=True)
    texts = []
    for path in validation:
        texts += pq.read_table(path, columns=["text"])["text"].to_pylist()
    models = {}
    for vocab in config["vocab_sizes"]:
        folder = out / f"zh-unigram-{vocab}"
        folder.mkdir()
        started = time.time()
        train_one(config, sample_path, folder / "tokenizer", vocab)
        checks = verify(folder / "tokenizer.model", {**config, "vocab": vocab}, texts)
        models[str(vocab)] = {"directory": folder.relative_to(out).as_posix(),
                              "tokenizer_model_sha256": sha256(folder / "tokenizer.model"),
                              "train_seconds": round(time.time() - started, 1), **checks}
        print(json.dumps({"vocab": vocab, **models[str(vocab)]}, ensure_ascii=False), flush=True)
    manifest = {"format": "vimeml_zh13_tokenizers_v1", "status": "complete", "config": config,
                "export": {k: v for k, v in identity.items() if k != "sums"},
                "sample": sample_info, "models": models,
                "validation_files": [p.relative_to(export).as_posix() for p in validation],
                "test_or_blind_read": False, "seconds": round(time.time() - began, 1)}
    write_json(out / "manifest.json", manifest)
    return manifest
