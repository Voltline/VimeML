"""ja-gen stage 3: encode colloquial paragraphs into a 12K [[data.extra]] token store.

Reads the output of `scripts/ja/corpus/ja_colloq.py export` (`train/*.parquet` and
`validation/*.parquet`, columns `id`, `text`) and builds a `vimeml_ja_kana_v1` store with
three kinds (`plain`, `span`, `sentence`) ready to mount under `[[data.extra]]` in
`vimeml.training.joint_ja_kana`.

Each paragraph is deterministically assigned by `doc_hash(label + "/role", id)`:
- `plain_fraction` (default 0.30) -> plain LM windows (`first = 1`, <=128 tokens)
- remaining -> reading sample (`span` or `sentence` drawn by `kind_weights`) using the
  exact D4 reading table and D5 user-typed reading rules of `scripts/ja/corpus/kana_samples.py`.
No paragraph is both plain text and a reading sample. Train plain lines identical to any
validation line are dropped. Parts are merged in deterministic job order.

    python -X utf8 scripts/ja/tokenizer/encode_ja_colloq.py --config configs/ja/data-ja-colloq-v1.toml --workers 16
    python -X utf8 scripts/ja/tokenizer/encode_ja_colloq.py --limit-jobs 1 --output /tmp/ja-colloq-smoke
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sys
import time
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/ja/corpus"))

import kana_samples as ks  # noqa: E402
from reading_pairs import load_table  # noqa: E402
from vimeml.posttraining.zh_pinyin import encode  # noqa: E402
from vimeml.posttraining.zh_pinyin_data import Part, merge  # noqa: E402
from vimeml.tokenizer.encode_zh13 import doc_hash  # noqa: E402
from vimeml.tokenizer.zh13 import META_SPACE, sha256, write_json  # noqa: E402

FORMAT = "vimeml_ja_kana_v1"
KINDS = ("plain", "span", "sentence")
SPLITS = ("train", "validation")
TWO64 = 2 ** 64

_state = {}


def line_key(text):
    return hashlib.sha256(text.encode("utf-8")).digest()[:8]


def encode_plain_windows(sp, part, stats, text, min_tokens=4):
    if META_SPACE in text:
        stats["plain_meta_space"] += 1
        return
    pieces = sp.encode(text)
    if sp.decode(pieces) != text:
        stats["plain_lossy"] += 1
        return
    ids = [2] + pieces + [3]
    stats["plain_lines"] += 1
    for start in range(0, len(ids), 128):
        window = ids[start:start + 128]
        if len(window) < min_tokens:
            stats["plain_short_windows"] += 1
            continue
        part.add("plain", window, 1)
        stats["plain_windows"] += 1
        stats["plain_tokens"] += len(window)


def encode_paragraph(sp, config, part, stats, key, raw_text, split, exclude_plain):
    """Deterministically route one colloquial paragraph to plain or span/sentence and encode."""
    text = ks.clean(raw_text)
    if len(text) < 4:
        stats["short_paragraphs"] += 1
        return
    if split == "train" and line_key(text) in exclude_plain:
        stats["validation_duplicates"] += 1
        return
    label = config["label"]
    plain_cut = int(float(config["plain_fraction"]) * TWO64)
    if doc_hash(label + "/role", key) < plain_cut:
        encode_plain_windows(sp, part, stats, text, min_tokens=int(config["plain_minimum_tokens"]))
        return

    rng = random.Random(doc_hash(label + "/sample", key))
    weights = config["kind_weights"]
    kind = rng.choices(list(weights), weights=list(weights.values()))[0]
    stats[f"{kind}_drawn"] += 1
    result = (ks.span_sample if kind == "span" else ks.sentence_sample)(config["sample"], rng, text, stats)
    if result is None:
        return
    context, reading, output = result
    encoded = encode(sp, context, reading, output)
    if encoded is None:
        stats[f"{kind}_skip_encode"] += 1
        return
    ids, first, removed = encoded
    part.add(kind, ids, first)
    r = ids.index(5)
    stats[f"{kind}_examples"] += 1
    stats[f"{kind}_tokens"] += len(ids)
    stats[f"{kind}_output_tokens"] += len(ids) - first - 1
    stats[f"{kind}_reading_tokens"] += first - r - 2
    stats[f"{kind}_context_tokens"] += r - 2
    stats[f"{kind}_reading_chars"] += len(reading)
    stats[f"{kind}_output_chars"] += len(output)
    stats[f"{kind}_context_trimmed"] += int(removed > 0)


def build_jobs(export_dir, row_groups_per_job=2, limit_jobs=0):
    import pyarrow.parquet as pq

    export_dir = Path(export_dir)
    out = []
    for split in SPLITS:
        split_jobs = []
        for path in sorted((export_dir / split).glob("*.parquet")):
            groups = pq.ParquetFile(path).metadata.num_row_groups
            for g in range(0, groups, row_groups_per_job):
                split_jobs.append({
                    "name": f"{split}-{path.stem}-{g:04d}",
                    "split": split,
                    "path": str(path),
                    "groups": [g, min(groups, g + row_groups_per_job)],
                })
        if limit_jobs:
            split_jobs = split_jobs[:limit_jobs]
        out.extend(split_jobs)
    return out


def validation_line_keys(export_dir):
    import pyarrow.parquet as pq

    keys = set()
    val_dir = Path(export_dir) / "validation"
    if not val_dir.is_dir():
        return keys
    for path in sorted(val_dir.glob("*.parquet")):
        pf = pq.ParquetFile(path)
        for g in range(pf.metadata.num_row_groups):
            col = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
            for raw in col:
                cleaned = ks.clean(raw)
                if len(cleaned) >= 4:
                    keys.add(line_key(cleaned))
    return keys


def init_worker(config, exclude):
    import sentencepiece as spm
    from vimeml.benchmarks.readings import DualReading

    ks._state["config"] = config
    ks._state["reader"] = DualReading()
    ks._state["table"] = load_table(config["table_path"])
    _state.update(
        config=config,
        exclude=exclude,
        sp=spm.SentencePieceProcessor(model_file=config["tokenizer_path"]),
    )


def run_job(job):
    import pyarrow.parquet as pq

    config, sp, exclude = _state["config"], _state["sp"], _state["exclude"]
    part, stats = Part(KINDS), Counter()
    pf = pq.ParquetFile(job["path"])
    split = job["split"]
    for g in range(*job["groups"]):
        table = pf.read_row_group(g, columns=["id", "text"]).to_pydict()
        for key, text in zip(table["id"], table["text"]):
            encode_paragraph(sp, config, part, stats, key, text, split, exclude)
    part.save(Path(config["parts_dir"]) / job["name"])
    return job["name"], {k: len(part.lengths[k]) for k in KINDS}, dict(stats)


def write_preview(output_dir, sp, count=20):
    lines = []
    for kind in KINDS:
        d = Path(output_dir) / kind
        val_tok = d / "validation.tokens.bin"
        if not val_tok.is_file():
            continue
        ids = np.fromfile(val_tok, dtype="<u2")
        offsets = np.fromfile(d / "validation.offsets.bin", dtype="<u8")
        first = np.fromfile(d / "validation.first.bin", dtype="u1")
        for i in range(min(count, len(first))):
            seq = ids[offsets[i]:offsets[i + 1]].tolist()
            if first[i] == 1:
                lines.append({"kind": kind, "text": sp.decode(seq)})
            else:
                r, o = seq.index(5), seq.index(6)
                lines.append({
                    "kind": kind,
                    "context": sp.decode(seq[2:r]),
                    "reading": sp.decode(seq[r + 1:o]),
                    "output": sp.decode(seq[int(first[i]):-1]),
                })
    with open(Path(output_dir) / "preview.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_store(config, export_dir, tokenizer_dir, table_path, output_dir, workers=16, limit_jobs=0):
    out = Path(output_dir)
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    began = time.time()

    tok_dir = Path(tokenizer_dir)
    tok = json.loads((tok_dir / "manifest.json").read_text("utf-8"))
    tok_model = tok_dir / tok["model"]["directory"] / "tokenizer.model"
    if tok.get("status") != "complete" or sha256(tok_model) != tok["model"]["tokenizer_model_sha256"]:
        raise SystemExit("tokenizer manifest incomplete or tokenizer.model changed")

    exp_dir = Path(export_dir)
    exp_manifest_path = exp_dir / "manifest.json"
    if not exp_manifest_path.is_file():
        raise SystemExit(f"Missing colloquial export manifest: {exp_manifest_path}")

    exclude = validation_line_keys(exp_dir)
    run_cfg = dict(
        config,
        tokenizer_path=str(tok_model),
        table_path=str(table_path),
        parts_dir=str(out / "parts"),
    )
    todo = build_jobs(exp_dir, int(config["row_groups_per_job"]), limit_jobs=limit_jobs)
    if not todo:
        raise SystemExit(f"No parquet jobs found under {exp_dir}")
    print(f"{len(todo)} jobs, {len(exclude)} validation lines excluded from train", flush=True)

    manifest = dict(format=FORMAT, status="partial", label=config["label"], stores={}, stats={})
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(run_cfg, exclude)) as pool:
        for split in SPLITS:
            mine = [j for j in todo if j["split"] == split]
            names, stats = [], Counter()
            for n, (name, _counts, part_stats) in enumerate(pool.map(run_job, mine), 1):
                names.append(name)
                stats.update(part_stats)
                if n % 20 == 0 or n == len(mine):
                    print(f"{split} {n}/{len(mine)} jobs {time.time() - began:.0f}s", flush=True)
            manifest["stores"][split] = merge(run_cfg["parts_dir"], names, out, split, KINDS)
            manifest["stats"][split] = dict(sorted(stats.items()))

    shutil.rmtree(run_cfg["parts_dir"])
    import sentencepiece as spm
    from vimeml.benchmarks.readings import DualReading

    write_preview(out, spm.SentencePieceProcessor(model_file=str(tok_model)))
    manifest.update(
        status="complete",
        tokenizer_sha256=tok["model"]["tokenizer_model_sha256"],
        tokenizer_manifest_sha256=sha256(tok_dir / "manifest.json"),
        table_sha256=sha256(table_path),
        export_manifest_sha256=sha256(exp_manifest_path),
        dictionary_versions=DualReading().versions,
        validation_lines_excluded=len(exclude),
        config={k: v for k, v in config.items()},
        limit_jobs=limit_jobs,
        code_sha256={
            p: sha256(ROOT / p)
            for p in (
                "scripts/ja/tokenizer/encode_ja_colloq.py",
                "scripts/ja/corpus/kana_samples.py",
                "src/vimeml/posttraining/zh_pinyin.py",
                "src/vimeml/posttraining/zh_pinyin_data.py",
            )
        },
        libraries={"sentencepiece": importlib.metadata.version("sentencepiece")},
        seconds=round(time.time() - began, 1),
    )
    write_json(out / "manifest.json", manifest)
    m_sha = sha256(out / "manifest.json")
    print(json.dumps({
        "stores": {s: {k: v["examples"] for k, v in st.items()} for s, st in manifest["stores"].items()},
        "manifest_sha256": m_sha,
    }, ensure_ascii=False), flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs/ja/data-ja-colloq-v1.toml")
    p.add_argument("--export-dir", type=Path, default=None)
    p.add_argument("--tokenizer-dir", type=Path, default=None)
    p.add_argument("--table", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit-jobs", type=int, default=0)
    args = p.parse_args()

    config = tomllib.loads(args.config.read_text("utf-8"))
    formal = ROOT / config["output_dir"]
    if args.limit_jobs and (args.output is None or args.output.resolve() == formal.resolve()):
        p.error("--limit-jobs requires --output outside output_dir")
    out = args.output or formal
    export_dir = args.export_dir or (ROOT / config["export_dir"])
    tokenizer_dir = args.tokenizer_dir or (ROOT / config["tokenizer_dir"])
    table_path = args.table or (ROOT / config["table"])
    workers = args.workers or int(config.get("workers", 16))
    encode_store(config, export_dir, tokenizer_dir, table_path, out, workers=workers, limit_jobs=args.limit_jobs)


if __name__ == "__main__":
    main()
