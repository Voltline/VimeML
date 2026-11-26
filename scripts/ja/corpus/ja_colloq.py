"""Mine colloquial and conversational paragraphs from a Japanese export (ja-gen-export-v1).

Mirrors `scripts/zh/corpus/zh_colloq.py` for Stage 3 (ja-gen colloquial post-training):
- Stage 1 (`scan`): per-paragraph Japanese colloquial rule pre-filter. Counts sentence/clause-final
  colloquial particles, spoken contractions, and casual interjections per 100 Japanese characters.
  Documents are split at the document level by `doc_hash("ja-colloq-heldout", id) < 0.02`:
  `heldout` is reserved strictly for evaluation and never enters training.
- Stage 2 (`sample`): draws a deterministic stratified sample across rule bands for LLM labelling.
- Stage 3 (`train`): fits two L2 logistic regressions (`chat` and `spoken`) over hashed character
  1-3-grams plus rule score and log length.
- Stage 4 (`select`): scores every paragraph in `export-v1` and keeps paragraphs with
  `p_chat >= KEEP_CHAT` or `p_spoken >= KEEP_SPOKEN`.
- Stage 5 (`screen-input` & `export`): prepares deduplicated shuffled JSONL for LLM quality
  screening, and exports screened `train`/`validation` parquets with optional Aho-Corasick
  evaluation overlap filtering (`--guard`).

    python -X utf8 scripts/ja/corpus/ja_colloq.py scan outputs/ja-corpus/export-v1 outputs/ja-colloq/scan-v1 --workers 16
    python -X utf8 scripts/ja/corpus/ja_colloq.py sample outputs/ja-colloq/scan-v1 outputs/ja-colloq/label-v1/cases.jsonl --per-band 1000
    python -X utf8 scripts/ja/corpus/ja_colloq.py train outputs/ja-colloq/model-v1.npz --cases cases.jsonl --labels labels.jsonl
    python -X utf8 scripts/ja/corpus/ja_colloq.py select outputs/ja-corpus/export-v1 outputs/ja-colloq/model-v1.npz outputs/ja-colloq/select-v1
    python -X utf8 scripts/ja/corpus/ja_colloq.py export outputs/ja-colloq/select-v1 outputs/ja-colloq/export-colloq-v1
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.corpus.ja_filter import JA, KANA  # noqa: E402
from vimeml.tokenizer.encode_zh13 import doc_hash  # noqa: E402

# Clause-final Japanese colloquial particles (followed by clause-end punctuation, quotes, space or EOL)
END_LOOKAHEAD = r"(?=[。！？!?、…～〜ー」』）)\s]|$)"
FINAL_PARTICLES = re.compile(
    r"(?:よね|だよね|ですよね|ますね|ますよ|でしょ|でしょう|かなあ|かなぁ|かな|かも|じゃん|だし|ってば|っけ|もん|なあ|なぁ|ねぇ|ねえ|よー|よ〜|わよ|なの|のよ|[ねよわさぞぜ])"
    + END_LOOKAHEAD
)
# Spoken contractions and casual discourse markers
CONTRACTIONS = re.compile(
    r"(?:ちゃう|ちゃった|ちゃって|じゃう|じゃった|じゃって|てる(?=[。！？!?、…～〜」』）)\s]|$)|てた(?=[。！？!?、…～〜」』）)\s]|$)|"
    r"とく(?=[。！？!?、…～〜」』）)\s]|$)|といた|なきゃ|なくちゃ|じゃない|っていう|ってか|なんだよね|なんだけど|けどさ|だけどさ|まじ|マジ|ヤバい|やばい|ヤバイ)"
)
CASUAL_LAUGH = re.compile(r"(?:[（(]笑[)）]|笑(?=[。！？!?…～〜\s]|$)|[wｗ]{2,})")

MIN_JA = 6
MIN_KANA_SHARE = 0.20
HELDOUT_LABEL = "ja-colloq-heldout"
HELDOUT_FRACTION = 0.02
ZERO_LABEL = "ja-colloq-zero"
ZERO_SAMPLE = 0.005
BANDS = [("zero", 0, 0), ("low", 1e-9, 1.5), ("mid", 1.5, 4.0), ("high", 4.0, float("inf"))]
TWO64 = 2 ** 64

HASH_FEATURES = 2 ** 20
KEEP_CHAT = 0.5
KEEP_SPOKEN = 0.8
SPOKEN_LABEL = "ja-colloq-spoken"
TARGETS = {
    "chat": lambda r: r["label"] == 3,
    "spoken": lambda r: r["label"] >= 2 and r["kind"] in ("chat", "fiction"),
}
CANON = re.compile(r"[^\wぁ-ゖァ-ヺー一-龠々]+")


def normalized(text):
    return CANON.sub("", unicodedata.normalize("NFKC", text)).casefold()


def features(text):
    """Return (ja_chars, markers, score_per_100_ja) or None if too short / non-Japanese."""
    ja = len(JA.findall(text))
    if ja < MIN_JA:
        return None
    kana = len(KANA.findall(text))
    if kana < MIN_KANA_SHARE * ja:
        return None
    n_final = len(FINAL_PARTICLES.findall(text))
    n_contr = len(CONTRACTIONS.findall(text))
    n_laugh = len(CASUAL_LAUGH.findall(text))
    markers = n_final + n_contr + n_laugh
    return ja, markers, 100.0 * markers / ja


def band(score):
    for name, low, high in BANDS:
        if (score == 0 and name == "zero") or (score > 0 and low <= score < high):
            return name
    raise ValueError(score)


def _scan_file(job):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path, out, max_groups = job
    handle = pq.ParquetFile(path)
    out = Path(out)
    part_path = out.with_suffix(".parquet.part")
    final_path = out.with_suffix(".parquet")
    schema = pa.schema([
        ("id", pa.string()),
        ("para", pa.int32()),
        ("text", pa.string()),
        ("ja", pa.int32()),
        ("markers", pa.int32()),
        ("score", pa.float64()),
        ("band", pa.string()),
        ("split", pa.string()),
    ])
    stats = Counter()
    groups = min(handle.metadata.num_row_groups, max_groups or 1 << 30)
    writer = pq.ParquetWriter(part_path, schema, compression="zstd")
    try:
        for group in range(groups):
            rows = {k: [] for k in ("id", "para", "text", "ja", "markers", "score", "band", "split")}
            table = handle.read_row_group(group, columns=["id", "text"]).to_pydict()
            for doc_id, text in zip(table["id"], table["text"]):
                stats["documents"] += 1
                split = "heldout" if doc_hash(HELDOUT_LABEL, doc_id) < HELDOUT_FRACTION * TWO64 else "train"
                for i, raw in enumerate(text.split("\n")):
                    para = raw.strip()
                    if not para:
                        continue
                    f = features(para)
                    if f is None:
                        stats["short_or_low_kana"] += 1
                        continue
                    ja, markers, score = f
                    b = band(score)
                    stats[f"{b}_paragraphs"] += 1
                    stats[f"{b}_ja"] += ja
                    stats[f"{split}_{b}_ja"] += ja
                    if b == "zero" and doc_hash(ZERO_LABEL, f"{doc_id}#{i}") >= ZERO_SAMPLE * TWO64:
                        continue
                    rows["id"].append(doc_id)
                    rows["para"].append(i)
                    rows["text"].append(para)
                    rows["ja"].append(ja)
                    rows["markers"].append(markers)
                    rows["score"].append(score)
                    rows["band"].append(b)
                    rows["split"].append(split)
            if rows["id"]:
                writer.write_table(pa.table(rows, schema=schema))
        writer.close()
        part_path.replace(final_path)
    except BaseException:
        writer.close()
        if part_path.exists():
            part_path.unlink()
        raise
    out.with_suffix(".json").write_text(
        json.dumps({"file": Path(path).name, "row_groups": groups, **stats}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return Path(path).name, stats


def scan(export, output, workers=8, max_groups=0, limit_files=0, only=None):
    export, output = Path(export), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted((export / "train").glob("web-*.parquet"))
    if only:
        wanted = set(only)
        files = [p for p in files if p.stem in wanted or p.name in wanted]
    if limit_files:
        files = files[:limit_files]
    jobs = [(str(p), str(output / p.stem), max_groups) for p in files
            if not (output / p.stem).with_suffix(".json").exists()]
    if jobs:
        with ProcessPoolExecutor(workers) as pool:
            for future in as_completed([pool.submit(_scan_file, j) for j in jobs]):
                name, stats = future.result()
                print(name, dict(stats), flush=True)
    summary = Counter()
    for p in sorted(output.glob("*.json")):
        if p.name == "summary.json":
            continue
        d = json.loads(p.read_text("utf-8"))
        summary.update({k: v for k, v in d.items() if isinstance(v, int) and k != "row_groups"})
    (output / "summary.json").write_text(
        json.dumps({"export": str(export), "files": len(files), "rules": {
            "min_ja": MIN_JA, "min_kana_share": MIN_KANA_SHARE,
            "heldout_fraction": HELDOUT_FRACTION, "zero_sample": ZERO_SAMPLE, "bands": BANDS[1:]},
            **summary}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def sample(scan_dir, out, per_band=1000, seed=20261126, split="train"):
    """Exact uniform reservoir sampling per band, streaming row groups in O(per_band) memory."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    reservoir = {name: [None] * per_band for name, *_ in BANDS}
    filled = {name: 0 for name, *_ in BANDS}
    seen = {name: 0 for name, *_ in BANDS}
    band_names = [name for name, *_ in BANDS]

    for p in sorted(Path(scan_dir).glob("*.parquet")):
        pf = pq.ParquetFile(p)
        for g in range(pf.metadata.num_row_groups):
            meta = pf.read_row_group(g, columns=["band", "split"])
            split_mask = pc.equal(meta["split"], split)
            picks = {}  # (band, slot) -> local_row_idx
            for name in band_names:
                mask = pc.and_(split_mask, pc.equal(meta["band"], name))
                idxs = np.nonzero(mask.to_numpy(zero_copy_only=False))[0]
                m = len(idxs)
                if m == 0:
                    continue
                cur_seen = seen[name]
                need = min(m, per_band - filled[name])
                for k in range(need):
                    picks[(name, filled[name] + k)] = int(idxs[k])
                filled[name] += need
                rem = m - need
                if rem > 0:
                    base = cur_seen + need
                    counts = np.arange(base + 1, base + rem + 1, dtype=np.int64)
                    draws = rng.integers(0, counts)
                    for h in np.nonzero(draws < per_band)[0]:
                        picks[(name, int(draws[h]))] = int(idxs[need + int(h)])
                seen[name] = cur_seen + m
            if not picks:
                continue
            unique_rows = sorted(set(picks.values()))
            pos_map = {r: i for i, r in enumerate(unique_rows)}
            payload = pf.read_row_group(g, columns=["id", "para", "text", "score"]).take(pa.array(unique_rows)).to_pylist()
            for (name, slot), local_idx in picks.items():
                r = payload[pos_map[local_idx]]
                reservoir[name][slot] = {
                    "id": f"{r['id']}#{r['para']}",
                    "band": name,
                    "score": round(float(r["score"]), 3),
                    "text": r["text"],
                }

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for name in band_names:
            chosen = reservoir[name][:filled[name]]
            for rec in chosen:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            print(name, seen[name], "->", len(chosen), flush=True)
    print("wrote", n, out, flush=True)


def matrix(texts):
    import numpy as np
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(analyzer="char", ngram_range=(1, 3), n_features=HASH_FEATURES,
                           alternate_sign=False, norm="l2")
    extra = []
    for t in texts:
        f = features(t)
        extra.append([f[2] / 10.0, np.log1p(f[0]) / 5.0] if f else [0.0, 0.0])
    return sp.hstack([hv.transform(texts), sp.csr_matrix(np.array(extra))]).tocsr()


def train(out, cases_paths, labels_paths, C=8.0):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rows = []
    for cases_path, labels_path in zip(cases_paths, labels_paths, strict=True):
        cases = {json.loads(l)["id"]: json.loads(l) for l in open(cases_path, encoding="utf-8")}
        rows += [(cases[r["id"]]["text"], r) for r in map(json.loads, open(labels_path, encoding="utf-8"))]
    X = matrix([t for t, _ in rows])
    saved = {}
    for name, fn in TARGETS.items():
        y = np.array([fn(r) for _, r in rows])
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y)
        saved[f"{name}_w"], saved[f"{name}_b"] = m.coef_[0].astype(np.float32), np.float32(m.intercept_[0])
        print(name, "positives", int(y.sum()), "of", len(y))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, hash_features=HASH_FEATURES, C=C, labels=len(rows), **saved)


def _select_file(job):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    path, model_path, out, max_groups = job
    model = np.load(model_path)
    out = Path(out)
    part_path = out.with_suffix(".parquet.part")
    final_path = out.with_suffix(".parquet")
    if part_path.exists():
        part_path.unlink()
    schema = pa.schema([
        ("id", pa.string()),
        ("para", pa.int64()),
        ("text", pa.string()),
        ("ja", pa.int64()),
        ("p_chat", pa.float32()),
        ("p_spoken", pa.float32()),
        ("split", pa.string()),
    ])
    cols = ("id", "para", "text", "ja", "p_chat", "p_spoken", "split")
    rows, stats, hist = {k: [] for k in cols}, Counter(), [0] * 10
    pending = []
    writer = pq.ParquetWriter(part_path, schema, compression="zstd")

    def write_chunk():
        if not rows["id"]:
            return
        writer.write_table(pa.table(rows, schema=schema))
        for k in cols:
            rows[k].clear()

    def flush():
        if not pending:
            return
        X = matrix([p[2] for p in pending])
        pc = 1.0 / (1.0 + np.exp(-(X @ model["chat_w"] + model["chat_b"])))
        ps = 1.0 / (1.0 + np.exp(-(X @ model["spoken_w"] + model["spoken_b"])))
        for (doc_id, i, text, ja, split), a, b in zip(pending, pc, ps):
            hist[min(9, int(a * 10))] += ja
            if a >= KEEP_CHAT or b >= KEEP_SPOKEN:
                for k, v in zip(cols, (doc_id, i, text, ja, float(a), float(b), split)):
                    rows[k].append(v)
                stats["kept_paragraphs"] += 1
                stats["kept_ja"] += ja
        pending.clear()
        if len(rows["id"]) >= 50000:
            write_chunk()

    handle = pq.ParquetFile(path)
    groups = min(handle.metadata.num_row_groups, max_groups or 1 << 30)
    try:
        for group in range(groups):
            table = handle.read_row_group(group, columns=["id", "text"]).to_pydict()
            for doc_id, text in zip(table["id"], table["text"]):
                stats["documents"] += 1
                split = "heldout" if doc_hash(HELDOUT_LABEL, doc_id) < HELDOUT_FRACTION * TWO64 else "train"
                for i, raw in enumerate(text.split("\n")):
                    para = raw.strip()
                    if not para:
                        continue
                    f = features(para)
                    if f is None:
                        continue
                    stats["paragraphs"] += 1
                    stats["ja"] += f[0]
                    pending.append((doc_id, i, para, f[0], split))
                    if len(pending) >= 20000:
                        flush()
        flush()
        write_chunk()
        if stats["kept_paragraphs"] == 0:
            writer.write_table(pa.table({k: [] for k in cols}, schema=schema))
    finally:
        writer.close()
    part_path.replace(final_path)
    out.with_suffix(".json").write_text(
        json.dumps({"file": Path(path).name, "row_groups": groups, **stats, "ja_by_p_chat_decile": hist}, indent=1),
        encoding="utf-8",
    )
    return Path(path).name, stats


def select(export, model, output, workers=8, max_groups=0, limit_files=0, file_range=None):
    export, output = Path(export), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted((export / "train").glob("web-*.parquet"))
    files = files[:limit_files] if limit_files else files
    if file_range:
        a, b = (int(x) if x else None for x in file_range.split(":"))
        files = files[a:b]
    jobs = [(str(p), str(model), str(output / p.stem), max_groups) for p in files
            if not (output / p.stem).with_suffix(".json").exists()]
    if jobs:
        with ProcessPoolExecutor(workers) as pool:
            for future in as_completed([pool.submit(_select_file, j) for j in jobs]):
                name, stats = future.result()
                print(name, dict(stats), flush=True)
    summary, hist = Counter(), [0] * 10
    for p in sorted(output.glob("*.json")):
        if p.name == "summary.json":
            continue
        d = json.loads(p.read_text("utf-8"))
        summary.update({k: v for k, v in d.items() if isinstance(v, int) and k != "row_groups"})
        hist = [a + b for a, b in zip(hist, d["ja_by_p_chat_decile"])]
    (output / "summary.json").write_text(
        json.dumps({"export": str(export), "model": str(model), "files": len(files),
                    "keep_chat": KEEP_CHAT, "keep_spoken": KEEP_SPOKEN, **summary,
                    "ja_by_p_chat_decile": hist}, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(summary), hist)


def text_id(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def screen_input(export_dir, out, seed=20261126):
    import pyarrow.parquet as pq

    seen = {}
    for p in sorted(Path(export_dir).glob("*/*.parquet")):
        for text in pq.read_table(p, columns=["text"])["text"].to_pylist():
            seen.setdefault(text_id(text), text)
    rows = sorted(seen.items())
    random.Random(seed).shuffle(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for i, t in rows:
            f.write(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n")
    print(len(rows), "texts")


def eval_guard(patterns_file):
    """Build an Aho-Corasick matcher from normalized evaluation patterns (>= 12 chars, one per line)."""
    import ahocorasick

    automaton = ahocorasick.Automaton()
    for line in Path(patterns_file).read_text("utf-8").splitlines():
        p = normalized(line)
        if len(p) >= 12:
            automaton.add_word(p, p)
    automaton.make_automaton()
    return lambda text: next(automaton.iter(normalized(text)), None) is not None


def screened_ok_glob(screen_dir, pattern):
    ok, bad = set(), set()
    for path in sorted(Path(screen_dir).glob(pattern)):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            (ok if r["ok"] else bad).add(r["id"])
    return ok - bad


def export(select_dir, out, chat_min=0.7, spoken_min=0.8, spoken_units=500000, rows_per_group=20000,
           name="v1", screen_dir=None, guard=None):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    select_dir, out = Path(select_dir), Path(out)
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} is complete; use a new directory")
    files = sorted(select_dir.glob("*.parquet"))
    cols = ["id", "para", "text", "p_chat", "p_spoken", "split"]
    ok = screened_ok_glob(screen_dir, f"{name}*-screen-*.jsonl") if screen_dir else None
    blocked = eval_guard(guard) if guard else None
    fname = f"web-colloq-{name}.parquet"

    def spoken_mask(t):
        return pc.and_(pc.and_(pc.equal(t["split"], "train"), pc.greater_equal(t["p_spoken"], spoken_min)),
                       pc.less(t["p_chat"], KEEP_CHAT))

    spoken_total = sum(pc.sum(spoken_mask(pq.read_table(p, columns=["p_chat", "p_spoken", "split"]))).as_py() or 0
                       for p in files)
    cut = min(1.0, spoken_units / max(1, spoken_total)) * TWO64
    schema = pa.schema([("id", pa.string()), ("text", pa.string()), ("origin", pa.string()),
                        ("p_chat", pa.float32()), ("p_spoken", pa.float32())])
    for split in ("train", "validation"):
        (out / split).mkdir(parents=True, exist_ok=True)
    writers = {s: pq.ParquetWriter(out / s / fname, schema, compression="zstd") for s in ("train", "validation")}
    counts, buffers = Counter(), {s: [] for s in writers}

    def flush(split, force=False):
        rows = buffers[split]
        while len(rows) >= rows_per_group or (force and rows):
            chunk, rows[:] = rows[:rows_per_group], rows[rows_per_group:]
            writers[split].write_table(pa.Table.from_pylist(chunk, schema=schema), row_group_size=rows_per_group)

    for p in files:
        t = pq.read_table(p, columns=cols)
        spoken = spoken_mask(t).to_pylist()
        for r, is_spoken in zip(t.to_pylist(), spoken):
            key = f'{r["id"]}#{r["para"]}'
            if r["split"] == "train" and r["p_chat"] >= chat_min:
                split, origin = "train", "chat"
            elif is_spoken and doc_hash(SPOKEN_LABEL, key) < cut:
                split, origin = "train", "spoken"
            elif r["split"] == "heldout" and r["p_chat"] >= chat_min:
                split, origin = "validation", "chat"
            else:
                continue
            if ok is not None and text_id(r["text"]) not in ok:
                counts[f"{split}_{origin}_screen_dropped"] += 1
                continue
            if blocked and blocked(r["text"]):
                counts[f"{split}_{origin}_eval_overlap_dropped"] += 1
                continue
            buffers[split].append(dict(id=key, text=r["text"], origin=origin,
                                       p_chat=r["p_chat"], p_spoken=r["p_spoken"]))
            counts[f"{split}_{origin}_rows"] += 1
            counts[f"{split}_{origin}_ja"] += len(JA.findall(r["text"]))
            flush(split)
    for s in writers:
        flush(s, force=True)
        writers[s].close()
    (out / "manifest.json").write_text(
        json.dumps({"format": "vimeml_ja_colloq_export_v1", "name": name, "select_dir": str(select_dir),
                    "chat_min": chat_min, "spoken_min": spoken_min, "spoken_units": spoken_units,
                    "screened": screen_dir is not None, "guarded": guard is not None,
                    "counts": dict(counts)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scan")
    sc.add_argument("export")
    sc.add_argument("output")
    sc.add_argument("--workers", type=int, default=8)
    sc.add_argument("--max-groups", type=int, default=0)
    sc.add_argument("--limit-files", type=int, default=0)
    sc.add_argument("--only", nargs="*", default=None)

    sp = sub.add_parser("sample")
    sp.add_argument("scan_dir")
    sp.add_argument("out")
    sp.add_argument("--per-band", type=int, default=1000)
    sp.add_argument("--seed", type=int, default=20261126)
    sp.add_argument("--split", default="train")

    tr = sub.add_parser("train")
    tr.add_argument("out")
    tr.add_argument("--cases", nargs="+", required=True)
    tr.add_argument("--labels", nargs="+", required=True)
    tr.add_argument("--C", type=float, default=8.0)

    sl = sub.add_parser("select")
    sl.add_argument("export")
    sl.add_argument("model")
    sl.add_argument("output")
    sl.add_argument("--workers", type=int, default=8)
    sl.add_argument("--max-groups", type=int, default=0)
    sl.add_argument("--limit-files", type=int, default=0)
    sl.add_argument("--file-range", default=None)

    si = sub.add_parser("screen-input")
    si.add_argument("export_dir")
    si.add_argument("out")
    si.add_argument("--seed", type=int, default=20261126)

    ex = sub.add_parser("export")
    ex.add_argument("select_dir")
    ex.add_argument("out")
    ex.add_argument("--chat-min", type=float, default=0.7)
    ex.add_argument("--spoken-min", type=float, default=0.8)
    ex.add_argument("--spoken-units", type=int, default=500000)
    ex.add_argument("--name", default="v1")
    ex.add_argument("--screen-dir", default=None)
    ex.add_argument("--guard", default=None)

    a = p.parse_args()
    if a.cmd == "scan":
        scan(a.export, a.output, workers=a.workers, max_groups=a.max_groups,
             limit_files=a.limit_files, only=a.only)
    elif a.cmd == "sample":
        sample(a.scan_dir, a.out, per_band=a.per_band, seed=a.seed, split=a.split)
    elif a.cmd == "train":
        train(a.out, a.cases, a.labels, C=a.C)
    elif a.cmd == "select":
        select(a.export, a.model, a.output, workers=a.workers, max_groups=a.max_groups,
               limit_files=a.limit_files, file_range=a.file_range)
    elif a.cmd == "screen-input":
        screen_input(a.export_dir, a.out, seed=a.seed)
    elif a.cmd == "export":
        export(a.select_dir, a.out, chat_min=a.chat_min, spoken_min=a.spoken_min,
               spoken_units=a.spoken_units, name=a.name, screen_dir=a.screen_dir, guard=a.guard)


if __name__ == "__main__":
    main()
