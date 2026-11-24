"""Mine colloquial paragraphs from a zh export (export-r1 / export-r2) for the v2-20m colloquial arms.

Stage 1 (this script) is a rule pre-filter; stage 2 labels a stratified sample with an LLM and
trains a classifier (separate tools). Rules work per paragraph (clean_paragraphs, as the pinyin
data builder splits documents):

- A "final particle" is one of PARTICLES at a clause end: followed by punctuation, whitespace,
  another particle or the end of the paragraph (so 哈尔滨, 罗里吧嗦 do not count); 哈哈 counts once
  per run. score = 100 * final particles / Han characters.
- Paragraphs with fewer than MIN_HAN Han characters are skipped.
- Documents in the zh13 memorization probe are skipped (same rule as the pinyin data builder).
- split: doc_hash("zh-colloq-heldout", id) < HELDOUT_FRACTION -> "heldout" (evaluation only,
  never trained on), else "train". Document-level, so a document never straddles the split.

scan writes, per export file, a parquet of every paragraph with score > 0 plus a deterministic
ZERO_SAMPLE share of score-0 paragraphs (negatives for labelling), and a JSON of band counts over
all paragraphs. sample draws a stratified labelling set from the scan output.

train fits two logistic regressions on LLM labels ({id, label 0-3, kind} rows joined to case files by
id) over hashed character 1-3-grams plus the rule score and log length: "chat" = label 3, "spoken" =
label >= 2 and kind chat or fiction. select scores every paragraph of an export with both and keeps
rows with p_chat >= KEEP_CHAT or p_spoken >= KEEP_SPOKEN, with a histogram of Han characters per
p_chat decile over all paragraphs; thresholds for training data are set later from these columns.

  python scripts/zh/corpus/zh_colloq.py scan outputs/zh-corpus/export-r1 outputs/zh-colloq/scan-r1 --pools artifacts/training-data/zh13-pools/pools.json
  python scripts/zh/corpus/zh_colloq.py sample outputs/zh-colloq/scan-r1 outputs/zh-colloq/label-r1/cases.jsonl --per-band 1000
  python scripts/zh/corpus/zh_colloq.py train outputs/zh-colloq/model-v1.npz --cases c1.jsonl c2.jsonl --labels l1.jsonl l2.jsonl
  python scripts/zh/corpus/zh_colloq.py select outputs/zh-corpus/export-r1 outputs/zh-colloq/model-v1.npz outputs/zh-colloq/select-r1 --pools ...
  python scripts/zh/corpus/zh_colloq.py export outputs/zh-colloq/select-r1 outputs/zh-colloq/export-colloq-r1
  python scripts/zh/corpus/zh_colloq.py export-chat outputs/zh-corpus/chat-ext-v2 outputs/zh-colloq/screen outputs/zh-colloq/export-colloq-chat

Second round (export-r2 part, more chat): select on the downloaded r2 files, export without screening, write the
screen input, screen it, then export again keeping ok paragraphs and dropping eval overlaps:
  python scripts/zh/corpus/zh_colloq.py export outputs/zh-colloq/select-r2 outputs/zh-colloq/export-colloq-r2-raw --name r2 --spoken-units N
  python scripts/zh/corpus/zh_colloq.py screen-input outputs/zh-colloq/export-colloq-r2-raw outputs/zh-colloq/screen/r2-in.jsonl
  python scripts/zh/corpus/zh_colloq.py export outputs/zh-colloq/select-r2 outputs/zh-colloq/export-colloq-r2 --name r2 --spoken-units N \
      --screen-dir outputs/zh-colloq/screen --guard outputs/benchmarks/ime-pinyin-zh-colloq-v1/frozen-r1
  python scripts/zh/corpus/zh_colloq.py export-chat ... --quota douban=N --quota mmchat=N --quota tieba=N --guard <frozen>
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.tokenizer.encode_zh13 import clean_paragraphs, doc_hash  # noqa: E402

PARTICLES = "吧呢啊嘛呀啦哦呗咯嘞噢哇喔呐咩嘻"
HAN = re.compile(r"[一-鿿]")
FINAL = re.compile(rf"[{PARTICLES}](?=[{PARTICLES}]|[\s，。！？、；：,.!?;:~～…）)」』”’\"']|$)")
HAHA = re.compile(r"哈{2,}|嘿{2,}|呵{2,}")
MIN_HAN = 4
HELDOUT_LABEL = "zh-colloq-heldout"
HELDOUT_FRACTION = 0.02
ZERO_LABEL = "zh-colloq-zero"
ZERO_SAMPLE = 0.01
BANDS = [("zero", 0, 0), ("low", 1e-9, 1), ("mid", 1, 3), ("high", 3, float("inf"))]
TWO64 = 2 ** 64


def features(text):
    han = len(HAN.findall(text))
    if han < MIN_HAN:
        return None
    finals = len(FINAL.findall(HAHA.sub("哈", text))) + len(HAHA.findall(text))
    return han, finals, 100.0 * finals / han


def band(score):
    for name, low, high in BANDS:
        if (score == 0 and name == "zero") or (score > 0 and low <= score < high):
            return name
    raise ValueError(score)


def _scan_file(job):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path, out, pools, max_groups = job
    handle = pq.ParquetFile(path)
    rows = {k: [] for k in ("id", "para", "text", "han", "finals", "score", "band", "split")}
    stats = Counter()
    probe = _probe(pools)
    groups = min(handle.metadata.num_row_groups, max_groups or 1 << 30)
    for group in range(groups):
        table = handle.read_row_group(group, columns=["id", "text", "script"]).to_pydict()
        for doc_id, text, script in zip(table["id"], table["text"], table["script"]):
            stats["documents"] += 1
            if probe and probe(doc_id):
                stats["probe_skipped"] += 1
                continue
            split = "heldout" if doc_hash(HELDOUT_LABEL, doc_id) < HELDOUT_FRACTION * TWO64 else "train"
            paragraphs, _ = clean_paragraphs(text)
            for i, para in enumerate(paragraphs):
                f = features(para)
                if f is None:
                    stats["short_paragraphs"] += 1
                    continue
                han, finals, score = f
                b = band(score)
                stats[f"{b}_paragraphs"] += 1
                stats[f"{b}_han"] += han
                stats[f"{split}_{b}_han"] += han
                if b == "zero" and doc_hash(ZERO_LABEL, f"{doc_id}#{i}") >= ZERO_SAMPLE * TWO64:
                    continue
                for k, v in (("id", doc_id), ("para", i), ("text", para), ("han", han), ("finals", finals),
                             ("score", score), ("band", b), ("split", split)):
                    rows[k].append(v)
                stats["script_" + script] += 1
    out = Path(out)
    pq.write_table(pa.table(rows), out.with_suffix(".parquet"), compression="zstd")
    out.with_suffix(".json").write_text(json.dumps({"file": Path(path).name, "row_groups": groups, **stats},
                                                   ensure_ascii=False, indent=1), "utf-8")
    return Path(path).name, stats


def scan(export, output, pools_path=None, workers=8, max_groups=0, limit_files=0):
    export, output = Path(export), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted((export / "train").glob("web-*.parquet")) + sorted((export / "train").glob("review-*.parquet"))
    if limit_files:
        files = files[:limit_files]
    pools = json.loads(Path(pools_path).read_text("utf-8")) if pools_path else None
    jobs = [(str(p), str(output / p.stem), pools, max_groups) for p in files
            if not (output / p.stem).with_suffix(".json").exists()]
    total = Counter()
    with ProcessPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(_scan_file, j) for j in jobs]):
            name, stats = future.result()
            total.update(stats)
            print(name, dict(stats), flush=True)
    summary = Counter()
    for p in sorted(output.glob("*.json")):
        if p.name == "summary.json":
            continue
        d = json.loads(p.read_text("utf-8"))
        summary.update({k: v for k, v in d.items() if isinstance(v, int)})
    (output / "summary.json").write_text(json.dumps({"export": str(export), "files": len(files), "rules": {
        "particles": PARTICLES, "min_han": MIN_HAN, "heldout_fraction": HELDOUT_FRACTION,
        "zero_sample": ZERO_SAMPLE, "bands": BANDS[1:]}, **summary}, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def sample(scan_dir, out, per_band=1000, seed=20261122, split="train"):
    import pyarrow.parquet as pq
    pools = {name: [] for name, *_ in BANDS}
    for p in sorted(Path(scan_dir).glob("*.parquet")):
        t = pq.read_table(p, columns=["id", "para", "text", "han", "score", "band", "split"]).to_pylist()
        for r in t:
            if r["split"] == split:
                pools[r["band"]].append(r)
    rng = random.Random(seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for name, rows in pools.items():
            chosen = rng.sample(rows, min(per_band, len(rows)))
            for r in chosen:
                f.write(json.dumps({"id": f"{r['id']}#{r['para']}", "band": name, "score": round(r["score"], 3),
                                    "text": r["text"]}, ensure_ascii=False) + "\n")
                n += 1
            print(name, len(rows), "->", len(chosen))
    print("wrote", n, out)


HASH_FEATURES = 2 ** 20
KEEP_CHAT = 0.5
KEEP_SPOKEN = 0.8
TARGETS = {"chat": lambda r: r["label"] == 3,
           "spoken": lambda r: r["label"] >= 2 and r["kind"] in ("chat", "fiction")}


def matrix(texts):
    import numpy as np
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(analyzer="char", ngram_range=(1, 3), n_features=HASH_FEATURES,
                           alternate_sign=False, norm="l2")
    extra = []
    for t in texts:
        f = features(t)
        extra.append([f[2] / 10, np.log1p(f[0]) / 5] if f else [0.0, 0.0])
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
    np.savez(out, hash_features=HASH_FEATURES, C=C, labels=len(rows), **saved)


def _select_file(job):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    path, model_path, out, pools, max_groups = job
    model = np.load(model_path)
    probe = _probe(pools)
    cols = ("id", "para", "text", "han", "p_chat", "p_spoken", "split", "script")
    rows, stats, hist = {k: [] for k in cols}, Counter(), [0] * 10
    pending = []

    def flush():
        if not pending:
            return
        X = matrix([p[2] for p in pending])
        pc = 1 / (1 + np.exp(-(X @ model["chat_w"] + model["chat_b"])))
        ps = 1 / (1 + np.exp(-(X @ model["spoken_w"] + model["spoken_b"])))
        for (doc_id, i, text, han, split, script), a, b in zip(pending, pc, ps):
            hist[min(9, int(a * 10))] += han
            if a >= KEEP_CHAT or b >= KEEP_SPOKEN:
                for k, v in zip(cols, (doc_id, i, text, han, float(a), float(b), split, script)):
                    rows[k].append(v)
                stats["kept_paragraphs"] += 1
                stats["kept_han"] += han
        pending.clear()

    handle = pq.ParquetFile(path)
    groups = min(handle.metadata.num_row_groups, max_groups or 1 << 30)
    for group in range(groups):
        table = handle.read_row_group(group, columns=["id", "text", "script"]).to_pydict()
        for doc_id, text, script in zip(table["id"], table["text"], table["script"]):
            stats["documents"] += 1
            if probe and probe(doc_id):
                stats["probe_skipped"] += 1
                continue
            split = "heldout" if doc_hash(HELDOUT_LABEL, doc_id) < HELDOUT_FRACTION * TWO64 else "train"
            for i, para in enumerate(clean_paragraphs(text)[0]):
                f = features(para)
                if f is None:
                    continue
                stats["paragraphs"] += 1
                stats["han"] += f[0]
                pending.append((doc_id, i, para, f[0], split, script))
                if len(pending) >= 20000:
                    flush()
    flush()
    out = Path(out)
    pq.write_table(pa.table(rows), out.with_suffix(".parquet"), compression="zstd")
    out.with_suffix(".json").write_text(json.dumps({"file": Path(path).name, "row_groups": groups, **stats,
                                                    "han_by_p_chat_decile": hist}, indent=1), "utf-8")
    return Path(path).name, stats


def _probe(pools):
    if not pools:
        return None
    p20, memo_cut = pools["thresholds"]["p20"], pools["memo_threshold"]
    pool_label, memo_label = pools["labels"]["pool"], pools["labels"]["memo"]
    return lambda d: doc_hash(pool_label, d) < p20 and doc_hash(memo_label, d) < memo_cut


def select(export, model, output, pools_path=None, workers=8, max_groups=0, limit_files=0, file_range=None):
    export, output = Path(export), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted((export / "train").glob("web-*.parquet")) + sorted((export / "train").glob("review-*.parquet"))
    files = files[:limit_files] if limit_files else files
    if file_range:  # "a:b" slice of the sorted file list, to split one export across machines
        a, b = (int(x) if x else None for x in file_range.split(":"))
        files = files[a:b]
    pools = json.loads(Path(pools_path).read_text("utf-8")) if pools_path else None
    jobs = [(str(p), str(model), str(output / p.stem), pools, max_groups) for p in files
            if not (output / p.stem).with_suffix(".json").exists()]
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
        hist = [a + b for a, b in zip(hist, d["han_by_p_chat_decile"])]
    (output / "summary.json").write_text(json.dumps({"export": str(export), "model": str(model), "files": len(files),
                                                     "keep_chat": KEEP_CHAT, "keep_spoken": KEEP_SPOKEN, **summary,
                                                     "han_by_p_chat_decile": hist}, indent=1), "utf-8")
    print(json.dumps(summary), hist)


SPOKEN_LABEL = "zh-colloq-spoken"


def screened_ok_glob(screen_dir, pattern):
    """Ids marked ok in the matching screen files and never marked not ok (a text screened by two models, where
    their ranges meet, is kept only if both say ok)."""
    ok, bad = set(), set()
    for path in sorted(Path(screen_dir).glob(pattern)):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            (ok if r["ok"] else bad).add(r["id"])
    return ok - bad


def screen_note(screen_dir, patterns):
    """Manifest description of the quality screen: the models that labelled the matching files."""
    models = sorted({json.loads(line).get("model", "?") for pattern in patterns
                     for path in Path(screen_dir).glob(pattern) for line in open(path, encoding="utf-8")})
    return f"LLM quality screen ({', '.join(models)}), ok only; a text any model marked not ok is dropped"


def text_id(text):
    import hashlib
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def eval_guard(frozen):
    """True for a text that contains any overlap pattern (screen_overlap_zh.py: the Han run around the target and
    the whole message, each if >= 8 characters) of an ime-pinyin-zh-colloq-v1 item, development and test. New
    training rows added after the set was frozen are filtered with this instead of dropping items."""
    import ahocorasick
    sys.path.insert(0, str(ROOT / "scripts/zh/benchmarks"))
    from screen_overlap_zh import patterns
    items = [json.loads(l) for split in ("development", "test")
             for l in open(Path(frozen) / f"{split}.jsonl", encoding="utf-8")]
    automaton = ahocorasick.Automaton()
    for pattern in patterns(items):
        automaton.add_word(pattern, pattern)
    automaton.make_automaton()
    return lambda text: next(automaton.iter(text), None) is not None


def screen_input(export_dir, out, seed=20261122):
    """Distinct texts of an export (train and validation parquet files) as screen input rows {id, text},
    id = sha1(text)[:16], shuffled so that a partly screened input is a random subset."""
    import pyarrow.parquet as pq
    seen = {}
    for p in sorted(Path(export_dir).glob("*/*.parquet")):
        for text in pq.read_table(p, columns=["text"])["text"].to_pylist():
            seen.setdefault(text_id(text), text)
    rows = sorted(seen.items())
    random.Random(seed).shuffle(rows)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for i, t in rows:
            f.write(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n")
    print(len(rows), "texts")


def export(select_dir, out, chat_min=0.7, spoken_min=0.8, spoken_units=730000, rows_per_group=20000,
           name="r1", screen_dir=None, guard=None):
    """Write select output as a frozen export for build_zh_pinyin.py: train/web-colloq-r1.parquet holds every
    train paragraph with p_chat >= chat_min ("chat") plus a doc_hash sample of about spoken_units paragraphs with
    p_spoken >= spoken_min and p_chat < KEEP_CHAT ("spoken", mostly fiction dialogue); validation/ holds the
    heldout chat paragraphs. One row per paragraph, id "<doc id>#<paragraph>", script from the document.
    name sets the file name (web-colloq-<name>.parquet). With screen_dir, only paragraphs whose text the quality
    screen marked ok in <screen_dir>/<name>*-screen-*.jsonl are written (export-r2 had no review; r1 had);
    with guard (a frozen eval directory), paragraphs containing an eval overlap pattern are dropped."""
    import hashlib
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    select_dir, out = Path(select_dir), Path(out)
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} is complete; use a new directory")
    files = sorted(p for p in select_dir.glob("*.parquet"))
    cols = ["id", "para", "text", "p_chat", "p_spoken", "split", "script"]
    ok = screened_ok_glob(screen_dir, f"{name}*-screen-*.jsonl") if screen_dir else None
    blocked = eval_guard(guard) if guard else None
    fname = f"web-colloq-{name}.parquet"

    def spoken_mask(t):
        return pc.and_(pc.and_(pc.equal(t["split"], "train"), pc.greater_equal(t["p_spoken"], spoken_min)),
                       pc.less(t["p_chat"], KEEP_CHAT))

    spoken_total = sum(pc.sum(spoken_mask(pq.read_table(p, columns=["p_chat", "p_spoken", "split"]))).as_py() or 0
                       for p in files)
    cut = min(1.0, spoken_units / max(1, spoken_total)) * TWO64
    schema = pa.schema([("id", pa.string()), ("text", pa.string()), ("script", pa.string()), ("origin", pa.string()),
                        ("p_chat", pa.float32()), ("p_spoken", pa.float32())])
    for split in ("train", "validation"):
        (out / split).mkdir(parents=True, exist_ok=True)
    writers = {s: pq.ParquetWriter(out / s / fname, schema, compression="zstd")
               for s in ("train", "validation")}
    counts = Counter()
    buffers = {s: [] for s in writers}

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
            buffers[split].append(dict(id=key, text=r["text"], script=r["script"], origin=origin,
                                       p_chat=r["p_chat"], p_spoken=r["p_spoken"]))
            counts[f"{split}_{origin}_paragraphs"] += 1
            counts[f"{split}_{origin}_characters"] += len(r["text"])
            flush(split)
        print(p.name, dict(counts), flush=True)
    for split, w in writers.items():
        flush(split, force=True)
        w.close()
    sums = {f"{s}/{fname}": hashlib.sha256((out / s / fname).read_bytes()).hexdigest()
            for s in writers}
    with open(out / "SHA256SUMS", "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"{d}  {n}\n" for n, d in sums.items()))
    summary = select_dir / "summary.json"
    extra = {}
    if screen_dir:
        extra["screen"] = screen_note(screen_dir, [f"{name}*-screen-*.jsonl"])
        extra["screen_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(Path(screen_dir).glob(f"{name}*-screen-*.jsonl"))}
    if guard:
        extra["eval_guard_manifest_sha256"] = hashlib.sha256((Path(guard) / "manifest.json").read_bytes()).hexdigest()
    manifest = dict(kind=f"zh-colloq-{name}-export", select_dir=str(select_dir), **extra,
                    select_summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    chat_min=chat_min, spoken_min=spoken_min, spoken_below_p_chat=KEEP_CHAT,
                    spoken_label=SPOKEN_LABEL, spoken_total=spoken_total, spoken_units=spoken_units,
                    heldout_label=HELDOUT_LABEL, counts=dict(counts), sha256=sums)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), "utf-8")
    print(json.dumps(manifest["counts"]))

CHAT_LABEL = "zh-colloq-chat"
CHAT_QUOTAS = {"douban": 800000, "mmchat": 600000, "tieba": 120000}   # 20 : 15 : 3 of the colloquial share
CHAT_VALIDATION = 6000


def screened_ok(screen_dir, source):
    """Text ids (sha1(text)[:16]) the quality screen marked ok (any model marking it not ok drops it), over every screen file of source."""
    return screened_ok_glob(screen_dir, f"{source}*-screen-*.jsonl")


def export_chat(chat_dir, screen_dir, out, quotas=CHAT_QUOTAS, validation_cap=CHAT_VALIDATION, guard=None):
    """Write the screened chat-ext turns as a frozen chat export for build_zh_pinyin.py
    (train/chat-lccc-train.parquet, validation/chat-lccc-validation.parquet; text, dialogue, turn, script).

    Only turns whose text passed the quality screen are eligible (an unscreened turn counts as not ok; the
    screen inputs are shuffled distinct texts, so a partly screened source is a random subset). Train: per
    source, the quota is filled by doc_hash(CHAT_LABEL, "<source>/<dialogue>/<turn>") below quota / eligible;
    a source with fewer eligible turns than its quota (tieba) repeats every turn quota / eligible times, the
    fractional part by the same hash of "<key>#<copy>", each copy under its own dialogue "<source>.r<copy>-<d>"
    so the builder draws it with its own seed. Validation: the screened heldout turns, at most validation_cap
    per source by the same hash. The builder keys a turn "lccc/<dialogue>/<turn>"; dialogue is "<source>-<d>".
    With guard (a frozen eval directory), turns containing an eval overlap pattern are not eligible."""
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    chat_dir, out = Path(chat_dir), Path(out)
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} is complete; use a new directory")
    schema = pa.schema([("text", pa.string()), ("dialogue", pa.string()), ("turn", pa.int32()),
                        ("script", pa.string())])
    counts, rows = Counter(), {"train": [], "validation": []}
    blocked = eval_guard(guard) if guard else None
    for source, quota in quotas.items():
        ok = screened_ok(screen_dir, source)
        for split, name, cap in (("train", "train.jsonl", quota), ("validation", "heldout.jsonl", validation_cap)):
            eligible = []
            for line in open(chat_dir / source / name, encoding="utf-8"):
                r = json.loads(line)
                counts[f"{split}_{source}_turns"] += 1
                if hashlib.sha1(r["text"].encode()).hexdigest()[:16] in ok:
                    if blocked and blocked(r["text"]):
                        counts[f"{split}_{source}_eval_overlap_dropped"] += 1
                        continue
                    eligible.append(r)
            counts[f"{split}_{source}_screened_ok"] += len(eligible)
            ratio = cap / max(1, len(eligible))
            whole, frac = (int(ratio), ratio - int(ratio)) if split == "train" and ratio > 1 else (0, min(ratio, 1.0))
            for r in eligible:
                key = f'{source}/{r["dialogue"]}/{r["turn"]}'
                copies = [k for k in range(whole)]
                if doc_hash(CHAT_LABEL, f"{key}#{whole}" if whole else key) < frac * TWO64:
                    copies.append(whole)
                for k in copies:
                    dialogue = f'{source}.r{k}-{r["dialogue"]}' if k else f'{source}-{r["dialogue"]}'
                    rows[split].append(dict(text=r["text"], dialogue=dialogue, turn=int(r["turn"]), script=r["script"]))
                    counts[f"{split}_{source}_rows"] += 1
                    counts[f"{split}_{source}_characters"] += len(r["text"])
        print(source, {k: v for k, v in counts.items() if f"_{source}_" in k}, flush=True)
    sums = {}
    for split, chunk in rows.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        name = f"{split}/chat-lccc-{split}.parquet"
        pq.write_table(pa.Table.from_pylist(chunk, schema=schema), out / name, row_group_size=20000,
                       compression="zstd")
        sums[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    with open(out / "SHA256SUMS", "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"{d}  {n}\n" for n, d in sums.items()))
    screens = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for source in quotas for p in sorted(Path(screen_dir).glob(f"{source}*-screen-*.jsonl"))}
    manifests = {s: hashlib.sha256((chat_dir / s / "manifest.json").read_bytes()).hexdigest() for s in quotas}
    manifest = dict(kind="zh-colloq-chat-export", label=CHAT_LABEL, quotas=quotas, validation_cap=validation_cap,
                    screen=screen_note(screen_dir, [f"{source}*-screen-*.jsonl" for source in quotas]),
                    screen_sha256=screens,
                    chat_ext_manifest_sha256=manifests,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    counts=dict(sorted(counts.items())), sha256=sums)
    if guard:
        manifest["eval_guard_manifest_sha256"] = hashlib.sha256((Path(guard) / "manifest.json").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), "utf-8")
    print(json.dumps(manifest["counts"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("export")
    s.add_argument("output")
    s.add_argument("--pools")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--max-groups", type=int, default=0)
    s.add_argument("--limit-files", type=int, default=0)
    m = sub.add_parser("sample")
    m.add_argument("scan_dir")
    m.add_argument("out")
    m.add_argument("--per-band", type=int, default=1000)
    m.add_argument("--seed", type=int, default=20261122)
    t = sub.add_parser("train")
    t.add_argument("out")
    t.add_argument("--cases", nargs="+", required=True)
    t.add_argument("--labels", nargs="+", required=True)
    t.add_argument("--C", type=float, default=8.0)
    e = sub.add_parser("select")
    e.add_argument("export")
    e.add_argument("model")
    e.add_argument("output")
    e.add_argument("--pools")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--max-groups", type=int, default=0)
    e.add_argument("--limit-files", type=int, default=0)
    e.add_argument("--range", dest="file_range", help="a:b slice of the sorted export files")
    x = sub.add_parser("export")
    x.add_argument("select_dir")
    x.add_argument("out")
    x.add_argument("--chat-min", type=float, default=0.7)
    x.add_argument("--spoken-units", type=int, default=730000)
    x.add_argument("--name", default="r1", help="file name web-colloq-<name>.parquet and screen file prefix")
    x.add_argument("--screen-dir", help="keep only paragraphs marked ok in <dir>/<name>*-screen-*.jsonl")
    x.add_argument("--guard", help="frozen eval directory: drop paragraphs with an eval overlap pattern")
    c = sub.add_parser("export-chat")
    c.add_argument("chat_dir")
    c.add_argument("screen_dir")
    c.add_argument("out")
    c.add_argument("--validation-cap", type=int, default=CHAT_VALIDATION)
    c.add_argument("--quota", action="append", default=[], metavar="SOURCE=ROWS",
                   help="train rows per source (repeat; default douban 800000, mmchat 600000, tieba 120000)")
    c.add_argument("--guard", help="frozen eval directory: turns with an eval overlap pattern are not eligible")
    i = sub.add_parser("screen-input")
    i.add_argument("export_dir")
    i.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "export-chat":
        quotas = {k: int(v) for k, v in (q.split("=") for q in a.quota)} if a.quota else CHAT_QUOTAS
        export_chat(a.chat_dir, a.screen_dir, a.out, quotas, a.validation_cap, a.guard)
    elif a.cmd == "screen-input":
        screen_input(a.export_dir, a.out)
    elif a.cmd == "export":
        export(a.select_dir, a.out, a.chat_min, spoken_units=a.spoken_units, name=a.name, screen_dir=a.screen_dir,
               guard=a.guard)
    elif a.cmd == "scan":
        scan(a.export, a.output, a.pools, a.workers, a.max_groups, a.limit_files)
    elif a.cmd == "sample":
        sample(a.scan_dir, a.out, a.per_band, a.seed)
    elif a.cmd == "train":
        train(a.out, a.cases, a.labels, a.C)
    else:
        select(a.export, a.model, a.output, a.pools, a.workers, a.max_groups, a.limit_files, a.file_range)


if __name__ == "__main__":
    main()
