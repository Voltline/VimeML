"""Build the Chinese prediction benchmark ime-predict-zh-v1 (docs/zh-CN/benchmarks/predict-zh-plan.md).

usage:
  python -I -X utf8 scripts/benchmarks/prepare_predict_ime_zh.py scan OUT [--sources ...] [--workers 10]
      Leak filter, before any sampling: every 16-Han-character n-gram of every message of every
      candidate dialogue is searched in export-r1/train (Han-normalized, as in the v1 leak check).
      Candidates are all dialogues of the chosen sources that never entered ime-standard-zh-v1
      (its draft, which is a superset of development and blind). Writes OUT/scan.json (aggregates
      and the hit dialogue keys) and OUT/hits.jsonl (local only).

Dialogue texts stay in ignored directories. Nothing here reads a model score.
"""
import argparse
import collections
import glob
import json
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets"
STANDARD_DRAFT = ROOT / "outputs/benchmarks/ime-standard-zh-v1/draft-r1/items.jsonl"
TRAIN = ROOT / "outputs/zh-corpus/export-r1/train"
SOURCES = ("crosswoz", "risawoz", "kdconv", "naturalconv", "cped")
HAN = re.compile("[〇㐀-䶿一-鿿豈-﫿]+")
N = 16
MULTIPLIER = 0x9E3779B97F4A7C15
SEPARATOR = 0


def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = "".join(c for c in s if unicodedata.category(c)[0] != "C" or c in "\n\t")
    return " ".join(s.split())


def han(s):
    return "".join(HAN.findall(s))


def protect_values(values):
    out = set()
    for v in values:
        if isinstance(v, (list, tuple)):
            out |= protect_values(v)
            continue
        if not isinstance(v, str):
            continue
        v = re.sub(r"（[^）]*）|\([^)]*\)", "", norm(v).replace(" ", ""))
        if len(v) >= 2 and HAN.search(v):
            out.add(v)
    return out


# Every loader yields whole dialogues: messages are (text, human, eligible). `human` marks a person's
# turn; `eligible` marks turns prediction targets may come from. `protected` holds annotated entity and
# slot values, which targets must not cover.
def load_crosswoz(raw):
    for split in ("train", "val", "test"):
        with zipfile.ZipFile(raw / "ime-standard-zh-v1/raw/CrossWOZ/data/crosswoz" / f"{split}.json.zip") as z:
            data = json.loads(z.read(z.namelist()[0]))
        for did, d in data.items():
            messages, protected = [], set()
            for m in d["messages"]:
                protected |= protect_values([a[3] for a in m.get("dialog_act", [])])
                protected |= protect_values([s[3] for s in m.get("user_state", [])])
                user = m["role"] == "usr"
                messages.append((m["content"], user, user))
            yield dict(source="crosswoz", source_split=split, dialogue_id=str(did), messages=messages, protected=protected)


def load_risawoz(raw):
    names = {"all_train10000.json": "train", "all_dev600.json": "dev", "all_test600_new.json": "test"}
    with zipfile.ZipFile(raw / "ime-standard-zh-v1/raw/RiSAWOZ/RiSAWOZ-data/source-data.zip") as z:
        for n, split in names.items():
            for d in json.loads(z.read("source-data/" + n)):
                messages, protected = [], set()
                for t in d["dialogue"]:
                    protected |= protect_values([a[3] for a in t.get("user_actions", []) if len(a) > 3])
                    protected |= protect_values([a[3] for a in t.get("system_actions", []) if len(a) > 3])
                    protected |= protect_values(list(t.get("belief_state", {}).get("inform slot-values", {}).values()))
                    messages.append((t["user_utterance"], True, True))
                    if t.get("system_utterance"):
                        messages.append((t["system_utterance"], False, False))
                yield dict(source="risawoz", source_split=split, dialogue_id=d["dialogue_id"], messages=messages,
                           protected=protected)


def load_kdconv(raw):
    for domain in ("film", "music", "travel"):
        for split in ("train", "dev", "test"):
            data = json.loads((raw / "ime-standard-zh-v1/raw/KdConv/data" / domain / f"{split}.json").read_text(encoding="utf-8"))
            for k, d in enumerate(data):
                protected, messages = protect_values([d.get("name", "")]), []
                for m in d["messages"]:
                    for a in m.get("attrs", []):
                        protected |= protect_values([a.get("name", "")])
                        if len(a.get("attrvalue", "")) <= 15:
                            protected |= protect_values([a["attrvalue"]])
                    messages.append((m["message"], True, not m.get("attrs")))  # targets avoid turns quoting knowledge
                yield dict(source="kdconv", source_split=split, dialogue_id=f"{domain}/{split}/{k}", messages=messages,
                           protected=protected)


def load_naturalconv(raw):
    base = raw / "ime-predict-zh-v1/raw/NaturalConv"
    split_of = {}
    for split in ("train", "dev", "test"):
        for line in (base / f"{split}.txt").read_text(encoding="utf-8").split():
            split_of[line] = split
    for d in json.loads((base / "dialog_release.json").read_text(encoding="utf-8")):
        yield dict(source="naturalconv", source_split=split_of[d["dialog_id"]], dialogue_id=d["dialog_id"],
                   messages=[(m, True, True) for m in d["content"]], protected=set())


def load_cped(raw):
    """CPED (GitHub release, data/CPED): TV-series dialogue transcripts, one row per utterance."""
    import csv
    base = raw / "ime-predict-zh-v1/raw/CPED"
    for split in ("train", "valid", "test"):
        dialogues = collections.defaultdict(list)
        with open(base / f"{split}_split.csv", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                dialogues[row["Dialogue_ID"]].append((row["Utterance_ID"], row["Utterance"]))
        for did in sorted(dialogues):
            yield dict(source="cped", source_split=split, dialogue_id=did,
                       messages=[(t, True, True) for _, t in sorted(dialogues[did])], protected=set())


LOADERS = dict(crosswoz=load_crosswoz, risawoz=load_risawoz, kdconv=load_kdconv, naturalconv=load_naturalconv,
               cped=load_cped)


def key(d):
    return f'{d["source"]}|{d["source_split"]}|{d["dialogue_id"]}'


def candidates(sources):
    """All dialogues of the given sources, minus those drafted for ime-standard-zh-v1."""
    used = set()
    with open(STANDARD_DRAFT, encoding="utf-8") as stream:
        for line in stream:
            i = json.loads(line)
            used.add(f'{i["source"]}|{i["source_split"]}|{i["dialogue_id"]}')
    counts = collections.defaultdict(collections.Counter)
    for source in sources:
        for d in LOADERS[source](RAW):
            counts[source]["dialogues"] += 1
            if key(d) in used:
                counts[source]["used_by_standard"] += 1
                continue
            counts[source]["candidates"] += 1
            yield d, counts
    candidates.counts = counts


def hashes(codes):
    """64-bit polynomial hash of every N-window of a uint32 code array (wrapping arithmetic)."""
    import numpy as np
    n = len(codes) - N + 1
    h = np.zeros(n, dtype=np.uint64)
    wide = codes.astype(np.uint64)
    multiplier = np.uint64(MULTIPLIER)
    for k in range(N):
        h *= multiplier
        h += wide[k:k + n]
    return h


PATTERNS = None


def init(path):
    global PATTERNS
    import numpy as np
    PATTERNS = np.load(path)


def scan_file(path):
    """Return (file, documents, [(gram, document index, snippet)]): the first hit of each gram per document."""
    import numpy as np
    import pyarrow.parquet as pq
    documents, found = 0, []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=20000, columns=["text"]):
        texts = [(t or "").replace("\x00", "") for t in batch.column(0).to_pylist()]
        codes = np.frombuffer("\x00".join(texts).encode("utf-32-le"), dtype=np.uint32)
        keep = ((codes >= 0x4E00) & (codes <= 0x9FFF)) | ((codes >= 0x3400) & (codes <= 0x4DBF)) \
            | ((codes >= 0xF900) & (codes <= 0xFAFF)) | (codes == 0x3007) | (codes == SEPARATOR)
        codes = codes[keep]
        if len(codes) >= N:
            h = hashes(codes)
            slot = np.searchsorted(PATTERNS, h)
            slot[slot == len(PATTERNS)] = 0
            hit = np.flatnonzero(PATTERNS[slot] == h)
            if len(hit):
                separators = np.flatnonzero(codes == SEPARATOR)
                seen = set()
                for i in hit.tolist():
                    window = codes[i:i + N]
                    if (window == SEPARATOR).any():
                        continue
                    document = int(np.searchsorted(separators, i))
                    gram = window.tobytes().decode("utf-32-le")
                    if (document, gram) in seen:
                        continue
                    seen.add((document, gram))
                    lo, hi = max(0, i - 40), i + N + 40
                    snippet = codes[lo:hi].tobytes().decode("utf-32-le").replace("\x00", "|")
                    found.append((gram, documents + document, snippet))
        documents += len(texts)
    return Path(path).name, documents, found


def scan(args):
    import numpy as np
    from multiprocessing import Pool
    out = Path(args.out)
    out.mkdir(parents=True)
    owners = collections.defaultdict(set)
    scannable = collections.Counter()
    total = collections.Counter()
    for d, counts in candidates(args.sources):
        any_message = False
        for text, _, _ in d["messages"]:
            h = han(norm(text))
            for k in range(len(h) - N + 1):
                owners[h[k:k + N]].add(key(d))
            any_message |= len(h) >= N
        scannable[d["source"]] += any_message
        total[d["source"]] += 1
    counts = candidates.counts
    grams = sorted(owners)
    codes = np.frombuffer("".join(grams).encode("utf-32-le"), dtype=np.uint32).reshape(len(grams), N)
    multiplier, h = np.uint64(MULTIPLIER), np.zeros(len(grams), dtype=np.uint64)
    for k in range(N):
        h *= multiplier
        h += codes[:, k].astype(np.uint64)
    if len(np.unique(h)) != len(h):
        raise SystemExit("hash collision among benchmark n-grams; change MULTIPLIER")
    np.save(out / "patterns.npy", np.sort(h))
    print(json.dumps({"candidates": dict(total), "grams": len(grams)}, ensure_ascii=False), flush=True)
    files = sorted(glob.glob(str(TRAIN / "*.parquet")))
    documents, hits = {}, []
    with Pool(args.workers, init, (str(out / "patterns.npy"),)) as pool:
        for name, n, found in pool.imap_unordered(scan_file, files):
            documents[name] = n
            for gram, document, snippet in found:
                if gram in owners:  # exact check; a bare hash match is not a hit
                    hits.append(dict(file=name, document=document, gram=gram, dialogues=sorted(owners[gram]), snippet=snippet))
            print(name, n, len(found), flush=True)
    hit_dialogues = collections.defaultdict(set)
    for hit in hits:
        for k in hit["dialogues"]:
            hit_dialogues[k.split("|")[0]].add(k)
    with open(out / "hits.jsonl", "w", encoding="utf-8", newline="\n") as stream:
        for hit in sorted(hits, key=lambda x: (x["file"], x["document"], x["gram"])):
            stream.write(json.dumps(hit, ensure_ascii=False) + "\n")
    summary = dict(
        n=N, train_files=len(files), train_documents=sum(documents.values()), grams=len(grams),
        hit_grams=len({h["gram"] for h in hits}), hit_documents=len({(h["file"], h["document"]) for h in hits}),
        sources={s: dict(counts[s], with_scannable_message=scannable[s], hit_dialogues=len(hit_dialogues[s]),
                         hit_rate=len(hit_dialogues[s]) / max(1, total[s])) for s in args.sources},
        dropped={s: sorted(hit_dialogues[s]) for s in args.sources})
    (out / "scan.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary["sources"], ensure_ascii=False, indent=1))


SEED = 20261111
PER_SOURCE = 1200
POSITIONS = 3
BAD = re.compile(r"https?://|www\.|@\w|<[a-z/]|&[a-z]+;|\[[A-Z_]+\]|�", re.I)
MIN_HAN_SHARE = 0.6
HAN_CHAR = re.compile("[〇㐀-䶿一-鿿豈-﫿]")
PARTICLES = set("的了呢吗吧啊呀哦哈嘛啦着过地得么呗哟喔噢嘞咯")


def clean(text):
    """Mechanical message filter: 4-120 characters, mostly Han, no markup, links or replacement characters."""
    return 4 <= len(text) <= 120 and not BAD.search(text) and len(HAN_CHAR.findall(text)) >= MIN_HAN_SHARE * len(text)


def blocked_positions(text, protected):
    blocked = set()
    for value in protected:
        for m in re.finditer(re.escape(value), text):
            blocked.update(range(m.start(), m.end()))
    return blocked


def next_positions(text, blocked):
    """Track A positions: between two Han characters, with at least two Han characters before the next
    non-Han character, and the first six characters of that continuation not on an annotated value."""
    out = []
    for run in HAN.finditer(text):
        for p in range(run.start() + 1, run.end() - 1):
            if not blocked.intersection(range(p, min(run.end(), p + 6))):
                out.append((p, run.end()))
    return out


def build(args):
    import hashlib
    import importlib.util
    import random
    spec = importlib.util.spec_from_file_location("standard_zh", ROOT / "scripts/benchmarks/prepare_standard_ime_zh.py")
    standard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(standard)
    out = Path(args.out)
    out.mkdir(parents=True)
    scans = [json.loads((Path(s) / "scan.json").read_text(encoding="utf-8")) for s in args.scan]
    scan_summary = dict(n=scans[0]["n"], train_documents=scans[0]["train_documents"], sources={}, dropped={})
    for s in scans:
        if (s["n"], s["train_documents"]) != (scan_summary["n"], scan_summary["train_documents"]):
            raise SystemExit("scans used different n-gram lengths or training exports")
        scan_summary["sources"].update(s["sources"])
        scan_summary["dropped"].update(s["dropped"])
    missing = set(args.sources) - set(scan_summary["sources"])
    if missing:
        raise SystemExit(f"no leak scan for {sorted(missing)}")
    scan_summary["sources"] = {s: scan_summary["sources"][s] for s in args.sources}
    dropped = {k for keys in scan_summary["dropped"].values() for k in keys}
    reader = standard.Reader()
    pools = collections.defaultdict(list)
    for d, _ in candidates(args.sources):
        if d["source"] in args.sources and key(d) not in dropped:
            pools[d["source"]].append(d)
    items_a, items_c, messages_b = [], [], []
    stats = collections.defaultdict(collections.Counter)
    for source in args.sources:
        rng = random.Random(f"{SEED}/{source}")
        dialogues = sorted(pools[source], key=lambda d: (d["source_split"], d["dialogue_id"]))
        stats[source]["clean_candidates"] = len(dialogues)
        rng.shuffle(dialogues)
        kept = []
        for d in dialogues:
            if len(kept) >= PER_SOURCE:
                break
            texts = [norm(t) for t, _, _ in d["messages"]]
            eligible = [k for k, (_, human, ok) in enumerate(d["messages"]) if human and ok and clean(texts[k])]
            blocked = {k: blocked_positions(texts[k], d["protected"]) for k in eligible}
            positions = [(k, p, end) for k in eligible for p, end in next_positions(texts[k], blocked[k])]
            if len(positions) < POSITIONS:
                stats[source]["too_few_positions"] += 1
                continue
            # Track C: one clause-final 2-6 character span with an unambiguous automatic reading.
            spans = []
            for k in eligible:
                for s, e, r in standard.spans(texts[k], d["protected"], reader):
                    if 2 <= e - s <= 6:
                        spans.append((k, s, e, r))
            rng.shuffle(spans)
            completion = None
            for k, s, e, r in spans:
                syllables, readers = reader.readings(texts[k][r:e])
                if syllables is None or any(len(set(v)) > 1 for v in zip(*readers)):
                    continue  # no reading, or the three readers disagree somewhere in the clause
                completion = (k, s, e, syllables[s - r:])
                break
            if completion is None:
                stats[source]["no_completion_span"] += 1
                continue
            kept.append((d, texts, rng.sample(positions, POSITIONS), completion))
        if len(kept) < PER_SOURCE:
            raise SystemExit(f"{source}: only {len(kept)} usable dialogues")
        stats[source]["dialogues"] = len(kept)
        order = sorted(kept, key=lambda x: hashlib.sha256(f'{SEED}/{x[0]["dialogue_id"]}'.encode()).hexdigest())
        for n, (d, texts, positions, (ck, cs, ce, syllables)) in enumerate(order):
            split = ("development", "blind")[n % 2]
            base = f'zh-pred-{source}-{hashlib.sha256(key(d).encode()).hexdigest()[:10]}'
            common = dict(source=source, source_split=d["source_split"], dialogue_id=d["dialogue_id"], split=split,
                          dialogue=f"{source}:{d['dialogue_id']}")
            for j, (k, p, end) in enumerate(sorted(positions)):
                items_a.append(dict(id=f"{base}-a{j + 1}", **common, turn=k, previous_turns=texts[:k],
                                    prefix=texts[k][:p], continuation=texts[k][p:end]))
            for k in sorted({k for k, _, _ in positions}):
                messages_b.append(dict(id=f"{base}-b{k}", **common, turn=k, previous_turns=texts[:k], message=texts[k]))
            items_c.append(dict(id=f"{base}-c", **common, turn=ck, previous_turns=texts[:ck], prefix=texts[ck][:cs],
                                target=texts[ck][cs:ce], syllables=syllables,
                                keys_first_syllable=syllables[0], keys_initials="".join(x[0] for x in syllables)))
    files = {"items-a.jsonl": items_a, "messages-b.jsonl": messages_b, "items-c.jsonl": items_c}
    digests = {}
    for name, rows in files.items():
        rows.sort(key=lambda r: r["id"])
        if len({r["id"] for r in rows}) != len(rows):
            raise SystemExit(f"duplicate ids in {name}")
        with open(out / name, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        digests[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    count = lambda rows: dict(sorted(collections.Counter(f'{r["source"]}/{r["split"]}' for r in rows).items()))
    manifest = dict(version="ime-predict-zh-v1-r1", seed=SEED, per_source=PER_SOURCE, positions=POSITIONS,
                    sources=list(args.sources), leak_scan=dict(n=scan_summary["n"], train_documents=scan_summary["train_documents"],
                                                               by_source=scan_summary["sources"]),
                    build={k: dict(v) for k, v in stats.items()},
                    counts={"track_a": count(items_a), "track_b_messages": count(messages_b), "track_c": count(items_c)},
                    labels="source text; track C readings are automatic (pypinyin, pypinyin over jieba, g2pM agree)",
                    sha256=digests, python=sys.version.split()[0])
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("build", "counts", "sha256")}, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("scan")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    p = commands.add_parser("build")
    p.add_argument("out")
    p.add_argument("--scan", nargs="+", required=True, help="directories written by scan; together they cover --sources")
    p.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    {"scan": scan, "build": build}[args.command](args)


if __name__ == "__main__":
    main()
