"""Build the public Chinese pinyin-generation test set ime-pinyin-zh-v1 (whole clauses, three kinds of keys).

usage:
  python -I -X utf8 scripts/benchmarks/prepare_pinyin_ime_zh.py build OUT [--per-source 300]
      Samples one clause per dialogue and writes OUT/items.jsonl (input to screen_overlap_zh.py) and
      OUT/records.jsonl.
  python -I -X utf8 scripts/benchmarks/screen_overlap_zh.py OUT/items.jsonl OUT/overlap \\
      outputs/zh-corpus/export-r1/train/*.parquet
  python -I -X utf8 scripts/benchmarks/prepare_pinyin_ime_zh.py freeze OUT FROZEN
      Drops every clause the overlap screen flagged and writes FROZEN/{development,test}.jsonl and
      FROZEN/manifest.json.

Sources and exclusions:
  - CrossWOZ, RiSAWOZ, KdConv and CPED, loaded and filtered as in ime-predict-zh-v1 (person turns;
    KdConv turns that quote knowledge are skipped; annotated entity and slot values are protected);
  - a dialogue is a candidate only if it never entered ime-standard-zh-v1 (its draft) or
    ime-predict-zh-v1 (any split), and the 16-gram leak scan of ime-predict-zh-v1 did not flag it;
  - the frozen set then also drops every clause of 8 or more characters that occurs verbatim in
    export-r1/train (screen_overlap_zh.py).

Items:
  - a clause is a maximal Han run bounded by the message ends or punctuation on both sides, 5-20
    characters, not touching a protected value, not starting with a particle; each clause text
    appears once in the set;
  - the target length rotates over 5-7, 8-12 and 13-20 characters, relaxed when a dialogue has
    no clause of the wanted length;
  - readings are automatic and kept only where pypinyin, pypinyin over jieba words and g2pM all
    agree on every syllable (no review);
  - keys follow the private benchmark's format: full pinyin (u-umlaut as v) plus up to five
    distinct word-level abbreviations. Words come from jieba (HMM off). The first abbreviation covers
    every word (the "initials" kind: first letter of each syllable, zh/ch/sh become z/c/s); in the
    others every word is independently abbreviated with probability 0.5, and any that covers every
    word again is dropped as a duplicate, so they are all "mixed". (The private benchmark has an
    initials input for 45% of its sentences; here every sentence has one, so the kind can be read
    on its own.)
  - dialogues are split development/test by a seeded hash, half each. Only development is read
    for now; test is held back.
Dialogue texts stay in ignored directories. Nothing here reads a model score or a candidate list.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREDICT_FROZEN = ROOT / "outputs/benchmarks/ime-predict-zh-v1/frozen-r1"
PREDICT_SCANS = [ROOT / "outputs/benchmarks/ime-predict-zh-v1/leak-scan-r1",
                 ROOT / "outputs/benchmarks/ime-predict-zh-v1/leak-scan-r2-cped"]
SOURCES = ("crosswoz", "risawoz", "kdconv", "cped")
SEED = 20261114
BUCKETS = ((5, 7), (8, 12), (13, 20))
VARIANTS = 5
ATTEMPTS = 200
VERSION = "ime-pinyin-zh-v1-r1"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/benchmarks/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clauses(text, blocked, han):
    """(start, end) of every Han run bounded by the message ends or punctuation on both sides."""
    out = []
    for m in han.finditer(text):
        s, e = m.start(), m.end()
        before = s == 0 or unicodedata.category(text[s - 1]).startswith(("P", "Z"))
        after = e == len(text) or unicodedata.category(text[e]).startswith(("P", "Z"))
        if before and after and not blocked.intersection(range(s, e)):
            out.append((s, e))
    return out


def abbreviations(words, rng):
    """All words, then up to VARIANTS - 1 more distinct non-empty sets, each word in with probability 0.5."""
    n = len(words)
    found = [tuple(range(n))]
    for _ in range(ATTEMPTS):
        if len(found) == VARIANTS:
            break
        pick = tuple(k for k in range(n) if rng.random() < 0.5)
        if pick and pick not in found:
            found.append(pick)
    return found


def record(text, syllables, jieba, rng):
    words, pos = [], 0
    for w in jieba.cut(text, HMM=False):
        py = syllables[pos:pos + len(w)]
        words.append(dict(text=w, start=pos, end=pos + len(w), pinyin=py, full_input="".join(py),
                          initial_input="".join(x[0] for x in py)))
        pos += len(w)
    assert pos == len(text)
    out = []
    for pick in abbreviations(words, rng):
        chosen = set(pick)
        word_inputs = [w["initial_input"] if k in chosen else w["full_input"] for k, w in enumerate(words)]
        syl = [x for k, w in enumerate(words) for x in ([y[0] for y in w["pinyin"]] if k in chosen else w["pinyin"])]
        out.append(dict(input="".join(word_inputs), syllables=syl, word_inputs=word_inputs,
                        abbreviated_word_indices=list(pick)))
    return dict(text=text, length=len(text), full_pinyin=" ".join(syllables), full_input="".join(syllables),
                words=words, abbreviations=out, abbreviation_count=len(out))


def build(args):
    predict, standard = load("prepare_predict_ime_zh"), load("prepare_standard_ime_zh")
    out = Path(args.out)
    out.mkdir(parents=True)
    used = set()
    for name in ("items-a.jsonl", "messages-b.jsonl", "items-c.jsonl"):
        with open(PREDICT_FROZEN / name, encoding="utf-8") as stream:
            for line in stream:
                i = json.loads(line)
                used.add(f'{i["source"]}|{i["source_split"]}|{i["dialogue_id"]}')
    dropped = set()
    for scan in PREDICT_SCANS:
        for keys in json.loads((scan / "scan.json").read_text(encoding="utf-8"))["dropped"].values():
            dropped |= set(keys)
    reader = standard.Reader()
    pools = collections.defaultdict(list)
    stats = collections.defaultdict(collections.Counter)
    for d, _ in predict.candidates(SOURCES):
        k = predict.key(d)
        if k in used:
            stats[d["source"]]["used_by_predict"] += 1
        elif k in dropped:
            stats[d["source"]]["leak_scan_hit"] += 1
        else:
            pools[d["source"]].append(d)
    for source, found in predict.candidates.counts.items():
        stats[source].update(found)
    seen_text, items, records = set(), [], []
    for source in SOURCES:
        rng = random.Random(f"{SEED}/{source}")
        dialogues = sorted(pools[source], key=lambda d: (d["source_split"], d["dialogue_id"]))
        stats[source]["clean_candidates"] = len(dialogues)
        rng.shuffle(dialogues)
        made = 0
        for d in dialogues:
            if made >= args.per_source:
                break
            want = BUCKETS[made % len(BUCKETS)]
            texts = [predict.norm(t) for t, _, _ in d["messages"]]
            options = []
            for k, (_, human, ok) in enumerate(d["messages"]):
                if not (human and ok and predict.clean(texts[k])):
                    continue
                blocked = predict.blocked_positions(texts[k], d["protected"])
                for s, e in clauses(texts[k], blocked, predict.HAN):
                    c = texts[k][s:e]
                    if 5 <= len(c) <= 20 and c[0] not in predict.PARTICLES and c not in seen_text:
                        options.append((k, c))
            rng.shuffle(options)
            pick = None
            for relax in (0, 1):
                for k, c in options:
                    if relax == 0 and not want[0] <= len(c) <= want[1]:
                        continue
                    syllables, readers = reader.readings(c)
                    if syllables is None or any(len(set(v)) > 1 for v in zip(*readers)):
                        stats[source]["reading_disagreement"] += 1
                        continue
                    pick = (k, c, syllables)
                    break
                if pick:
                    stats[source][f"relax{relax}"] += 1
                    break
            if pick is None:
                stats[source]["no_clause"] += 1
                continue
            k, c, syllables = pick
            seen_text.add(c)
            rec = record(c, syllables, reader.jieba, rng)
            meta = dict(source=source, source_split=d["source_split"], dialogue_id=d["dialogue_id"], turn=k)
            records.append(dict(meta, **rec))
            made += 1
        if made < args.per_source:
            raise SystemExit(f"{source}: only {made} usable dialogues")
        stats[source]["items"] = made
    for source in SOURCES:
        rows = sorted((r for r in records if r["source"] == source),
                      key=lambda r: hashlib.sha256(f'{SEED}/{r["dialogue_id"]}'.encode()).hexdigest())
        for n, r in enumerate(rows):
            r["split"] = ("development", "test")[n % 2]
            r["id"] = f'zh-pin-{source}-{hashlib.sha256((source + "|" + r["source_split"] + "|" + r["dialogue_id"]).encode()).hexdigest()[:10]}'
    records.sort(key=lambda r: r["id"])
    if len({r["id"] for r in records}) != len(records):
        raise SystemExit("duplicate ids")
    for r in records:
        items.append(dict(id=r["id"], source=r["source"], split=r["split"], message=r["text"],
                          span=[0, len(r["text"])], target=r["text"]))
    write(out / "items.jsonl", items)
    write(out / "records.jsonl", records)
    manifest = dict(version=VERSION + "-draft", seed=SEED, per_source=args.per_source, sources=list(SOURCES),
                    build={k: dict(v) for k, v in stats.items()}, counts=counts(records), python=sys.version.split()[0])
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


def kind(r, a):
    return "initials" if len(a["abbreviated_word_indices"]) == len(r["words"]) else "mixed"


def counts(records):
    c = collections.Counter()
    for r in records:
        c[f'{r["split"]}/sentences'] += 1
        c[f'{r["split"]}/full'] += 1
        for a in r["abbreviations"]:
            c[f'{r["split"]}/{kind(r, a)}'] += 1
        c[f'{r["split"]}/length-' + next(f"{lo}-{hi}" for lo, hi in BUCKETS if lo <= r["length"] <= hi)] += 1
    return dict(sorted(c.items()))


def write(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def freeze(args):
    draft, out = Path(args.draft), Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; use a new directory")
    summary = json.loads((draft / "overlap/overlap-summary.json").read_text(encoding="utf-8"))
    flagged = {json.loads(l)["id"] for l in open(draft / "overlap/overlap.jsonl", encoding="utf-8") if l.strip()}
    records = [json.loads(l) for l in open(draft / "records.jsonl", encoding="utf-8") if l.strip()]
    kept = [r for r in records if r["id"] not in flagged]
    out.mkdir(parents=True)
    digests = {}
    for split in ("development", "test"):
        rows = [r for r in kept if r["split"] == split]
        write(out / f"{split}.jsonl", rows)
        digests[f"{split}.jsonl"] = hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest()
    draft_manifest = json.loads((draft / "manifest.json").read_text(encoding="utf-8"))
    by_source = collections.Counter(r["source"] for r in records if r["id"] in flagged)
    manifest = dict(version=VERSION, seed=draft_manifest["seed"], sources=draft_manifest["sources"],
                    build=draft_manifest["build"],
                    overlap=dict(min_len=summary["min_len"], corpus_docs=sum(summary["corpus_docs"].values()),
                                 screened=summary["screened"], dropped=len(flagged), dropped_by_source=dict(by_source)),
                    counts=counts(kept),
                    labels="source text; readings automatic (pypinyin, pypinyin over jieba, g2pM agree); "
                           "abbreviations synthetic (all initials, then word level p=0.5)",
                    sha256=digests, python=sys.version.split()[0])
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("overlap", "counts")}, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("build")
    p.add_argument("out")
    p.add_argument("--per-source", type=int, default=300)
    p = commands.add_parser("freeze")
    p.add_argument("draft")
    p.add_argument("out")
    args = parser.parse_args()
    {"build": build, "freeze": freeze}[args.command](args)


if __name__ == "__main__":
    main()
