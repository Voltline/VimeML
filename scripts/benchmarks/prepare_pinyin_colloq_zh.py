"""Build the colloquial pinyin-generation set ime-pinyin-zh-colloq-v1 (the primary metric of the v2-20m
colloquial post-training). Same record format as ime-pinyin-zh-v1, so pinyin_gen_zh.py runs it with --frozen.

usage:
  python -X utf8 scripts/benchmarks/prepare_pinyin_colloq_zh.py build OUT
      writes OUT/items.jsonl (input to screen_overlap_zh.py), OUT/records.jsonl, OUT/contexts.jsonl
  python -I -X utf8 scripts/benchmarks/screen_overlap_zh.py OUT/items.jsonl OUT/overlap \\
      outputs/zh-corpus/export-r1/train/*.parquet outputs/zh-corpus/lccc-v2/train.jsonl \\
      outputs/zh-corpus/chat-ext-v2/*/train.jsonl
  python -X utf8 scripts/benchmarks/prepare_pinyin_colloq_zh.py freeze OUT FROZEN
      drops every flagged item; writes FROZEN/{development,test}.jsonl, FROZEN/{development,test}-contexts.jsonl
      and FROZEN/manifest.json

Sources (all evaluation-only splits, never in any training mix):
  lccc    LCCC test (outputs/zh-corpus/lccc-v2/test.jsonl), one message per dialogue
  r1      export-r1 paragraphs in the zh-colloq heldout split with p_chat >= P_CHAT (zh_colloq.py select),
          one paragraph per document
  douban, mmchat, tieba
          chat-ext-v2 heldout turns that passed the LLM quality screen; one message per dialogue,
          tieba up to TIEBA_PER_POST per post (its heldout split has only 45 posts)

Items: as ime-pinyin-zh-v1 (prepare_pinyin_ime_zh.py): a clause is a maximal Han run bounded by message ends or
punctuation, not starting with a predict-set particle, each clause text once; readings kept only where pypinyin,
pypinyin over jieba and g2pM agree; keys are full pinyin, all initials and up to four mixed abbreviations.
Differences: clauses are 4-20 characters (buckets 4-7, 8-12, 13-20, rotating), and every other item prefers a
clause ending in a sentence-final particle (zh_colloq.PARTICLES; relaxed when the message has none), so the
"final particle" stratum is about half the set. Context is infield only (the text before the clause in its
own message, contexts_pinyin_ime_zh.infield): the keyboard sees nothing of earlier turns.
Splits: development/test by a seeded hash, half each. Test is held back for the confirmation run.
"""
import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/benchmarks"))
sys.path.insert(0, str(ROOT / "scripts/corpus"))
import prepare_pinyin_ime_zh as P  # noqa: E402
from contexts_pinyin_ime_zh import infield  # noqa: E402

VERSION = "ime-pinyin-zh-colloq-v1-r1"
SEED = 20261122
BUCKETS = ((4, 7), (8, 12), (13, 20))
FINAL = set("吧呢啊嘛呀啦哦呗咯嘞噢哇喔呐咩")
P_CHAT = 0.7
TIEBA_PER_POST = 8
PER_SOURCE = {"lccc": 600, "r1": 600, "douban": 300, "mmchat": 300, "tieba": 200}
LCCC = ROOT / "outputs/zh-corpus/lccc-v2/test.jsonl"
SELECT = ROOT / "outputs/zh-colloq/select-r1"
CHAT_EXT = ROOT / "outputs/zh-corpus/chat-ext-v2"
SCREEN = ROOT / "outputs/zh-colloq/screen"


def text_id(t):
    return hashlib.sha1(t.encode()).hexdigest()[:16]


def screened_ok(source):
    ok = set()
    for path in sorted(SCREEN.glob(f"{source}*-screen-*.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["ok"]:
                ok.add(r["id"])
    return ok


def units(source):
    """(unit key, group key, message text) in a fixed order; group = the dialogue or document."""
    if source == "lccc":
        for line in open(LCCC, encoding="utf-8"):
            r = json.loads(line)
            yield f'{r["dialogue"]}/{r["turn"]}', str(r["dialogue"]), r["text"]
    elif source == "r1":
        import pyarrow.parquet as pq
        for path in sorted(SELECT.glob("*.parquet")):
            t = pq.read_table(path, columns=["id", "para", "text", "p_chat", "split"]).to_pylist()
            for r in t:
                if r["split"] == "heldout" and r["p_chat"] >= P_CHAT:
                    yield f'{r["id"]}#{r["para"]}', r["id"], r["text"]
    else:
        ok = screened_ok(source)
        for line in open(CHAT_EXT / source / "heldout.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if text_id(r["text"]) in ok:
                yield f'{r["dialogue"]}/{r["turn"]}', str(r["dialogue"]), r["text"]


def build(args):
    predict, standard = P.load("prepare_predict_ime_zh"), P.load("prepare_standard_ime_zh")
    out = Path(args.out)
    out.mkdir(parents=True)
    reader = standard.Reader()
    seen_text, records, stats = set(), [], collections.defaultdict(collections.Counter)
    for source, want_n in PER_SOURCE.items():
        rng = random.Random(f"{SEED}/{source}")
        groups = collections.defaultdict(list)
        for key, group, text in units(source):
            groups[group].append((key, predict.norm(text)))
        order = sorted(groups)
        rng.shuffle(order)
        cap = TIEBA_PER_POST if source == "tieba" else 1
        stats[source]["groups"] = len(order)
        made, rounds = 0, 0
        taken = collections.Counter()
        while made < want_n and rounds < cap:
            rounds += 1
            for group in order:
                if made >= want_n:
                    break
                if taken[group] >= rounds:
                    continue
                bucket = BUCKETS[made % len(BUCKETS)]
                prefer_final = made % 2 == 0
                options = []
                for key, text in groups[group]:
                    if not 2 <= len(text) <= 300:
                        continue
                    for s, e in P.clauses(text, set(), predict.HAN):
                        c = text[s:e]
                        if 4 <= len(c) <= 20 and c[0] not in predict.PARTICLES and c not in seen_text:
                            options.append((key, text, s, c))
                rng.shuffle(options)
                pick = None
                for relax in (0, 1, 2):
                    for key, text, s, c in options:
                        if relax < 2 and prefer_final and c[-1] not in FINAL:
                            continue
                        if relax < 1 and not bucket[0] <= len(c) <= bucket[1]:
                            continue
                        syllables, readers = reader.readings(c)
                        if syllables is None or any(len(set(v)) > 1 for v in zip(*readers)):
                            stats[source]["reading_disagreement"] += 1
                            continue
                        pick = (key, text, s, c, syllables)
                        break
                    if pick:
                        stats[source][f"relax{relax}"] += 1
                        break
                if pick is None:
                    stats[source]["no_clause"] += 1
                    taken[group] = cap  # nothing usable left in this group
                    continue
                key, text, s, c, syllables = pick
                seen_text.add(c)
                taken[group] += 1
                rec = P.record(c, syllables, reader.jieba, rng)
                records.append(dict(source=source, unit=key, group=group, message=text, span=[s, s + len(c)],
                                    infield=infield(text[:s]), final_particle=c[-1] in FINAL, **rec))
                made += 1
        if made < want_n:
            raise SystemExit(f"{source}: only {made} usable items")
        stats[source]["items"] = made
    for source in PER_SOURCE:
        rows = sorted((r for r in records if r["source"] == source),
                      key=lambda r: hashlib.sha256(f'{SEED}/{r["group"]}'.encode()).hexdigest())
        for n, r in enumerate(rows):
            r["split"] = ("development", "test")[n % 2]
            r["id"] = f'zh-colloq-{source}-{hashlib.sha256((source + "|" + r["unit"]).encode()).hexdigest()[:10]}'
    records.sort(key=lambda r: r["id"])
    if len({r["id"] for r in records}) != len(records):
        raise SystemExit("duplicate ids")
    P.write(out / "items.jsonl", [dict(id=r["id"], source=r["source"], split=r["split"], message=r["message"],
                                       span=r["span"], target=r["text"]) for r in records])
    P.write(out / "records.jsonl", records)
    manifest = dict(version=VERSION + "-draft", seed=SEED, per_source=PER_SOURCE, p_chat=P_CHAT,
                    build={k: dict(v) for k, v in stats.items()}, counts=counts(records),
                    python=sys.version.split()[0])
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


def counts(records):
    c = collections.Counter()
    for r in records:
        sp = r["split"]
        c[f"{sp}/sentences"] += 1
        c[f'{sp}/source-{r["source"]}'] += 1
        c[f"{sp}/final-particle"] += r["final_particle"]
        c[f"{sp}/with-infield"] += bool(r["infield"])
        c[f"{sp}/length-" + next(f"{lo}-{hi}" for lo, hi in BUCKETS if lo <= r["length"] <= hi)] += 1
        for a in r["abbreviations"]:
            c[f"{sp}/{P.kind(r, a)}"] += 1
    return dict(sorted(c.items()))


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
        P.write(out / f"{split}.jsonl", rows)
        P.write(out / f"{split}-contexts.jsonl", [dict(id=r["id"], infield=r["infield"]) for r in rows])
        for name in (f"{split}.jsonl", f"{split}-contexts.jsonl"):
            digests[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    draft_manifest = json.loads((draft / "manifest.json").read_text(encoding="utf-8"))
    by_source = collections.Counter(r["source"] for r in records if r["id"] in flagged)
    manifest = dict(version=VERSION, seed=draft_manifest["seed"], per_source=draft_manifest["per_source"],
                    p_chat=draft_manifest["p_chat"], build=draft_manifest["build"],
                    overlap=dict(min_len=summary["min_len"], corpus_docs=sum(summary["corpus_docs"].values()),
                                 screened=summary["screened"], dropped=len(flagged), dropped_by_source=dict(by_source)),
                    counts=counts(kept),
                    labels="source text; readings automatic (pypinyin, pypinyin over jieba, g2pM agree); "
                           "abbreviations synthetic (all initials, then word level p=0.5); context infield only",
                    sha256=digests, python=sys.version.split()[0])
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("overlap", "counts")}, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("build")
    p.add_argument("out")
    p = commands.add_parser("freeze")
    p.add_argument("draft")
    p.add_argument("out")
    args = parser.parse_args()
    {"build": build, "freeze": freeze}[args.command](args)


if __name__ == "__main__":
    main()
