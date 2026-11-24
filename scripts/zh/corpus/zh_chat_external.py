"""Clean external Chinese chat corpora into per-turn chat text, with the same rules as LCCC (zh_lccc.py).

Sources (local copies under datasets/chat-external/, never committed):
  douban  Douban Conversation Corpus (MarkWuNLP/MultiTurnResponseSelection): lines "label\\tturn...\\tresponse",
          word-segmented; only label 1 lines; a dialogue is its turns plus the true response, deduplicated.
  mmchat  MMChat (silver/mmchat), the authors' LCCC-filtered dialogues mmchat_lccc_filtered/dialog_lccc_flt.jsonl.gz.
  tieba   Baidu Tieba Sun Xiaochuan bar (Orphanage/Baidu_Tieba_SunXiaochuan), original.json: one post = one dialogue
          (title, first-floor text, replies).

Per turn: scrape_cleanup first (forum page residue, mostly Tieba: a trailing "来自Android客户端12楼<timestamp>" is
stripped, a leading "@name:" reply prefix is stripped; folded sub-floor threads ("我也说一句", "点击查看", glued
timestamps), other @mentions, [lbk]/[rbk] placeholders other than [lbk]表情[rbk], Markdown structure and video
player text drop the turn). Then zh_lccc.clean_turn (detokenise, Traditional -> Simplified, MIN_HAN, the stage-A term lists: a hit drops
the turn, a lone weak term goes to review.jsonl). Then:
- turns of HELDOUT_MIN_CHARS or more that occur in LCCC validation or test are removed (LCCC test is the
  colloquial evaluation source); with --dedup-lccc-train such turns that occur in LCCC train are removed too
  (MMChat overlaps LCCC heavily, and LCCC train is already in the pinyin data);
- split: doc_hash("zh-chat-ext-heldout", "<source>/<dialogue>") < HELDOUT_FRACTION -> heldout (evaluation only);
- train turns repeated n times keep ceil(sqrt(n)) copies, per source (as LCCC).
Writes <out>/<source>/{train,heldout,review}.jsonl ({text, dialogue, turn, script}) and <out>/<source>/manifest.json;
each source is done once per --out and can be run separately.

  python scripts/zh/corpus/zh_chat_external.py --root datasets/chat-external --lccc outputs/zh-corpus/lccc-v2 --out outputs/zh-corpus/chat-ext-v1
"""
import argparse
import collections
import gzip
import json
import math
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "zh" / "corpus"))
from vimeml.corpus.zh_filter import Thresholds, load_rules  # noqa: E402
from vimeml.tokenizer.encode_zh13 import doc_hash  # noqa: E402
from zh_lccc import HELDOUT_MIN_CHARS, clean_turn, sha256  # noqa: E402

HELDOUT_LABEL = "zh-chat-ext-heldout"
HELDOUT_FRACTION = 0.02
FILES = {"douban": "douban/DoubanConversaionCorpus.zip",
         "mmchat": "mmchat/mmchat_lccc_filtered/dialog_lccc_flt.jsonl.gz",
         "tieba": "tieba-sunxiaochuan/original.json"}


CLIENT_TAIL = re.compile(r"来自\S{0,12}客户端(\d+楼)?(20\d\d-\d{1,2}-\d{3,4}:\d\d)?\s*$")
REPLY_HEAD = re.compile(r"^@[^\s:：@]{1,24}[:：]")
DROP = [("subfloor", re.compile(r"我也说一句|点击查看|20\d\d-\d{1,2}-\d{3,4}:\d\d")),
        ("mention", re.compile(r"@\S")),
        ("placeholder", re.compile(r"\[lbk\]|\[rbk\]")),
        ("markdown", re.compile(r"\*\*|###|-{3,}")),
        ("video", re.compile(r"\d\d:\d\d/\d\d:\d\d"))]


def scrape_cleanup(raw):
    text = REPLY_HEAD.sub("", CLIENT_TAIL.sub("", raw)).replace("[lbk]表情[rbk]", "").strip()
    for reason, pattern in DROP:
        if pattern.search(text):
            return None, reason
    return text, None


def read_douban(path):
    seen = set()
    with zipfile.ZipFile(path) as z:
        for name in ("train.txt", "dev.txt", "test.txt"):
            with z.open(f"DoubanConversaionCorpus/{name}") as f:
                for line in f:
                    parts = line.decode("utf-8").rstrip("\n").split("\t")
                    if parts[0] != "1":
                        continue
                    turns = tuple(t for t in parts[1:] if t.strip())
                    if turns and turns not in seen:
                        seen.add(turns)
                        yield list(turns)


def read_mmchat(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            turns = json.loads(line)
            if turns:
                yield turns


def read_tieba(path):
    for post in json.loads(Path(path).read_text("utf-8")):
        turns = [post.get("标题") or "", post.get("楼主内容") or "", *(post.get("回复列表") or [])]
        turns = [t for t in turns if isinstance(t, str) and t.strip()]
        if turns:
            yield turns


READERS = {"douban": read_douban, "mmchat": read_mmchat, "tieba": read_tieba}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT / "datasets/chat-external"))
    p.add_argument("--lccc", default=str(ROOT / "outputs/zh-corpus/lccc-v2"))
    p.add_argument("--terms", default=str(ROOT / "configs/zh/zh-filter-terms.toml"))
    p.add_argument("--out", default=str(ROOT / "outputs/zh-corpus/chat-ext-v1"))
    p.add_argument("--sources", nargs="+", default=list(FILES))
    p.add_argument("--dedup-lccc-train", action="store_true")
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    done = [s for s in a.sources if (out / s / "manifest.json").exists()]
    if done:
        raise SystemExit(f"{done} already complete in {out}; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    rules, th = load_rules(a.terms), Thresholds()
    lccc_heldout = set()
    for split in ("validation", "test"):
        with open(Path(a.lccc) / f"{split}.jsonl", encoding="utf-8") as f:
            lccc_heldout.update(t for t in (json.loads(l)["text"] for l in f) if len(t) >= HELDOUT_MIN_CHARS)
    lccc_train = set()
    if a.dedup_lccc_train:
        with open(Path(a.lccc) / "train.jsonl", encoding="utf-8") as f:
            lccc_train.update(t for t in (json.loads(l)["text"] for l in f) if len(t) >= HELDOUT_MIN_CHARS)
    for source in a.sources:
        c, rows, review = collections.Counter(), {"train": [], "heldout": []}, []
        for d, turns in enumerate(READERS[source](root / FILES[source])):
            c["dialogues"] += 1
            split = "heldout" if doc_hash(HELDOUT_LABEL, f"{source}/{d}") < HELDOUT_FRACTION * 2 ** 64 else "train"
            for j, raw in enumerate(turns):
                raw, scrape = scrape_cleanup(raw)
                if scrape:
                    c[f"scrape_{scrape}"] += 1
                    continue
                text, script, decision, reason, found = clean_turn(raw, rules, th)
                c[decision] += 1
                c[f"script_{script}"] += 1
                if reason:
                    c[reason] += 1
                row = dict(text=text, dialogue=d, turn=j, script=script)
                if decision == "review":
                    review.append(dict(row, hits=found))
                    continue
                if decision != "keep":
                    continue
                if len(text) >= HELDOUT_MIN_CHARS and text in lccc_heldout:
                    c["lccc_heldout_overlap"] += 1
                    continue
                if len(text) >= HELDOUT_MIN_CHARS and text in lccc_train:
                    c["lccc_train_overlap"] += 1
                    continue
                rows[split].append(row)
        freq = collections.Counter(r["text"] for r in rows["train"])
        emitted, train = collections.Counter(), []
        for r in rows["train"]:
            if emitted[r["text"]] >= math.isqrt(freq[r["text"]] - 1) + 1:
                c["duplicate"] += 1
                continue
            emitted[r["text"]] += 1
            train.append(r)
        rows["train"] = train
        c["top_repeats"] = [[t, n] for t, n in freq.most_common(20)]
        (out / source).mkdir(exist_ok=True)
        for split, rs in rows.items():
            with open(out / source / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for r in rs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            c[f"{split}_turns"] = len(rs)
            c[f"{split}_chars"] = sum(len(r["text"]) for r in rs)
        with open(out / source / "review.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in review:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c["review"] = len(review)
        manifest = dict(source=source, file=FILES[source], sha256=sha256(root / FILES[source]),
                        terms_sha256=sha256(a.terms), script_sha256=sha256(__file__),
                        lccc_manifest_sha256=sha256(Path(a.lccc) / "manifest.json"),
                        heldout_label=HELDOUT_LABEL, heldout_fraction=HELDOUT_FRACTION,
                        heldout_min_chars=HELDOUT_MIN_CHARS, dedup_lccc_train=a.dedup_lccc_train,
                        repeat_policy="ceil(sqrt(n)) copies per source", counts=dict(c))
        (out / source / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
        print(source, {k: v for k, v in c.items() if k != "top_repeats"}, flush=True)


if __name__ == "__main__":
    main()
