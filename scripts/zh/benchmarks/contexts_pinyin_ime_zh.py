"""Left contexts of ime-pinyin-zh-v1: what Vime gives the model before the caret.

infield: the text before the sentence in its own message, by VimeLanguageModel.sentenceContext: the last 64
characters, cut after the last full stop, question mark, exclamation mark or newline. history: the previous
turn, a newline, then infield. Rebuilt from the source dialogues the set was sampled from.

usage:
  python -X utf8 scripts/zh/benchmarks/contexts_pinyin_ime_zh.py [--check]
writes FROZEN/development-contexts.jsonl; --check only compares a rebuild with the existing file.
The test split's contexts are built only by the preregistered confirmation run (build(..., "test")).
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/zh/benchmarks"))
import prepare_pinyin_ime_zh as P  # noqa: E402

FROZEN = ROOT / "outputs/benchmarks/ime-pinyin-zh-v1/frozen-r1"
LIMIT = 64
BREAKS = "\n。！？!?"


def infield(text, limit=LIMIT):
    tail = text[-limit:]
    cut = max((i for i, ch in enumerate(tail) if ch in BREAKS), default=-1)
    return tail[cut + 1:].lstrip()


def build(frozen, split):
    """JSONL text of split's contexts, in the order the source dialogues are read."""
    if split not in ("development", "test"):
        raise ValueError("split is development or test")
    predict = P.load("prepare_predict_ime_zh")
    records = [json.loads(l) for l in open(Path(frozen) / f"{split}.jsonl", encoding="utf-8") if l.strip()]
    want = {(r["source"], r["source_split"], r["dialogue_id"]): r for r in records}
    lines = []
    for dialogue, _ in predict.candidates(P.SOURCES):
        r = want.get((dialogue["source"], dialogue["source_split"], dialogue["dialogue_id"]))
        if r is None:
            continue
        texts = [predict.norm(t) for t, _, _ in dialogue["messages"]]
        start = texts[r["turn"]].find(r["text"])
        if start < 0:
            raise ValueError(f"{r['id']}: sentence not in its turn")
        field = infield(texts[r["turn"]][:start])
        previous = texts[r["turn"] - 1] if r["turn"] > 0 else ""
        lines.append(json.dumps({"id": r["id"], "infield": field,
                                 "history": f"{previous}\n{field}" if previous else field}, ensure_ascii=False) + "\n")
    if len(lines) != len(records):
        raise ValueError(f"{len(records) - len(lines)} {split} records not found in the sources")
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen", default=str(FROZEN))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = build(args.frozen, "development")
    path = Path(args.frozen) / "development-contexts.jsonl"
    if args.check:
        same = path.read_text(encoding="utf-8") == text
        print("development contexts", "equal the existing file" if same else "DIFFER from the existing file")
        sys.exit(0 if same else 1)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}: {text.count(chr(10))} records")


if __name__ == "__main__":
    main()
