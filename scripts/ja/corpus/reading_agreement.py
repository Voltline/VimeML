"""ja-gen stage 1.2 gate: Sudachi / UniDic whole-reading agreement on a 10,000-sentence sample.

Sentences (4-64 characters, split at 。！？ and line breaks) are drawn with a fixed seed from the
three text sources of the ja-gen models:

  lm     the V4 LM corpus (outputs/corpus-fast-v1/train.txt)        4,000
  chat   the V4.1 chat bank (artifacts/token-data/corpus-v41-chat-8k) 2,000
  web    the FineWeb-2 jpn_Jpan export (outputs/ja-corpus/export-v1) 4,000

A sentence is typeable when it holds only kana, kanji and Japanese punctuation (no Latin
letters, digits or other symbols); only typeable sentences can become reading samples. For each
typeable sentence `DualReading.analyse` gives both readings. Reported: typeable share, agreement
among typeable sentences (the gate: stop below 60%), unknown-word share, and the most frequent
disagreements as (UniDic, Sudachi) reading fragments. Sentences stay local; summary.json holds
aggregates plus fragment counts only.

    python -X utf8 scripts/ja/corpus/reading_agreement.py --output outputs/ja-corpus/reading-agreement-v1
"""
import argparse
import collections
import difflib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/ja/benchmarks'))
from vimeml.benchmarks.readings import DualReading  # noqa: E402

SEED = 20261126
QUOTAS = {'lm': 4000, 'chat': 2000, 'web': 4000}
GATE = 0.60
SPLIT = re.compile(r'(?<=[。！？!?])|\n')
TYPEABLE = re.compile(r'[ぁ-ゖゝゞァ-ヺー-ヾ一-鿿㐀-䶿々〆ヶ、。！？「」『』（）・…〜～ 　]+')
LM = ROOT / 'outputs/corpus-fast-v1/train.txt'
WEB = ROOT / 'outputs/ja-corpus/export-v1/train'


def sentences(text):
    for s in SPLIT.split(text):
        s = s.strip()
        if 4 <= len(s) <= 64:
            yield s


def reservoir(stream, k, rng):
    sample = []
    for n, item in enumerate(stream):
        if n < k:
            sample.append(item)
        else:
            j = rng.randrange(n + 1)
            if j < k:
                sample[j] = item
    return sample


def lm_sentences(rng, stride=50):
    with LM.open(encoding='utf-8') as f:
        for n, line in enumerate(f):
            if n % stride == 0:
                yield from sentences(line)


def chat_sentences():
    from prepare_standard_ime_v2 import chat_lines
    for split, line in chat_lines():
        if split == 'train':
            yield from sentences(line)


def web_sentences(rng):
    import pyarrow.parquet as pq
    files = sorted(WEB.glob('*.parquet'))
    for path in files[::max(1, len(files) // 8)]:
        pf = pq.ParquetFile(path)
        for group in sorted(rng.sample(range(pf.metadata.num_row_groups), min(3, pf.metadata.num_row_groups))):
            for text in pf.read_row_group(group, columns=['text']).column('text').to_pylist()[::40]:
                yield from sentences(text)


def fragments(a, b):
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != 'equal':
            out.append(f'{a[max(0, i1 - 1):i2 + 1]} / {b[max(0, j1 - 1):j2 + 1]}')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(SEED)
    pools = {'lm': reservoir(lm_sentences(rng), QUOTAS['lm'], rng),
             'chat': reservoir(chat_sentences(), QUOTAS['chat'], rng),
             'web': reservoir(web_sentences(rng), QUOTAS['web'], rng)}
    reader = DualReading()
    summary, frag = {'seed': SEED, 'dictionary_versions': reader.versions, 'gate': GATE, 'sources': {}}, {}
    with (args.output / 'samples.jsonl').open('w', encoding='utf-8') as out:
        for source, rows in pools.items():
            counts, diffs = collections.Counter(), collections.Counter()
            for text in rows:
                counts['sentences'] += 1
                if not TYPEABLE.fullmatch(text):
                    counts['not_typeable'] += 1
                    continue
                r = reader.analyse(text)
                counts['typeable'] += 1
                counts['agree'] += r['agreement']
                counts['agree_known'] += r['agreement'] and not r['unknown']
                counts['unknown_word'] += bool(r['unknown'])
                if not r['agreement']:
                    diffs.update(fragments(r['unidic'], r['sudachi']))
                out.write(json.dumps(dict(source=source, text=text, **r), ensure_ascii=False) + '\n')
            summary['sources'][source] = dict(counts, typeable_share=round(counts['typeable'] / counts['sentences'], 3),
                                              agreement=round(counts['agree'] / max(1, counts['typeable']), 3))
            frag[source] = diffs
    total = sum((collections.Counter({k: v for k, v in s.items() if isinstance(v, int)})
                 for s in summary['sources'].values()), collections.Counter())
    summary['all'] = dict(total, typeable_share=round(total['typeable'] / total['sentences'], 3),
                          agreement=round(total['agree'] / total['typeable'], 3))
    summary['gate_passed'] = summary['all']['agreement'] >= GATE
    merged = sum(frag.values(), collections.Counter())
    summary['top_disagreements'] = merged.most_common(60)
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('sources', 'all', 'gate_passed')}, ensure_ascii=False, indent=1))
    for f, n in summary['top_disagreements'][:30]:
        print(f'{n:>5}  {f}')


if __name__ == '__main__':
    main()
