"""Word-level UniDic / Sudachi reading disagreements, for the known-difference table (deviation D4).

  extract   readings of a seeded sample of typeable sentences (same sources and sentence rule as
            reading_agreement.py, larger quotas). Both tokenizations are aligned on shared
            character boundaries; each minimal common segment whose two readings differ after
            normalization (symbol readings and non-kana characters removed, ヂ/ヅ folded to ジ/ズ) is one (surface, UniDic reading,
            Sudachi reading) pair. Writes pairs.json (counts per pair, sentences with only
            table-eligible differences) and contexts.jsonl (up to three example sentences per
            pair; stays local, it quotes corpus text).

  table     merges two LLM judgment passes (the two readings shown in opposite order) into the
            known-difference table: a pair enters only when both passes give the same verdict and
            that verdict is one reading or "both" (then the reading more users type). Pairs judged
            context-dependent, neither, or differently by the two passes stay out, as do one-kanji
            words of at most two characters, and pairs seen fewer than min_count times (mostly names
            and segmentation fragments; their sentences are dropped). configs/ja/ja-reading-review.toml
            holds the hand review: min_count, exclusions, and overrides that set the reading of a
            surface whatever the two passes said.

  measure   agreement on the reading-agreement-v1 samples: raw (the 1.2 gate), normalized, and
            normalized with the table resolving the remaining differences.

    python -X utf8 scripts/ja/corpus/reading_pairs.py extract --output outputs/ja-corpus/reading-pairs-v1
    python -X utf8 scripts/ja/corpus/reading_pairs.py table --pass1-in .. --pass1-out .. --pass2-in .. \\
        --pass2-out .. --table configs/ja/ja-reading-table.tsv
    python -X utf8 scripts/ja/corpus/reading_pairs.py measure
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/ja/corpus'))
sys.path.insert(0, str(ROOT / 'scripts/ja/benchmarks'))
from vimeml.benchmarks.readings import KANA, DualReading, katakana  # noqa: E402
from reading_agreement import TYPEABLE, chat_sentences, lm_sentences, reservoir, web_sentences  # noqa: E402

SEED = 20261127
QUOTAS = {'lm': 100000, 'chat': 100000, 'web': 100000}
KANJI = re.compile(r'[一-鿿㐀-䶿々]')
JAPANESE = re.compile(r'[ぁ-ゖゝゞァ-ヺー-ヾ一-鿿㐀-䶿々〆ヶ]')
FOLD = str.maketrans({'ヂ': 'ジ', 'ヅ': 'ズ'})


def unidic_words(reader, text):
    out, offset = [], 0
    for w in reader.mecab(text):
        s = w.surface
        start = text.index(s, offset) if s else offset
        k = katakana(s)
        reading = k if KANA.fullmatch(k) else (w.feature.kana or s)
        out.append((start, start + len(s), reading))
        offset = start + len(s)
    return out


def sudachi_words(reader, text):
    out = []
    for w in reader.sudachi.tokenize(text, reader.mode):
        s, k = w.surface(), katakana(w.surface())
        out.append((w.begin(), w.end(), k if KANA.fullmatch(k) else w.reading_form()))
    return out


def segments(text, a, b):
    """Minimal spans whose two ends are word boundaries in both analyses."""
    common = sorted({e for _, e, _ in a} & {e for _, e, _ in b})
    out, prev = [], 0
    for end in common:
        ra = ''.join(r for s, e, r in a if prev <= s and e <= end)
        rb = ''.join(r for s, e, r in b if prev <= s and e <= end)
        out.append((text[prev:end], ra, rb))
        prev = end
    return out


def norm(surface, reading):
    """Reading as typed: symbols carry no reading (Sudachi reads some as キゴウ), ヂ/ヅ fold to ジ/ズ."""
    if not JAPANESE.search(surface):
        return ''
    if '記号' not in surface:
        reading = reading.replace('キゴウ', '')
    return ''.join(c for c in reading.translate(FOLD) if 'ァ' <= c <= 'ヺ' or c == 'ー')


def cmd_extract(args):
    args.output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(SEED)
    pools = {'lm': reservoir(lm_sentences(rng, stride=5), QUOTAS['lm'], rng),
             'chat': reservoir(chat_sentences(), QUOTAS['chat'], rng),
             'web': reservoir(web_sentences(rng), QUOTAS['web'], rng)}
    reader = DualReading()
    pairs, contexts, counts = collections.Counter(), collections.defaultdict(list), collections.Counter()
    for source, rows in pools.items():
        for text in rows:
            counts[f'{source}/sentences'] += 1
            if not TYPEABLE.fullmatch(text):
                continue
            counts[f'{source}/typeable'] += 1
            diff = differences(reader, text)
            if not diff:
                counts[f'{source}/agree_normalized'] += 1
                continue
            for d in set(diff):
                pairs[d] += 1
                if len(contexts[d]) < 3:
                    contexts[d].append(text)
    ranked = [dict(surface=s, unidic=u, sudachi=v, count=n) for (s, u, v), n in pairs.most_common()]
    (args.output / 'pairs.json').write_text(json.dumps(dict(seed=SEED, quotas=QUOTAS, versions=reader.versions,
                                                            counts=dict(counts), pairs=ranked),
                                                       ensure_ascii=False, indent=1), encoding='utf-8')
    with (args.output / 'contexts.jsonl').open('w', encoding='utf-8') as f:
        for p in ranked:
            f.write(json.dumps(dict(p, contexts=contexts[(p['surface'], p['unidic'], p['sudachi'])]),
                               ensure_ascii=False) + '\n')
    print(json.dumps(dict(counts), indent=1))
    print(len(ranked), 'pairs;', sum(1 for p in ranked if p['count'] >= 5), 'with count >= 5')
    for p in ranked[:40]:
        print(p['count'], p['surface'], p['unidic'], p['sudachi'])


def judged(inp, out):
    """{id: verdict} from one LLM pass; verdict is unidic, sudachi, both:<common>, context or neither."""
    rows = {r['id']: r for r in map(json.loads, inp.open(encoding='utf-8'))}
    result = {}
    for o in map(json.loads, out.open(encoding='utf-8')):
        r = rows[o['id']]
        side = {'a': r['a_is'], 'b': 'unidic' if r['a_is'] == 'sudachi' else 'sudachi'}
        c = o['choice']
        result[o['id']] = side[c] if c in side else f'both:{side[o["common"]]}' if c == 'both' else c
    return result


def one_kanji_word(surface):
    """A single kanji, alone or with one kana: its reading depends on the sentence (方, 行っ, 今)."""
    return len(KANJI.findall(surface)) == 1 and len(surface.strip()) <= 2


def cmd_table(args):
    import tomllib
    pairs = json.loads((args.pairs / 'pairs.json').read_text(encoding='utf-8'))['pairs']
    review = tomllib.loads(args.review.read_text(encoding='utf-8'))
    min_count, override = review['min_count'], review.get('override', {})
    exclude = set(review.get('exclude', {}).get('surfaces', []))
    first, second = judged(args.pass1_in, args.pass1_out), judged(args.pass2_in, args.pass2_out)
    rows, tally = [], collections.Counter()
    for i, p in enumerate(pairs):
        v1, v2 = first.get(i), second.get(i)
        verdict = v1 if v1 == v2 else 'inconsistent' if v1 and v2 else 'missing'
        key, surface = (p['surface'], p['unidic'], p['sudachi']), p['surface'].strip()
        tally[verdict.split(':')[0]] += 1
        if p['count'] < min_count:
            tally['below_min_count'] += 1
        elif surface in exclude or one_kanji_word(surface):
            tally['excluded'] += 1
        elif surface in override:
            rows.append((*key, override[surface], 'hand', p['count']))
        elif verdict in ('unidic', 'sudachi') or verdict.startswith('both:'):
            rows.append((*key, p[verdict.split(':')[-1]], verdict.split(':')[0], p['count']))
    args.table.parent.mkdir(parents=True, exist_ok=True)
    with args.table.open('w', encoding='utf-8', newline='\n') as f:
        f.write('# surface\tunidic\tsudachi\treading\tverdict\tcount\n')
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')
    print(dict(tally), len(rows), 'table rows')


def load_table(path):
    """{(surface, unidic, sudachi): reading} from the known-difference table."""
    rows = (line.rstrip('\n').split('\t') for line in path.open(encoding='utf-8') if not line.startswith('#'))
    return {tuple(r[:3]): r[3] for r in rows}


def differences(reader, text):
    """Segments whose normalized readings differ: [(surface, unidic, sudachi)]."""
    diff = [(s, norm(s, ra), norm(s, rb)) for s, ra, rb in
            segments(text, unidic_words(reader, text), sudachi_words(reader, text))]
    return [d for d in diff if d[1] != d[2]]


def cmd_measure(args):
    table, reader = load_table(args.table), DualReading()
    counts = collections.Counter()
    for r in map(json.loads, (args.samples / 'samples.jsonl').open(encoding='utf-8')):
        if not TYPEABLE.fullmatch(r['text']):
            continue
        diff = differences(reader, r['text'])
        for level, ok in (('typeable', True), ('raw', r['agreement']), ('normalized', not diff),
                          ('table', all(d in table for d in diff))):
            counts[f"{r['source']}/{level}"] += ok
            counts[level] += ok
    out = {k: dict(sentences=counts[f'{k}typeable'],
                   **{lv: round(counts[f'{k}{lv}'] / counts[f'{k}typeable'], 4)
                      for lv in ('raw', 'normalized', 'table')})
           for k in ('lm/', 'chat/', 'web/', '')}
    print(json.dumps({k.rstrip('/') or 'all': v for k, v in out.items()}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('table')
    p.add_argument('--pairs', type=Path, default=ROOT / 'outputs/ja-corpus/reading-pairs-v1')
    for name in ('pass1-in', 'pass1-out', 'pass2-in', 'pass2-out'):
        p.add_argument(f'--{name}', type=Path, required=True)
    p.add_argument('--review', type=Path, default=ROOT / 'configs/ja/ja-reading-review.toml')
    p.add_argument('--table', type=Path, required=True)
    p = sub.add_parser('measure')
    p.add_argument('--samples', type=Path, default=ROOT / 'outputs/ja-corpus/reading-agreement-v1')
    p.add_argument('--table', type=Path, default=ROOT / 'configs/ja/ja-reading-table.tsv')
    args = parser.parse_args()
    {'extract': cmd_extract, 'table': cmd_table, 'measure': cmd_measure}[args.command](args)


if __name__ == '__main__':
    main()
