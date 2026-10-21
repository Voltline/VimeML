"""Top-1 error decomposition of existing FP32 rerank scores on the four nonblind tracks.

Reads only completed `scores.jsonl` files; no model is run and no reference or blind
set is read. For every wrong Top-1 the choice is compared with the closest reference:

- pool miss: no reference is among the frozen AzooKey candidates (unreachable);
- notation (A): same words, different script/okurigana (`vimeml.benchmarks.notation`,
  conservative rule; the OR of its two rules is an upper bound);
- semantic (S): everything else (homophones, segmentation, proper nouns).

Outputs aggregate counts and per-case labels to a separate output directory.
"""
import argparse
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.training.data import write_json
from vimeml.benchmarks.notation import error_class, kana_share, relation

TRACKS = ['standard-development', 'legacy-regression', 'ajimee', 'historical-development']
MODELS = {
    'v21': ROOT / 'outputs/evaluation-v4-20261015/v21',
    'v42-e20': ROOT / 'outputs/evaluation-v42-20261020',
    'v43-10ep': ROOT / 'outputs/evaluation-v43-20261020/epoch10',
    'v43-e30': ROOT / 'outputs/evaluation-v43-20261020/epoch30',
    'v5-10ep': ROOT / 'outputs/evaluation-v5-20261021/epoch10',
    'v5-e30': ROOT / 'outputs/evaluation-v5-20261021/epoch30',
}
SIZE_PAIRS = [('v43-10ep', 'v5-10ep'), ('v43-e30', 'v5-e30'), ('v42-e20', 'v5-e30'), ('v21', 'v42-e20')]
def exact_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def read_rows(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def classify(row, choice):
    kind, answer = error_class(row, choice)
    if kind == 'correct':
        return {'class': kind}
    if kind == 'pool_miss':
        return {'class': kind, 'choice_is_notation_variant': any(relation(choice, a) == 'A' for a in row['answers'])}
    scores = {c['text']: c['log_probability_sum'] for c in row['lm_scores']['contextual']['candidates']}         if 'lm_scores' in row else {}
    reachable = [a for a in row['answers'] if a in scores]
    best = max(reachable, key=scores.get) if reachable else None
    margin = scores[choice] - scores[best] if best is not None and choice in scores else None
    return {'class': kind, 'answer': answer, 'margin_nats': margin,
            'more_kana': kana_share(choice) > kana_share(answer)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='Separate empty output directory')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory')
    summary, cases = {}, {}
    for track in TRACKS:
        rows = {name: read_rows(folder / track / 'scores.jsonl') for name, folder in MODELS.items()}
        base = rows['v21']
        for name, data in rows.items():
            assert [r['id'] for r in data] == [r['id'] for r in base], (track, name)
            assert all(r['answers'] == b['answers'] and r['orders']['azookey'] == b['orders']['azookey']
                       for r, b in zip(data, base)), (track, name)
        n = len(base)
        reachable = sum(any(a in [c['text'] for c in r['candidates']] for a in r['answers']) for r in base)
        labels = {name: [classify(r, r['orders']['lm_context_sum'][0]) for r in data] for name, data in rows.items()}
        labels['azookey'] = [classify({k: v for k, v in r.items() if k != 'lm_scores'}, r['orders']['azookey'][0])
                             for r in base]
        track_summary = {'cases': n, 'pool_reachable': reachable, 'models': {}}
        for name, items in labels.items():
            count = Counter(item['class'] for item in items)
            wrong = [i for i in items if i['class'] in ('notation', 'notation_upper', 'semantic')]
            margins = [i['margin_nats'] for i in wrong if i.get('margin_nats') is not None]
            notation = [i for i in items if i['class'] == 'notation']
            track_summary['models'][name] = {
                'top1': count['correct'], 'pool_miss': count['pool_miss'],
                'pool_miss_choice_notation_variant': sum(i.get('choice_is_notation_variant', False) for i in items),
                'notation': count['notation'], 'notation_upper_only': count['notation_upper'],
                'semantic': count['semantic'],
                'notation_lenient_top1': count['correct'] + count['notation'],
                'notation_lenient_top1_upper': count['correct'] + count['notation'] + count['notation_upper'],
                'notation_more_kana': sum(i['more_kana'] for i in notation),
                'wrong_margin_below_1nat': sum(m < 1 for m in margins), 'wrong_with_margin': len(margins),
            }
        model_names = list(MODELS)
        track_summary['union_oracle_top1'] = sum(any(labels[m][i]['class'] == 'correct' for m in model_names)
                                                 for i in range(n))
        track_summary['all_models_wrong_semantic'] = sum(all(labels[m][i]['class'] == 'semantic' for m in model_names)
                                                         for i in range(n))
        pairs = {}
        for a, b in SIZE_PAIRS:
            result = {}
            for metric, ok in [('exact', lambda x: x['class'] == 'correct'),
                               ('notation_lenient', lambda x: x['class'] in ('correct', 'notation'))]:
                gain = sum(ok(y) and not ok(x) for x, y in zip(labels[a], labels[b]))
                loss = sum(ok(x) and not ok(y) for x, y in zip(labels[a], labels[b]))
                result[metric] = {'gain': gain, 'loss': loss, 'net': gain - loss, 'exact_mcnemar_p': exact_p(gain, loss)}
            pairs[f'{a}->{b}'] = result
        track_summary['paired'] = pairs
        summary[track] = track_summary
        cases[track] = [{'id': r['id'], **{m: labels[m][i] for m in labels}} for i, r in enumerate(base)]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'summary.json', {'models': {k: str(v.relative_to(ROOT)) for k, v in MODELS.items()},
                                      'tracks': summary, 'blind_read': False, 'reference_read': False,
                                      'rule': 'notation = equal UniDic lemma sequence AND nested kanji sets'})
    write_json(out / 'cases.json', cases)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
