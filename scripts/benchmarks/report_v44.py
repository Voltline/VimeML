"""Preregistered V4.4 analysis (docs/zh-CN/training-v44-plan.md) on completed FP32 scores.

Reads only the `scores.jsonl` files of V4.2 epoch 20 and the six V4.4 arms; no model is
run. Reports per track the exact, notation-lenient and pool-conditional Top-1 of every
model, the seed noise band (V4.2 + s1-s3), all six seed pairs, each treatment against
V4.2 under ime-metrics-v2 statistics, and the registered decision for chat0, chat10 and
crop60 on Standard. Training-side LM/chat BPC histories are added from the run records.
"""
import argparse
import hashlib
import json
import statistics as stats
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.training.data import write_json
from vimeml.benchmarks.notation import error_class
from vimeml.benchmarks import statistics

TRACKS = ['standard-development', 'legacy-regression', 'ajimee', 'historical-development']
PRIMARY = 'standard-development'
GUARDED = ['standard-development', 'legacy-regression', 'ajimee']
SMOKE, SMOKE_DROP = 'historical-development', 3
SEEDS = ['v42-e20', 's1', 's2', 's3']
TREATMENTS = ['chat0', 'chat10', 'crop60']
SCORES = ROOT / 'outputs/evaluation-v44-20261105'
MODELS = {'v42-e20': ROOT / 'outputs/evaluation-v42-20261020',
          **{arm: SCORES / arm for arm in SEEDS[1:] + TREATMENTS}}
RECORDS = ROOT / 'outputs/training-v44-20261105/remote'
V42_HISTORY = ROOT / 'outputs/training-v42-20261020/remote/records/epoch-validation-history.json'
OUTCOMES = {'exact': ('correct',), 'notation_lenient': ('correct', 'notation')}
ALPHA = .05


def read_rows(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def holm(pvalues):
    """Holm step-down adjusted p-values, in input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.] * len(pvalues), 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def band(values):
    return {'values': values, 'mean': stats.mean(values), 'min': min(values), 'max': max(values),
            'range': max(values) - min(values), 'sd': stats.stdev(values)}


def history(path):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    return [{'epoch': r['epoch'], 'lm_bpc': r['lm']['bpc'], 'chat_bpc': r['chat']['bpc']} for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='Separate empty output directory')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory')
    for arm in SEEDS[1:] + TREATMENTS:
        if json.loads((SCORES / arm / 'summary.json').read_text(encoding='utf-8'))['status'] != 'complete':
            parser.error(f'Score {arm} first')
    summary = {'plan': 'docs/zh-CN/training-v44-plan.md', 'models': {k: str(v.relative_to(ROOT)) for k, v in MODELS.items()},
               'score_sha256': {}, 'blind_read': False, 'reference_read': False, 'tracks': {}}
    for track in TRACKS:
        rows = {name: read_rows(folder / track / 'scores.jsonl') for name, folder in MODELS.items()}
        base = rows['v42-e20']
        for name, data in rows.items():
            summary['score_sha256'][f'{name}/{track}'] = hashlib.sha256(
                (MODELS[name] / track / 'scores.jsonl').read_bytes()).hexdigest()
            assert [r['id'] for r in data] == [r['id'] for r in base], (track, name)
            assert all(r['answers'] == b['answers'] and r['orders']['azookey'] == b['orders']['azookey']
                       for r, b in zip(data, base)), (track, name)
        reachable = sum(any(a in {c['text'] for c in r['candidates']} for a in r['answers']) for r in base)
        labels = statistics.clusters(base, template=None)
        classes = {name: [error_class(r, r['orders']['lm_context_sum'][0])[0] for r in data]
                   for name, data in rows.items()}
        entry = {'cases': len(base), 'pool_reachable': reachable, 'author_clusters': len(set(labels)),
                 'models': {}, 'seed_band': {}, 'seed_pairs': {}, 'vs_v42': {}}
        for name, items in classes.items():
            exact = items.count('correct')
            entry['models'][name] = {'exact': exact, 'notation_lenient': exact + items.count('notation'),
                                     'pool_conditional': exact / reachable,
                                     'errors': {c: items.count(c) for c in sorted(set(items) - {'correct'})}}
        for metric in OUTCOMES:
            entry['seed_band'][metric] = band([entry['models'][m][metric] for m in SEEDS])

        def compare(a, b):
            return {metric: statistics.paired([c in ok for c in classes[a]], [c in ok for c in classes[b]], labels)
                    for metric, ok in OUTCOMES.items()}
        for a, b in combinations(SEEDS, 2):
            entry['seed_pairs'][f'{a}->{b}'] = compare(a, b)
        for arm in SEEDS[1:] + TREATMENTS:
            entry['vs_v42'][arm] = compare('v42-e20', arm)
        summary['tracks'][track] = entry
        print(track, json.dumps({k: (v['exact'], v['notation_lenient']) for k, v in entry['models'].items()}), flush=True)
    primary = summary['tracks'][PRIMARY]
    raw = [primary['vs_v42'][arm]['exact']['cluster_sign_flip_p'] for arm in TREATMENTS]
    decision = {}
    for arm, p, adjusted in zip(TREATMENTS, raw, holm(raw)):
        exact = primary['models'][arm]['exact']
        seed = primary['seed_band']['exact']
        guards = {track: summary['tracks'][track]['vs_v42'][arm]['notation_lenient'] for track in GUARDED}
        regressions = [t for t, g in guards.items() if g['net'] < 0 and g['cluster_sign_flip_p'] < ALPHA]
        outside = exact < seed['min'] or exact > seed['max']
        smoke = summary['tracks'][SMOKE]
        drop = smoke['models']['v42-e20']['exact'] - smoke['models'][arm]['exact']
        decision[arm] = {'standard_exact': exact, 'net_vs_v42': primary['vs_v42'][arm]['exact']['net'],
                         'outside_seed_range': outside, 'cluster_p': p, 'holm_p': adjusted,
                         'notation_lenient_regressions': regressions, 'smoke_flag': drop > SMOKE_DROP,
                         'effective': outside and adjusted < ALPHA and not regressions}
    summary['decision'] = decision
    summary['bpc'] = {'v42-e20': history(V42_HISTORY),
                      **{arm: history(RECORDS / arm / 'records/epoch-validation-history.json')
                         for arm in SEEDS[1:] + TREATMENTS}}
    final = {arm: rows[-1] for arm, rows in summary['bpc'].items()}
    summary['bpc_seed_band'] = {key: band([final[m][key] for m in SEEDS]) for key in ('lm_bpc', 'chat_bpc')}
    print(json.dumps(decision, indent=1), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'summary.json', summary)


if __name__ == '__main__':
    main()
