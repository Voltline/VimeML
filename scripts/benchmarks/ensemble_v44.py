"""Exploratory, not preregistered: log-probability ensembles of the four V4.2-recipe seeds.

Reads the completed contextual full-vocabulary suffix logP sums of V4.2 epoch 20 and
V4.4 s1-s3 from their `scores.jsonl` files; no model is run. An ensemble ranks each
candidate by the sum of its member log-probabilities (equal-weight product of experts),
ties kept in AzooKey order exactly as the single-model ranking. Reports every 2-, 3- and
4-member ensemble per track, and the 4-member ensemble against each single seed under
ime-metrics-v2 author-cluster statistics. The idea was formed after the preregistered
analysis on the same development tracks, so any gain here still needs a fresh set.
"""
import argparse
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
SCORES = ROOT / 'outputs/evaluation-v44-20261105'
MODELS = {'v42-e20': ROOT / 'outputs/evaluation-v42-20261020',
          **{arm: SCORES / arm for arm in ('s1', 's2', 's3')}}
OUTCOMES = {'exact': ('correct',), 'notation_lenient': ('correct', 'notation')}


def read_rows(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='Separate empty output directory')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory')
    summary = {'exploratory': True, 'models': {k: str(v.relative_to(ROOT)) for k, v in MODELS.items()}, 'tracks': {}}
    for track in TRACKS:
        rows = {name: read_rows(folder / track / 'scores.jsonl') for name, folder in MODELS.items()}
        base = rows['v42-e20']
        labels = statistics.clusters(base, template=None)
        logp = {name: [{c['text']: c['log_probability_sum'] for c in r['lm_scores']['contextual']['candidates']}
                       for r in data] for name, data in rows.items()}

        def classes(members):
            result = []
            for i, row in enumerate(base):
                order = row['orders']['azookey']
                total = {t: sum(logp[m][i][t] for m in members) for t in order}
                result.append(error_class(row, sorted(order, key=lambda t: -total[t])[0])[0])
            return result
        single = {name: classes([name]) for name in MODELS}
        for name, data in rows.items():
            assert single[name] == [error_class(r, r['orders']['lm_context_sum'][0])[0] for r in data], (track, name)
        entry = {'single': {}, 'ensembles': {}, 'full_vs_single': {}}
        for name, items in single.items():
            entry['single'][name] = {m: sum(c in ok for c in items) for m, ok in OUTCOMES.items()}
        full = None
        for size in (2, 3, 4):
            for members in combinations(MODELS, size):
                items = classes(members)
                entry['ensembles']['+'.join(members)] = {
                    **{m: sum(c in ok for c in items) for m, ok in OUTCOMES.items()},
                    'errors': {c: items.count(c) for c in sorted(set(items) - {'correct'})}}
                if size == 4:
                    full = items
        for size in (1, 2, 3, 4):
            values = ([entry['single'][n]['exact'] for n in MODELS] if size == 1 else
                      [v['exact'] for k, v in entry['ensembles'].items() if k.count('+') == size - 1])
            entry[f'size{size}_exact'] = {'mean': stats.mean(values), 'min': min(values), 'max': max(values)}
        for name, items in single.items():
            entry['full_vs_single'][name] = {
                m: statistics.paired([c in ok for c in items], [c in ok for c in full], labels)
                for m, ok in OUTCOMES.items()}
        summary['tracks'][track] = entry
        print(track, json.dumps({f'size{s}': entry[f'size{s}_exact'] for s in (1, 2, 3, 4)}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'summary.json', summary)


if __name__ == '__main__':
    main()
