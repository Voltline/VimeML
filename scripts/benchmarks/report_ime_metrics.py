"""Registered IME metrics (configs/ime-metrics-v2.json) on completed FP32 rerank scores.

Reads only completed `scores.jsonl` files; no model is run. For each model and track it
reports exact, notation-lenient and pool-conditional Top-1, and for each model pair the
paired gain/loss with per-item McNemar, the cluster sign-flip p-value and cluster
bootstrap interval under author, template and union clusters.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.training.data import write_json
from vimeml.benchmarks.notation import error_class
from vimeml.benchmarks import statistics

POLICY = ROOT / 'configs/ime-metrics-v2.json'
TRACKS = ['standard-development', 'legacy-regression', 'ajimee', 'historical-development']
MODELS = {
    'v21': ROOT / 'outputs/evaluation-v4-20261015/v21',
    'v42-e20': ROOT / 'outputs/evaluation-v42-20261020',
    'v43-10ep': ROOT / 'outputs/evaluation-v43-20261020/epoch10',
    'v43-e30': ROOT / 'outputs/evaluation-v43-20261020/epoch30',
    'v5-10ep': ROOT / 'outputs/evaluation-v5-20261021/epoch10',
    'v5-e30': ROOT / 'outputs/evaluation-v5-20261021/epoch30',
}
PAIRS = [('v21', 'v42-e20'), ('v42-e20', 'v43-e30'), ('v42-e20', 'v5-e30'),
         ('v43-10ep', 'v5-10ep'), ('v43-e30', 'v5-e30')]
OUTCOMES = {'exact': ('correct',), 'notation_lenient': ('correct', 'notation')}


def read_rows(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def code_hashes(policy):
    return {name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for name, path in policy['code'].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='Separate empty output directory')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory')
    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    summary = {'policy': str(POLICY.relative_to(ROOT)), 'policy_sha256': hashlib.sha256(POLICY.read_bytes()).hexdigest(),
               'code_sha256': code_hashes(policy), 'models': {k: str(v.relative_to(ROOT)) for k, v in MODELS.items()},
               'blind_read': False, 'reference_read': False, 'tracks': {}}
    for track in TRACKS:
        rows = {name: read_rows(folder / track / 'scores.jsonl') for name, folder in MODELS.items()}
        base = rows['v21']
        for name, data in rows.items():
            assert [r['id'] for r in data] == [r['id'] for r in base], (track, name)
            assert all(r['answers'] == b['answers'] and r['orders']['azookey'] == b['orders']['azookey']
                       for r, b in zip(data, base)), (track, name)
        reachable = [any(a in {c['text'] for c in r['candidates']} for a in r['answers']) for r in base]
        labels = {'author': statistics.clusters(base, template=None),
                  'template': statistics.clusters([{'answers': r['answers']} for r in base]),
                  'union': statistics.clusters(base)}
        classes = {name: [error_class(r, r['orders']['lm_context_sum'][0])[0] for r in data] for name, data in rows.items()}
        entry = {'cases': len(base), 'pool_reachable': sum(reachable),
                 'clusters': {k: len(set(v)) for k, v in labels.items()}, 'models': {}, 'paired': {}}
        for name, items in classes.items():
            exact = items.count('correct')
            entry['models'][name] = {'exact': exact, 'notation_lenient': exact + items.count('notation'),
                                     'pool_conditional': exact / sum(reachable)}
        for a, b in PAIRS:
            result = {}
            for metric, ok in OUTCOMES.items():
                before = [c in ok for c in classes[a]]
                after = [c in ok for c in classes[b]]
                result[metric] = {scheme: statistics.paired(before, after, lab) for scheme, lab in labels.items()}
            entry['paired'][f'{a}->{b}'] = result
        summary['tracks'][track] = entry
        print(track, json.dumps({k: entry['models'][k] for k in entry['models']}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'summary.json', summary)


if __name__ == '__main__':
    main()
