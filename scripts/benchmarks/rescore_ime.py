"""Training-free rescoring of existing FP32 scores: PMI, token-length and engine-score fusion.

Uses only the per-candidate values already stored in completed `scores.jsonl` files
(contextual and context-free suffix logP sums, scored token counts, AzooKey scores);
no model is run and no reference or blind set is read.

Protocol (written to plan.json by --plan-only before any result is computed):
- every method's parameter is selected on Standard development exact Top-1 alone
  (ties: smallest parameter magnitude, i.e. closest to the baseline);
- the primary candidate is the method with the best selected Standard score
  (ties: fewer parameters); all others are descriptive;
- Legacy, AJIMEE and Historical are held out; Standard is also split by source
  (select on one, test on the other);
- every result reports exact Top-1, notation-lenient Top-1 and semantic errors,
  paired against the baseline with exact McNemar tests.
"""
import argparse
import datetime
import hashlib
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.training.data import write_json
from vimeml.benchmarks.notation import error_class

TUNE = 'standard-development'
HELD_OUT = ['legacy-regression', 'ajimee', 'historical-development']
MODELS = {
    'v42-e20': ROOT / 'outputs/evaluation-v42-20261020',
    'v5-e30': ROOT / 'outputs/evaluation-v5-20261021/epoch30',
    'v21': ROOT / 'outputs/evaluation-v4-20261015/v21',
}
PMI = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
LENGTH = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0]
ENGINE = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
MARGIN = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
PLANS = {
    1: {'baseline': [()], 'pmi': [(l,) for l in PMI], 'length': [(a,) for a in LENGTH],
        'engine': [(b,) for b in ENGINE], 'pmi+engine': [(l, b) for l in PMI for b in ENGINE]},
    # Registered after plan 1: engine fusion only where the LM's own top-2 margin is below tau.
    2: {'baseline': [()], 'gated-engine': [(b, t) for b in ENGINE for t in MARGIN]},
}
METHODS = PLANS[1]


def plan_for(version):
    plan = {
    'tune_track': TUNE, 'held_out_tracks': HELD_OUT, 'models': {k: str(v.relative_to(ROOT)) for k, v in MODELS.items()},
    'score': {'baseline': 'c', 'pmi': 'c - lambda*f', 'length': 'c + alpha*n', 'engine': 'c + beta*z',
              'pmi+engine': 'c - lambda*f + beta*z'},
    'symbols': {'c': 'contextual suffix logP sum', 'f': 'context-free logP sum of the same candidate',
                'n': 'scored candidate tokens', 'z': 'AzooKey candidate score'},
    'grids': {k: v for k, v in PLANS[version].items() if k != 'baseline'},
    'selection': 'per method: max Standard exact Top-1, ties -> smallest parameter magnitude; '
                 'primary = best selected method on Standard, ties -> fewer parameters; others descriptive',
    'cross_validation': 'Standard split by source: select on one source, report on the other',
    'ties': 'stable sort over the frozen AzooKey candidate order',
    'metrics': ['exact Top-1', 'notation-lenient Top-1 (vimeml.benchmarks.notation, conservative)', 'semantic errors'],
    'blind_read': False, 'reference_read': False,
    }
    if version == 2:
        plan['plan_version'] = 2
        plan['score'] = {'baseline': 'c', 'gated-engine': 'c + beta*z if c(top1) - c(top2) < tau under c, else c'}
        plan['follows'] = 'plan 1 primary (ungated engine fusion) failed on held-out AJIMEE; gate designed on Standard only'
    return plan


def exact_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def plan_digest(version):
    return hashlib.sha256(json.dumps(plan_for(version), sort_keys=True).encode()).hexdigest()


def load(folder, track):
    rows = []
    with open(folder / track / 'scores.jsonl', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            ctx = row['lm_scores']['contextual']['candidates']
            free = row['lm_scores']['context_free']['candidates']
            assert [c['text'] for c in row['candidates']] == [c['text'] for c in ctx] == [c['text'] for c in free]
            row['features'] = [(c['log_probability_sum'], f['log_probability_sum'], c['scored_tokens'], z['score'])
                               for c, f, z in zip(ctx, free, row['candidates'])]
            rows.append(row)
    return rows


def score(method, params, c, f, n, z):
    if method == 'baseline':
        return c
    if method == 'pmi':
        return c - params[0] * f
    if method == 'length':
        return c + params[0] * n
    if method == 'engine':
        return c + params[0] * z
    return c - params[0] * f + params[1] * z


def top1(row, method, params):
    if method == 'gated-engine':
        ctx = sorted((x[0] for x in row['features']), reverse=True)
        if len(ctx) < 2 or ctx[0] - ctx[1] >= params[1]:
            return top1(row, 'baseline', ())
        method, params = 'engine', params[:1]
    values = [score(method, params, *x) for x in row['features']]
    best = max(range(len(values)), key=lambda i: (values[i], -i))
    return row['candidates'][best]['text']


CACHE = {}


def classify(row, choice):
    key = (row['id'], choice)
    if key not in CACHE:
        CACHE[key] = error_class(row, choice)[0]
    return CACHE[key]


def evaluate(rows, method, params):
    classes = [classify(r, top1(r, method, params)) for r in rows]
    return classes, {'exact': classes.count('correct'),
                     'lenient': classes.count('correct') + classes.count('notation'),
                     'semantic': classes.count('semantic')}


def paired(base, other):
    out = {}
    for metric, ok in [('exact', ('correct',)), ('lenient', ('correct', 'notation'))]:
        gain = sum(b not in ok and o in ok for b, o in zip(base, other))
        loss = sum(b in ok and o not in ok for b, o in zip(base, other))
        out[metric] = {'gain': gain, 'loss': loss, 'net': gain - loss, 'exact_mcnemar_p': exact_p(gain, loss)}
    return out


def select(rows, method):
    best = None
    for params in METHODS[method]:
        exact = evaluate(rows, method, params)[1]['exact']
        key = (exact, -sum(abs(p) for p in params))
        if best is None or key > best[0]:
            best = (key, params)
    return best[1], best[0][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true', help='Write plan.json before computing any result')
    parser.add_argument('--plan', type=int, choices=sorted(PLANS), default=1)
    args = parser.parse_args()
    global METHODS
    METHODS = PLANS[args.plan]
    out = args.output.resolve()
    if args.plan_only:
        if out.exists() and any(out.iterdir()):
            parser.error('Use an empty output directory for the plan')
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'plan.json', {**plan_for(args.plan), 'plan_sha256': plan_digest(args.plan),
                                       'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        print('plan written', plan_digest(args.plan))
        return
    plan = json.loads((out / 'plan.json').read_text(encoding='utf-8'))
    if plan['plan_sha256'] != plan_digest(args.plan) or (out / 'summary.json').exists():
        parser.error('Plan changed after registration, or results already exist')
    results = {}
    for model, folder in MODELS.items():
        data = {track: load(folder, track) for track in [TUNE, *HELD_OUT]}
        for track, rows in data.items():
            assert all(top1(r, 'baseline', ()) == r['orders']['lm_context_sum'][0] for r in rows), (model, track)
        base = {track: evaluate(rows, 'baseline', ()) for track, rows in data.items()}
        chosen = {}
        for method in METHODS:
            params, tune_exact = select(data[TUNE], method)
            entry = {'params': list(params), 'tracks': {}}
            for track, rows in data.items():
                classes, metrics = evaluate(rows, method, params)
                entry['tracks'][track] = {**metrics, 'vs_baseline': paired(base[track][0], classes)}
            cv = {}
            for source in sorted({r['source'] for r in data[TUNE]}):
                fit = [r for r in data[TUNE] if r['source'] == source]
                test = [r for r in data[TUNE] if r['source'] != source]
                cv_params = select(fit, method)[0]
                classes, metrics = evaluate(test, method, cv_params)
                cv[f'select_{source}'] = {'params': list(cv_params), 'test_n': len(test), **metrics,
                                          'vs_baseline': paired(evaluate(test, 'baseline', ())[0], classes)}
            entry['cross_validation'] = cv
            chosen[method] = entry
        ranked = sorted((m for m in METHODS if m != 'baseline'),
                        key=lambda m: (-chosen[m]['tracks'][TUNE]['exact'], len(chosen[m]['params'])))
        results[model] = {'primary': ranked[0], 'methods': chosen}
        print(model, 'primary', ranked[0], chosen[ranked[0]]['params'], flush=True)
    write_json(out / 'summary.json', {'plan_sha256': plan['plan_sha256'], 'results': results,
                                      'blind_read': False, 'reference_read': False})


if __name__ == '__main__':
    main()
