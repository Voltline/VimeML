"""Preregistered confirmation scoring of Standard v2 blind and the homophone challenge.

`plan` writes configs/ime-confirmation-v2.json from the constants below: hypotheses,
model snapshots (SHA-256, step), candidate pool file hashes and code hashes. It reads
no benchmark text. The plan must be committed before `run`.

`run` refuses unless the plan and this script are committed and unchanged, every hash
still matches, and no completed consumption receipt exists. It first rescores the exposed
v1 development n20 pool with every model and requires the recorded rankings to be
reproduced, and only then opens the blind and homophone pools. Scores are written once;
the report follows configs/ime-metrics-v2.json and the plan's decision rules.

`report` recomputes summary.json from a completed run without scoring again.
"""
import argparse
import copy
import importlib.metadata
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.training.data import file_sha, write_json
from vimeml.benchmarks.comparison import rows as read_rows

PLAN = ROOT / 'configs/ime-confirmation-v2.json'
IMPORTED = 'artifacts/benchmarks/ime-candidates-v2-imported'
RECEIPT = ROOT / IMPORTED / 'consumption.json'
POOLS = {
    'standard-v2-blind': {'n20': 'standard-v2-blind-n20', 'n50': 'standard-v2-blind-n50'},
    'homophone-v1': {'n20': 'homophone-v1-n20', 'n50': 'homophone-v1-n50'},
}
PREFLIGHT = 'standard-v1-development-n20'
MODELS = {
    'v21': {'checkpoint': 'artifacts/models/tiny-ja-v2.1-e16k-d320-l6-extend5/best.pt',
            'tokenizer': 'artifacts/tokenizers/ja-unigram-16k-v2', 'loader': 'v2',
            'role': 'Deployed model (FP32 weights of the INT8 deployment)',
            'recorded_development': 'outputs/evaluation-v4-20261015/v21/standard-development/scores.jsonl'},
    'v42-e20': {'checkpoint': 'outputs/training-v42-20261020/remote/checkpoints/best-lm.pt',
                'tokenizer': 'artifacts/tokenizers/v4-a/ja-unigram-8192', 'loader': 'v4',
                'role': 'Optional replacement; device-measured; selected by LM BPC before IME evaluation',
                'recorded_development': 'outputs/evaluation-v42-20261020/standard-development/scores.jsonl'},
    'v43-e30': {'checkpoint': 'outputs/training-v43-20261020/remote/checkpoints/epoch-30.pt',
                'tokenizer': 'artifacts/tokenizers/v4-a/ja-unigram-8192', 'loader': 'v4',
                'role': 'V4.2 recipe plus web data; best LM BPC checkpoint',
                'recorded_development': 'outputs/evaluation-v43-20261020/epoch30/standard-development/scores.jsonl'},
    'v5-e30': {'checkpoint': 'outputs/training-v5-20261021/remote/checkpoints/epoch-30.pt',
               'tokenizer': 'artifacts/tokenizers/v4-a/ja-unigram-8192', 'loader': 'v5',
               'role': 'V4.3 data and budget at d512/L8; best LM BPC checkpoint',
               'recorded_development': 'outputs/evaluation-v5-20261021/epoch30/standard-development/scores.jsonl'},
}
PRIMARY = {'id': 'H1', 'before': 'v21', 'after': 'v42-e20', 'track': 'standard-v2-blind', 'pool': 'n20',
           'question': 'Does V4.2 epoch 20 improve exact Top-1 over the deployed V2.1?'}
SECONDARY = [
    {'id': 'S1', 'before': 'v21', 'after': 'v42-e20', 'track': 'homophone-v1', 'pool': 'n20',
     'question': 'V2.1 -> V4.2 on written-text homophones'},
    {'id': 'S2', 'before': 'v43-e30', 'after': 'v5-e30', 'track': 'standard-v2-blind', 'pool': 'n20',
     'question': 'Model size only (same data and budget), conversational'},
    {'id': 'S3', 'before': 'v43-e30', 'after': 'v5-e30', 'track': 'homophone-v1', 'pool': 'n20',
     'question': 'Model size only (same data and budget), written-text homophones'},
    {'id': 'S4', 'before': 'v42-e20', 'after': 'v43-e30', 'track': 'standard-v2-blind', 'pool': 'n20',
     'question': 'Added web data, conversational'},
]
CODE = ['configs/ime-metrics-v2.json', 'scripts/benchmarks/confirm_ime_v2.py',
        'src/vimeml/benchmarks/notation.py', 'src/vimeml/benchmarks/statistics.py',
        'src/vimeml/benchmarks/evaluate_ajimee.py', 'src/vimeml/benchmarks/standard_ime.py',
        'src/vimeml/benchmarks/expanded_ime.py', 'src/vimeml/benchmarks/comparison.py',
        'src/vimeml/training/evaluate_ime.py', 'src/vimeml/training/infer.py',
        'src/vimeml/training/infer_v4.py', 'src/vimeml/training/infer_v5.py']
PACKAGES = ('torch', 'fugashi', 'unidic-lite', 'sentencepiece')
POOL_FILES = ('manifest.json', 'case-map.json', 'ajimee-results/ajimee-input.json',
              'ajimee-results/azookey-candidates.json')
OUTCOMES = {'exact': ('correct',), 'notation_lenient': ('correct', 'notation')}
ALPHA = 0.05


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def pool_hashes(name):
    folder = ROOT / IMPORTED / name
    return {f: file_sha(folder / f) for f in POOL_FILES}


def make_plan(args):
    if PLAN.exists():
        raise ValueError('The plan already exists; a changed plan needs a new version and a new blind release.')
    if RECEIPT.exists():
        raise ValueError('These pools have already been opened.')
    import torch
    models = {}
    for name, spec in MODELS.items():
        path = ROOT / spec['checkpoint']
        saved = torch.load(path, map_location='cpu', weights_only=True)
        models[name] = {**spec, 'sha256': file_sha(path), 'bytes': path.stat().st_size, 'step': saved['step']}
        del saved
    release_ids = {track: json.loads((ROOT / IMPORTED / pools['n20'] / 'manifest.json').read_text(encoding='utf-8'))['release_id']
                   for track, pools in POOLS.items()}
    write_json(PLAN, {
        'version': 'ime-confirmation-v2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'metric_policy': 'configs/ime-metrics-v2.json',
        'tracks': {track: {'release_id': release_ids[track], 'cases': {'standard-v2-blind': 729, 'homophone-v1': 619}[track],
                           'pools': {k: {'directory': f'{IMPORTED}/{v}', 'sha256': pool_hashes(v)} for k, v in pools.items()}}
                   for track, pools in POOLS.items()},
        'preflight': {'pool': f'{IMPORTED}/{PREFLIGHT}', 'sha256': pool_hashes(PREFLIGHT),
                      'requirement': 'Every model reproduces its recorded development lm_context_sum order on all 728 cases before any blind or homophone file is read'},
        'models': models,
        'scoring': {'semantics': 'FP32 full-vocabulary suffix logP sum; joint context+candidate tokenization; no final EOS, truncation or tuning; stable ties',
                    'precision': 'FP32 on CUDA; TF32 disabled', 'packages': versions()},
        'baseline': 'AzooKey first candidate (no language model), descriptive only',
        'primary': {**PRIMARY, 'metric': 'exact Top-1',
                    'claim_rule': 'net > 0 and author-cluster sign-flip p < 0.05; and no notation-lenient loss with cluster p < 0.05 for the same pair on either n20 track',
                    'if_confirmed': 'V4.2 is a confirmed exact Top-1 improvement over the deployed V2.1',
                    'if_not_confirmed': 'No confirmed difference; V2.1 stays deployed and V4.2 stays an optional variant'},
        'secondary': {'tests': SECONDARY, 'metric': 'exact Top-1',
                      'multiplicity': 'Holm over the four cluster sign-flip p-values at 0.05',
                      'claim_rule': 'Holm-adjusted p < 0.05 (two-sided; direction from net) and the same notation-lenient guard as H1'},
        'descriptive': ['notation-lenient and pool-conditional Top-1 for every model and track',
                        'pool coverage at n20 and n50, opened with scoring',
                        'n50 pools: every metric and pair, descriptive only (metrics v2 keeps n20 primary)',
                        'every model pair on both tracks, unadjusted, descriptive only',
                        'template and union cluster sensitivity',
                        'Standard by source (WRIME, JMultiWOZ)',
                        'homophone: exact Top-1 when the reference is the more frequent vs the second lemma of its group; groups with both items correct (complete groups only)',
                        'error classes'],
        'clusters': 'Author (WRIME user, JMultiWOZ user) on Standard; homophone group on the challenge track',
        'execution': 'One run. A run that stops for a technical failure before writing summary.json may be repeated into a new directory under the same plan, recorded in the receipt. Results are never rerun to change an outcome',
        'deviations': 'Any change after the first run (bug fix, metric correction) is reported as a deviation next to the original output, which is kept',
        'publication': 'Aggregates and case IDs only; no benchmark text, candidates or homophone group lists',
        'code_sha256': {path: file_sha(ROOT / path) for path in CODE},
    })
    print(f'Wrote {PLAN.relative_to(ROOT)}; commit it with this script before `run`.')


def check_committed(plan):
    paths = ['configs/ime-confirmation-v2.json', *plan['code_sha256']]
    git('ls-files', '--error-unmatch', *paths)
    dirty = git('status', '--porcelain', '--', *paths)
    if dirty:
        raise ValueError(f'Commit the plan and code first:\n{dirty}')
    for path, digest in plan['code_sha256'].items():
        if file_sha(ROOT / path) != digest:
            raise ValueError(f'{path} differs from the plan')
    if versions() != plan['scoring']['packages']:
        raise ValueError(f'Package versions differ from the plan: {versions()}')


def load_model(spec):
    checkpoint, tokenizer = ROOT / spec['checkpoint'], ROOT / spec['tokenizer']
    if spec['loader'] == 'v2':
        from vimeml.training.infer import JapaneseLM as Model
    elif spec['loader'] == 'v4':
        from vimeml.training.infer_v4 import JapaneseLMV4 as Model
    else:
        from vimeml.training.infer_v5 import JapaneseLMV5 as Model
    lm = Model(checkpoint, tokenizer, 'cuda')
    if lm.metadata['checkpoint_step'] != spec['step'] or file_sha(checkpoint) != spec['sha256']:
        raise ValueError(f'{checkpoint} differs from the planned snapshot')
    return lm


def load_pool(entry):
    from vimeml.benchmarks.standard_ime import load_standard_export
    folder = ROOT / entry['directory']
    if {f: file_sha(folder / f) for f in POOL_FILES} != entry['sha256']:
        raise ValueError(f'{folder} differs from the plan')
    return load_standard_export(folder)[2]


def score(lm, rows):
    from vimeml.benchmarks.evaluate_ajimee import rerank
    import torch
    rows = copy.deepcopy(rows)
    with torch.inference_mode():
        rerank(lm, rows)
    return rows


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows),
                    encoding='utf-8', newline='\n')


def run(args):
    import torch
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    check_committed(plan)
    receipt = json.loads(RECEIPT.read_text(encoding='utf-8')) if RECEIPT.exists() else {'attempts': []}
    if receipt.get('plan_sha256', file_sha(PLAN)) != file_sha(PLAN):
        raise ValueError('These pools were opened under a different plan.')
    if any(a['status'] == 'complete' for a in receipt['attempts']):
        raise ValueError('This plan has already been scored; use `report` on the completed output.')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    if not torch.cuda.is_available():
        raise ValueError('FP32 CUDA scoring is required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(4)
    commit = git('rev-parse', 'HEAD').strip()
    plan_sha = file_sha(PLAN)

    # Preflight on exposed development: the pipeline must reproduce recorded rankings.
    development = load_pool({'directory': plan['preflight']['pool'], 'sha256': plan['preflight']['sha256']})
    models = {}
    for name, spec in plan['models'].items():
        lm = load_model(spec)
        scored = score(lm, development)
        recorded = read_rows(ROOT / spec['recorded_development'])
        same = sum(a['id'] == b['id'] and a['orders']['lm_context_sum'] == b['orders']['lm_context_sum']
                   for a, b in zip(recorded, scored, strict=True))
        print(f'preflight {name}: {same}/{len(scored)} recorded development orders reproduced', flush=True)
        if same != len(scored):
            raise ValueError(f'{name} does not reproduce its recorded development scores; blind pools stay closed.')
        models[name] = lm

    out.mkdir(parents=True)
    receipt['plan_sha256'] = plan_sha
    receipt['attempts'].append({'output': str(out.relative_to(ROOT)), 'commit': commit, 'status': 'started',
                                'started_utc': datetime.now(timezone.utc).isoformat()})
    write_json(RECEIPT, receipt)
    for track, info in plan['tracks'].items():
        for pool, entry in info['pools'].items():
            rows = load_pool(entry)
            if len(rows) != info['cases']:
                raise ValueError(f'{track} {pool}: {len(rows)} cases')
            for name, lm in models.items():
                dest = out / track / pool
                dest.mkdir(parents=True, exist_ok=True)
                write_rows(dest / f'{name}.jsonl', score(lm, rows))
                print(f'scored {track} {pool} {name}', flush=True)
    write_json(out / 'run.json', {'plan_sha256': plan_sha, 'commit': commit, 'gpu': torch.cuda.get_device_name(),
                                  'cuda': torch.version.cuda, 'packages': versions(),
                                  'completed_utc': datetime.now(timezone.utc).isoformat()})
    receipt['attempts'][-1].update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(RECEIPT, receipt)
    report(argparse.Namespace(output=out))


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def track_report(rows):
    from vimeml.benchmarks.notation import error_class
    from vimeml.benchmarks import statistics
    base = next(iter(rows.values()))
    for data in rows.values():
        assert [r['id'] for r in data] == [r['id'] for r in base]
        assert all(r['answers'] == b['answers'] and r['orders']['azookey'] == b['orders']['azookey']
                   for r, b in zip(data, base))
    reachable = [any(a in {c['text'] for c in r['candidates']} for a in r['answers']) for r in base]
    labels = {'author': statistics.clusters(base, template=None),
              'template': statistics.clusters([{'answers': r['answers']} for r in base]),
              'union': statistics.clusters(base)}
    classes = {'azookey': [error_class(r, r['orders']['azookey'][0])[0] for r in base]}
    classes.update({name: [error_class(r, r['orders']['lm_context_sum'][0])[0] for r in data] for name, data in rows.items()})
    sources = sorted({r['source'] for r in base})
    homophone = all(str(r['provenance'].get('group_id', '')).startswith('homophone:') for r in base)
    entry = {'cases': len(base), 'pool_reachable': sum(reachable),
             'clusters': {k: len(set(v)) for k, v in labels.items()}, 'models': {}, 'paired': {}}
    for name, items in classes.items():
        exact = sum(c == 'correct' for c in items)
        model = {'exact': exact, 'notation_lenient': exact + items.count('notation'),
                 'pool_conditional': exact / sum(reachable), 'error_classes': dict(Counter(items))}
        if len(sources) > 1:
            model['by_source'] = {s: sum(c == 'correct' for c, r in zip(items, base) if r['source'] == s) for s in sources}
        if homophone:
            ranks = [r['provenance']['group_rank'] for r in base]
            model['by_group_rank'] = {str(k): sum(c == 'correct' for c, g in zip(items, ranks) if g == k) for k in (0, 1)}
            groups = Counter(r['provenance']['group_id'] for r in base)
            correct = Counter(r['provenance']['group_id'] for c, r in zip(items, base) if c == 'correct')
            model['complete_groups_both_correct'] = sum(correct[g] == 2 for g, n in groups.items() if n == 2)
        entry['models'][name] = model
    if homophone:
        groups = Counter(r['provenance']['group_id'] for r in base)
        entry['groups'] = {'all': len(groups), 'complete': sum(n == 2 for n in groups.values()),
                           'cases_by_group_rank': dict(Counter(str(r['provenance']['group_rank']) for r in base))}
    names = list(classes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            entry['paired'][f'{a}->{b}'] = {
                metric: {scheme: statistics.paired([c in ok for c in classes[a]], [c in ok for c in classes[b]], lab)
                         for scheme, lab in labels.items()}
                for metric, ok in OUTCOMES.items()}
    return entry, reachable


def report(args):
    out = args.output.resolve()
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    run_info = json.loads((out / 'run.json').read_text(encoding='utf-8'))
    tracks, reach = {}, {}
    for track, info in plan['tracks'].items():
        tracks[track] = {}
        for pool in info['pools']:
            rows = {name: read_rows(out / track / pool / f'{name}.jsonl') for name in plan['models']}
            tracks[track][pool], reach[track, pool] = track_report(rows)
        n20, n50 = reach[track, 'n20'], reach[track, 'n50']
        tracks[track]['coverage'] = {'n20': sum(n20), 'n50': sum(n50),
                                     'gained': sum(b and not a for a, b in zip(n20, n50)),
                                     'lost': sum(a and not b for a, b in zip(n20, n50))}

    def test(spec):
        pair = tracks[spec['track']][spec['pool']]['paired'][f"{spec['before']}->{spec['after']}"]
        guard = {t: tracks[t]['n20']['paired'][f"{spec['before']}->{spec['after']}"]['notation_lenient']['author']
                 for t in plan['tracks']}
        lenient_loss = [t for t, g in guard.items() if g['net'] < 0 and g['cluster_sign_flip_p'] < ALPHA]
        return pair['exact']['author'], lenient_loss

    exact, lenient_loss = test(plan['primary'])
    primary = {**plan['primary'], 'exact': exact, 'notation_lenient_losses': lenient_loss,
               'confirmed': exact['net'] > 0 and exact['cluster_sign_flip_p'] < ALPHA and not lenient_loss}
    results = {s['id']: test(s) for s in plan['secondary']['tests']}
    adjusted = holm({k: v[0]['cluster_sign_flip_p'] for k, v in results.items()})
    secondary = []
    for spec in plan['secondary']['tests']:
        exact, losses = results[spec['id']]
        significant = adjusted[spec['id']] < ALPHA and not losses
        secondary.append({**spec, 'exact': exact, 'holm_p': adjusted[spec['id']], 'notation_lenient_losses': losses,
                          'claim': ('improvement' if exact['net'] > 0 else 'regression') if significant else 'none'})
    summary = {'plan_sha256': file_sha(PLAN), 'run': run_info, 'primary': primary, 'secondary': secondary,
               'tracks': tracks, 'report_script_sha256': file_sha(Path(__file__))}
    write_json(out / 'summary.json', summary)
    print(json.dumps({'primary': {k: primary[k] for k in ('id', 'exact', 'notation_lenient_losses', 'confirmed')},
                      'secondary': [{k: s[k] for k in ('id', 'holm_p', 'claim')} for s in secondary],
                      'models': {t: {p: {m: {k: v[k] for k in ('exact', 'notation_lenient')}
                                         for m, v in tracks[t][p]['models'].items()} for p in ('n20', 'n50')}
                                 for t in tracks},
                      'coverage': {t: tracks[t]['coverage'] for t in tracks}}, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('plan')
    for name in ('run', 'report'):
        subs.add_parser(name).add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'plan': make_plan, 'run': run, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
