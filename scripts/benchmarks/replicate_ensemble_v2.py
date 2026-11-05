"""Preregistered second use of Standard v2 blind and the homophone set: V4.2-recipe seed ensemble.

The four V4.2-recipe seeds (V4.2 epoch 20 and V4.4 s1-s3) were combined only after the V4.4
analysis on development (docs/zh-CN/training-v44.md). Both pools were consumed once by
confirmation v2, and no model or ensemble rule was tuned on them since, so this is a
replication: its outcome is descriptive evidence, not a deployment confirmation.

`plan` writes configs/ime-ensemble-replication-v2.json (hypotheses, snapshots, the reused
V4.2 confirmation score files, pool and code hashes); it reads no benchmark text and must be
committed and pushed before `run`. `run` refuses unless the plan and code are committed and
unchanged, s1-s3 reproduce their recorded development orders, and no completed attempt
exists; it then scores s1-s3 once on the n20 and n50 pools, reuses the recorded V4.2 scores,
and builds the ensembles. `report` recomputes summary.json without scoring. Loading,
scoring and per-track statistics are those of `confirm_ime_v2.py`.
"""
import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/benchmarks'))
import confirm_ime_v2 as v2
from vimeml.training.data import file_sha, write_json
from vimeml.benchmarks.comparison import rows as read_rows

PLAN = ROOT / 'configs/ime-ensemble-replication-v2.json'
RECEIPT = ROOT / v2.IMPORTED / 'ensemble-replication.json'
CONFIRMATION = 'outputs/confirmation-v2-20261021'
SEEDS = {arm: {'checkpoint': f'outputs/training-v44-20261105/remote/{arm}/checkpoints/epoch-20.pt',
               'tokenizer': 'artifacts/tokenizers/v4-a/ja-unigram-8192', 'loader': 'v4',
               'recorded_development': f'outputs/evaluation-v44-20261105/{arm}/standard-development/scores.jsonl'}
         for arm in ('s1', 's2', 's3')}
MEMBERS = ['v42-e20', 's1', 's2', 's3']
ENSEMBLE = 'ens4'
PRIMARY = {'id': 'R1', 'before': 'v42-e20', 'after': ENSEMBLE, 'track': 'standard-v2-blind', 'pool': 'n20',
           'question': 'Does the four-seed ensemble improve exact Top-1 over the deployed V4.2 epoch 20?'}
SECONDARY = [{'id': 'R2', 'before': 'v42-e20', 'after': ENSEMBLE, 'track': 'homophone-v1', 'pool': 'n20',
              'question': 'The same on written-text homophones'}]
CODE = ['scripts/benchmarks/replicate_ensemble_v2.py', *v2.CODE]


def make_plan(args):
    if PLAN.exists():
        raise ValueError('The plan already exists.')
    if RECEIPT.exists():
        raise ValueError('These pools have already been opened for this replication.')
    import torch
    confirmation = json.loads(v2.PLAN.read_text(encoding='utf-8'))
    models = {}
    for name, spec in SEEDS.items():
        path = ROOT / spec['checkpoint']
        saved = torch.load(path, map_location='cpu', weights_only=True)
        models[name] = {**spec, 'sha256': file_sha(path), 'bytes': path.stat().st_size, 'step': saved['step']}
        del saved
    reused = {f'{track}/{pool}': {'file': f'{CONFIRMATION}/{track}/{pool}/v42-e20.jsonl',
                                  'sha256': file_sha(ROOT / CONFIRMATION / track / pool / 'v42-e20.jsonl')}
              for track, info in confirmation['tracks'].items() for pool in info['pools']}
    write_json(PLAN, {
        'version': 'ime-ensemble-replication-v2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'metric_policy': 'configs/ime-metrics-v2.json',
        'status_of_sets': ('Second use. Both pools were consumed once by configs/ime-confirmation-v2.json; '
                           'the ensemble idea came from development after that. Outcomes are descriptive replication '
                           'evidence; any deployment claim needs a fresh blind release'),
        'tracks': confirmation['tracks'],
        'preflight': {**confirmation['preflight'],
                      'requirement': 's1-s3 reproduce their recorded development lm_context_sum order on all 728 cases before any blind or homophone file is read'},
        'models': models,
        'reused_scores': {'model': 'v42-e20', 'source': 'configs/ime-confirmation-v2.json run', 'files': reused},
        'ensemble': {'name': ENSEMBLE, 'members': MEMBERS,
                     'rule': 'Rank candidates by the equal-weight sum of the members\' contextual full-vocabulary suffix logP sums; ties keep AzooKey order'},
        'scoring': {**confirmation['scoring'], 'packages': v2.versions()},
        'primary': {**PRIMARY, 'metric': 'exact Top-1',
                    'claim_rule': 'net > 0 and author-cluster sign-flip p < 0.05; and no notation-lenient loss with cluster p < 0.05 for the same pair on either n20 track',
                    'if_replicated': 'The development ensemble gain replicates on the consumed blind; distill the ensemble (V4.5)',
                    'if_not_replicated': 'No replicated gain; ensemble distillation is not pursued on this evidence'},
        'secondary': {'tests': SECONDARY, 'metric': 'exact Top-1',
                      'claim_rule': 'two-sided author/homophone-group cluster p < 0.05 (single test, no adjustment) and the same notation-lenient guard'},
        'descriptive': ['seed band (mean, range, SD) of the four single seeds on each track and pool',
                        'every two- and three-member ensemble',
                        'ensemble against each single seed',
                        'n50 pools, error classes, by source and homophone diagnostics as in confirmation v2'],
        'execution': 'One run; a run that stops for a technical failure before summary.json may be repeated into a new directory, recorded in the receipt. Never rerun to change an outcome',
        'publication': 'Aggregates and case IDs only',
        'code_sha256': {path: file_sha(ROOT / path) for path in CODE},
    })
    print(f'Wrote {PLAN.relative_to(ROOT)}; commit and push it with this script before `run`.')


def check_committed(plan):
    paths = [str(PLAN.relative_to(ROOT)), *plan['code_sha256']]
    v2.git('ls-files', '--error-unmatch', *paths)
    if v2.git('status', '--porcelain', '--', *paths):
        raise ValueError('Commit the plan and code first')
    if v2.git('rev-list', '--count', '@{u}..HEAD').strip() != '0':
        raise ValueError('Push the plan before opening the pools')
    for path, digest in plan['code_sha256'].items():
        if file_sha(ROOT / path) != digest:
            raise ValueError(f'{path} differs from the plan')
    if v2.versions() != plan['scoring']['packages']:
        raise ValueError(f'Package versions differ from the plan: {v2.versions()}')


def run(args):
    import torch
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    check_committed(plan)
    receipt = json.loads(RECEIPT.read_text(encoding='utf-8')) if RECEIPT.exists() else {'attempts': []}
    if any(a['status'] == 'complete' for a in receipt['attempts']):
        raise ValueError('This plan has already been scored; use `report`.')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    for entry in plan['reused_scores']['files'].values():
        if file_sha(ROOT / entry['file']) != entry['sha256']:
            raise ValueError(f"{entry['file']} differs from the plan")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(4)
    development = v2.load_pool({'directory': plan['preflight']['pool'], 'sha256': plan['preflight']['sha256']})
    models = {}
    for name, spec in plan['models'].items():
        lm = v2.load_model(spec)
        scored = v2.score(lm, development)
        recorded = read_rows(ROOT / spec['recorded_development'])
        same = sum(a['id'] == b['id'] and a['orders']['lm_context_sum'] == b['orders']['lm_context_sum']
                   for a, b in zip(recorded, scored, strict=True))
        print(f'preflight {name}: {same}/{len(scored)} recorded development orders reproduced', flush=True)
        if same != len(scored):
            raise ValueError(f'{name} does not reproduce its development scores; pools stay closed.')
        models[name] = lm
    out.mkdir(parents=True)
    commit = v2.git('rev-parse', 'HEAD').strip()
    receipt['plan_sha256'] = file_sha(PLAN)
    receipt['attempts'].append({'output': str(out.relative_to(ROOT)), 'commit': commit, 'status': 'started',
                                'started_utc': datetime.now(timezone.utc).isoformat()})
    write_json(RECEIPT, receipt)
    for track, info in plan['tracks'].items():
        for pool, entry in info['pools'].items():
            rows = v2.load_pool(entry)
            if len(rows) != info['cases']:
                raise ValueError(f'{track} {pool}: {len(rows)} cases')
            dest = out / track / pool
            dest.mkdir(parents=True, exist_ok=True)
            for name, lm in models.items():
                v2.write_rows(dest / f'{name}.jsonl', v2.score(lm, rows))
                print(f'scored {track} {pool} {name}', flush=True)
    write_json(out / 'run.json', {'plan_sha256': file_sha(PLAN), 'commit': commit, 'gpu': torch.cuda.get_device_name(),
                                  'cuda': torch.version.cuda, 'packages': v2.versions(),
                                  'completed_utc': datetime.now(timezone.utc).isoformat()})
    receipt['attempts'][-1].update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(RECEIPT, receipt)
    report(argparse.Namespace(output=out))


def ensemble_rows(rows, members):
    """Rows whose lm_context_sum order is the members' summed contextual logP ranking.

    Where the context is not scorable (e.g. longer than the 128-token window) every
    single model falls back to the AzooKey order; the shared tokenizer makes that
    fallback identical across members, and the ensemble keeps it too.
    """
    base = rows[members[0]]
    result = []
    for i, row in enumerate(base):
        order = row['orders']['azookey']
        fallback = {rows[name][i]['fallbacks'].get('lm_context_sum') for name in members}
        if len(fallback) != 1:
            raise ValueError(f"{row['id']}: members disagree on the contextual fallback")
        if fallback != {None}:
            merged = copy.copy(row)
            merged['orders'] = {**row['orders'], 'lm_context_sum': list(order)}
            result.append(merged)
            continue
        total = {t: 0.0 for t in order}
        for name in members:
            assert rows[name][i]['id'] == row['id']
            for c in rows[name][i]['lm_scores']['contextual']['candidates']:
                total[c['text']] += c['log_probability_sum']
        merged = copy.copy(row)
        merged['orders'] = {**row['orders'], 'lm_context_sum': sorted(order, key=lambda t: -total[t])}
        result.append(merged)
    return result


def report(args):
    import statistics as stats
    from vimeml.benchmarks.notation import error_class
    out = args.output.resolve()
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    tracks = {}
    for track, info in plan['tracks'].items():
        tracks[track] = {}
        for pool in info['pools']:
            rows = {'v42-e20': read_rows(ROOT / plan['reused_scores']['files'][f'{track}/{pool}']['file'])}
            rows.update({name: read_rows(out / track / pool / f'{name}.jsonl') for name in plan['models']})
            rows = {name: rows[name] for name in MEMBERS}
            rows[ENSEMBLE] = ensemble_rows(rows, MEMBERS)
            entry, _ = v2.track_report(rows)
            exact = [entry['models'][m]['exact'] for m in MEMBERS]
            lenient = [entry['models'][m]['notation_lenient'] for m in MEMBERS]
            entry['seed_band'] = {key: {'mean': stats.mean(v), 'min': min(v), 'max': max(v), 'sd': stats.stdev(v)}
                                  for key, v in (('exact', exact), ('notation_lenient', lenient))}
            subsets = {}
            for size in (2, 3):
                for members in combinations(MEMBERS, size):
                    merged = ensemble_rows(rows, list(members))
                    items = [error_class(r, r['orders']['lm_context_sum'][0])[0] for r in merged]
                    subsets['+'.join(members)] = {'exact': items.count('correct'),
                                                  'notation_lenient': items.count('correct') + items.count('notation')}
            entry['subset_ensembles'] = subsets
            entry['ensemble_size_exact'] = {'1': stats.mean(exact), **{
                str(size): stats.mean(v['exact'] for k, v in subsets.items() if k.count('+') == size - 1)
                for size in (2, 3)}, '4': entry['models'][ENSEMBLE]['exact']}
            tracks[track][pool] = entry

    def test(spec):
        key = f"{spec['before']}->{spec['after']}"
        guard = {t: tracks[t]['n20']['paired'][key]['notation_lenient']['author'] for t in plan['tracks']}
        losses = [t for t, g in guard.items() if g['net'] < 0 and g['cluster_sign_flip_p'] < v2.ALPHA]
        exact = tracks[spec['track']][spec['pool']]['paired'][key]['exact']['author']
        return exact, losses

    exact, losses = test(plan['primary'])
    primary = {**plan['primary'], 'exact': exact, 'notation_lenient_losses': losses,
               'replicated': exact['net'] > 0 and exact['cluster_sign_flip_p'] < v2.ALPHA and not losses}
    secondary = []
    for spec in plan['secondary']['tests']:
        exact, losses = test(spec)
        significant = exact['cluster_sign_flip_p'] < v2.ALPHA and not losses
        secondary.append({**spec, 'exact': exact, 'notation_lenient_losses': losses,
                          'claim': ('improvement' if exact['net'] > 0 else 'regression') if significant else 'none'})
    run_info = json.loads((out / 'run.json').read_text(encoding='utf-8'))
    write_json(out / 'summary.json', {'plan_sha256': file_sha(PLAN), 'run': run_info, 'primary': primary,
                                      'secondary': secondary, 'tracks': tracks,
                                      'report_script_sha256': file_sha(Path(__file__))})
    print(json.dumps({'primary': {k: primary[k] for k in ('id', 'exact', 'notation_lenient_losses', 'replicated')},
                      'secondary': [{k: s[k] for k in ('id', 'exact', 'claim')} for s in secondary],
                      'band_and_sizes': {t: {p: {'band': tracks[t][p]['seed_band']['exact'],
                                                 'sizes': tracks[t][p]['ensemble_size_exact']}
                                             for p in tracks[t]} for t in tracks}}, ensure_ascii=False, indent=1))


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
