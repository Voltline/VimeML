"""Development-only candidate-pool checks for the candidate export v2.

Compares the imported n-best 20/50 and empty-context pools of Standard v1
development with the original export, then rescores the n-best 50 pool with
V2.1 and V4.2 (FP32, same semantics as the frozen comparisons) and reports
exact / notation-lenient Top-1 against the recorded n-best 20 scores.
Blind and homophone tracks are refused.
"""
import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from vimeml.training.data import file_sha, write_json
from vimeml.benchmarks.standard_ime import load_standard_export
from vimeml.benchmarks.evaluate_ajimee import rerank
from vimeml.benchmarks.comparison import rows as read_rows
from vimeml.benchmarks.notation import error_class
from vimeml.benchmarks import statistics

IMPORTED = ROOT / 'artifacts/benchmarks/ime-candidates-v2-imported'
ORIGINAL = ROOT / 'artifacts/benchmarks/ime-standard-ja-v1-candidates/development'
ALLOWED = ('standard-v1-development-n20', 'standard-v1-development-n50', 'standard-v1-development-nocontext-n20')
MODELS = {
    'v21': (ROOT / 'artifacts/models/tiny-ja-v2.1-e16k-d320-l6-extend5/best.pt',
            ROOT / 'artifacts/tokenizers/ja-unigram-16k-v2',
            ROOT / 'outputs/evaluation-v4-20261015/v21/standard-development/scores.jsonl'),
    'v42-e20': (ROOT / 'outputs/training-v42-20261020/remote/checkpoints/best-lm.pt',
                ROOT / 'artifacts/tokenizers/v4-a/ja-unigram-8192',
                ROOT / 'outputs/evaluation-v42-20261020/standard-development/scores.jsonl'),
}
OUTCOMES = {'exact': ('correct',), 'notation_lenient': ('correct', 'notation')}


def load(name):
    if name not in ALLOWED:
        raise ValueError('Only exposed development pools are allowed here')
    return load_standard_export(IMPORTED / name)[2]


def texts(row):
    return [c['text'] for c in row['candidates']]


def covered(row):
    return bool(set(texts(row)) & set(row['answers']))


def pool_checks(n20, n50, empty):
    original = load_standard_export(ORIGINAL)[2]
    by_id = {r['id']: r for r in original}
    assert [r['id'] for r in n20] == [r['id'] for r in n50] == [r['id'] for r in empty]
    context = [i for i, r in enumerate(n20) if r['left_context']]
    gained = [i for i, (a, b) in enumerate(zip(n20, n50)) if not covered(a) and covered(b)]
    return {
        'determinism_n20': {
            'cases': len(n20),
            'identical_inputs': sum((r['query'], r['left_context']) == (by_id[r['id']]['query'], by_id[r['id']]['left_context']) for r in n20),
            'identical_candidate_lists': sum(texts(r) == texts(by_id[r['id']]) for r in n20)},
        'n50': {
            'covered_n20': sum(map(covered, n20)), 'covered_n50': sum(map(covered, n50)),
            'gained': len(gained), 'lost': sum(covered(a) and not covered(b) for a, b in zip(n20, n50)),
            'n20_subset_of_n50': sum(set(texts(a)) <= set(texts(b)) for a, b in zip(n20, n50)),
            'same_azookey_top1': sum(texts(a)[0] == texts(b)[0] for a, b in zip(n20, n50)),
            'gained_by_source': dict(Counter(n20[i]['source'] for i in gained)),
            'mean_pool_size': sum(len(texts(r)) for r in n50) / len(n50)},
        'empty_context_n20': {
            'with_context': len(context),
            'identical_candidate_lists': sum(texts(n20[i]) == texts(empty[i]) for i in context)},
    }


def rescore(name, n50):
    checkpoint, tokenizer, _ = MODELS[name]
    if name == 'v21':
        from vimeml.training.infer import JapaneseLM as Model
    else:
        from vimeml.training.infer_v4 import JapaneseLMV4 as Model
    lm = Model(checkpoint, tokenizer, 'cuda')
    rows = copy.deepcopy(n50)
    with torch.inference_mode():
        rerank(lm, rows)
    del lm
    torch.cuda.empty_cache()
    return rows


def consistency(recorded, scored):
    """Contextual scores on candidates shared with the recorded n20 run.

    Sums start after the prefix all candidates share, so a new n50 candidate that
    shortens it shifts every score in the case by one constant; only the spread of
    the per-candidate differences within a case can change a ranking.
    """
    spread, prefix_changed, order_kept = 0.0, 0, 0
    for a, b in zip(recorded, scored, strict=True):
        assert a['id'] == b['id']
        ca, cb = a['lm_scores']['contextual'], b['lm_scores']['contextual']
        old = {c['text']: c['log_probability_sum'] for c in ca['candidates']}
        new = {c['text']: c['log_probability_sum'] for c in cb['candidates']}
        diffs = [new[t] - old[t] for t in old if t in new]
        spread = max(spread, max(diffs) - min(diffs))
        prefix_changed += ca['common_prefix_tokens_including_bos'] != cb['common_prefix_tokens_including_bos']
        order_kept += [t for t in a['orders']['lm_context_sum'] if t in new] == \
            [t for t in b['orders']['lm_context_sum'] if t in old]
    return {'max_within_case_spread': spread, 'common_prefix_changed': prefix_changed,
            'shared_order_identical': order_kept, 'cases': len(recorded)}


def compare(recorded, scored):
    labels = statistics.clusters(recorded, template=None)
    before = [error_class(r, r['orders']['lm_context_sum'][0])[0] for r in recorded]
    after = [error_class(r, r['orders']['lm_context_sum'][0])[0] for r in scored]
    result = {'n20': dict(Counter(before)), 'n50': dict(Counter(after))}
    for metric, ok in OUTCOMES.items():
        result[metric] = statistics.paired([c in ok for c in before], [c in ok for c in after], labels)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='Separate empty output directory')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory')
    if not torch.cuda.is_available():
        parser.error('FP32 CUDA scoring is required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    n20, n50, empty = (load(n) for n in ALLOWED)
    summary = {'blind_read': False, 'homophone_read': False, 'script_sha256': file_sha(Path(__file__)),
               'pools': pool_checks(n20, n50, empty), 'models': {}}
    print(json.dumps(summary['pools'], ensure_ascii=False), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, (checkpoint, _, recorded_path) in MODELS.items():
        recorded = read_rows(recorded_path)
        assert [r['id'] for r in recorded] == [r['id'] for r in n50]
        scored = rescore(name, n50)
        (out / f'{name}-n50-scores.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in scored), encoding='utf-8')
        entry = {'checkpoint_sha256': file_sha(checkpoint), 'shared_scores': consistency(recorded, scored),
                 **compare(recorded, scored)}
        summary['models'][name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != 'checkpoint_sha256'}, ensure_ascii=False), flush=True)
    write_json(out / 'summary.json', summary)


if __name__ == '__main__':
    main()
