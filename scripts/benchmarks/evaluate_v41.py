"""Reproduce frozen V4.1 FP32 comparisons in a separate local output directory.

Requires the hash-verified model download and the four pinned nonblind pools.
The original October 16 evaluation script remains with its recorded outputs.
"""
import argparse
import copy
import datetime
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from vimeml.training.data import file_sha, write_json
from vimeml.training.infer_v4 import JapaneseLMV4
from vimeml.benchmarks.evaluate_ajimee import rerank, metrics, summarize
from vimeml.benchmarks.standard_ime import source_metrics
from vimeml.benchmarks.comparison import paired, rows as read_rows
from vimeml.posttraining.infer import TRACKS, read_snapshot, load_conditional_checkpoint, fp32_evaluation, evaluate_snapshot

OUT = ROOT / 'outputs/evaluation-v41-20261016'
MODEL = ROOT / 'artifacts/models/tiny-ja-v41-e8k-d384-l7'
TOKENIZER = ROOT / 'artifacts/tokenizers/v4-a/ja-unigram-8192'
BASELINES = {'v21': ROOT / 'outputs/evaluation-v4-20261015/v21',
             'v4b': ROOT / 'outputs/evaluation-v4-b12cosine-20261015/best'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def check_base(base, frozen):
    assert len(base) == len(frozen)
    for a, b in zip(base, frozen, strict=True):
        for k in ['id', 'query', 'left_context', 'answers', 'candidates']:
            assert a[k] == b[k], (k, b['id'])
        assert a['orders']['azookey'] == b['orders']['azookey']


def comparisons(destination, scored, track):
    for name, folder in BASELINES.items():
        base = read_rows(folder / track / 'scores.jsonl')
        check_base(base, scored)
        result = paired(base, scored)
        result['paired'].pop('significant_at_0_05', None)
        write_json(destination / f'paired-{name}.json', result)


def replay_check(name, scored, conditional=False):
    epoch = 13 if name == 'best-ordinary' else 16
    folder = 'standard-development' if conditional else 'standard-ordinary'
    remote = {r['id']: r for r in read_rows(MODEL / f'epoch-evaluation/epoch-{epoch}/{folder}/scores.jsonl')}
    differences, top1_differences = [], []
    for r in scored:
        if r['orders']['lm_context_sum'] != remote[r['id']]['ranked']:
            differences.append(r['id'])
        if r['orders']['lm_context_sum'][0] != remote[r['id']]['ranked'][0]:
            top1_differences.append(r['id'])
    result = {'epoch': epoch, 'cases': len(scored), 'complete_order_differences': differences,
              'top1_choice_differences': top1_differences,
              'top1_correct': metrics(scored, 'lm_context_sum')['top1_correct'],
              'remote_top1_correct': sum(r['correct'] for r in remote.values())}
    if result['top1_correct'] != result['remote_top1_correct']:
        raise ValueError(('Local FP32 Top-1 differs from frozen training-time result', result))
    return result


def ordinary(name, inputs):
    dest = OUT / name / 'ordinary'
    lm = JapaneseLMV4(MODEL / f'{name}.pt', TOKENIZER, 'cuda')
    reports = {}
    for track, frozen in inputs.items():
        target = dest / track
        if (target / 'metrics.json').exists():
            reports[track] = read(target / 'metrics.json')
            continue
        rows = copy.deepcopy(frozen)
        began = time.perf_counter()
        rerank(lm, rows)
        result = {'model': lm.metadata, 'metrics': summarize(rows),
                  'elapsed_seconds': time.perf_counter() - began,
                  'source_metrics': source_metrics(rows, metrics) if 'source' in rows[0] else None}
        if track == 'standard-development':
            result['remote_replay'] = replay_check(name, rows)
        save_rows(target / 'scores.jsonl', rows)
        comparisons(target, rows, track)
        write_json(target / 'metrics.json', result)
        reports[track] = result
        print(name, 'ordinary', track, result['metrics']['all']['lm_context_sum']['top1_correct'], flush=True)
    generated = []
    with torch.inference_mode():
        for prompt in read(ROOT / 'configs/inference-prompts.json'):
            generated.append({**prompt, 'results': [lm.generate(prompt['prompt'], max_new_tokens=16, temperature=t, seed=42) for t in [0, .8]]})
    write_json(dest / 'generation.json', {'qualitative_only': True, 'prompts_sha256': file_sha(ROOT / 'configs/inference-prompts.json'), 'samples': generated})
    del lm
    gc.collect()
    torch.cuda.empty_cache()
    return reports


def conditional(inputs):
    model, codec, metadata = load_conditional_checkpoint(MODEL / 'best-rerank.pt', TOKENIZER / 'tokenizer.model', 'cuda')
    reports = {}
    with fp32_evaluation(model):
        for condition in ['context_reading', 'empty_reading', 'empty_context']:
            reports[condition] = {}
            for track, frozen in inputs.items():
                if condition != 'context_reading' and track != 'standard-development':
                    continue
                target = OUT / 'best-rerank' / condition / track
                if (target / 'comparison-metrics.json').exists():
                    reports[condition][track] = read(target / 'comparison-metrics.json')
                    continue
                began = time.perf_counter()
                result = evaluate_snapshot(model, codec, frozen, target, condition)
                records = read_rows(target / 'scores.jsonl')
                common = copy.deepcopy(frozen)
                for row, record in zip(common, records, strict=True):
                    assert row['id'] == record['id'] and row['answers'] == record['answers']
                    assert row['orders']['azookey'] == record['candidate_order']
                    row['orders']['lm_context_sum'] = record['ranked']
                    if record['fallback']:
                        row['fallbacks']['lm_context_sum'] = record['fallback']
                report = {'model': metadata, 'condition': condition, 'raw_metrics': result,
                          'metrics': summarize(common), 'elapsed_seconds': time.perf_counter() - began,
                          'source_metrics': source_metrics(common, metrics) if 'source' in common[0] else None}
                if condition == 'context_reading' and track == 'standard-development':
                    report['remote_replay'] = replay_check('last', common, conditional=True)
                save_rows(target / 'comparison-scores.jsonl', common)
                comparisons(target, common, track)
                write_json(target / 'comparison-metrics.json', report)
                reports[condition][track] = report
                print('best-rerank', condition, track, result['correct'], flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return reports


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Separate output directory; completed results are never overwritten')
    args = parser.parse_args()
    OUT = args.output.resolve()
    if OUT.exists() and any(OUT.iterdir()):
        parser.error('Use an empty output directory to preserve prior evaluations')
    if not torch.cuda.is_available():
        parser.error('The recorded V4.1 comparison uses an FP32 CUDA GPU')
    receipt = read(ROOT / 'outputs/training-v41-20261016/download-receipt.json')
    assert receipt['status'] == 'verified'
    inputs = {track: read_snapshot(track) for track in TRACKS}
    hashes = {r['relative']: r['sha256'] for r in receipt['files']}
    for name in ['best-ordinary', 'last', 'best-rerank']:
        p = MODEL / f'{name}.pt'
        assert file_sha(p) == hashes[p.relative_to(ROOT).as_posix()]
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'primary': 'best-ordinary.pt selected on Standard development at epoch13 before these local comparisons',
            'secondary': 'last.pt epoch16; best-rerank.pt epoch16; conditional empty-READ/CTX on Standard only; no coefficients tuned',
            'tracks': {k: {'cases': len(v), 'sha256': TRACKS[k][1]} for k, v in inputs.items()},
            'checkpoints': {n: file_sha(MODEL / f'{n}.pt') for n in ['best-ordinary', 'last', 'best-rerank']},
            'precision': 'primitive FP32; TF32 disabled; no quantization',
            'statistics': 'Exact paired McNemar descriptive p-values; correlated development/regression tracks, repeated checkpoint selection; no independent-test or multiplicity-adjusted claims',
            'blind_read': False, 'test_read': False, 'script_sha256': file_sha(Path(__file__))}
    if (OUT / 'plan.json').exists():
        previous = read(OUT / 'plan.json')
        assert previous['checkpoints'] == plan['checkpoints'] and previous['script_sha256'] == plan['script_sha256']
    else:
        write_json(OUT / 'plan.json', plan)
        (OUT / 'evaluation-script.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    write_json(OUT / 'environment.json', {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(), 'precision': 'FP32', 'tf32': False})
    results = {}
    with torch.inference_mode():
        for name in ['best-ordinary', 'last']:
            results[name] = ordinary(name, inputs)
        results['conditional'] = conditional(inputs)
    for track in TRACKS:
        read_snapshot(track)
    write_json(OUT / 'summary.json', {'status': 'complete', 'results': results, 'blind_read': False, 'test_read': False})
    print('LOCAL_ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
