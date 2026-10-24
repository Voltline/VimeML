"""Local FP32 comparison of the WSD-schedule runs on the four pinned nonblind pools.

Same scorer and semantics as `evaluate_v5.py`. Two runs: V5-50ep (d512/L8, V4.3 mixture)
and V4.2-50ep (d384/L7, V4.2 mixture), both warmup -> 3e-4 stable to 85% -> cosine to 6e-5.
Epoch 50 is the best complete-LM-BPC checkpoint of each run (primary). Epoch 42 is the
last epoch of the stable phase, named before local IME evaluation as a within-run control:
same weights lineage, about 0.13 higher BPC. It is descriptive only, and must be scored
before epoch 50, which pairs against it. The fixed reference and blind sets are not read.
Each run writes a separate empty output directory.
"""
import argparse
import copy
import datetime
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from vimeml.training.data import file_sha, write_json
from vimeml.training.infer_v4 import JapaneseLMV4
from vimeml.training.infer_v5 import JapaneseLMV5
from vimeml.training.model_v4 import PARAMETERS as V4_PARAMETERS
from vimeml.training.model_v5 import PARAMETERS as V5_PARAMETERS
from vimeml.posttraining.infer import TRACKS, read_snapshot
from vimeml.benchmarks.evaluate_ajimee import rerank, summarize, metrics
from vimeml.benchmarks.standard_ime import source_metrics
from vimeml.benchmarks.comparison import paired, rows as read_rows

DOWNLOAD = ROOT / 'outputs/training-wsd-20261024'
RESULTS = ROOT / 'outputs/evaluation-wsd-20261024'
# The V5 run is the repository root; the V4.2 run is stored under v42-50ep/.
RUNS = {
    'v5-50ep': {'prefix': '', 'loader': JapaneseLMV5, 'parameters': V5_PARAMETERS,
                'steps': {42: 4516228, 50: 5376440}},
    'v42-50ep': {'prefix': 'v42-50ep/', 'loader': JapaneseLMV4, 'parameters': V4_PARAMETERS,
                 'steps': {42: 2272072, 50: 2704835}},
}
BASELINES = {
    'v21': ROOT / 'outputs/evaluation-v4-20261015/v21',
    'v42-epoch20': ROOT / 'outputs/evaluation-v42-20261020',
    'v43-epoch30': ROOT / 'outputs/evaluation-v43-20261020/epoch30',
    'v5-epoch30': ROOT / 'outputs/evaluation-v5-20261021/epoch30',
    'v5-10ep': ROOT / 'outputs/evaluation-v5-20261021/epoch10',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', choices=sorted(RUNS), required=True)
    parser.add_argument('--epoch', type=int, choices=[42, 50], required=True)
    args = parser.parse_args()
    run = RUNS[args.run]
    relative = f"{run['prefix']}checkpoints/epoch-{args.epoch}.pt"
    checkpoint = DOWNLOAD / 'remote' / relative
    step = run['steps'][args.epoch]
    out = RESULTS / args.run / f'epoch{args.epoch}'
    if out.exists() and any(out.iterdir()):
        parser.error('Completed results are never overwritten')
    if not torch.cuda.is_available():
        parser.error('The WSD comparison uses an FP32 CUDA GPU')
    baselines = dict(BASELINES)
    if args.epoch == 50:
        stable = RESULTS / args.run / 'epoch42'
        if read(stable / 'summary.json')['status'] != 'complete':
            parser.error('Score epoch 42 of this run first')
        baselines['wsd-epoch42'] = stable
    receipt = read(DOWNLOAD / 'download-receipt.json')
    assert receipt['status'] == 'verified'
    digest = file_sha(checkpoint)
    assert digest == next(r['sha256'] for r in receipt['files'] if r['path'] == relative)
    frozen = {track: read_snapshot(track) for track in TRACKS}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    lm = run['loader'](checkpoint, ROOT / 'artifacts/tokenizers/v4-a/ja-unigram-8192', 'cuda')
    assert lm.metadata['checkpoint_step'] == step
    assert lm.model.parameter_count() == run['parameters']
    assert all(torch.isfinite(p).all().item() for p in lm.model.parameters())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'plan.json', {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'repo': receipt['repo_id'], 'revision': receipt['revision'], 'run': args.run,
        'checkpoint_sha256': digest, 'checkpoint_step': step, 'checkpoint_epoch': args.epoch,
        'selection': 'epoch50 = best complete LM validation BPC (primary); epoch42 = last stable-phase epoch, within-run control named before local IME evaluation, descriptive only',
        'tracks': {k: {'cases': len(v), 'sha256': TRACKS[k][1]} for k, v in frozen.items()},
        'semantics': 'full-vocabulary suffix logP sum; joint context+candidate tokenization; no final EOS, truncation, or tuning; stable ties',
        'precision': 'FP32; TF32 disabled', 'blind_read': False, 'test_read': False,
        'script_sha256': file_sha(Path(__file__)),
    })
    write_json(out / 'environment.json', {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'parameters': lm.model.parameter_count(), 'model': lm.metadata})
    results = {}
    with torch.inference_mode():
        for track, original in frozen.items():
            rows = copy.deepcopy(original)
            began = time.perf_counter()
            rerank(lm, rows)
            dest = out / track
            dest.mkdir()
            (dest / 'scores.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
            result = {'metrics': summarize(rows), 'elapsed_seconds': time.perf_counter() - began,
                      'source_metrics': source_metrics(rows, metrics) if 'source' in rows[0] else None}
            write_json(dest / 'metrics.json', result)
            result['comparisons'] = {}
            for name, folder in baselines.items():
                base = read_rows(folder / track / 'scores.jsonl')
                assert len(base) == len(rows)
                for a, b in zip(base, rows, strict=True):
                    assert all(a[k] == b[k] for k in ['id', 'query', 'left_context', 'answers', 'candidates'])
                    assert a['orders']['azookey'] == b['orders']['azookey']
                comparison = paired(base, rows)
                comparison['paired'].pop('significant_at_0_05', None)
                write_json(dest / f'paired-{name}.json', comparison)
                result['comparisons'][name] = comparison['paired']
            results[track] = result
            print(json.dumps({'track': track, 'top1': result['metrics']['all']['lm_context_sum']['top1_correct'],
                              'comparisons': result['comparisons']}, ensure_ascii=False), flush=True)
    for track in TRACKS:
        read_snapshot(track)
    assert file_sha(checkpoint) == digest
    write_json(out / 'summary.json', {'status': 'complete', 'results': results, 'blind_read': False, 'test_read': False})
    print('WSD_LOCAL_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
