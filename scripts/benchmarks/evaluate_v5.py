"""Local FP32 comparison of V5 epoch checkpoints on the four pinned nonblind pools.

Same scorer and semantics as `evaluate_v43.py`. Epoch 30 is the best complete-LM-BPC
checkpoint (= best-lm.pt); epoch 29 is descriptive only, named by the cluster agent from
dev IME scores, so it is not a selection candidate. Epoch 10 is the controlled-10ep
best-lm, paired with V4.3 controlled-10ep as a size-only comparison. The fixed reference and
blind sets are not read. Each run writes a separate empty output directory.
"""
import argparse
import copy
import datetime
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from vimeml.training.data import file_sha, write_json
from vimeml.training.infer_v5 import JapaneseLMV5
from vimeml.training.model_v5 import PARAMETERS
from vimeml.posttraining.infer import TRACKS, read_snapshot
from vimeml.benchmarks.evaluate_ajimee import rerank, summarize, metrics
from vimeml.benchmarks.standard_ime import source_metrics
from vimeml.benchmarks.comparison import paired, rows as read_rows

OUT = None
DOWNLOAD = ROOT / 'outputs/training-v5-20261021'
STEPS = {10: 1075287, 29: 3118362, 30: 3225888}
# The controlled 10-epoch run stores its checkpoints under controlled-10ep/.
RUNS = {10: 'controlled-10ep/'}
BASELINES = {
    'v21': ROOT / 'outputs/evaluation-v4-20261015/v21',
    'v42-epoch20': ROOT / 'outputs/evaluation-v42-20261020',
    'v43-epoch30': ROOT / 'outputs/evaluation-v43-20261020/epoch30',
    'v43-epoch27': ROOT / 'outputs/evaluation-v43-20261020/epoch27',
    'v43-10ep': ROOT / 'outputs/evaluation-v43-20261020/epoch10',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Separate empty output directory; completed results are never overwritten')
    parser.add_argument('--epoch', type=int, choices=sorted(STEPS), required=True)
    args = parser.parse_args()
    relative = f"{RUNS.get(args.epoch, '')}checkpoints/epoch-{args.epoch}.pt"
    checkpoint = DOWNLOAD / 'remote' / relative
    step = STEPS[args.epoch]
    OUT = args.output.resolve()
    if OUT.exists() and any(OUT.iterdir()):
        parser.error('Use an empty output directory to preserve prior evaluations')
    if not torch.cuda.is_available():
        parser.error('The V5 comparison uses an FP32 CUDA GPU')
    receipt = read(DOWNLOAD / 'download-receipt.json')
    assert receipt['status'] == 'verified'
    digest = file_sha(checkpoint)
    assert digest == next(r['sha256'] for r in receipt['files'] if r['path'] == relative)
    frozen = {track: read_snapshot(track) for track in TRACKS}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    lm = JapaneseLMV5(checkpoint, ROOT / 'artifacts/tokenizers/v4-a/ja-unigram-8192', 'cuda')
    assert lm.metadata['checkpoint_step'] == step
    assert lm.model.parameter_count() == PARAMETERS
    assert all(torch.isfinite(p).all().item() for p in lm.model.parameters())
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'plan.json', {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'repo': receipt['repo_id'], 'revision': receipt['revision'],
        'checkpoint_sha256': digest, 'checkpoint_step': step, 'checkpoint_epoch': args.epoch,
        'selection': 'epoch30 = best complete LM validation BPC (primary); epoch29 = cluster-agent pick by dev IME, descriptive only; epoch10 = controlled-10ep best-lm, size-only comparison with V4.3 controlled-10ep',
        'tracks': {k: {'cases': len(v), 'sha256': TRACKS[k][1]} for k, v in frozen.items()},
        'semantics': 'full-vocabulary suffix logP sum; joint context+candidate tokenization; no final EOS, truncation, or tuning; stable ties',
        'precision': 'FP32; TF32 disabled', 'blind_read': False, 'test_read': False,
        'script_sha256': file_sha(Path(__file__)),
    })
    write_json(OUT / 'environment.json', {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'parameters': lm.model.parameter_count(), 'model': lm.metadata})
    results = {}
    with torch.inference_mode():
        for track, original in frozen.items():
            rows = copy.deepcopy(original)
            began = time.perf_counter()
            rerank(lm, rows)
            dest = OUT / track
            dest.mkdir()
            (dest / 'scores.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
            result = {'metrics': summarize(rows), 'elapsed_seconds': time.perf_counter() - began,
                      'source_metrics': source_metrics(rows, metrics) if 'source' in rows[0] else None}
            write_json(dest / 'metrics.json', result)
            result['comparisons'] = {}
            for name, folder in BASELINES.items():
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
    write_json(OUT / 'summary.json', {'status': 'complete', 'results': results, 'blind_read': False, 'test_read': False})
    print('V5_LOCAL_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
