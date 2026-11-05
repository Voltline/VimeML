"""Local FP32 scoring of the V4.4 ablation arms on the four pinned nonblind pools.

Same scorer and semantics as `evaluate_wsd.py`. Each arm is scored only at epoch 20, the
point fixed by the preregistration (docs/zh-CN/training-v44-plan.md), and paired per item
against the deployed V4.2 epoch 20. Cluster statistics and the noise band are computed
afterwards from the score files by `report_v44.py`. The fixed reference and blind sets are
not read. Each arm writes a separate empty output directory.
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
from vimeml.training.model_v4 import PARAMETERS as V4_PARAMETERS
from vimeml.posttraining.infer import TRACKS, read_snapshot
from vimeml.benchmarks.evaluate_ajimee import rerank, summarize, metrics
from vimeml.benchmarks.standard_ime import source_metrics
from vimeml.benchmarks.comparison import paired, rows as read_rows

DOWNLOAD = ROOT / 'outputs/training-v44-20261105'
RESULTS = ROOT / 'outputs/evaluation-v44-20261105'
# Final committed step of each arm (records/summary.json), checked against the checkpoint.
STEPS = {'s1': 1081944, 's2': 1081940, 's3': 1081937, 'chat0': 984700, 'chat10': 1189984, 'crop60': 1077365}
EPOCH = 20
BASELINES = {'v42-epoch20': ROOT / 'outputs/evaluation-v42-20261020'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=sorted(STEPS), required=True)
    args = parser.parse_args()
    relative = f'{args.arm}/checkpoints/epoch-{EPOCH}.pt'
    checkpoint = DOWNLOAD / 'remote' / relative
    step = STEPS[args.arm]
    out = RESULTS / args.arm
    if out.exists() and any(out.iterdir()):
        parser.error('Completed results are never overwritten')
    if not torch.cuda.is_available():
        parser.error('The V4.4 comparison uses an FP32 CUDA GPU')
    baselines = dict(BASELINES)
    receipt = read(DOWNLOAD / 'download-receipt.json')
    assert receipt['status'] == 'verified'
    digest = file_sha(checkpoint)
    assert digest == next(r['sha256'] for r in receipt['files'] if r['path'] == relative)
    frozen = {track: read_snapshot(track) for track in TRACKS}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    lm = JapaneseLMV4(checkpoint, ROOT / 'artifacts/tokenizers/v4-a/ja-unigram-8192', 'cuda')
    assert lm.metadata['checkpoint_step'] == step
    assert lm.model.parameter_count() == V4_PARAMETERS
    assert all(torch.isfinite(p).all().item() for p in lm.model.parameters())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'plan.json', {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'repo': receipt['repo_id'], 'revision': receipt['revision'], 'arm': args.arm,
        'checkpoint_sha256': digest, 'checkpoint_step': step, 'checkpoint_epoch': EPOCH,
        'selection': 'epoch 20 of every arm, fixed by the preregistration before training',
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
    print('V44_LOCAL_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
