"""Pack one finished V4.4 arm for the private HF training-output repo.

Writes outputs/v44-upload/<arm>/: the epoch-20 checkpoint without optimizer state
(the preregistered evaluation point), run records, the arm's mixture request and
manifest, the exact sources, W&B run metadata and offline runs, and SHA256SUMS.
Intermediate checkpoints stay on the cluster and are not uploaded.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
from vimeml.training.data import file_sha,write_json
from vimeml.training.joint_v44 import ARMS,MIXTURE_EPOCHS

RECORDS=('config.json','plan.json','environment.json','summary.json','progress.json',
         'metrics.jsonl','full-validation.json','subset-validation.json')
SOURCES=('scripts/corpus/prepare_v44.py','scripts/training/cluster_v44.sh','scripts/training/train_v44.py',
         'scripts/training/export_v44.py','src/vimeml/training/joint_v44.py','src/vimeml/training/v41_data.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=sorted(set(ARMS)-{'control'}),required=True)
    args=parser.parse_args()
    config_path=ROOT/f'configs/train-v44-{args.arm}.toml'
    config=tomllib.loads(config_path.read_text('utf-8'))
    output=ROOT/config['output_dir']
    summary=json.loads((output/'summary.json').read_text('utf-8'))
    if summary['status']!='complete':
        raise ValueError(f'{args.arm} has not completed training')
    bundle=ROOT/'outputs/v44-upload'/args.arm
    if bundle.exists():
        raise ValueError(f'{bundle} exists; remove it to re-export')
    saved=torch.load(output/f'epoch-{MIXTURE_EPOCHS}.pt',map_location='cpu',weights_only=True)
    if saved['epoch']!=MIXTURE_EPOCHS-1 or saved['step']!=summary['max_steps'] or saved['config']!=config:
        raise ValueError('Epoch-20 checkpoint does not match the completed run')
    history=saved['training_state']['epoch_results']
    if [r['epoch'] for r in history]!=list(range(1,MIXTURE_EPOCHS+1)):
        raise ValueError('Incomplete epoch validation history')
    del saved['optimizer']
    (bundle/'checkpoints').mkdir(parents=True)
    torch.save(saved,bundle/f'checkpoints/epoch-{MIXTURE_EPOCHS}.pt')
    records=bundle/'records'
    records.mkdir()
    for name in RECORDS:
        if (output/name).exists():
            shutil.copyfile(output/name,records/name)
    write_json(records/'epoch-validation-history.json',history)
    log=ROOT/f'logs/v44/{args.arm}.log'
    if log.exists():
        shutil.copyfile(log,records/'train.log')
    mixture=ROOT/config['data_mixture']['directory']
    (bundle/'mixture').mkdir()
    for name in ('request.json','manifest.json'):
        shutil.copyfile(mixture/name,bundle/'mixture'/name)
    if file_sha(bundle/'mixture/manifest.json')!=config['dataset_sha256']:
        raise ValueError('Mixture manifest differs from the pinned SHA')
    (bundle/'source').mkdir()
    for relative in (config_path.relative_to(ROOT).as_posix(),*SOURCES):
        shutil.copyfile(ROOT/relative,bundle/'source'/relative.replace('/','__'))
    tracking=ROOT/'artifacts/tracking'/output.name
    if tracking.exists():
        shutil.copytree(tracking,bundle/'tracking',ignore=shutil.ignore_patterns('cache','data','config','netrc'))
    files=sorted(p for p in bundle.rglob('*') if p.is_file())
    (bundle/'SHA256SUMS').write_bytes(''.join(f'{file_sha(p)}  {p.relative_to(bundle).as_posix()}\n'
                                              for p in files).encode())
    last=history[-1]
    print(json.dumps({'arm':args.arm,'bundle':str(bundle),'files':len(files)+1,
        'epoch20_step':summary['max_steps'],'lm_bpc':last['lm']['bpc'],'chat_bpc':last['chat']['bpc'],
        'checkpoint_sha256':file_sha(bundle/f'checkpoints/epoch-{MIXTURE_EPOCHS}.pt')}),flush=True)


if __name__=='__main__':
    main()
