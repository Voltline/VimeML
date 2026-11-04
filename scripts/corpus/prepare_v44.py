"""Prepare one V4.4 ablation arm's frozen 20-epoch LM+chat mixture.

Derives the arm request from the frozen V4.1 mixture exactly as V4.2 did (same
joined banks, exclusion masks and crop minimum; conversion pool removed; 20
epochs), then overrides only the arm's seed, chat fraction or crop probability.
Request and manifest are written with LF newlines so the manifest SHA is the
same on Windows and Linux; a configured arm must reproduce its pinned SHA.

The 'control' arm rebuilds V4.2 itself: epochs 0-15 must equal frozen V4.1, and
all 20 must equal the V4.2 mixture when that directory is present.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vimeml.training.data import file_sha
from vimeml.training.joint_v44 import ARMS,MIXTURE_EPOCHS
from vimeml.training.v41_data import prepare_maps

OLD=ROOT/'artifacts/training-data/corpus-v41-mixed-c128'
V42=ROOT/'artifacts/training-data/corpus-v42-mixed-c128'


def directory(arm):
    return ROOT/f'artifacts/training-data/corpus-v44-{arm}-c128'


def request_for(arm):
    request=json.loads((OLD/'request.json').read_text('utf-8'))
    if request['epochs']!=16:
        raise ValueError('Unexpected V4.1 mixture shape')
    seed,chat,crop=ARMS[arm]
    for key in ('conversion_dir','conversion_manifest_sha256','conversion_keep',
                'conversion_keep_sha256','conversion_examples','numpy_version'):
        request.pop(key)
    request.update(arm=arm,epochs=MIXTURE_EPOCHS,seed=seed,chat_fraction=chat,crop_probability=crop,
        old_keep=directory(arm).relative_to(ROOT).as_posix()+'/old.keep.npy',
        derived_from='artifacts/training-data/corpus-v41-mixed-c128',
        derivation=('V4.2 request (V4.1 with epochs 16->20 and the conversion pool removed) '
                    f'with seed={seed}, chat_fraction={chat}, crop_probability={crop}'))
    return request


def write_lf(path,text):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(text.encode('utf-8'))
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=sorted(ARMS),required=True)
    args=parser.parse_args()
    target=directory(args.arm)
    request=request_for(args.arm)
    if (target/'request.json').exists():
        if json.loads((target/'request.json').read_text('utf-8'))!=request:
            raise ValueError(f'{target} holds a different request')
    else:
        target.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(OLD/'old.keep.npy',target/'old.keep.npy')
        write_lf(target/'request.json',json.dumps(request,ensure_ascii=False,indent=1)+'\n')
    result=prepare_maps(target,root=ROOT)
    manifest=target/'manifest.json'
    raw=manifest.read_bytes()
    if b'\r\n' in raw:
        write_lf(manifest,raw.decode('utf-8').replace('\r\n','\n'))
    if len(result['epochs'])!=MIXTURE_EPOCHS:
        raise ValueError('prepare_maps did not produce the 20-epoch plan')
    report={'arm':args.arm,'directory':target.relative_to(ROOT).as_posix(),'epochs':MIXTURE_EPOCHS,
        'dataset_sha256':file_sha(manifest),
        'chat_fraction':[round(r['chat_fraction'],6) for r in result['epochs'][:2]],
        'chat_effective_passes':round(result['epochs'][0]['chat_effective_passes'],3),
        'windows_epoch1':result['epochs'][0]['windows']}
    if args.arm=='control':
        old=json.loads((OLD/'manifest.json').read_text('utf-8'))['epochs']
        if result['epochs'][:16]!=old:
            raise ValueError('Control maps differ from frozen V4.1 epochs 0-15')
        report['identical_to_v41']='epochs 0-15'
        if (V42/'manifest.json').exists():
            if result['epochs']!=json.loads((V42/'manifest.json').read_text('utf-8'))['epochs']:
                raise ValueError('Control maps differ from frozen V4.2')
            report['identical_to_v42']='epochs 0-19'
    else:
        config=tomllib.loads((ROOT/f'configs/train-v44-{args.arm}.toml').read_text('utf-8'))
        if config['data_mixture']['directory']!=report['directory']:
            raise ValueError('Config points at another mixture directory')
        if config['dataset_sha256'] and config['dataset_sha256']!=report['dataset_sha256']:
            raise ValueError(f"{args.arm} mixture SHA {report['dataset_sha256']} differs from the pinned "
                             f"{config['dataset_sha256']}; do not train, report this mismatch")
        report['matches_pinned_sha']=bool(config['dataset_sha256'])
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
