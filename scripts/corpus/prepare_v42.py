"""Prepare the V4.2 LM+chat mixture: 20 epochs, conversion pool removed.

Derives the new request from the frozen V4.1 mixture: identical joined banks,
exclusion masks, seed, crop and chat fraction; only the epoch budget changes
from 16 to 20. Epoch maps 0-15 must reproduce the V4.1 maps byte-identically.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

OLD=ROOT/'artifacts/training-data/corpus-v41-mixed-c128'
NEW=ROOT/'artifacts/training-data/corpus-v42-mixed-c128'
EPOCHS=20


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true',
        help='Verify an existing V4.2 mixture without writing anything')
    args=parser.parse_args()
    request=json.loads((OLD/'request.json').read_text('utf-8'))
    old_manifest=json.loads((OLD/'manifest.json').read_text('utf-8'))
    if request['epochs']!=16 or len(old_manifest['epochs'])!=16:
        raise ValueError('Unexpected V4.1 mixture shape')
    if (NEW/'manifest.json').exists():
        from vimeml.training.v41_data import prepare_maps
        from vimeml.training.data import file_sha
        result=prepare_maps(NEW,root=ROOT)
        if len(result['epochs'])!=EPOCHS:
            raise ValueError('Existing V4.2 mixture is not a 20-epoch plan')
        for epoch in range(16):
            if result['epochs'][epoch]!=old_manifest['epochs'][epoch]:
                raise ValueError(f'V4.2 map epoch {epoch} differs from frozen V4.1')
        print(json.dumps({'status':'verified','epochs':EPOCHS,
            'dataset_sha256':file_sha(NEW/'manifest.json')}))
        return
    if args.verify_only:
        raise ValueError('No V4.2 mixture to verify')
    NEW.mkdir(parents=True,exist_ok=True)
    keep=shutil.copyfile(OLD/'old.keep.npy',NEW/'old.keep.npy')
    derived=dict(request)
    for key in ('conversion_dir','conversion_manifest_sha256','conversion_keep',
                'conversion_keep_sha256','conversion_examples'):
        derived.pop(key)
    derived['epochs']=EPOCHS
    derived['old_keep']='artifacts/training-data/corpus-v42-mixed-c128/old.keep.npy'
    derived['derived_from']='artifacts/training-data/corpus-v41-mixed-c128'
    derived['derivation']=('V4.1 request with epochs 16->20 and the conversion pool removed; '
        'chat quota and exclusion masks unchanged')
    derived['numpy_version']=np.__version__
    (NEW/'request.json').write_text(json.dumps(derived,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    from vimeml.training.v41_data import prepare_maps
    from vimeml.training.data import file_sha
    result=prepare_maps(NEW,root=ROOT)
    if len(result['epochs'])!=EPOCHS:
        raise ValueError('prepare_maps did not produce the 20-epoch plan')
    for epoch in range(16):
        if result['epochs'][epoch]!=old_manifest['epochs'][epoch]:
            raise ValueError(f'Rebuilt V4.2 map epoch {epoch} differs from frozen V4.1')
    print(json.dumps({'status':'prepared','epochs':EPOCHS,
        'verified_identical_to_v41':'epochs 0-15','dataset_sha256':file_sha(NEW/'manifest.json')}))


if __name__=='__main__':
    main()
