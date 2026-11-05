"""Fetch the four frozen V4.5 teachers from the private HF training-output repos.

Downloads each registered epoch-20 checkpoint (`joint_v45.TEACHERS`) at its fixed
revision into artifacts/teachers/v45/<name>.pt and verifies its SHA-256; an existing
file with the right hash is kept. Needs read access to the private datasets. Run it once
before launching; `--verify-only` (used by cluster_v45.sh) checks without downloading.
"""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vimeml.training.data import file_sha
from vimeml.training.joint_v45 import TEACHERS,teacher_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    for name,spec in sorted(TEACHERS.items()):
        target=teacher_file(name)
        if target.exists() and file_sha(target)==spec['sha256']:
            print(f'{name}: verified {target.relative_to(ROOT)}',flush=True)
            continue
        if args.verify_only:
            raise SystemExit(f'{name}: {target.relative_to(ROOT)} missing or changed; run prepare_v45_teachers.py first')
        from huggingface_hub import hf_hub_download
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
            path=Path(hf_hub_download(spec['repo_id'],spec['path'],repo_type='dataset',
                                      revision=spec['revision'],local_dir=scratch))
            if file_sha(path)!=spec['sha256']:
                raise ValueError(f'{name}: downloaded file differs from the registered SHA-256')
            shutil.move(str(path),target)
        print(f'{name}: downloaded and verified {target.relative_to(ROOT)}',flush=True)


if __name__=='__main__':
    main()
