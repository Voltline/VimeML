"""Download the FineWeb2 cmn_Hani shards for the Chinese corpus at a pinned revision.

The 370 train files each cover only a few Common Crawl dumps, so the selection is
evenly spaced over the sorted file list (every STRIDE-th file from OFFSET) rather
than a contiguous block. Files are fetched round-robin over the five file groups
(000_ ... 004_), so a partial download (`--max-files N`) still spans the crawl years.

Each file is checked against its LFS SHA-256. Files already present with the right
hash are kept, so the script can be re-run after an interruption. The manifest
(datasets/fineweb-2-cmn-hani/manifest.json) lists the files present and verified.

Proxy: huggingface_hub honours HTTPS_PROXY / HTTP_PROXY; set them before running.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[2]
REPO='HuggingFaceFW/fineweb-2'
REVISION='af9c13333eb981300149d5ca60a8e9d659b276b9'
PREFIX='data/cmn_Hani/train/'
OFFSET,STRIDE=9,18
TARGET=ROOT/'datasets'/'fineweb-2-cmn-hani'


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<24),b''):
            h.update(block)
    return h.hexdigest()


def selection():
    from huggingface_hub import HfApi
    info=HfApi().dataset_info(REPO,revision=REVISION,files_metadata=True)
    files=sorted((s for s in info.siblings if s.rfilename.startswith(PREFIX)),key=lambda s:s.rfilename)
    if len(files)!=370:
        raise ValueError(f'expected 370 train files at {REVISION}, found {len(files)}')
    return [dict(path=s.rfilename,size=s.size,sha256=s.lfs.sha256) for s in files[OFFSET::STRIDE]]


def download_order(chosen):
    """Round-robin over the 000_..004_ groups so any prefix of the order spans them."""
    groups={}
    for c in chosen:
        groups.setdefault(Path(c['path']).name[:3],[]).append(c)
    rounds=itertools.zip_longest(*(groups[k] for k in sorted(groups)))
    return [c for r in rounds for c in r if c is not None]


def verified(target,c):
    return target.exists() and target.stat().st_size==c['size'] and sha256(target)==c['sha256']


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list',action='store_true',help='print the download order and exit')
    parser.add_argument('--max-files',type=int,default=None,help='stop after the first N files of the order')
    args=parser.parse_args()
    order=download_order(selection())
    if args.max_files is not None:
        order=order[:args.max_files]
    print(f'{len(order)} files, {sum(c["size"] for c in order)/2**30:.1f} GiB',flush=True)
    if args.list:
        for i,c in enumerate(order,1): print(f'{i:2d}',c['path'],f'{c["size"]/2**30:.2f} GiB')
        return
    from huggingface_hub import hf_hub_download
    TARGET.mkdir(parents=True,exist_ok=True)
    present=[]
    for i,c in enumerate(order,1):
        target=TARGET/Path(c['path']).name
        if verified(target,c):
            print(f'[{i}/{len(order)}] kept {target.name}',flush=True)
        else:
            with tempfile.TemporaryDirectory(dir=TARGET) as scratch:
                path=Path(hf_hub_download(REPO,c['path'],repo_type='dataset',revision=REVISION,local_dir=scratch))
                if sha256(path)!=c['sha256']:
                    raise ValueError(f'{c["path"]}: SHA-256 mismatch')
                shutil.move(str(path),target)
            print(f'[{i}/{len(order)}] downloaded {target.name}',flush=True)
        present.append(dict(c,local=target.name))
    manifest_path=TARGET/'manifest.json'
    known={}
    if manifest_path.exists():
        known={f['path']:f for f in json.loads(manifest_path.read_text(encoding='utf-8'))['files']}
    known.update({f['path']:f for f in present})
    manifest=dict(repo=REPO,revision=REVISION,offset=OFFSET,stride=STRIDE,license='ODC-By 1.0 (FineWeb2)',
                  files=sorted(known.values(),key=lambda f:f['path']))
    manifest_path.write_text(json.dumps(manifest,indent=1),encoding='utf-8')
    print(f'manifest: {len(known)} verified files',flush=True)


if __name__=='__main__':
    main()
