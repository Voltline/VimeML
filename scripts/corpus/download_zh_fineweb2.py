"""Download the FineWeb2 cmn_Hani shards for the Chinese corpus at a pinned revision.

The 370 train files each cover only a few Common Crawl dumps, so the selection is
evenly spaced over the sorted file list (every STRIDE-th file from OFFSET) rather
than a contiguous block. Files are fetched round-robin over the five file groups
(000_ ... 004_), so a partial download (`--max-files N`) still spans the crawl years.

Batches (--batch):
  r1  every STRIDE-th file from OFFSET: 21 files, the export-r1 corpus.
  r2  R2_FILES files spread evenly over the 349 files r1 did not take: file
      (2i+1)*349 // (2*R2_FILES) of that list for i = 0..R2_FILES-1. The order puts the even i
      first, so `--max-files 30` is itself an evenly spread half and the rest follows later.
      Goes to its own directory, so stage A and the export of r2 never see r1 files.

Each file is checked against its LFS SHA-256. Files already present with the right
hash are kept, so the script can be re-run after an interruption. The manifest
(<target>/manifest.json) lists the files present and verified.

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
R2_FILES=60
TARGETS=dict(r1=ROOT/'datasets'/'fineweb-2-cmn-hani',r2=ROOT/'datasets'/'fineweb-2-cmn-hani-r2')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<24),b''):
            h.update(block)
    return h.hexdigest()


def selection(batch):
    from huggingface_hub import HfApi
    info=HfApi().dataset_info(REPO,revision=REVISION,files_metadata=True)
    files=sorted((s for s in info.siblings if s.rfilename.startswith(PREFIX)),key=lambda s:s.rfilename)
    if len(files)!=370:
        raise ValueError(f'expected 370 train files at {REVISION}, found {len(files)}')
    files=[dict(path=s.rfilename,size=s.size,sha256=s.lfs.sha256) for s in files]
    r1=files[OFFSET::STRIDE]
    if batch=='r1':
        return r1
    rest=[f for f in files if f not in r1]
    return [rest[(2*i+1)*len(rest)//(2*R2_FILES)] for i in range(R2_FILES)]


def download_order(chosen,batch):
    """r1: round-robin over the 000_..004_ groups so any prefix of the order spans them.
    r2: even positions of the selection first, then odd, so each half is evenly spread."""
    if batch=='r2':
        return chosen[0::2]+chosen[1::2]
    groups={}
    for c in chosen:
        groups.setdefault(Path(c['path']).name[:3],[]).append(c)
    rounds=itertools.zip_longest(*(groups[k] for k in sorted(groups)))
    return [c for r in rounds for c in r if c is not None]


def verified(target,c):
    return target.exists() and target.stat().st_size==c['size'] and sha256(target)==c['sha256']


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch',choices=sorted(TARGETS),default='r1')
    parser.add_argument('--list',action='store_true',help='print the download order and exit')
    parser.add_argument('--max-files',type=int,default=None,help='stop after the first N files of the order')
    args=parser.parse_args()
    order=download_order(selection(args.batch),args.batch)
    TARGET=TARGETS[args.batch]
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
    rule=dict(offset=OFFSET,stride=STRIDE) if args.batch=='r1' else dict(r1_offset=OFFSET,r1_stride=STRIDE,files=R2_FILES)
    manifest=dict(repo=REPO,revision=REVISION,batch=args.batch,**rule,license='ODC-By 1.0 (FineWeb2)',
                  files=sorted(known.values(),key=lambda f:f['path']))
    manifest_path.write_text(json.dumps(manifest,indent=1),encoding='utf-8')
    print(f'manifest: {len(known)} verified files',flush=True)


if __name__=='__main__':
    main()
