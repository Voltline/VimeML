"""Download the FineWeb-2 jpn_Jpan shards for the ja-gen 200M corpus (plan stage 1.1) through a mirror.

Same pinned revision as the Chinese corpus. The 175 train files are sorted and every STRIDE-th file
from OFFSET is taken (22 files, 82.1 GiB, against 84.6 GiB for the Chinese r1 batch), so the
selection spans all five file groups (000_ ... 004_) and the crawl years.

Downloads go through a Hugging Face mirror (HF_ENDPOINT, default https://hf-mirror.com); the script
refuses to run against huggingface.co. Each file is checked against its LFS SHA-256. Files already
present with the right hash are kept, and partial downloads in <target>/.download resume, so the
script can be re-run after an interruption. The manifest (<target>/manifest.json) lists the files
present and verified.

    python scripts/ja/corpus/download_ja_fineweb2.py [--list] [--max-files N]
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# No proxy: on Windows, urllib (and so httpx/requests) falls back to the system proxy in the registry
# when no proxy variable is set. A NO_PROXY variable makes the environment win, and '*' bypasses all.
for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
    os.environ.pop(name, None)
os.environ['NO_PROXY'] = os.environ['no_proxy'] = '*'
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')

ROOT = Path(__file__).resolve().parents[3]
REPO = 'HuggingFaceFW/fineweb-2'
REVISION = 'af9c13333eb981300149d5ca60a8e9d659b276b9'
PREFIX = 'data/jpn_Jpan/train/'
FILES = 175
OFFSET, STRIDE = 4, 8
TARGET = ROOT / 'datasets' / 'fineweb-2-jpn-jpan'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def selection():
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    files = sorted((s for s in info.siblings if s.rfilename.startswith(PREFIX)), key=lambda s: s.rfilename)
    if len(files) != FILES:
        raise ValueError(f'expected {FILES} train files at {REVISION}, found {len(files)}')
    return [dict(path=s.rfilename, size=s.size, sha256=s.lfs.sha256) for s in files[OFFSET::STRIDE]]


def verified(target, c):
    return target.exists() and target.stat().st_size == c['size'] and sha256(target) == c['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list', action='store_true', help='print the selection and exit')
    parser.add_argument('--max-files', type=int, default=None, help='stop after the first N files')
    args = parser.parse_args()
    endpoint = os.environ['HF_ENDPOINT'].rstrip('/')
    if 'huggingface.co' in endpoint:
        parser.error(f'HF_ENDPOINT is {endpoint}; use a mirror, not a direct connection.')
    import urllib.request
    proxies = {k: v for k, v in urllib.request.getproxies().items() if k != 'no'}
    if proxies:
        parser.error(f'a proxy is still active: {proxies}')
    print(f'endpoint {endpoint}, no proxy', flush=True)
    order = selection()
    if args.max_files is not None:
        order = order[:args.max_files]
    print(f'{len(order)} files, {sum(c["size"] for c in order) / 2**30:.1f} GiB', flush=True)
    if args.list:
        for i, c in enumerate(order, 1):
            print(f'{i:2d}', c['path'], f'{c["size"] / 2**30:.2f} GiB')
        return
    from huggingface_hub import hf_hub_download
    TARGET.mkdir(parents=True, exist_ok=True)
    staging = TARGET / '.download'
    present = []
    for i, c in enumerate(order, 1):
        target = TARGET / Path(c['path']).name
        if verified(target, c):
            print(f'[{i}/{len(order)}] kept {target.name}', flush=True)
        else:
            path = Path(hf_hub_download(REPO, c['path'], repo_type='dataset', revision=REVISION,
                                        local_dir=staging))
            if path.stat().st_size != c['size'] or sha256(path) != c['sha256']:
                path.unlink()
                raise ValueError(f'{c["path"]}: size or SHA-256 mismatch; removed, re-run to fetch again')
            shutil.move(str(path), target)
            print(f'[{i}/{len(order)}] downloaded {target.name}', flush=True)
        present.append(dict(c, local=target.name))
    manifest = dict(repo=REPO, revision=REVISION, offset=OFFSET, stride=STRIDE, endpoint='mirror',
                    license='ODC-By 1.0 (FineWeb2)', files=present)
    (TARGET / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    if len(present) == len(selection()):
        shutil.rmtree(staging, ignore_errors=True)
    print(f'manifest: {len(present)} verified files', flush=True)


if __name__ == '__main__':
    main()
