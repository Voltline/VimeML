"""Verify a downloaded VimeML release. Standard library only; no inference or network."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if not name or '\\' in name or relative.is_absolute() or any(p in {'.', '..'} for p in relative.parts):
        raise ValueError(f'Invalid release path: {name}')
    result = root.joinpath(*relative.parts)
    if not result.resolve().is_relative_to(root.resolve()) or result.is_symlink():
        raise ValueError(f'Unsafe release path: {name}')
    return result


def verify(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'RELEASE.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'vimeml_hf_release_v1' or manifest.get('license') != 'gpl-2.0':
        raise ValueError('Unsupported release format or license.')
    expected = set(manifest['files']) | {'RELEASE.json'}
    sums = {}
    for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or name in sums:
            raise ValueError('Invalid checksum entry.')
        sums[name] = digest
    if set(sums) != expected:
        raise ValueError('Checksum file list differs from RELEASE.json.')
    for name, digest in sums.items():
        path = safe_path(root, name)
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f'Release checksum mismatch: {name}')
        if name != 'RELEASE.json' and (manifest['files'][name]['sha256'] != digest or
                manifest['files'][name]['bytes'] != path.stat().st_size):
            raise ValueError(f'Release inventory mismatch: {name}')
    # Hub local-dir metadata, the Hub's optional root .gitattributes and Python
    # bytecode are not model payload. Only inventoried files are upload candidates.
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and '.cache' not in p.relative_to(root).parts
              and '__pycache__' not in p.relative_to(root).parts and p.suffix != '.pyc'
              and p.relative_to(root).as_posix() != '.gitattributes'}
    if actual != expected | {'SHA256SUMS.txt'}:
        raise ValueError(f'Unexpected release files: {sorted(actual - expected - {"SHA256SUMS.txt"})}')
    return manifest, sorted(expected | {'SHA256SUMS.txt'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    manifest, files = verify(args.release)
    print(f"Verified {len(files)} files: {manifest['repo_id']} / source {manifest['source_commit']}")


if __name__ == '__main__':
    main()
