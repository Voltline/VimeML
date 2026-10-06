"""Archive Mac project snapshots, experiment evidence and Vime client for local handoff."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from datetime import date


SKIP_PARTS = {'.git', '__pycache__', '.pytest_cache', '.build', '.swiftpm', 'xcuserdata'}
SKIP_NAMES = {'.DS_Store'}


def digest(source):
    if isinstance(source, bytes):
        return hashlib.sha256(source).hexdigest()
    with source.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, client, out = args.project.resolve(), args.client.resolve(), args.output.resolve()
    if out.exists():
        raise SystemExit('Output already exists; use a new handoff version.')
    out.mkdir(parents=True)
    manifest = {'format': 'vimeml_mac_handoff_v1', 'date': date.today().isoformat(),
                'vime_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=client, text=True).strip(),
                'vime_git_status': subprocess.check_output(['git', 'status', '--short'], cwd=client, text=True),
                'archives': {}, 'excluded': [],
                'policy': 'Local transfer only; snapshots are not automatic merge instructions or Git commits.'}

    def entries(base, roots, prefix):
        files = []
        for name in roots:
            source = base / name
            if not source.exists():
                continue
            paths = [source] if source.is_file() else sorted(source.rglob('*'))
            for path in paths:
                relative = path.relative_to(base)
                if not path.is_file():
                    continue
                if (SKIP_PARTS.intersection(relative.parts) or path.name in SKIP_NAMES or path.suffix == '.pyc'
                        or path.name == '.env' or path.name.startswith('.env.')
                        or path.name in {'.pypirc', '.envrc'}):
                    manifest['excluded'].append(str(Path(prefix) / relative))
                    continue
                if path.is_symlink():
                    raise SystemExit(f'Unexpected symlink requires review: {path}')
                files.append(((Path(prefix) / relative).as_posix(), path))
        return files

    def archive(name, files):
        indexed = []
        with zipfile.ZipFile(out / name, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zip_file:
            for relative, data in files:
                value = digest(data)
                if isinstance(data, bytes):
                    zip_file.writestr(relative, data)
                    size = len(data)
                else:
                    zip_file.write(data, relative)
                    size = data.stat().st_size
                indexed.append({'path': relative, 'bytes': size, 'sha256': value})
        # Verify every entry against its source hash, including models and raw trace files.
        with zipfile.ZipFile(out / name) as zip_file:
            for entry in indexed:
                with zip_file.open(entry['path']) as stream:
                    value = hashlib.file_digest(stream, 'sha256').hexdigest()
                if value != entry['sha256']:
                    raise SystemExit(f'Archive verification failed: {entry["path"]}')
        manifest['archives'][name] = {'file_count': len(indexed),
                                     'uncompressed_bytes': sum(e['bytes'] for e in indexed),
                                     'archive_bytes': (out / name).stat().st_size,
                                     'entries': indexed}
        print(f'{name}: {len(indexed)} files, {(out / name).stat().st_size / 1048576:.1f} MiB; SHA256 verified', flush=True)

    source_roots = ['annotations', 'configs', 'docs', 'examples', 'scripts', 'src', 'tests']
    source_roots += [p.name for p in root.iterdir() if p.is_file() and p.name not in SKIP_NAMES]
    archive('VimeML-source.zip', entries(root, source_roots, 'VimeML'))
    results_roots = ['outputs', 'runs'] + [str(p.relative_to(root)) for p in (root / 'artifacts').iterdir()
                                          if p.name not in {'models', 'token-data'}]
    archive('VimeML-results.zip', entries(root, results_roots, 'VimeML'))
    archive('VimeML-training-inputs.zip', entries(root, ['artifacts/models', 'artifacts/token-data'], 'VimeML'))

    client_paths = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=client).decode().split('\0')
    client_entries = entries(client, sorted(set(p for p in client_paths if p)), 'Vime')
    patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD', '--', '.', ':(exclude)**/xcuserdata/**'], cwd=client)
    client_entries += [('CLIENT_TRACKED_CHANGES.patch', patch),
                       ('CLIENT_GIT_STATUS.txt', manifest['vime_git_status'].encode()),
                       ('CLIENT_GIT_HEAD.txt', (manifest['vime_git_head'] + '\n').encode()),
                       ('Vime/Docs/MacStageHandoff.md', (root / 'docs/reports/mac-20261006/handoff.md').read_bytes()),
                       ('Vime/Docs/iPhoneKeyboardMemory.md', (root / 'docs/reports/mac-20261006/iphone-keyboard-memory.md').read_bytes())]
    archive('Vime-client.zip', client_entries)
    manifest['not_archived_roots'] = ['VimeML/venv (rebuild from requirements and environment snapshot)',
                                      'VimeML/handoff (this generated output)',
                                      'Vime/.git (base commit and patch preserved instead)']
    (out / 'README.md').write_bytes((root / 'docs/reports/mac-20261006/handoff.md').read_bytes())
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    sums = []
    for path in sorted(out.iterdir()):
        if path.is_file():
            with path.open('rb') as stream:
                value = hashlib.file_digest(stream, 'sha256').hexdigest()
            sums.append(f'{value}  {path.name}')
    (out / 'SHA256SUMS.txt').write_text('\n'.join(sums) + '\n')
    print(f'Handoff ready: {out}', flush=True)


if __name__ == '__main__':
    main()
