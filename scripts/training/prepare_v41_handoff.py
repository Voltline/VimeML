"""Freeze V4.1 code and dependency identities; rebuild epoch maps on the training host."""
import json
from pathlib import Path
import sys
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from vimeml.training.joint_v41 import build_plan
    from vimeml.training.data import file_sha, write_json
    config_path = ROOT / 'configs/train-v41.toml'
    config = tomllib.loads(config_path.read_text('utf-8'))
    plan = build_plan(config)
    destination = ROOT / 'outputs/handoff-v41'
    destination.mkdir(parents=True, exist_ok=True)
    code = sorted((ROOT / 'src/vimeml').rglob('*.py'))
    code += [ROOT / name for name in ('scripts/training/train_v41.py',
        'scripts/training/prepare_v41_handoff.py', 'scripts/corpus/prepare_v41.py',
        'configs/train-v41.toml', 'requirements.txt', 'requirements-cuda.txt', 'pyproject.toml')]
    archive = destination / 'v41-code.tar.gz'
    def clean(info):
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ''
        info.mode = 0o644
        return info
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in sorted(set(code)):
            bundle.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False, filter=clean)
    maps = {f"{config['data_mixture']['directory']}/{r['file']}" for r in
            json.loads((ROOT / config['data_mixture']['directory'] / 'manifest.json').read_text('utf-8'))['epochs']}
    payload = [{'path': path, 'bytes': (ROOT / path).stat().st_size, 'sha256': digest}
               for path, digest in plan['required_files'].items() if path not in maps]
    result = {'format': 'vimeml_v41_handoff_v1',
        'code_archive': {'name': archive.name, 'bytes': archive.stat().st_size, 'sha256': file_sha(archive)},
        'code_files': {p.relative_to(ROOT).as_posix(): file_sha(p) for p in sorted(set(code))},
        'payload_files': payload, 'rebuild_maps': sorted(maps), 'signatures': plan['signatures'],
        'seed': config['training']['seed'], 'epochs': 16, 'max_steps': plan['settings']['max_steps'],
        'initialization': 'random', 'training_started': False}
    write_json(destination / 'manifest.json', result)
    write_json(destination / 'training-plan.json', plan)
    print(json.dumps({'seed': result['seed'], 'epochs': 16, 'max_steps': result['max_steps'],
        'payload_files': len(payload), 'maps_rebuilt_remotely': len(maps),
        'code_archive_bytes': archive.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
