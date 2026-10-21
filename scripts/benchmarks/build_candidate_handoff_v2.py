"""Second Mac candidate handoff: new frozen tracks at n-best 20 and 50, plus pool checks.

Tracks and why they are exported:

- standard-v2-blind, homophone-v1: new frozen releases (n20 for comparability with the
  registered protocol, n50 for the next benchmark version);
- standard-v1-development: n50 measures how many pool misses a larger n-best recovers;
  n20 repeats the original export to check determinism;
- standard-v1-development-nocontext (n20): the same inputs with empty left context,
  to test whether AzooKey pools depend on context with Zenzai off (free-running
  incremental evaluation is valid only if they do not);
- incremental-proto1: segmented development prototype.

No language model runs during export. Commands: package, import.
"""
import argparse
import copy
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/benchmarks'))
from vimeml.benchmarks.expanded_ime import CONVERTER
from vimeml.benchmarks.standard_ime import FORMAT, load_standard_export
from vimeml.training.data import write_json
from prepare_standard_ime_v2 import read, fresh
from standard_ime import track_files

BENCH = ROOT / 'artifacts/benchmarks'
VERSION = 'ime-candidates-v2'
TRACKS = {
    'standard-v2-blind': (BENCH / 'ime-standard-ja-v2-blind-ai-expert', 'blind', (20, 50)),
    'homophone-v1': (BENCH / 'ime-homophone-ja-v1-ai-expert', 'challenge', (20, 50)),
    'standard-v1-development': (BENCH / 'ime-standard-ja-v1-ai-expert-r1', 'development', (20, 50)),
    'standard-v1-development-nocontext': (BENCH / 'ime-standard-ja-v1-ai-expert-r1', 'development', (20,)),
    'incremental-proto1': (BENCH / 'ime-incremental-ja-proto1', 'segmented', (20, 50)),
}
FILES = ('ajimee-input.json', 'case-map.json', 'azookey-candidates.json', 'converter-version.txt',
         'dictionary-versions.txt', 'swift-version.txt', 'export-flags.txt', 'completed.txt', 'export.log')


def items(track):
    release, split, _ = TRACKS[track]
    rows = read(release / split / 'evaluation_items.json')
    if track.endswith('-nocontext'):
        rows = copy.deepcopy(rows)
        for row in rows:
            row['context_text'] = ''
    return rows


def package(args):
    out = args.output
    fresh(out)
    tracks = {}
    for track, (release, split, n_best) in TRACKS.items():
        track_files(out / track, items(track))
        (out / track / 'evaluation_items.json').unlink()
        (out / track / 'stats.json').unlink()
        manifest = read(release / 'manifest.json')
        tracks[track] = {'release': release.name, 'release_id': manifest.get('release_id'), 'split': split,
                         'n_best': list(n_best), 'cases': len(read(out / track / 'case-map.json')),
                         'input_sha256': hashlib.sha256((out / track / 'ajimee-input.json').read_bytes()).hexdigest()}
    (out / 'tracks.tsv').write_text(''.join(f'{t}\t{n}\n' for t, v in tracks.items() for n in v['n_best']), encoding='utf-8')
    write_json(out / 'manifest.json', {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                                       'converter': CONVERTER, 'flags': 'typo off, Zenzai disabled, stable off',
                                       'tracks': tracks, 'lm_scoring': False})
    for name in ('export_azookey_tracks.sh', 'run_candidates_mac.sh'):
        shutil.copyfile(ROOT / 'scripts/benchmarks' / name, out / name)
    shutil.copyfile(ROOT / 'templates/benchmarks/candidates-v2-mac.md', out / 'README.md')
    for source in ('wrime', 'jmultiwoz'):
        shutil.copyfile(ROOT / f'datasets/ime-standard-ja-v1/{source}/LICENSE', out / f'LICENSE-{source}.txt')
    (out / 'NOTICE.md').write_text(
        (BENCH / 'ime-standard-ja-v2-blind-ai-expert/NOTICE.md').read_text(encoding='utf-8') + '\n' +
        (BENCH / 'ime-homophone-ja-v1-ai-expert/NOTICE.md').read_text(encoding='utf-8'), encoding='utf-8')
    with zipfile.ZipFile(out.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                archive.write(path, str(path.relative_to(out.parent)))
    print(json.dumps(tracks, indent=1))


def import_results(args):
    """Each (track, n-best) becomes a loadable standard export; inputs must match the package byte for byte."""
    handoff = read(args.package / 'manifest.json')
    fresh(args.output)
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        for track, info in handoff['tracks'].items():
            for n in info['n_best']:
                payload = {}
                for filename in FILES:
                    suffix = f'azookey-results/{track}/n{n}/{filename}'
                    matches = [name for name in names if name == suffix or name.endswith('/' + suffix)]
                    if len(matches) != 1:
                        raise ValueError(f'Expected exactly one {suffix}.')
                    payload[filename] = archive.read(matches[0])
                for filename in ('ajimee-input.json', 'case-map.json'):
                    if payload[filename] != (args.package / track / filename).read_bytes():
                        raise ValueError(f'{track}/n{n}: returned {filename} differs from the package.')
                dest = args.output / f'{track}-n{n}'
                rows = items(track)
                track_files(dest, rows)
                (dest / 'ajimee-results').mkdir()
                for filename, content in payload.items():
                    (dest / 'ajimee-results' / filename).write_bytes(content)
                release, split, _ = TRACKS[track]
                manifest = {**read(release / 'manifest.json'), 'format': FORMAT, 'status': 'complete', 'split': split,
                            'track': track, 'n_best': n, 'stats': read(dest / 'stats.json'),
                            'candidate_provenance': f'{VERSION} Mac export', 'handoff_created_utc': handoff['created_utc']}
                write_json(dest / 'manifest.json', manifest)
                _, _, loaded = load_standard_export(dest)
                print(f'{track}-n{n}: {len(loaded)} cases loaded', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command', required=True)
    sub = subs.add_parser('package')
    sub.add_argument('--output', type=Path, default=BENCH / f'{VERSION}-handoff')
    sub = subs.add_parser('import')
    sub.add_argument('--archive', type=Path, required=True)
    sub.add_argument('--package', type=Path, default=BENCH / f'{VERSION}-handoff')
    sub.add_argument('--output', type=Path, default=BENCH / f'{VERSION}-imported')
    args = parser.parse_args()
    {'package': package, 'import': import_results}[args.command](args)


if __name__ == '__main__':
    main()
