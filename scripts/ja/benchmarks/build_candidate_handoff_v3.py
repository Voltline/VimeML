"""Third Mac candidate handoff (ja-gen plan, stage 0): AzooKey candidates for the new kana set and a deeper
Standard development pool.

- standard-v1-development n100: with the existing n20/n50 imports, separates pool misses a deeper
  AzooKey list recovers from those only generation can recover (stage 0.1);
- kana-ja-v1-development n20, kana-ja-v1-test n20: the deployed setting, exported once (stage 0.2).
  Test candidates are exported and imported but not scored; the CliTool log of test is not imported,
  because CliTool may print a coverage summary.

Same converter, dictionary and flags as v1/v2. No language model runs during export. Commands: package, import.
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/ja/benchmarks'))
from vimeml.benchmarks.expanded_ime import CONVERTER
from vimeml.benchmarks.standard_ime import FORMAT, load_standard_export
from vimeml.training.data import write_json
from prepare_standard_ime_v2 import read, fresh
from standard_ime import track_files

BENCH = ROOT / 'artifacts/benchmarks'
VERSION = 'ime-candidates-v3'
TRACKS = {
    'standard-v1-development': (BENCH / 'ime-standard-ja-v1-ai-expert-r1', 'development', (100,)),
    'kana-ja-v1-development': (BENCH / 'ime-kana-ja-v1', 'development', (20,)),
    'kana-ja-v1-test': (BENCH / 'ime-kana-ja-v1', 'test', (20,)),
}
SEALED = {'kana-ja-v1-test'}
FILES = ('ajimee-input.json', 'case-map.json', 'azookey-candidates.json', 'converter-version.txt',
         'dictionary-versions.txt', 'swift-version.txt', 'export-flags.txt', 'completed.txt', 'export.log')


def items(track):
    release, split, _ = TRACKS[track]
    return read(release / split / 'evaluation_items.json')


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
    (out / 'tracks.tsv').write_text(''.join(f'{t}\t{n}\n' for t, v in tracks.items() for n in v['n_best']),
                                    encoding='utf-8', newline='\n')
    write_json(out / 'manifest.json', {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                                       'converter': CONVERTER, 'flags': 'typo off, Zenzai disabled, stable off',
                                       'tracks': tracks, 'sealed_tracks': sorted(SEALED), 'lm_scoring': False})
    shutil.copyfile(ROOT / 'scripts/ja/benchmarks/export_azookey_tracks.sh', out / 'export_azookey_tracks.sh')
    for source in ('wrime', 'jmultiwoz'):
        shutil.copyfile(ROOT / f'datasets/ime-standard-ja-v1/{source}/LICENSE', out / f'LICENSE-{source}.txt')
    shutil.copyfile(BENCH / 'ime-kana-ja-v1/NOTICE.md', out / 'NOTICE.md')
    print(json.dumps(tracks, indent=1))


def import_results(args):
    """Each (track, n-best) becomes a loadable standard export; inputs must match the package byte for byte."""
    handoff = read(args.package / 'manifest.json')
    fresh(args.output)
    for track, info in handoff['tracks'].items():
        for n in info['n_best']:
            source = args.results / track / f'n{n}'
            payload = {name: (source / name).read_bytes() for name in FILES
                       if not (track in SEALED and name == 'export.log')}
            for filename in ('ajimee-input.json', 'case-map.json'):
                if payload[filename] != (args.package / track / filename).read_bytes():
                    raise ValueError(f'{track}/n{n}: returned {filename} differs from the package.')
            if payload['converter-version.txt'].decode().strip() != CONVERTER:
                raise ValueError(f'{track}/n{n}: converter revision differs.')
            dest = args.output / f'{track}-n{n}'
            track_files(dest, items(track))
            (dest / 'ajimee-results').mkdir()
            for filename, content in payload.items():
                (dest / 'ajimee-results' / filename).write_bytes(content)
            release, split, _ = TRACKS[track]
            manifest = {**read(release / 'manifest.json'), 'format': FORMAT, 'status': 'complete', 'split': split,
                        'track': track, 'n_best': n, 'stats': read(dest / 'stats.json'), 'sealed': track in SEALED,
                        'candidate_provenance': f'{VERSION} Mac export', 'handoff_created_utc': handoff['created_utc']}
            write_json(dest / 'manifest.json', manifest)
            if track in SEALED:
                print(f'{track}-n{n}: imported, sealed (not loaded)', flush=True)
                continue
            _, _, loaded = load_standard_export(dest)
            print(f'{track}-n{n}: {len(loaded)} cases loaded', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command', required=True)
    sub = subs.add_parser('package')
    sub.add_argument('--output', type=Path, default=BENCH / f'{VERSION}-handoff')
    sub = subs.add_parser('import')
    sub.add_argument('--results', type=Path, required=True, help='returned azookey-results directory')
    sub.add_argument('--package', type=Path, default=BENCH / f'{VERSION}-handoff')
    sub.add_argument('--output', type=Path, default=BENCH / f'{VERSION}-imported')
    args = parser.parse_args()
    {'package': package, 'import': import_results}[args.command](args)


if __name__ == '__main__':
    main()
