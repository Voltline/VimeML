"""Mac candidate handoff for the sealed ime-kana-ja-colloq-v1 test split (ja-gen stage 4 confirmation).

AzooKey N-best 20 with the deployed setting, exactly as for the development split (converter d59a28e4c7ca,
typo off, Zenzai disabled). Test candidates are exported and imported but not scored here; the CliTool log is
not imported, because CliTool may print a coverage summary. The imported pool is read once, by
scripts/ja/benchmarks/confirm_ja_gen.py. Commands: package, import.

  python -X utf8 scripts/ja/benchmarks/build_candidate_handoff_colloq_test.py package
  python -X utf8 scripts/ja/benchmarks/build_candidate_handoff_colloq_test.py import --results <azookey-results>
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
from vimeml.benchmarks.standard_ime import FORMAT
from vimeml.training.data import write_json
from prepare_standard_ime_v2 import read, fresh
from standard_ime import track_files

BENCH = ROOT / 'artifacts/benchmarks'
VERSION = 'ime-candidates-colloq-test-v1'
RELEASE = BENCH / 'ime-kana-ja-colloq-v1'
RELEASE_MANIFEST_SHA256 = '6538b128474cb938cc942aa645b18ea8282d3218e43a33a77ba0186e6dc66e9f'
TRACK, SPLIT, N_BEST = 'kana-ja-colloq-v1-test', 'test', 20
FILES = ('ajimee-input.json', 'case-map.json', 'azookey-candidates.json', 'converter-version.txt',
         'dictionary-versions.txt', 'swift-version.txt', 'export-flags.txt', 'completed.txt')


def items():
    if hashlib.sha256((RELEASE / 'manifest.json').read_bytes()).hexdigest() != RELEASE_MANIFEST_SHA256:
        raise ValueError(f'{RELEASE.name}: manifest differs from the frozen release')
    return read(RELEASE / SPLIT / 'evaluation_items.json')


def package(args):
    out = args.output
    fresh(out)
    track_files(out / TRACK, items())
    (out / TRACK / 'evaluation_items.json').unlink()
    (out / TRACK / 'stats.json').unlink()
    manifest = read(RELEASE / 'manifest.json')
    track = {'release': RELEASE.name, 'release_id': manifest.get('release_id'), 'split': SPLIT, 'n_best': [N_BEST],
             'cases': len(read(out / TRACK / 'case-map.json')),
             'input_sha256': hashlib.sha256((out / TRACK / 'ajimee-input.json').read_bytes()).hexdigest()}
    (out / 'tracks.tsv').write_text(f'{TRACK}\t{N_BEST}\n', encoding='utf-8', newline='\n')
    write_json(out / 'manifest.json', {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                                       'converter': CONVERTER, 'flags': 'typo off, Zenzai disabled, stable off',
                                       'tracks': {TRACK: track}, 'sealed_tracks': [TRACK], 'lm_scoring': False})
    shutil.copyfile(ROOT / 'scripts/ja/benchmarks/export_azookey_tracks.sh', out / 'export_azookey_tracks.sh')
    print(json.dumps(track, indent=1))


def import_results(args):
    """The returned inputs must match the package byte for byte; the result is a sealed standard export."""
    handoff = read(args.package / 'manifest.json')
    fresh(args.output)
    source = args.results / TRACK / f'n{N_BEST}'
    payload = {name: (source / name).read_bytes() for name in FILES}
    for filename in ('ajimee-input.json', 'case-map.json'):
        if payload[filename] != (args.package / TRACK / filename).read_bytes():
            raise ValueError(f'{TRACK}: returned {filename} differs from the package.')
    if payload['converter-version.txt'].decode().strip() != CONVERTER:
        raise ValueError(f'{TRACK}: converter revision differs.')
    dest = args.output / f'{TRACK}-n{N_BEST}'
    track_files(dest, items())
    (dest / 'ajimee-results').mkdir()
    for filename, content in payload.items():
        (dest / 'ajimee-results' / filename).write_bytes(content)
    manifest = {**read(RELEASE / 'manifest.json'), 'format': FORMAT, 'status': 'complete', 'split': SPLIT,
                'track': TRACK, 'n_best': N_BEST, 'stats': read(dest / 'stats.json'), 'sealed': True,
                'candidate_provenance': f'{VERSION} Mac export', 'handoff_created_utc': handoff['created_utc']}
    write_json(dest / 'manifest.json', manifest)
    for name in ('manifest.json', 'evaluation_items.json', 'ajimee-results/azookey-candidates.json'):
        print(hashlib.sha256((dest / name).read_bytes()).hexdigest(), f'{dest.relative_to(ROOT).as_posix()}/{name}')
    print(f'{TRACK}-n{N_BEST}: imported, sealed (not loaded)', flush=True)


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
