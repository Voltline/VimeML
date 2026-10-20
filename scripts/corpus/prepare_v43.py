"""Prepare the V4.3 three-store mixture: old web + expansion + chat.

The old web store and its frozen keep mask are reused unchanged from the V4.1/V4.2
mixture. The expansion store's keep mask excludes windows whose sentence matches a
frozen benchmark provenance text (when the evaluation inputs are supplied).
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

OLD_MIX = ROOT / 'artifacts/training-data/corpus-v41-mixed-c128'
NEW = ROOT / 'artifacts/training-data/corpus-v43-mixed-c128'
EPOCHS = 30


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-inputs', type=Path, default=None,
                        help='Frozen evaluation track JSONs; provenance texts are excluded from the expansion pool')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    from vimeml.training.data import write_json
    from vimeml.training.v43_data import prepare_maps_v43, window_sentences, sentence_keys
    if (NEW / 'manifest.json').exists():
        result = prepare_maps_v43(NEW, root=ROOT)
        print(json.dumps({'status': 'verified', 'epochs': len(result['epochs']),
                          'dataset_sha256': sha(NEW / 'manifest.json')}))
        return
    if args.verify_only:
        raise ValueError('No V4.3 mixture to verify')
    old_request = json.loads((OLD_MIX / 'request.json').read_text('utf-8'))
    old_store, chat_store = old_request['joined_stores'][0], old_request['joined_stores'][1]
    exp_token = ROOT / 'artifacts/token-data/corpus-v43-exp8k'
    exp_index = ROOT / 'artifacts/training-data/corpus-v43-exp-c128'
    if not (exp_token / 'manifest.json').is_file() or not (exp_index / 'manifest.json').is_file():
        raise ValueError('Expansion token store is not built yet')
    NEW.mkdir(parents=True, exist_ok=True)
    exp_keep = np.ones(json.loads((exp_index / 'manifest.json').read_text('utf-8'))['splits']['train']['windows'],
                       dtype=bool)
    excluded = 0
    if args.benchmark_inputs and args.benchmark_inputs.is_dir():
        from vimeml.training.data_v2 import PrefixCropWindowDataset
        import unicodedata
        keys = sentence_keys(_benchmark_texts(args.benchmark_inputs))
        excluded_lines = set()
        dedup_manifest = json.loads((ROOT / 'outputs/corpus-expansion-v43-dedup/manifest.json').read_text('utf-8'))
        with (ROOT / 'outputs/corpus-expansion-v43-dedup/train.txt').open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream):
                text = line.rstrip('\r\n')
                if text and (' '.join(unicodedata.normalize('NFKC', text).split()).encode() and
                             hashlib.blake2b(' '.join(unicodedata.normalize('NFKC', text).split()).encode(),
                                             digest_size=16).digest() in keys):
                    excluded_lines.add(line_number)
        base = PrefixCropWindowDataset(exp_token, exp_index, seed=old_request['seed'],
                                       probability=0.0, min_remaining_tokens=8)
        try:
            keep = np.ones(len(base), dtype=bool)
            for start in range(0, len(base), 262144):
                ids = np.arange(start, min(len(base), start + 262144), dtype=np.int64)
                sentences = window_sentences(base, ids)
                keep[start:start + len(ids)] = ~np.isin(sentences, np.fromiter(excluded_lines, dtype=np.int64))
            excluded = int((~keep).sum())
            exp_keep = keep
        finally:
            base.close()
    np.save(NEW / 'expansion.keep.npy', exp_keep, allow_pickle=False)
    request = {
        'joined_stores': [
            {**old_store, 'keep': 'artifacts/training-data/corpus-v41-mixed-c128/old.keep.npy',
             'keep_sha256': old_request['old_keep_sha256']},
            {'token_dir': 'artifacts/token-data/corpus-v43-exp8k',
             'index_dir': 'artifacts/training-data/corpus-v43-exp-c128',
             'token_manifest_sha256': sha(exp_token / 'manifest.json'),
             'window_manifest_sha256': sha(exp_index / 'manifest.json'),
             'source_map': {'0': 4},
             'keep': 'artifacts/training-data/corpus-v43-mixed-c128/expansion.keep.npy',
             'keep_sha256': sha(NEW / 'expansion.keep.npy')},
            {**chat_store},
        ],
        'primary_store': 0,
        'token_manifest_sha256': old_store['token_manifest_sha256'],
        'window_manifest_sha256': old_store['window_manifest_sha256'],
        'index_dir': old_store['index_dir'],
        'epochs': EPOCHS, 'seed': old_request['seed'],
        'crop_probability': old_request['crop_probability'],
        'crop_minimum': old_request['crop_minimum'], 'chat_fraction': old_request['chat_fraction'],
        'old_keep': 'artifacts/training-data/corpus-v41-mixed-c128/old.keep.npy',
        'old_keep_sha256': old_request['old_keep_sha256'],
        'derived_from': 'V4.1/V4.2 data pools plus the V4.3 deduplicated expansion store',
        'expansion_benchmark_excluded_windows': excluded,
        'test_or_blind_read': False, 'numpy_version': np.__version__,
    }
    (NEW / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    result = prepare_maps_v43(NEW, root=ROOT)
    if len(result['epochs']) != EPOCHS:
        raise ValueError('prepare_maps_v43 did not produce the full plan')
    print(json.dumps({'status': 'prepared', 'epochs': EPOCHS,
                      'benchmark_excluded_windows': excluded,
                      'dataset_sha256': sha(NEW / 'manifest.json')}))


def _benchmark_texts(directory):
    rows = []
    for path in sorted(Path(directory).glob('*.json')):
        data = json.loads(path.read_text('utf-8'))
        rows.extend(row.get('provenance', {}).get('original_message')
                    for row in data if row.get('provenance', {}).get('original_message'))
    return [text for text in rows if text]


if __name__ == '__main__':
    main()
