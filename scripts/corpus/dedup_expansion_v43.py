"""Cross-pool deduplication of the V4.3 expansion corpus before encoding.

Removes expansion sentences that exactly duplicate any frozen-corpus sentence
(all three splits of the reproduction), any chat-bank sentence, or any frozen
benchmark provenance text (when supplied). Canonical keys use NFKC plus
whitespace collapse, matching the V4.1 mixture preparation.
"""
import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def key(text):
    return hashlib.blake2b(' '.join(unicodedata.normalize('NFKC', text).split()).encode(), digest_size=16).digest()


def load_txt_keys(path, report, name):
    keys = set()
    sentences = 0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            text = line.rstrip('\r\n')
            if text:
                keys.add(key(text))
                sentences += 1
    report[name] = sentences
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expansion', type=Path, default=ROOT / 'outputs/corpus-expansion-v43')
    parser.add_argument('--old', type=Path, default=ROOT / 'outputs/corpus-reproduction-v1')
    parser.add_argument('--chat-bank', type=Path, default=ROOT / 'artifacts/token-data/corpus-v41-chat-8k')
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model')
    parser.add_argument('--benchmark-inputs', type=Path, default=None,
                        help='Directory of frozen evaluation track JSONs; provenance texts are excluded')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/corpus-expansion-v43-dedup')
    args = parser.parse_args()
    report = {}
    held = set()
    for split in ('train', 'validation', 'test'):
        held |= load_txt_keys(args.old / f'{split}.txt', report, f'old_{split}')
    chat_keys = set()
    import sentencepiece as spm
    import struct
    processor = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    for split in ('train', 'validation'):
        tokens = (args.chat_bank / f'{split}.tokens.bin').read_bytes()
        raw = (args.chat_bank / f'{split}.offsets.bin').read_bytes()
        offsets = struct.unpack(f'<{len(raw) // 8}Q', raw)
        for start, end in zip(offsets, offsets[1:]):
            ids = [int.from_bytes(tokens[2 * i:2 * i + 2], 'little') for i in range(start + 1, end - 1)]
            text = processor.decode(ids)
            if text:
                chat_keys.add(key(text))
    report['chat_bank_sentences'] = len(chat_keys)
    held |= chat_keys
    if args.benchmark_inputs and args.benchmark_inputs.is_dir():
        benchmark = 0
        for path in sorted(args.benchmark_inputs.glob('*.json')):
            for row in json.loads(path.read_text('utf-8')):
                text = row.get('provenance', {}).get('original_message')
                if text:
                    held.add(key(text))
                    benchmark += 1
        report['benchmark_provenance_texts'] = benchmark
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f'Refusing to overwrite: {args.output}')
    args.output.mkdir(parents=True)
    import shutil
    from collections import Counter
    stats = {}
    for split in ('train', 'validation'):
        removed = Counter()
        kept = 0
        characters = 0
        with (args.expansion / f'{split}.txt').open(encoding='utf-8') as source, \
                (args.output / f'{split}.txt').open('w', encoding='utf-8', newline='\n') as target:
            for line in source:
                text = line.rstrip('\r\n')
                if not text:
                    continue
                if key(text) in held:
                    removed['cross_pool_duplicate'] += 1
                    continue
                target.write(text + '\n')
                kept += 1
                characters += len(text)
        stats[split] = {'sentences': kept, 'characters': characters,
                        'removed': dict(removed), 'primary_sources': {'fineweb': kept}}
        print(f'{split}: kept {kept:,}, removed {sum(removed.values()):,}', flush=True)
    for split in ('train', 'validation'):
        source = args.output / f'{split}.txt'
        with (args.output / f'{split}.sources.bin').open('wb') as stream:
            stream.write(b'\x00' * stats[split]['sentences'])
    manifest = {
        'format': 'vimeml_expansion_corpus_dedup_v1', 'status': 'complete',
        'expansion_source': str(args.expansion), 'old_corpus': str(args.old),
        'chat_bank': str(args.chat_bank), 'benchmark_inputs': str(args.benchmark_inputs) if args.benchmark_inputs else None,
        'canonical_key': 'blake2b16 of NFKC text with collapsed whitespace',
        'excluded_sentences': report, 'splits': stats,
        'source_ids': {'fineweb': 0}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'kept': {s: v['sentences'] for s, v in stats.items()}, 'excluded': report}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
