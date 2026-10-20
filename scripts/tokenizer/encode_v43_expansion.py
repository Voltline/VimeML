"""Encode the deduplicated V4.3 expansion corpus with the frozen 8K tokenizer.

Reuses the V4 TXT encoding implementation (scan/partition/encode/merge) but
carries the expansion corpus's own identity: no frozen-corpus checks, no
16K source store, all sentences are FineWeb2-Edu Japanese (source id 0).
"""
import argparse
import hashlib
import json
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from vimeml.tokenizer.encode_v4 import (SPLITS, collect, merge_txt_split, prepare_v4_indexes, scan_txt)
from vimeml.tokenizer.encode import dump_json, file_sha
from vimeml.tokenizer.encode_parallel import output_lock, part_paths
from vimeml.tokenizer.monitor import RunMonitor
from vimeml.posttraining.data import TOKENIZER_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/data-v43-expansion.toml')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    import sentencepiece as spm
    config = tomllib.loads(args.config.read_text(encoding='utf-8'))
    paths = {key: ROOT / value for key, value in config['paths'].items()}
    corpus, tokenizer = paths['corpus_dir'], paths['tokenizer_dir']
    output, index_dir = paths['output_dir'], paths['index_dir']
    encoding, checks = config['encoding'], config['checks']
    model = tokenizer / 'tokenizer.model'
    if file_sha(model) != TOKENIZER_SHA:
        raise ValueError('Frozen V4 8K tokenizer changed')
    dedup = json.loads((corpus / 'manifest.json').read_text('utf-8'))
    if dedup.get('format') != 'vimeml_expansion_corpus_dedup_v1' or dedup.get('status') != 'complete':
        raise ValueError('Expansion corpus has not passed cross-pool deduplication')
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Output is nonempty; choose a new path or --resume')
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        if (output / 'manifest.json').exists():
            complete = json.loads((output / 'manifest.json').read_text('utf-8'))
            if complete.get('status') == 'complete':
                if complete['tokenizer_model_sha256'] != TOKENIZER_SHA:
                    raise ValueError('Completed output belongs to another tokenizer')
                for name, digest in complete['output_sha256'].items():
                    if file_sha(output / name) != digest:
                        raise ValueError(f'Completed output checksum mismatch: {name}')
                prepare_v4_indexes(output, index_dir, checks)
                return
        started = time.monotonic()
        with RunMonitor(output, ROOT / 'runs/tokenizer/corpus-v43-expansion-encode' / str(time.time_ns()), True) as monitor:
            jobs, scans = [], {}
            hashes = {}
            for split in SPLITS:
                expected = dedup['splits'][split]
                selected, scans[split] = scan_txt(corpus / f'{split}.txt', split,
                    encoding['chunk_mib'] * 1024 * 1024, expected,
                    lambda done, total: monitor.update('checking_inputs', preflight_completed_sentences=done, preflight_total_sentences=total))
                hashes[str(corpus / f'{split}.txt')] = scans[split]['sha256']
                for i, job in enumerate(selected):
                    job.update(name=f'{split}-{i:05d}', source_input=str(corpus / f'{split}.sources.bin'))
                jobs.extend(selected)
            signature = hashlib.sha256(json.dumps({'hashes': hashes, 'encoding': encoding,
                'splits': list(SPLITS)}, sort_keys=True).encode()).hexdigest()
            parts = output / '.parts'
            parts.mkdir(exist_ok=True)
            for job in jobs:
                job.update(model=str(model), parts=str(parts), batch_size=encoding['batch_size'],
                           signature=signature, source_ids=dedup['source_ids'], verification='sha256')
            dump_json(output / 'encoding-plan.json', {'signature': signature, 'scans': scans,
                'corpus': str(corpus), 'source': 'deduplicated V4.3 expansion corpus'})
            results = collect(jobs, encoding['workers'], monitor)
            exported = {}
            for split in SPLITS:
                selected = [job for job in jobs if job['split'] == split]
                exported[split] = merge_txt_split(output, split, selected, results, dedup['splits'][split], monitor)
            dump_json(output / 'stats.json', {'actual_vocab_size': 8192, 'splits': exported})
            artifacts = [output / f'{s}.{suffix}' for s in SPLITS for suffix in ('tokens.bin', 'offsets.bin', 'rows.bin', 'sources.bin')]
            artifacts.append(output / 'stats.json')
            output_hashes = {p.name: file_sha(p) for p in artifacts}
            dump_json(output / 'manifest.json', {'status': 'complete', 'format': 'vimeml_sentence_tokens_v1',
                'vocab_size': 8192, 'token_dtype': 'uint16_le', 'offset_dtype': 'uint64_le', 'offset_unit': 'tokens',
                'special_ids': {'pad': 0, 'unk': 1, 'bos': 2, 'eos': 3}, 'reserved_marker_ids': [4, 5, 6],
                'sentencepiece_version': spm.__version__, 'tokenizer_model_sha256': TOKENIZER_SHA,
                'tokenizer_dir': str(tokenizer), 'corpus_dir': str(corpus), 'available_splits': list(SPLITS),
                'splits': exported, 'source_ids': dedup['source_ids'], 'input_sha256': hashes,
                'output_sha256': output_hashes, 'signature': signature,
                'sequence_policy': 'BOS + complete sentence + EOS; no truncation, concatenation or task markers.',
                'row_offset_source': 'corpus_txt', 'row_offset_dtype': 'uint64_le',
                'provenance_policy': 'rows.bin stores uint64 byte offsets into the deduplicated expansion TXT; sources.bin is all FineWeb2-Edu (id 0).',
                'input_verification': 'Full SHA256 before/after; every encoded text roundtrip; all sequence IDs checked; merged offsets/sizes/counts checked.',
                'test_and_blind_access': False, 'corpus_ready_for_lm_training': True,
                'corpus_quality_mode': 'rules_only (V4.3 expansion; no approved annotations for new documents)',
                'elapsed_seconds': time.monotonic() - started,
                'purpose': 'V4.3 data expansion: nine additional FineWeb2-Edu shards, train-oriented, cross-pool deduplicated.'})
            for job in jobs:
                for path in part_paths(job).values():
                    path.unlink(missing_ok=True)
            (output / '.parts').rmdir()
    prepare_v4_indexes(output, index_dir, checks)
    print(json.dumps({'tokens': exported}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
