"""Export of the Japanese web corpus for the ja-gen 200M (stage 1.1): ja-gen-export-v1.

  web     stage-a-v1 decision=keep documents, minus exact duplicates, every document on a listed
          spam host or spam farm, and every document that contains a message of a Japanese IME
          benchmark (plan deviation D2). There is no review step: decision=review is dropped.
  finish  SHA256SUMS and manifest.json over everything written.

Benchmark screen: the cleaned source message of every item in every evaluation_items.json under
artifacts/benchmarks (ime-kana-ja-v1, Standard, homophone, incremental), normalized as in the
benchmark overlap scan (NFKC, casefold, punctuation and spaces removed). Messages of at least
12 normalized characters are matched as substrings of the normalized document.

A fixed 0.05% of documents, chosen by SHA-256 of the id, goes to validation/. Output files are
zstd-compressed Parquet of about --file-bytes of text each; each input shard is written to
temporary names and renamed when complete; completed shards are skipped.

    python -I -X utf8 scripts/ja/corpus/ja_export.py web outputs/ja-corpus/export-v1
    python -I -X utf8 scripts/ja/corpus/ja_export.py finish outputs/ja-corpus/export-v1
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import tomllib
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/zh/corpus'))
from vimeml.corpus.zh_filter import in_domains, registrable  # noqa: E402
from zh_export import Writer, sha256  # noqa: E402

VERSION = 'ja-gen-export-v1'
STAGE_A = ROOT / 'outputs/ja-corpus/stage-a-v1'
TERMS = ROOT / 'configs/ja/ja-filter-terms.toml'
BENCH = ROOT / 'artifacts/benchmarks'
VALIDATION_RATE = 0.0005
COLUMNS = ['id', 'host', 'han', 'text']
CANON = re.compile(r'[^\wぁ-ゖァ-ヺー一-龠々]+')
MIN_MESSAGE = 12


def normalized(text):
    return CANON.sub('', unicodedata.normalize('NFKC', text)).casefold()


def is_validation(doc_id):
    digest = hashlib.sha256(('ja-gen-validation\0' + doc_id).encode()).digest()
    return int.from_bytes(digest[:8], 'big') < int(VALIDATION_RATE * (1 << 64))


def benchmark_messages():
    files, messages = [], set()
    for path in sorted(BENCH.rglob('evaluation_items.json')):
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
        found = {normalized(m) for r in rows if isinstance(r, dict)
                 for m in [(r.get('provenance') or {}).get('cleaned_message')] if m}
        found = {m for m in found if len(m) >= MIN_MESSAGE}
        if found:
            files.append(path.relative_to(BENCH).as_posix())
            messages |= found
    return files, sorted(messages)


def automaton(messages):
    import ahocorasick
    a = ahocorasick.Automaton()
    for m in messages:
        a.add_word(m, m)
    a.make_automaton()
    return a


def spam_rules():
    spam = json.loads((STAGE_A / 'spam-hosts.json').read_text(encoding='utf-8'))
    allow = tuple(tomllib.loads(TERMS.read_text(encoding='utf-8')).get('domains', {}).get('spam_allow', []))
    return {h['host'] for h in spam['hosts']}, {f['domain'] for f in spam['farms']}, allow


def export_shard(job):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    shard, out, limit, level, messages = job
    out = Path(out)
    marker = out / 'parts' / f'web-{shard}.json'
    if marker.exists():
        return json.loads(marker.read_text(encoding='utf-8'))
    for stale in out.glob(f'*/web-{shard}-*.parquet.part'):
        stale.unlink()
    hosts, farms, allow = spam_rules()
    duplicates = set(pq.read_table(STAGE_A / 'duplicates.parquet', columns=['id']).column('id').to_pylist())
    screen = automaton(messages)
    schema = pa.schema([('id', pa.string()), ('host', pa.string()), ('han', pa.int64()), ('text', pa.string())])
    writers = {s: Writer(out, s, f'web-{shard}', limit, level, schema) for s in ('train', 'validation')}
    counts = {'input': 0, 'review_dropped': 0, 'duplicate': 0, 'spam_host': 0, 'spam_farm': 0, 'benchmark': 0,
              'train': 0, 'validation': 0, 'train_han': 0, 'validation_han': 0}
    site_cache = {}
    source = pq.ParquetFile(STAGE_A / f'{shard}.parquet')
    started = time.time()
    for group in range(source.metadata.num_row_groups):
        table = source.read_row_group(group, columns=COLUMNS + ['decision'])
        counts['review_dropped'] += pc.sum(pc.equal(table['decision'], 'review')).as_py() or 0
        table = table.filter(pc.equal(table['decision'], 'keep')).select(COLUMNS)
        split = []
        for doc_id, host, text in zip(table['id'].to_pylist(), table['host'].to_pylist(), table['text'].to_pylist()):
            counts['input'] += 1
            if doc_id in duplicates:
                counts['duplicate'] += 1; split.append(None); continue
            if host not in site_cache:
                allowed = in_domains(host, allow)
                site_cache[host] = ('spam_host' if host in hosts and not allowed else
                                    'spam_farm' if not allowed and registrable(host) in farms else None)
            if site_cache[host]:
                counts[site_cache[host]] += 1; split.append(None); continue
            if next(screen.iter(normalized(text)), None) is not None:
                counts['benchmark'] += 1; split.append(None); continue
            split.append('validation' if is_validation(doc_id) else 'train')
        for name in ('train', 'validation'):
            part = table.filter(pa.array([s == name for s in split]))
            if len(part):
                writers[name].write(part, pc.sum(pc.binary_length(part['text'])).as_py())
                counts[name] += len(part)
                counts[f'{name}_han'] += pc.sum(part['han']).as_py()
    files = []
    for writer in writers.values():
        files += [p.relative_to(out).as_posix() for p in writer.finish()]
    result = {'shard': shard, **counts, 'files': files, 'seconds': round(time.time() - started, 1)}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    return result


def cmd_web(args):
    out = Path(args.out)
    run = json.loads((STAGE_A / 'run.json').read_text(encoding='utf-8'))
    config = tomllib.loads(Path(run['config']).read_text(encoding='utf-8'))
    wanted = sorted(Path(f['local']).stem for f in
                    json.loads((ROOT / config['input']['manifest']).read_text(encoding='utf-8'))['files'])
    missing = [s for s in wanted if not (STAGE_A / f'{s}.json').exists()]
    if missing:
        raise SystemExit(f'stage A unfinished: {len(missing)} shards missing')
    for name in ('duplicates.parquet', 'spam-hosts.json'):
        if not (STAGE_A / name).exists():
            raise SystemExit(f'run ja_stage_a.py dedup and report first ({name} missing)')
    files, messages = benchmark_messages()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'parts').mkdir(exist_ok=True)
    (out / 'parts' / 'benchmark-screen.json').write_text(json.dumps(
        {'files': files, 'messages': len(messages), 'min_chars': MIN_MESSAGE,
         'messages_sha256': hashlib.sha256('\n'.join(messages).encode()).hexdigest()}, indent=1) + '\n',
        encoding='utf-8')
    print(f'benchmark screen: {len(messages)} messages from {len(files)} files', flush=True)
    jobs = [(s, str(out), args.file_bytes, args.level, messages) for s in wanted]
    total = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(export_shard, jobs):
            print(f'{r["shard"]}: train {r["train"]:,} validation {r["validation"]:,} dup {r["duplicate"]:,} '
                  f'host {r["spam_host"]:,} farm {r["spam_farm"]:,} benchmark {r["benchmark"]:,} '
                  f'({r["seconds"]}s)', flush=True)
            for k, v in r.items():
                if isinstance(v, int):
                    total[k] = total.get(k, 0) + v
    print('web', json.dumps(total))


def cmd_finish(args):
    out = Path(args.out)
    if list(out.glob('*/*.part')):
        raise SystemExit('unfinished .part files remain')
    parts = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'parts').glob('web-*.json'))]
    files = sorted(out.glob('*/*.parquet'))
    lines = [f'{sha256(p)}  {p.relative_to(out).as_posix()}' for p in files]
    (out / 'SHA256SUMS').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    totals = {}
    for r in parts:
        for k, v in r.items():
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
    manifest = {
        'version': VERSION, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'inputs': {'stage_a_run_sha256': sha256(STAGE_A / 'run.json'),
                   'spam_hosts_sha256': sha256(STAGE_A / 'spam-hosts.json'),
                   'duplicates_sha256': sha256(STAGE_A / 'duplicates.parquet'), 'terms_sha256': sha256(TERMS),
                   'script_sha256': sha256(Path(__file__)), 'review': 'none: every review-queue document dropped',
                   'benchmark_screen': json.loads((out / 'parts' / 'benchmark-screen.json').read_text(encoding='utf-8'))},
        'validation_rate': VALIDATION_RATE, 'columns': COLUMNS, 'han': 'Japanese characters (kana + kanji)',
        'totals': totals, 'shards': len(parts), 'files': len(files), 'bytes': sum(p.stat().st_size for p in files),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('totals', 'shards', 'files', 'bytes')}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('web', 'finish'):
        p = commands.add_parser(name)
        p.add_argument('out')
        p.add_argument('--level', type=int, default=9)
        p.add_argument('--file-bytes', type=int, default=4 << 30, help='uncompressed text bytes per output file')
        p.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    {'web': cmd_web, 'finish': cmd_finish}[args.command](args)


if __name__ == '__main__':
    main()
