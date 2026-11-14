"""Final export of the Chinese corpus for upload (zh-v13-export-r1, and later batches such as r2).

  web     stage-a-v2 decision=keep documents, minus exact duplicates and every document on a
          listed spam host or spam farm (same host -> site rule and spam_allow as `zh_stage_a.py
          report`). Review documents are never part of this command.
  review  stage-a-v2 decision=review documents whose id is in --accept (one id per line). The
          accept list comes from the private review tooling (all votes keep); a review document
          that is not on the list is dropped, reviewed or not.
  chat    LCCC v2 train/validation/test utterances. lccc-v2/review.jsonl (weak-term hits that were
          never reviewed) is not exported.
  finish  SHA256SUMS and manifest.json over everything written.

--batch r2 exports the second FineWeb2 batch (stage-a-r2) as zh-v13-export-r2: web and finish
only. r2 has no review step: every decision=review document is dropped. Duplicates come from
`zh_stage_a.py dedup --against export-r1`, so documents already in r1 are dropped too; the spam
host and farm lists are the union of stage-a-v2's and stage-a-r2's. LCCC is not re-exported.

A fixed 0.05% of web and review documents, chosen by SHA-256 of the id, goes to validation/.
Output files are zstd-compressed Parquet of about --file-bytes of text each, so a failed download
costs one file. Each input shard is written to a temporary name and renamed when complete;
completed shards are skipped, nothing is overwritten.

usage:
  python -I -X utf8 scripts/corpus/zh_export.py web OUT [--workers 8]
  python -I -X utf8 scripts/corpus/zh_export.py review OUT --accept accept-ids.txt
  python -I -X utf8 scripts/corpus/zh_export.py chat OUT
  python -I -X utf8 scripts/corpus/zh_export.py finish OUT
  python -I -X utf8 scripts/corpus/zh_export.py web OUT --batch r2
  python -I -X utf8 scripts/corpus/zh_export.py finish OUT --batch r2
"""
import argparse
import hashlib
import json
import os
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.corpus.zh_filter import in_domains, registrable  # noqa: E402

BATCHES = {
    'r1': dict(version='zh-v13-export-r1', stage_a=ROOT / 'outputs/zh-corpus/stage-a-v2', shards=21,
               spam=[ROOT / 'outputs/zh-corpus/stage-a-v2']),
    'r2': dict(version='zh-v13-export-r2', stage_a=ROOT / 'outputs/zh-corpus/stage-a-r2', shards=None,
               spam=[ROOT / 'outputs/zh-corpus/stage-a-v2', ROOT / 'outputs/zh-corpus/stage-a-r2']),
}
LCCC = ROOT / 'outputs/zh-corpus/lccc-v2'
TERMS = ROOT / 'configs/zh-filter-terms.toml'
VALIDATION_RATE = 0.0005
COLUMNS = ['id', 'host', 'script', 'han', 'text']


def is_validation(doc_id):
    digest = hashlib.sha256(('zh-v13-validation\0' + doc_id).encode()).digest()
    return int.from_bytes(digest[:8], 'big') < int(VALIDATION_RATE * (1 << 64))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while block := stream.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def spam_rules(batch):
    hosts, farms = set(), set()
    for stage_a in BATCHES[batch]['spam']:
        spam = json.loads((stage_a / 'spam-hosts.json').read_text(encoding='utf-8'))
        hosts |= {h['host'] for h in spam['hosts']}
        farms |= {f['domain'] for f in spam['farms']}
    allow = tuple(tomllib.loads(TERMS.read_text(encoding='utf-8')).get('domains', {}).get('spam_allow', []))
    return hosts, farms, allow


def duplicate_ids(batch):
    import pyarrow.parquet as pq
    path = BATCHES[batch]['stage_a'] / 'duplicates.parquet'
    return set(pq.read_table(path, columns=['id']).column('id').to_pylist())


class Writer:
    """Rolls Parquet files of about `limit` text bytes under out/<split>/<prefix>-NNN.parquet."""

    def __init__(self, out, split, prefix, limit, level, schema):
        self.dir, self.prefix, self.limit, self.level, self.schema = out / split, prefix, limit, level, schema
        self.dir.mkdir(parents=True, exist_ok=True)
        self.writer, self.index, self.bytes, self.files = None, 0, 0, []

    def write(self, table, text_bytes):
        import pyarrow.parquet as pq
        if self.writer is None:
            path = self.dir / f'{self.prefix}-{self.index:03d}.parquet.part'
            self.writer = pq.ParquetWriter(path, self.schema, compression='zstd', compression_level=self.level)
            self.files.append(path)
        self.writer.write_table(table, row_group_size=len(table))
        self.bytes += text_bytes
        if self.bytes >= self.limit:
            self.close_file()

    def close_file(self):
        if self.writer is not None:
            self.writer.close()
            self.writer, self.index, self.bytes = None, self.index + 1, 0

    def finish(self):
        self.close_file()
        done = []
        for part in self.files:
            final = part.with_suffix('')
            os.replace(part, final)
            done.append(final)
        return done


def export_shard(job):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    shard, out, kind, limit, level, accept_path, batch = job
    out = Path(out)
    marker = out / 'parts' / f'{kind}-{shard}.json'
    if marker.exists():
        return json.loads(marker.read_text(encoding='utf-8'))
    for stale in out.glob(f'*/{kind}-{shard}-*.parquet.part'):
        stale.unlink()
    hosts, farms, allow = spam_rules(batch)
    duplicates = duplicate_ids(batch)
    accept = set(Path(accept_path).read_text(encoding='utf-8').split()) if accept_path else None
    schema = pa.schema([('id', pa.string()), ('host', pa.string()), ('script', pa.string()),
                        ('han', pa.int64()), ('text', pa.string())])
    writers = {split: Writer(out, split, f'{kind}-{shard}', limit, level, schema) for split in ('train', 'validation')}
    counts = {'input': 0, 'duplicate': 0, 'spam_host': 0, 'spam_farm': 0, 'not_accepted': 0,
              'train': 0, 'validation': 0, 'train_han': 0, 'validation_han': 0}
    site_cache = {}
    source = pq.ParquetFile(BATCHES[batch]['stage_a'] / f'{shard}.parquet')
    decision = 'keep' if kind == 'web' else 'review'
    started = time.time()
    for group in range(source.metadata.num_row_groups):
        table = source.read_row_group(group, columns=COLUMNS + ['decision'])
        table = table.filter(pc.equal(table['decision'], decision)).select(COLUMNS)
        ids, host_values = table['id'].to_pylist(), table['host'].to_pylist()
        keep, split = [], []
        for doc_id, host in zip(ids, host_values):
            counts['input'] += 1
            if doc_id in duplicates:
                counts['duplicate'] += 1; keep.append(False); split.append(None); continue
            if host not in site_cache:
                allowed = in_domains(host, allow)
                site_cache[host] = ('spam_host' if host in hosts and not allowed else
                                    'spam_farm' if not allowed and registrable(host) in farms else None)
            if site_cache[host]:
                counts[site_cache[host]] += 1; keep.append(False); split.append(None); continue
            if accept is not None and doc_id not in accept:
                counts['not_accepted'] += 1; keep.append(False); split.append(None); continue
            keep.append(True); split.append('validation' if is_validation(doc_id) else 'train')
        for name in ('train', 'validation'):
            mask = pa.array([k and s == name for k, s in zip(keep, split)])
            part = table.filter(mask)
            if len(part):
                text_bytes = pc.sum(pc.binary_length(part['text'])).as_py()
                writers[name].write(part, text_bytes)
                counts[name] += len(part)
                counts[f'{name}_han'] += pc.sum(part['han']).as_py()
    files = []
    for writer in writers.values():
        files += [p.relative_to(out).as_posix() for p in writer.finish()]
    result = {'shard': shard, 'kind': kind, **counts, 'files': files, 'seconds': round(time.time() - started, 1)}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    return result


def run_shards(args, kind):
    out = Path(args.out)
    batch = BATCHES[args.batch]
    stage_a = batch['stage_a']
    shards = sorted(p.name[:-len('.parquet')] for p in stage_a.glob('[0-9]*_*[0-9].parquet'))
    expected = batch['shards']
    if expected is None:  # every file of the batch's download manifest, each with a finished stage-A marker
        run = json.loads((stage_a / 'run.json').read_text(encoding='utf-8'))
        config = tomllib.loads(Path(run['config']).read_text(encoding='utf-8'))
        files = json.loads((ROOT / config['input']['manifest']).read_text(encoding='utf-8'))['files']
        wanted = sorted(Path(f['local']).stem for f in files)
        missing = [s for s in wanted if not (stage_a / f'{s}.json').exists()]
        if missing or shards != wanted:
            raise SystemExit(f'{stage_a.name}: stage A unfinished or mismatched ({len(missing)} shards missing)')
        expected = len(wanted)
    if len(shards) != expected:
        raise SystemExit(f'expected {expected} {stage_a.name} shards, found {len(shards)}')
    if not (stage_a / 'duplicates.parquet').exists():
        raise SystemExit(f'{stage_a.name}: run zh_stage_a.py dedup first')
    accept = str(Path(args.accept).resolve()) if kind == 'review' else None
    jobs = [(s, str(out), kind, args.file_bytes, args.level, accept, args.batch) for s in shards]
    total = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(export_shard, jobs):
            print(f'{result["shard"]}: train {result["train"]:,} validation {result["validation"]:,} '
                  f'dropped dup {result["duplicate"]:,} host {result["spam_host"]:,} farm {result["spam_farm"]:,} '
                  f'not accepted {result["not_accepted"]:,} ({result["seconds"]}s)', flush=True)
            for k, v in result.items():
                if isinstance(v, int):
                    total[k] = total.get(k, 0) + v
    print(kind, json.dumps(total))


def cmd_chat(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = Path(args.out)
    marker = out / 'parts' / 'chat.json'
    if marker.exists():
        print('chat already exported'); return
    result = {}
    for split in ('train', 'validation', 'test'):
        target = out / split / f'chat-lccc-{split}.parquet'
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = {'text': [], 'dialogue': [], 'turn': [], 'script': []}
        with open(LCCC / f'{split}.jsonl', encoding='utf-8') as stream:
            for line in stream:
                r = json.loads(line)
                for k in rows:
                    rows[k].append(r[k])
        part = target.with_name(target.name + '.part')
        pq.write_table(pa.table(rows), part, compression='zstd', compression_level=args.level, row_group_size=200_000)
        os.replace(part, target)
        result[split] = {'utterances': len(rows['text']), 'chars': sum(map(len, rows['text'])),
                         'file': target.relative_to(out).as_posix()}
        print(split, result[split], flush=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')


def cmd_finish(args):
    out = Path(args.out)
    parts = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'parts').glob('*.json'))}
    files = sorted(p for p in out.glob('*/*.parquet'))
    if list(out.glob('*/*.part')):
        raise SystemExit('unfinished .part files remain')
    lines = [f'{sha256(p)}  {p.relative_to(out).as_posix()}' for p in files]
    (out / 'SHA256SUMS').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    totals = {}
    for name, r in parts.items():
        kind = r.get('kind', 'chat')
        if kind == 'chat':
            totals['chat'] = {s: v['utterances'] for s, v in r.items()}
            continue
        t = totals.setdefault(kind, {})
        for k, v in r.items():
            if isinstance(v, int):
                t[k] = t.get(k, 0) + v
    batch = BATCHES[args.batch]
    stage_a = batch['stage_a']
    inputs = {'stage_a_run_sha256': sha256(stage_a / 'run.json'),
              'spam_hosts_sha256': sha256(stage_a / 'spam-hosts.json'),
              'duplicates_sha256': sha256(stage_a / 'duplicates.parquet'),
              'terms_sha256': sha256(TERMS), 'script_sha256': sha256(Path(__file__))}
    if args.batch == 'r1':
        inputs['lccc_manifest_sha256'] = sha256(LCCC / 'manifest.json')
    else:
        inputs['spam_hosts_sha256'] = {s.name: sha256(s / 'spam-hosts.json') for s in batch['spam']}
        inputs['review'] = 'none: every review-queue document dropped'
    manifest = {
        'version': batch['version'], 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'inputs': inputs,
        'validation_rate': VALIDATION_RATE, 'columns': {'web/review': COLUMNS, 'chat': ['text', 'dialogue', 'turn', 'script']},
        'totals': totals, 'files': len(files), 'bytes': sum(p.stat().st_size for p in files),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('totals', 'files', 'bytes')}, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('web', 'review', 'chat', 'finish'):
        p = commands.add_parser(name)
        p.add_argument('out')
        p.add_argument('--level', type=int, default=9)
        p.add_argument('--file-bytes', type=int, default=4 << 30, help='uncompressed text bytes per output file')
        p.add_argument('--workers', type=int, default=8)
        p.add_argument('--batch', choices=sorted(BATCHES), default='r1')
        if name == 'review':
            p.add_argument('--accept', required=True)
    args = parser.parse_args()
    if args.batch != 'r1' and args.command in ('review', 'chat'):
        raise SystemExit(f'{args.command} is r1 only: later batches have no review step and reuse LCCC')
    if args.command in ('web', 'review'):
        run_shards(args, args.command)
    else:
        {'chat': cmd_chat, 'finish': cmd_finish}[args.command](args)


if __name__ == '__main__':
    main()
