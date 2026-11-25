"""Stage A of the Japanese web corpus (ja-gen stage 1.1; rules in src/vimeml/corpus/ja_filter.py).

Same outputs and commands as the Chinese stage A (`scripts/zh/corpus/zh_stage_a.py`):

  run     classify every document of every shard -> <shard>.parquet (keep + review, full text),
          <shard>.dropped.parquet (no text), <shard>.json (counts; marks the shard complete).
          Completed shards are skipped, never overwritten. run.json records the inputs.
  dedup   exact duplicates (whitespace-insensitive) across shards, first occurrence kept in shard
          order -> duplicates.parquet (numpy over 16-byte hashes; the Chinese dict version needs
          too much memory for about 50M documents)
  report  totals, spam hosts and spam farms -> report.json, spam-hosts.json
  terms   pilot: hits per term over the finished shards, with up to three context snippets
          per term from the first shard -> term-pilot.json (stays local; it quotes web text)

The `han` column and `han_chars` counts hold Japanese characters (kana + kanji).

    python -X utf8 scripts/ja/corpus/ja_stage_a.py run [--shards 2] [--only 000_00004,000_00012]
"""
import argparse
import collections
import json
import os
import sys
import tomllib
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/zh/corpus'))
from vimeml.corpus.ja_filter import Thresholds, classify, load_rules  # noqa: E402
from zh_stage_a import DROP_COLUMNS, KEEP_COLUMNS, cmd_report, load_config, sha256, text_sha  # noqa: E402

CONFIG = ROOT / 'configs/ja/corpus-ja-stage-a.toml'


def process_shard(job):
    path, out_dir, terms, th, batch_rows = job
    import pyarrow as pa
    import pyarrow.parquet as pq
    name = Path(path).stem
    done = out_dir / f'{name}.json'
    if done.exists():
        return json.loads(done.read_text(encoding='utf-8'))
    rules = load_rules(terms)
    th = Thresholds(**th)
    tmp_keep = out_dir / f'{name}.parquet.tmp'
    tmp_drop = out_dir / f'{name}.dropped.parquet.tmp'
    writers = {}
    keep, drop = collections.defaultdict(list), collections.defaultdict(list)
    counts, reasons, ja_by = collections.Counter(), collections.Counter(), collections.Counter()

    def flush(force=False):
        for rows, columns, tmp in ((keep, KEEP_COLUMNS, tmp_keep), (drop, DROP_COLUMNS, tmp_drop)):
            if rows['id'] and (force or len(rows['id']) >= batch_rows):
                table = pa.table({c: rows[c] for c in columns})
                if tmp not in writers:
                    writers[tmp] = pq.ParquetWriter(tmp, table.schema, compression='zstd')
                writers[tmp].write_table(table)
                rows.clear()

    cols = ['text', 'id', 'dump', 'url', 'date', 'language_score', 'minhash_cluster_size']
    # One row group at a time: iter_batches pre-buffers and holds the column data of the whole
    # file until iteration ends (several GB per worker on a 4.5 GB shard).
    source = pq.ParquetFile(path)
    for group in range(source.metadata.num_row_groups):
        for r in source.read_row_group(group, columns=cols).to_pylist():
            host = (urlsplit(r['url']).hostname or '').lower()
            c = classify(r['text'], host, rules, th)
            counts[c['decision']] += 1
            ja_by[c['decision']] += c['han']
            reasons.update(c['reasons'])
            if c['decision'] == 'drop':
                for k, v in (('id', r['id']), ('host', host), ('dump', r['dump']), ('reasons', c['reasons']),
                             ('hits', c['hits']), ('han', c['han'])):
                    drop[k].append(v)
            else:
                row = dict(r, host=host, script=c['script'], decision=c['decision'], reasons=c['reasons'],
                           hits=c['hits'], han=c['han'], text=c['text'], text_sha=text_sha(c['text']))
                for k in KEEP_COLUMNS:
                    keep[k].append(row[k])
        flush()
    flush(force=True)
    for w in writers.values():
        w.close()
    for tmp in (tmp_keep, tmp_drop):
        if tmp.exists():
            os.replace(tmp, tmp.with_suffix(''))
    stats = dict(shard=name, documents=sum(counts.values()), decisions=dict(counts), han_chars=dict(ja_by),
                 reasons=dict(reasons), scripts={f'{d}/ja': n for d, n in sorted(counts.items())})
    done.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding='utf-8')
    return stats


def cmd_run(args):
    cfg, out = load_config(args.config)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / cfg['input']['manifest']).read_text(encoding='utf-8'))
    files = [ROOT / cfg['input']['dir'] / f['local'] for f in manifest['files']]
    if args.shards:
        files = files[:args.shards]
    if args.only:
        wanted = args.only.split(',')
        files = [f for f in files if f.stem in wanted]
        if len(files) != len(wanted):
            raise SystemExit(f'--only names {len(wanted)} shards, {len(files)} found in the manifest')
    terms = ROOT / cfg['rules']['terms']
    record = dict(config_sha256=sha256(args.config), terms_sha256=sha256(terms),
                  filter_sha256=sha256(ROOT / 'src/vimeml/corpus/ja_filter.py'),
                  shared_filter_sha256=sha256(ROOT / 'src/vimeml/corpus/zh_filter.py'),
                  script_sha256=sha256(__file__), input_manifest_sha256=sha256(ROOT / cfg['input']['manifest']),
                  thresholds=cfg['thresholds'])
    run_path = out / 'run.json'
    if run_path.exists():
        old = json.loads(run_path.read_text(encoding='utf-8'))
        changed = [k for k in record if old.get(k) != record[k]]
        if changed:
            raise SystemExit(f'{out} was produced with different {changed}; use a new output dir')
    run_path.write_text(json.dumps(dict(record, config=str(Path(args.config))), indent=1), encoding='utf-8')
    jobs = [(f, out, terms, cfg['thresholds'], cfg['run']['batch_rows']) for f in files]
    with ProcessPoolExecutor(args.workers or cfg['run']['workers']) as ex:
        for s in ex.map(process_shard, jobs):
            print(s['shard'], s['documents'], s['decisions'], flush=True)


def cmd_dedup(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    _, out = load_config(args.config)
    ids, shards, names, hashes = [], [], [], []
    for p in sorted(out.glob('*.parquet')):
        if p.name.endswith('.dropped.parquet') or p.name == 'duplicates.parquet':
            continue
        t = pq.read_table(p, columns=['id', 'text_sha'])
        ids.append(t.column('id').combine_chunks().cast(pa.large_string()))
        shards.append(np.full(len(t), len(names), dtype=np.int32))
        names.append(p.stem)
        hashes.append(np.array([bytes.fromhex(h) for h in t.column('text_sha').to_pylist()], dtype='S16'))
    ids = pa.chunked_array(ids).combine_chunks()
    shards, hashes = np.concatenate(shards), np.concatenate(hashes)
    _, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    first = first[inverse.reshape(-1)]
    dup = np.flatnonzero(first != np.arange(len(first)))
    pq.write_table(pa.table(dict(id=ids.take(pa.array(dup)).cast(pa.string()),
                                 shard=pa.array([names[c] for c in shards[dup].tolist()], pa.string()),
                                 first_id=ids.take(pa.array(first[dup])).cast(pa.string()))),
                   out / 'duplicates.parquet')
    print(f'{len(hashes) - len(dup)} unique, {len(dup)} exact duplicates', flush=True)


def cmd_terms(args):
    import pyarrow.parquet as pq
    cfg, out = load_config(args.config)
    rules = load_rules(ROOT / cfg['rules']['terms'])
    hits, docs = collections.Counter(), 0
    for p in sorted(out.glob('*.parquet')):
        if p.name == 'duplicates.parquet':
            continue
        t = pq.read_table(p, columns=['hits'])
        docs += len(t)
        for hs in t.column('hits').to_pylist():
            hits.update(hs)
    first = sorted(p for p in out.glob('*.json') if p.stem[:1].isdigit())[0].stem
    source = ROOT / cfg['input']['dir'] / f'{first}.parquet'
    samples = collections.defaultdict(list)
    pf = pq.ParquetFile(source)
    for group in range(pf.metadata.num_row_groups):
        batch = pf.read_row_group(group, columns=['text', 'url'])
        for text, url in zip(batch.column('text').to_pylist(), batch.column('url').to_pylist()):
            low = unicodedata.normalize('NFKC', text).lower()
            for m in rules.pattern.finditer(low):
                term = m.group(0)
                if len(samples[term]) < 3:
                    samples[term].append(dict(host=urlsplit(url).hostname,
                                              context=low[max(0, m.start() - 30):m.end() + 30].replace('\n', ' ')))
        if all(len(samples[t]) >= 3 for t in rules.category):
            break
    result = dict(documents=docs, terms={t: dict(category=rules.category[t], weak=t in rules.weak, hits=hits[t],
                                                 samples=samples.get(t, []))
                                         for t in sorted(rules.category, key=lambda t: -hits[t])})
    (out / 'term-pilot.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    for t, v in result['terms'].items():
        print(f'{v["hits"]:>7} {v["category"]:<9} {"weak " if v["weak"] else ""}{t}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['run', 'dedup', 'report', 'terms'])
    p.add_argument('--config', default=str(CONFIG))
    p.add_argument('--workers', type=int)
    p.add_argument('--shards', type=int, help='run: only the first N shards of the manifest')
    p.add_argument('--only', help='run: only these shards (comma-separated names, e.g. 000_00004); '
                                  'used to split one run between machines')
    p.add_argument('--against', help=argparse.SUPPRESS)
    p.add_argument('--min-host-docs', type=int, default=20)
    p.add_argument('--host-term-share', type=float, default=0.3)
    p.add_argument('--farm-hosts', type=int, default=10)
    a = p.parse_args()
    dict(run=cmd_run, dedup=cmd_dedup, report=cmd_report, terms=cmd_terms)[a.command](a)


if __name__ == '__main__':
    main()
