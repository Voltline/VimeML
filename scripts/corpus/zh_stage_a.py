"""Stage A of the Chinese web corpus (rules in src/vimeml/corpus/zh_filter.py).

  run     classify every document of every shard; per shard it writes
            <shard>.parquet          keep + review documents, full converted text
            <shard>.dropped.parquet  dropped documents: id, host, reasons, hits (no text)
            <shard>.json             counts; written last, marks the shard complete
          Completed shards are skipped, never overwritten. run.json records the inputs.
  dedup   exact duplicates (whitespace-insensitive) across shards, first occurrence kept
          in shard order -> duplicates.parquet
  report  totals, hosts whose pages mostly hit the term list, and spam farms (many
          hosts under one mixed letter-digit domain) -> report.json, spam-hosts.json.
          Sites in [domains] spam_allow of the terms file (checked by hand) are never listed.
          Every surviving document of a listed host or farm is dropped downstream.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vimeml.corpus.zh_filter import Thresholds,classify,in_domains,load_rules,random_looking,registrable  # noqa: E402

KEEP_COLUMNS=['id','url','host','dump','date','language_score','minhash_cluster_size','script',
              'decision','reasons','hits','han','text_sha','text']
DROP_COLUMNS=['id','host','dump','reasons','hits','han']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    cfg=tomllib.loads(Path(path).read_text(encoding='utf-8'))
    return cfg,ROOT/cfg['output']['dir']


def text_sha(text):
    return hashlib.blake2b(''.join(text.split()).encode('utf-8'),digest_size=16).hexdigest()


def process_shard(job):
    path,out_dir,terms,th,batch_rows=job
    import pyarrow as pa
    import pyarrow.parquet as pq
    name=Path(path).stem
    done=out_dir/f'{name}.json'
    if done.exists():
        return json.loads(done.read_text(encoding='utf-8'))
    rules=load_rules(terms); th=Thresholds(**th)
    f=pq.ParquetFile(path)
    tmp_keep=out_dir/f'{name}.parquet.tmp'; tmp_drop=out_dir/f'{name}.dropped.parquet.tmp'
    wk=wd=None
    keep=collections.defaultdict(list); drop=collections.defaultdict(list)
    counts=collections.Counter(); reasons=collections.Counter(); scripts=collections.Counter()
    han_by=collections.Counter()

    def flush(force=False):
        nonlocal wk,wd
        if keep['id'] and (force or len(keep['id'])>=batch_rows):
            t=pa.table({c:keep[c] for c in KEEP_COLUMNS})
            wk=wk or pq.ParquetWriter(tmp_keep,t.schema,compression='zstd')
            wk.write_table(t); keep.clear()
        if drop['id'] and (force or len(drop['id'])>=batch_rows):
            t=pa.table({c:drop[c] for c in DROP_COLUMNS})
            wd=wd or pq.ParquetWriter(tmp_drop,t.schema,compression='zstd')
            wd.write_table(t); drop.clear()

    cols=['text','id','dump','url','date','language_score','minhash_cluster_size']
    for batch in f.iter_batches(batch_size=4096,columns=cols):
        for r in batch.to_pylist():
            host=(urlsplit(r['url']).hostname or '').lower()
            c=classify(r['text'],host,rules,th)
            counts[c['decision']]+=1; scripts[(c['decision'],c['script'])]+=1
            han_by[c['decision']]+=c['han']
            reasons.update(c['reasons'])
            if c['decision']=='drop':
                for k,v in (('id',r['id']),('host',host),('dump',r['dump']),('reasons',c['reasons']),
                            ('hits',c['hits']),('han',c['han'])):
                    drop[k].append(v)
            else:
                row=dict(r,host=host,script=c['script'],decision=c['decision'],reasons=c['reasons'],
                         hits=c['hits'],han=c['han'],text=c['text'],text_sha=text_sha(c['text']))
                for k in KEEP_COLUMNS: keep[k].append(row[k])
        flush()
    flush(force=True)
    for w in (wk,wd):
        if w: w.close()
    for tmp in (tmp_keep,tmp_drop):
        if tmp.exists(): os.replace(tmp,tmp.with_suffix(''))
    stats=dict(shard=name,documents=sum(counts.values()),decisions=dict(counts),
               han_chars=dict(han_by),reasons=dict(reasons),
               scripts={f'{d}/{s}':n for (d,s),n in sorted(scripts.items())})
    done.write_text(json.dumps(stats,ensure_ascii=False,indent=1),encoding='utf-8')
    return stats


def cmd_run(args):
    cfg,out=load_config(args.config)
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/cfg['input']['manifest']).read_text(encoding='utf-8'))
    files=[ROOT/cfg['input']['dir']/f['local'] for f in manifest['files']]
    if args.shards: files=files[:args.shards]
    terms=ROOT/cfg['rules']['terms']
    run_path=out/'run.json'
    record=dict(config=str(Path(args.config)),config_sha256=sha256(args.config),terms_sha256=sha256(terms),
                filter_sha256=sha256(ROOT/'src/vimeml/corpus/zh_filter.py'),script_sha256=sha256(__file__),
                input_manifest_sha256=sha256(ROOT/cfg['input']['manifest']),
                opencc=__import__('opencc').__version__ if hasattr(__import__('opencc'),'__version__') else 'unknown',
                thresholds=cfg['thresholds'])
    if run_path.exists():
        old=json.loads(run_path.read_text(encoding='utf-8'))
        changed=[k for k in record if k!='config' and old.get(k)!=record[k]]
        if changed:
            raise SystemExit(f'{out} was produced with different {changed}; use a new output dir')
    run_path.write_text(json.dumps(record,indent=1),encoding='utf-8')
    jobs=[(f,out,terms,cfg['thresholds'],cfg['run']['batch_rows']) for f in files]
    with ProcessPoolExecutor(args.workers or cfg['run']['workers']) as ex:
        for s in ex.map(process_shard,jobs):
            print(s['shard'],s['documents'],s['decisions'],flush=True)


def cmd_dedup(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    _,out=load_config(args.config)
    seen={}; dup=collections.defaultdict(list)
    for p in sorted(out.glob('*.parquet')):
        if p.name.endswith('.dropped.parquet') or p.name=='duplicates.parquet': continue
        t=pq.read_table(p,columns=['id','text_sha'])
        for i,h in zip(t.column('id').to_pylist(),t.column('text_sha').to_pylist()):
            if h in seen:
                dup['id'].append(i); dup['shard'].append(p.stem); dup['first_id'].append(seen[h])
            else:
                seen[h]=i
    pq.write_table(pa.table({k:dup[k] for k in ('id','shard','first_id')}),out/'duplicates.parquet')
    print(f'{len(seen)} unique, {len(dup["id"])} exact duplicates',flush=True)


def cmd_report(args):
    import pyarrow.parquet as pq
    cfg,out=load_config(args.config)
    allow=tuple(tomllib.loads((ROOT/cfg['rules']['terms']).read_text(encoding='utf-8')).get('domains',{}).get('spam_allow',[]))
    stats=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(out.glob('*.json')) if p.name not in ('run.json','report.json','spam-hosts.json')]
    N=sum(s['documents'] for s in stats)
    tot=lambda key:sum((collections.Counter(s[key]) for s in stats),collections.Counter())
    decisions=tot('decisions'); reasons=tot('reasons'); han=tot('han_chars'); scripts=tot('scripts')
    host_n=collections.Counter(); host_term=collections.Counter(); host_keep=collections.Counter()
    for p in sorted(out.glob('*.dropped.parquet')):
        t=pq.read_table(p,columns=['host','reasons']).to_pydict()
        for h,rs in zip(t['host'],t['reasons']):
            host_n[h]+=1
            if any(r.startswith('term_') for r in rs): host_term[h]+=1
    for p in sorted(out.glob('*.parquet')):
        if p.name.endswith('.dropped.parquet') or p.name=='duplicates.parquet': continue
        for h in pq.read_table(p,columns=['host']).column('host').to_pylist():
            host_n[h]+=1; host_keep[h]+=1
    sites=collections.defaultdict(set); site_docs=collections.Counter(); site_keep=collections.Counter()
    for h,n in host_n.items():
        if in_domains(h,allow): continue
        r=registrable(h); sites[r].add(h); site_docs[r]+=n; site_keep[r]+=host_keep[h]
    farms=sorted(((r,len(hs),site_docs[r],site_keep[r]) for r,hs in sites.items()
                  if len(hs)>=args.farm_hosts and random_looking(r.split('.')[0])),key=lambda x:-x[3])
    dups=pq.read_metadata(out/'duplicates.parquet').num_rows if (out/'duplicates.parquet').exists() else None
    spam_hosts=sorted(((h,n,host_term[h],host_keep[h]) for h,n in host_n.items()
                       if n>=args.min_host_docs and host_term[h]/n>=args.host_term_share and not in_domains(h,allow)),key=lambda x:-x[3])
    report=dict(spam_allow=list(allow),shards=len(stats),documents=N,decisions=dict(decisions),
                decision_pct={k:round(100*v/N,2) for k,v in decisions.items()},
                han_chars_G={k:round(v/1e9,2) for k,v in han.items()},
                reasons={k:(v,round(100*v/N,2)) for k,v in reasons.most_common()},
                scripts=dict(scripts),exact_duplicates=dups,
                spam_hosts=dict(rule=f'>= {args.min_host_docs} docs and >= {args.host_term_share:.0%} term drops',
                                hosts=len(spam_hosts),kept_or_review_docs=sum(x[3] for x in spam_hosts),
                                top=spam_hosts[:200]),
                spam_farms=dict(rule=f'>= {args.farm_hosts} hosts under a mixed letter-digit domain',
                                farms=len(farms),kept_or_review_docs=sum(x[3] for x in farms),top=farms[:200]))
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=1),encoding='utf-8')
    (out/'spam-hosts.json').write_text(json.dumps(dict(rule=report['spam_hosts']['rule'],
        hosts=[dict(host=h,docs=n,term_drops=k,surviving=s) for h,n,k,s in spam_hosts],
        farm_rule=report['spam_farms']['rule'],
        farms=[dict(domain=r,hosts=k,docs=n,surviving=s) for r,k,n,s in farms]),ensure_ascii=False,indent=1),
        encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('spam_hosts','spam_farms')},ensure_ascii=False,indent=1))
    print('spam farms:',report['spam_farms']['farms'],'domains,',report['spam_farms']['kept_or_review_docs'],'surviving docs')
    print('spam hosts:',report['spam_hosts']['hosts'],'hosts,',report['spam_hosts']['kept_or_review_docs'],'surviving docs')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',choices=['run','dedup','report'])
    p.add_argument('--config',default=str(ROOT/'configs/corpus-zh-stage-a.toml'))
    p.add_argument('--workers',type=int)
    p.add_argument('--shards',type=int,help='run: only the first N shards of the manifest')
    p.add_argument('--min-host-docs',type=int,default=20)
    p.add_argument('--host-term-share',type=float,default=0.3)
    p.add_argument('--farm-hosts',type=int,default=10)
    a=p.parse_args()
    dict(run=cmd_run,dedup=cmd_dedup,report=cmd_report)[a.command](a)


if __name__=='__main__':
    main()
