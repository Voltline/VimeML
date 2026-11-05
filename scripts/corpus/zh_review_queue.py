"""Human review queue for the Chinese web corpus, served by templates/zh-review-site.

  export  every stage-A `review` document (weak_term, domain_tier2), minus exact duplicates
          and minus documents of spam hosts and spam farms (spam-hosts.json from
          `zh_stage_a.py report`; reviewers found them uniformly junk, so they are dropped,
          not reviewed), shuffled with a fixed seed and cut into batches:
            <out>/batches/NNNN.json   {batch, source, items:[{id,host,url,reasons,hits,text}]}
            <out>/index.json          per-batch counts by reason; the queue's SHA-256s
          Every candidate is exported; nothing is sampled away.
  merge   label files from the static page (labels-<batch>-<who>.json) and/or the review
          server's labels.jsonl export -> <out>/labels.jsonl (one line per id x reviewer)
          and agreement counts
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vimeml.corpus.zh_filter import registrable  # noqa: E402
LABELS={'keep','drop','unsure'}


def stage_dir(config):
    cfg=tomllib.loads(Path(config).read_text(encoding='utf-8'))
    return ROOT/cfg['output']['dir']


def cmd_export(a):
    """Three streaming passes, so memory stays small however large the queue is:
    1. per shard, read id/host/decision/reasons (no text) and pick the candidates;
    2. order them (sort by id, seeded shuffle) and give each a (batch, slot);
    3. per shard, read only the candidates' text and append it to the batch part files;
    then each batch is sorted by slot and written. Built in <out>.building, renamed at the end.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shutil
    src=stage_dir(a.config); out=Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} already exists; use a new --out')
    tmp=out.with_name(out.name+'.building')
    if tmp.exists(): shutil.rmtree(tmp)  # an interrupted export of this script
    (tmp/'parts').mkdir(parents=True); (tmp/'batches').mkdir()
    dup=set(pq.read_table(src/'duplicates.parquet',columns=['id']).column('id').to_pylist())
    spec=json.loads((src/'spam-hosts.json').read_text(encoding='utf-8'))
    spam={h['host'] for h in spec['hosts']}; farms={f['domain'] for f in spec['farms']}
    shards=[p for p in sorted(src.glob('*.parquet'))
            if not p.name.endswith('.dropped.parquet') and p.name!='duplicates.parquet']
    cand=[]; skipped=0  # (id, shard index, row, reasons)
    for k,p in enumerate(shards):
        t=pq.read_table(p,columns=['id','host','decision','reasons']).to_pydict()
        for row,(i,h,d,rs) in enumerate(zip(t['id'],t['host'],t['decision'],t['reasons'])):
            if i in dup: continue
            if d!='review': continue
            if h in spam or registrable(h) in farms: skipped+=1; continue
            cand.append((i,k,row,list(rs)))
        print(f'pass 1 {p.stem}: {len(cand)} candidates so far',flush=True)
        del t
    cand.sort(key=lambda x:x[0])
    random.Random(a.seed).shuffle(cand)
    where=[{} for _ in shards]  # shard -> row -> (position, reasons)
    for pos,(i,k,row,reasons) in enumerate(cand):
        where[k][row]=(pos,reasons)
    n_items=len(cand); total=collections.Counter(r for c in cand for r in c[3]); del cand
    for k,p in enumerate(shards):
        f=pq.ParquetFile(p); base=0; parts=collections.defaultdict(list)
        for b in f.iter_batches(batch_size=8192,columns=['id','host','url','hits','text']):
            local=[r-base for r in range(base,base+b.num_rows) if r in where[k]]
            if local:
                for r,x in zip(local,b.take(pa.array(local)).to_pylist()):
                    pos,reasons=where[k][base+r]
                    item=dict(id=x['id'],host=x['host'],url=x['url'],reasons=reasons,hits=list(x['hits']),text=x['text'])
                    parts[pos//a.batch_size].append(f'{pos}\t'+json.dumps(item,ensure_ascii=False))
            base+=b.num_rows
        for bi,lines in parts.items():
            with open(tmp/'parts'/f'{bi:05d}.tsv','a',encoding='utf-8') as fh: fh.write('\n'.join(lines)+'\n')
        where[k]=None
        print(f'pass 2 {p.stem}: {sum(map(len,parts.values()))} items written',flush=True)
    index=[]
    for bi in range((n_items+a.batch_size-1)//a.batch_size):
        name=f'{bi+1:04d}'; part=tmp/'parts'/f'{bi:05d}.tsv'
        # split on \n only: splitlines() would also cut at U+2028 etc., which JSON leaves raw
        lines=[l.split('\t',1) for l in part.read_text(encoding='utf-8').split('\n') if l]
        chunk=[json.loads(j) for _,j in sorted(lines,key=lambda x:int(x[0]))]
        body=json.dumps(dict(batch=name,source=str(src.relative_to(ROOT)),items=chunk),ensure_ascii=False)
        (tmp/'batches'/f'{name}.json').write_text(body,encoding='utf-8'); part.unlink()
        index.append(dict(batch=name,items=len(chunk),sha256=hashlib.sha256(body.encode('utf-8')).hexdigest(),
                          reasons=dict(collections.Counter(r for it in chunk for r in it['reasons']))))
    assert sum(b['items'] for b in index)==n_items
    (tmp/'parts').rmdir()
    (tmp/'index.json').write_text(json.dumps(dict(source=str(src.relative_to(ROOT)),seed=a.seed,batch_size=a.batch_size,
        items=n_items,skipped_spam_review_docs=skipped,reasons=dict(total),batches=index),ensure_ascii=False,indent=1),encoding='utf-8')
    tmp.rename(out)
    print(f'{n_items} items in {len(index)} batches; reasons {dict(total)}')


def cmd_merge(a):
    out=Path(a.out); index=json.loads((out/'index.json').read_text(encoding='utf-8'))
    known={b['batch'] for b in index['batches']}
    rows=[]; by_id=collections.defaultdict(dict)
    def add(name,batch,who,i,label,t):
        if batch not in known: raise ValueError(f'{name}: unknown batch {batch}')
        if label not in LABELS: raise ValueError(f'{name}: bad label {label}')
        who=who.strip(); by_id[i][who]=label
        rows.append(dict(id=i,batch=batch,reviewer=who,label=label,t=t,file=name))
    for p in sorted(Path(a.labels).glob('*.json')):  # static prototype page, one batch per file
        d=json.loads(p.read_text(encoding='utf-8'))
        for i,v in d['labels'].items(): add(p.name,d.get('batch'),d['reviewer'],i,v['label'],v.get('t'))
    for p in sorted(Path(a.labels).glob('*.jsonl')):  # review server export, one row per id x reviewer
        for line in p.read_text(encoding='utf-8').split('\n'):
            if not line.strip(): continue
            r=json.loads(line)
            if r.get('queue') not in (None,out.name): raise ValueError(f'{p.name}: row from queue {r["queue"]}')
            add(p.name,r['batch'],r['reviewer'],r['id'],r['label'],r.get('t'))
    with open(out/'labels.jsonl','w',encoding='utf-8') as fh:
        for r in rows: fh.write(json.dumps(r,ensure_ascii=False)+'\n')
    multi=[v for v in by_id.values() if len(v)>1]
    agree=sum(len(set(v.values()))==1 for v in multi)
    binary=sum(len({x=='keep' for x in v.values() if x!='unsure'})<=1 for v in multi)
    print(json.dumps(dict(labels=len(rows),ids=len(by_id),reviewers=len({r['reviewer'] for r in rows}),
        label_counts=dict(collections.Counter(r['label'] for r in rows)),double_labeled=len(multi),
        exact_agreement=agree,keep_vs_drop_agreement=binary),ensure_ascii=False,indent=1))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',choices=['export','merge'])
    p.add_argument('--config',default=str(ROOT/'configs/corpus-zh-stage-a.toml'))
    p.add_argument('--out',default=str(ROOT/'outputs/zh-corpus/review-v1'))
    p.add_argument('--labels',help='merge: folder of label files (*.json from the page, *.jsonl from the server)')
    p.add_argument('--batch-size',type=int,default=200)
    p.add_argument('--seed',type=int,default=20261105)
    a=p.parse_args()
    dict(export=cmd_export,merge=cmd_merge)[a.command](a)


if __name__=='__main__':
    main()
