"""Screen the Standard zh benchmark items for verbatim overlap with the vimeml-tiny-zh-v1 training corpus.

usage: screen_overlap_zh.py <items.jsonl> <out dir> <corpus file>...
  corpus files: stage-a parquet shards (text column; every row of a non-dropped shard, so both
  keep and review rows are screened) and LCCC jsonl files (screened line by line as raw text).

Two patterns per item, matched as exact substrings (Aho-Corasick):
  run      the maximal Han run of the message that contains the target, if >= MIN_LEN chars;
           robust to punctuation differences around it
  message  the whole message, if >= MIN_LEN chars
Shorter runs are generic phrases (苏州旅游) and are not screened. An item may name its own source document
("own_doc", e.g. an export-r1 heldout paragraph's document id): in parquet files with an id column, hits in
that document are not counted for that item. Writes <out>/overlap.jsonl
(one row per item with hits, counts per corpus file) and <out>/overlap-summary.json; never
overwrites. Corpus files are data and only read; run with python -I.
"""
import collections
import json
from multiprocessing import Pool
from pathlib import Path
import re
import sys

import ahocorasick

MIN_LEN=8
HAN=re.compile(r'[〇㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f]+')


def patterns(items):
    pats=collections.defaultdict(set)  # pattern -> {(id, kind)}
    for it in items:
        s,e=it['span']; msg=it['message']
        for m in HAN.finditer(msg):
            if m.start()<=s and e<=m.end() and len(m.group())>=MIN_LEN: pats[m.group()].add((it['id'],'run'))
        if len(msg)>=MIN_LEN: pats[msg].add((it['id'],'message'))
    return pats


def texts(path):
    """(document id or None, text)"""
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        pf=pq.ParquetFile(path)
        names=pf.schema_arrow.names
        if 'text' not in names: raise SystemExit(f'{path} has no text column')
        cols=['text']+(['id'] if 'id' in names else [])
        for batch in pf.iter_batches(batch_size=20000,columns=cols):
            ids=batch.column(1).to_pylist() if len(cols)==2 else [None]*batch.num_rows
            yield from zip(ids,batch.column(0).to_pylist())
    else:
        with open(path,encoding='utf-8') as fh:
            for line in fh: yield None,line


def scan(job):
    path,pats,own=job
    A=ahocorasick.Automaton()
    for p in pats: A.add_word(p,p)
    A.make_automaton()
    hits=collections.Counter(); own_hits=collections.Counter(); docs=0
    for doc,t in texts(path):
        docs+=1
        for p in {v for _,v in A.iter(t)}:
            hits[p]+=1   # documents containing the pattern
            if doc is not None and doc in own: own_hits[p,doc]+=1
    return path,docs,hits,own_hits


def main():
    items_path,out=sys.argv[1],Path(sys.argv[2]); corpus=sys.argv[3:]
    if (out/'overlap.jsonl').exists(): raise SystemExit(f'{out}/overlap.jsonl exists; use a new out dir')
    items=[json.loads(l) for l in open(items_path,encoding='utf-8') if l.strip()]
    pats=patterns(items); plist=sorted(pats)
    own_of={it['id']:it['own_doc'] for it in items if it.get('own_doc')}
    own=set(own_of.values())
    per_item=collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    total_docs={}
    with Pool(min(len(corpus),12)) as pool:
        for path,docs,hits,own_hits in pool.imap_unordered(scan,[(c,plist,own) for c in corpus]):
            total_docs[path]=docs; print(f'{Path(path).name}: {docs} docs, {len(hits)} patterns hit',flush=True)
            for p,n in hits.items():
                for i,kind in pats[p]:
                    n_other=n-own_hits.get((p,own_of.get(i)),0)
                    if n_other>0: per_item[i][kind][Path(path).name]+=n_other
    out.mkdir(parents=True,exist_ok=True)
    by_id={it['id']:it for it in items}
    with open(out/'overlap.jsonl','w',encoding='utf-8') as fh:
        for i in sorted(per_item):
            it=by_id[i]
            fh.write(json.dumps(dict(id=i,source=it['source'],split=it['split'],message=it['message'],
                target=it['target'],hits={k:dict(v) for k,v in per_item[i].items()}),ensure_ascii=False)+'\n')
    screened=collections.Counter((by_id[i]['split'],k) for i,k in set().union(*pats.values()))
    flagged=collections.Counter((by_id[i]['split'],k) for i,v in per_item.items() for k in v)
    summary=dict(items=len(items),min_len=MIN_LEN,corpus_docs=total_docs,
        screened={f'{s}/{k}':n for (s,k),n in sorted(screened.items())},
        flagged={f'{s}/{k}':n for (s,k),n in sorted(flagged.items())},
        flagged_items=len(per_item))
    (out/'overlap-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=1))


if __name__=='__main__':
    main()
