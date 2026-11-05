"""Pilot the Chinese content filter on a row-group sample of the cmn_Hani shards.

Samples PER_FILE evenly spaced row groups from every shard (about 1,000 documents
each), matches configs/zh-filter-terms.toml against the text (after t2s, so
traditional pages hit the same simplified terms), and writes to the output folder:

  pilot-report.json   overall and per-category hit rates, per-term document counts,
                      per-TLD and per-domain hit shares (domains with >= 20 sampled docs)
  pilot-review.md     per category, random hits with +-40 characters of context, and
                      the domains whose sampled pages hit most often

Nothing is filtered here; the report is for reviewing the lists.
"""
import argparse
import collections
import json
from pathlib import Path
import random
import tomllib
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[2]


def load_terms(path):
    spec=tomllib.loads(Path(path).read_text(encoding='utf-8'))
    terms={c:[t.lower() for t in ts] for c,ts in spec['categories'].items()}
    weak={t.lower() for t in spec.get('rules',{}).get('weak',[])}
    return terms,spec.get('domains',{}),weak


def counts_as_hit(hits,weak):
    """A document hits when it contains a non-weak term, or two different terms."""
    found={t for ts in hits.values() for t in ts}
    return bool(found-weak) or len(found)>=2


def scan(args):
    path,per_file,terms_path,seed=args
    import opencc
    import pyarrow.parquet as pq
    terms,_,_=load_terms(terms_path)
    t2s=opencc.OpenCC('t2s'); rng=random.Random(f'{seed}:{path.name}')
    f=pq.ParquetFile(path); n=f.metadata.num_row_groups
    groups=sorted(set(round(i*(n-1)/(per_file-1)) for i in range(per_file)))
    out=[]
    for g in groups:
        for r in f.read_row_group(g,columns=['text','url','id']).to_pylist():
            text=t2s.convert(r['text']).lower()
            hits={}
            for cat,ts in terms.items():
                found=[t for t in ts if t in text]
                if found: hits[cat]=found
            snippet={}
            for cat,found in hits.items():
                t=rng.choice(found); i=text.find(t)
                snippet[cat]=(t,text[max(0,i-40):i+len(t)+40].replace('\n',' '))
            out.append((urlsplit(r['url']).hostname or '',hits,snippet,r['id']))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',default=str(ROOT/'datasets/fineweb-2-cmn-hani'))
    p.add_argument('--terms',default=str(ROOT/'configs/zh-filter-terms.toml'))
    p.add_argument('--out',default=str(ROOT/'outputs/zh-corpus/term-pilot'))
    p.add_argument('--per-file',type=int,default=20)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    terms,domains,weak=load_terms(a.terms)
    files=sorted(Path(a.data).glob('*.parquet'))
    with ProcessPoolExecutor(a.workers) as ex:
        docs=[d for part in ex.map(scan,[(f,a.per_file,a.terms,a.seed) for f in files]) for d in part]
    N=len(docs); cats=list(terms)
    tlds=set(domains.get('block_tlds',[]))
    def tld(h): return h.rsplit('.',1)[-1]
    term_docs=collections.Counter(); cat_docs=collections.Counter()
    host_n=collections.Counter(); host_hit=collections.Counter()
    tld_n=collections.Counter(); tld_hit=collections.Counter()
    any_hit=0; tld_blocked=0; tld_or_hit=0; weak_only=0
    tiers={'block':domains.get('block',[]),'block_tier2':domains.get('block_tier2',[])}
    def tier(h):
        for name,ds in tiers.items():
            if any(h==d or h.endswith('.'+d) for d in ds): return name
        return None
    tier_n=collections.Counter()
    docs=[(h,hits if counts_as_hit(hits,weak) else {},s,i,bool(hits) and not counts_as_hit(hits,weak))
          for h,hits,s,i in docs]
    for host,hits,_,_,spared in docs:
        if spared:
            weak_only+=1
    for host,hits,_,_,_ in docs:
        host_n[host]+=1; tld_n[tld(host)]+=1
        t=tier(host)
        if t: tier_n[t]+=1
        tld_blocked+=tld(host) in tlds
        blocked=tld(host) in tlds or t is not None
        if hits:
            any_hit+=1; host_hit[host]+=1; tld_hit[tld(host)]+=1
            for c,found in hits.items():
                cat_docs[c]+=1; term_docs.update(found)
        tld_or_hit+=bool(hits) or blocked
    pct=lambda x:round(100*x/N,3)
    report=dict(documents=N,files=len(files),per_file=a.per_file,
                any_hit_pct=pct(any_hit),weak_only_spared_pct=pct(weak_only),blocked_tld_pct=pct(tld_blocked),
                blocked_domain_pct={k:pct(v) for k,v in sorted(tier_n.items())},dropped_total_pct=pct(tld_or_hit),
                categories={c:dict(docs=cat_docs[c],pct=pct(cat_docs[c])) for c in cats},
                terms={c:sorted(((t,term_docs[t]) for t in terms[c]),key=lambda x:-x[1]) for c in cats},
                tlds=[(t,n,round(100*tld_hit[t]/n,1)) for t,n in tld_n.most_common(25)],
                domains=sorted(((h,n,host_hit[h],round(100*host_hit[h]/n,1)) for h,n in host_n.items() if n>=20 and host_hit[h]),
                               key=lambda x:(-x[2],-x[3]))[:200])
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/'pilot-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=1),encoding='utf-8')
    rng=random.Random(a.seed)
    with open(out/'pilot-review.md','w',encoding='utf-8') as fh:
        fh.write(f'# 合规词表 pilot\n\n抽样 {N} 篇，任一类命中 {report["any_hit_pct"]}%，'
                 f'只含一个弱词而保留 {report["weak_only_spared_pct"]}%，屏蔽顶级域名 {report["blocked_tld_pct"]}%，'
                 f'屏蔽域名 {report["blocked_domain_pct"]}，合计丢弃 {report["dropped_total_pct"]}%。\n\n')
        for c in cats:
            fh.write(f'## {c}：{cat_docs[c]} 篇（{pct(cat_docs[c])}%）\n\n')
            fh.write('命中最多的词：'+'，'.join(f'{t} {n}' for t,n in report['terms'][c][:15] if n)+'\n\n')
            pool=[(h,s[c]) for h,hits,s,_,_ in docs if c in hits]
            for h,(t,ctx) in rng.sample(pool,min(40,len(pool))):
                fh.write(f'- `{t}` · {h} · …{ctx}…\n')
            fh.write('\n')
        spared=[(h,t,ctx) for h,_,s,_,keep in docs if keep for t,ctx in s.values()]
        fh.write(f'## 只含一个弱词、予以保留：{weak_only} 篇\n\n')
        for h,t,ctx in rng.sample(spared,min(40,len(spared))):
            fh.write(f'- `{t}` · {h} · …{ctx}…\n')
        fh.write('\n')
        fh.write('## 命中最多的域名（抽样中至少 20 篇）\n\n| 域名 | 抽样篇数 | 命中篇数 | 命中率 |\n| --- | ---: | ---: | ---: |\n')
        for h,n,k,r in report['domains'][:60]:
            fh.write(f'| {h} | {n} | {k} | {r}% |\n')
    print(json.dumps({k:report[k] for k in ('documents','any_hit_pct','weak_only_spared_pct','blocked_tld_pct','blocked_domain_pct','dropped_total_pct','categories')},ensure_ascii=False,indent=1))


if __name__=='__main__':
    main()
