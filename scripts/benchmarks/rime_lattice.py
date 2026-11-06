"""Whole-input candidate pools for the Standard Chinese IME benchmark: Rime first, then a
word lattice over the rime-ice dictionary.

For a key sequence Rime's sentence maker returns one whole-input sentence; its other
candidates cover only a prefix of the input. So the pool for an item is
  1. Rime's whole-input candidates, in Rime order (rime_candidates.py query), then
  2. the k best whole-input paths of a lattice built from the same rime-ice dictionaries
     (cn_dicts/8105, base, ext, tencent, others, as imported by rime_ice.dict.yaml),
deduplicated, up to --n. Every candidate records its origin, so the Rime-only recall and the
pool recall are reported separately, and the order without an LM is Rime's.

Lattice model: unigram over dictionary words, score = sum log(weight / total weight). Words
without a code (the tencent list) get codes the way librime encodes phrases: every
combination of their characters' readings, using only readings with at least 5% of the
character's total weight (librime's kMinimalWeightProportionForWordMaking; 了 liao is not
used for 了解 this way, 长 chang and zhang both are). Entries without a weight (others: extra
readings of known words) take the word's highest weight elsewhere, else 1.

  index  build <work>/lattice-index.pkl from the rime-ice checkout
  pool   <rime-input.jsonl> <rime-candidates.jsonl> <out.jsonl> --track full|abbrev
The full track splits keys into syllables (every split is tried: xian = xian | xi an); the
abbreviated track reads each key letter as the first letter of one syllable.
"""
import argparse
import collections
import heapq
import itertools
import json
import math
from pathlib import Path
import pickle

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'datasets/ime-standard-zh-v1/engine'
TABLES=['cn_dicts/8105','cn_dicts/base','cn_dicts/ext','cn_dicts/tencent','cn_dicts/others']
MIN_PROPORTION=0.05
PER_CODE=40      # words kept per code (by weight)
MAX_WORD=8       # syllables per dictionary word in the lattice
ABBREV_MAX_WORD=4


def entries(path):
    body=False
    for line in path.read_text(encoding='utf-8').split('\n'):
        if not body:
            body=line.strip()=='...'; continue
        if not line or line.startswith('#'): continue
        yield line.rstrip('\r').split('\t')


def build_index(shared):
    shared=Path(shared); rows=[]
    for t in TABLES:
        for parts in entries(shared/f'{t}.dict.yaml'):
            text=parts[0]
            code=parts[1] if len(parts)>=2 and not parts[1].isdigit() else None
            w=parts[-1] if len(parts)>=2 and parts[-1].isdigit() else None
            rows.append((t,text,code,int(w) if w is not None else None))
    char_codes=collections.defaultdict(collections.Counter)
    for t,text,code,w in rows:
        if t=='cn_dicts/8105' and code and len(text)==1: char_codes[text][code]+=w or 0
    usable={c:[k for k,v in cs.items() if v>=MIN_PROPORTION*sum(cs.values())] for c,cs in char_codes.items()}
    best_weight=collections.Counter()
    for t,text,code,w in rows:
        if w is not None: best_weight[text]=max(best_weight[text],w)
    table=collections.defaultdict(dict); unencodable=0
    for t,text,code,w in rows:
        w=max(1,w if w is not None else best_weight.get(text,1))  # some 8105 rare chars have weight 0
        if code: codes=[tuple(code.split())]
        elif all(c in usable for c in text):
            codes=list(itertools.islice(itertools.product(*(usable[c] for c in text)),64))
        else:
            unencodable+=1; continue
        for cd in codes:
            if len(cd)!=len(text): continue  # letters, digits or erhua spellings
            d=table[cd]; d[text]=max(d.get(text,0),w)
    total=sum(max(d.values()) for d in table.values())
    full={}; abbrev=collections.defaultdict(dict)
    for cd,d in table.items():
        top=heapq.nlargest(PER_CODE,d.items(),key=lambda x:x[1])
        full[' '.join(cd)]=[(text,math.log(w/total)) for text,w in top]
        if len(cd)<=ABBREV_MAX_WORD:
            a=abbrev[''.join(s[0] for s in cd)]
            for text,w in top: a[text]=max(a.get(text,0),w)
    abbrev={k:[(text,math.log(w/total)) for text,w in heapq.nlargest(PER_CODE,d.items(),key=lambda x:x[1])]
            for k,d in abbrev.items()}
    syllables=sorted({s for cd in table for s in cd if s.isalpha() and s.isascii()})
    return dict(full=full,abbrev=abbrev,syllables=syllables,
                stats=dict(rows=len(rows),codes=len(table),unencodable=unencodable,syllables=len(syllables)))


class Lattice:
    def __init__(self,index):
        self.full,self.abbrev=index['full'],index['abbrev']
        self.syl=set(index['syllables']); self.maxlen=max(map(len,self.syl))

    def edges_full(self,keys):
        """(start, end, text, logp): words whose syllables spell keys[start:end]."""
        n=len(keys); out=[]
        def walk(i,j,syls):
            if syls:
                for text,lp in self.full.get(' '.join(syls),()): out.append((i,j,text,lp))
            if len(syls)==MAX_WORD: return
            for k in range(j+1,min(n,j+self.maxlen)+1):
                if keys[j:k] in self.syl: walk(i,k,syls+[keys[j:k]])
        for i in range(n): walk(i,i,[])
        return out

    def edges_abbrev(self,keys):
        out=[]
        for i in range(len(keys)):
            for j in range(i+1,min(len(keys),i+ABBREV_MAX_WORD)+1):
                for text,lp in self.abbrev.get(keys[i:j],()): out.append((i,j,text,lp))
        return out

    def kbest(self,keys,track,k):
        edges=self.edges_full(keys) if track=='full' else self.edges_abbrev(keys)
        by_start=collections.defaultdict(list)
        for i,j,text,lp in edges: by_start[i].append((j,text,lp))
        best=collections.defaultdict(dict); best[0]['']=0.0  # position -> text -> best score
        for i in range(len(keys)):
            if not best[i]: continue
            here=heapq.nlargest(k,best[i].items(),key=lambda x:x[1])
            for j,text,lp in by_start[i]:
                d=best[j]
                for prefix,s in here:
                    t=prefix+text
                    if s+lp>d.get(t,-math.inf): d[t]=s+lp
        return [t for t,_ in heapq.nlargest(k,best[len(keys)].items(),key=lambda x:x[1])]


def cmd_index(a):
    out=Path(a.work)/'lattice-index.pkl'
    index=build_index(a.shared)
    out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,'wb') as fh: pickle.dump(index,fh,protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(dict(out=str(out),**index['stats'])))


def cmd_pool(a):
    with open(Path(a.work)/'lattice-index.pkl','rb') as fh: lat=Lattice(pickle.load(fh))
    rime={}
    for line in open(a.args[1],encoding='utf-8'):
        if line.strip(): r=json.loads(line); rime[r['id']]=r
    with open(a.args[0],encoding='utf-8') as fi, open(a.args[2],'w',encoding='utf-8') as fo:
        for line in fi:
            if not line.strip(): continue
            q=json.loads(line); r=rime[q['id']]; assert r['keys']==q['keys']
            pool=[dict(text=t,origin='rime') for t in r['candidates'][:a.n]]
            seen={c['text'] for c in pool}
            for t in lat.kbest(q['keys'],a.track,a.n):
                if len(pool)>=a.n: break
                if len(t)==q['n_han'] and t not in seen: pool.append(dict(text=t,origin='lattice')); seen.add(t)
            fo.write(json.dumps(dict(id=q['id'],keys=q['keys'],track=a.track,engine=r['engine'],candidates=pool),
                                ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',choices=['index','pool'])
    p.add_argument('args',nargs='*')
    p.add_argument('--shared',default=str(ENGINE/'rime-ice'))
    p.add_argument('--work',default=str(ROOT/'outputs/benchmarks/ime-standard-zh-v1/rime'))
    p.add_argument('--track',choices=['full','abbrev'],default='full')
    p.add_argument('--n',type=int,default=20)
    a=p.parse_args()
    dict(index=cmd_index,pool=cmd_pool)[a.command](a)


if __name__=='__main__':
    main()
