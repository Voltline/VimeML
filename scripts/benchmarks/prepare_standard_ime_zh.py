"""Draft items for the Standard Chinese IME benchmark (ime-standard-zh-v1).

Mirrors the Japanese Standard protocol (docs/zh-CN/benchmarks/standard-ime.md):
  - sources: user turns of CrossWOZ and RiSAWOZ (task dialogue), KdConv messages without
    knowledge attributes (chit-chat); one item per dialogue;
  - the given left context is only the prefix of the current message (at most 64 chars);
  - the target is the end of a clause: a run of Han characters that starts at a jieba word
    boundary and ends at punctuation or the end of the message, 2-12 characters;
  - targets never overlap an annotated entity or slot value (CrossWOZ dialog_act and
    user_state values, RiSAWOZ user_actions and belief_state values, KdConv entity names and
    short attribute values); names left unannotated are for the expert review to remove;
  - a target does not start with a particle that attaches to its left (的, 了, 呢 ...);
    the same (left context, target) appears once, and one target at most 3 times per source;
  - readings come from pypinyin over the whole clause (phrase context); every reading is
    checked in the expert review. Keys are full pinyin with `v` for u-umlaut and, as a separate
    track for targets of 2-6 characters, abbreviated pinyin (first letter of each syllable);
  - dialogues are split dev/blind per source by a seeded shuffle; target length (2-3, 4-6,
    7-12) and with/without left context rotate inside each source. Nothing reads an LM score,
    a candidate list or candidate difficulty.

  extract  <raw dir> -> <out>/items.jsonl, <out>/manifest.json, <out>/rime-{full,abbrev}.jsonl
  freeze   --draft <draft dir> --review <review dir> [--human <human-review json>] --out <dir>
           applies the per-item review records (<review>/chunk-*.jsonl: keep | fix | drop, corrected
           syllables, acceptable alternative targets) and, when given, the human decisions that
           override them (keep | drop | ai, and the global reading of 谁); regenerates the keys and
           the Rime inputs. The review is labelled ai_expert_reviewed: no native-speaker pass yet,
           so the labels are not formal gold.
The raw directory is downloaded data and is only read; run with python -I.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import unicodedata
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HAN_CHAR=r'〇㐀-䶿一-鿿豈-﫿'
HAN_RUN=re.compile(f'[{HAN_CHAR}]+')
BAD=re.compile(r'https?://|www\.|@\w|<[a-z/]|&[a-z]+;|\[[A-Z_]+\]',re.I)
BUCKETS=((2,3),(4,6),(7,12))
LEADING_PARTICLES=set('的了呢吗吧啊呀哦哈嘛啦着过地得么呗哟喔噢嘞咯')
MAX_SAME_TARGET=3
ABBREV_MAX=6
MAX_CONTEXT=64


def norm(s):
    s=unicodedata.normalize('NFC',s)
    s=''.join(c for c in s if unicodedata.category(c)[0]!='C' or c in '\n\t')
    return ' '.join(s.split())


def protect_values(values):
    out=set()
    for v in values:
        if isinstance(v,(list,tuple)):
            out|=protect_values(v); continue
        if not isinstance(v,str): continue
        v=norm(v).replace(' ','')
        v=re.sub(r'（[^）]*）|\([^)]*\)','',v)
        if len(v)>=2 and HAN_RUN.search(v): out.add(v)
    return out


def load_crosswoz(raw):
    for split in ('train','val','test'):
        with zipfile.ZipFile(raw/'CrossWOZ/data/crosswoz'/f'{split}.json.zip') as z:
            data=json.loads(z.read(z.namelist()[0]))
        for did,d in data.items():
            msgs=[];prot=set()
            for i,m in enumerate(d['messages']):
                if m['role']!='usr': continue
                prot|=protect_values([a[3] for a in m.get('dialog_act',[])])
                prot|=protect_values([s[3] for s in m.get('user_state',[])])
                msgs.append((i,m['content']))
            yield dict(source='crosswoz',source_split=split,dialogue_id=str(did),messages=msgs,protected=prot)


def load_risawoz(raw):
    names={'all_train10000.json':'train','all_dev600.json':'dev','all_test600_new.json':'test'}
    with zipfile.ZipFile(raw/'RiSAWOZ/RiSAWOZ-data/source-data.zip') as z:
        for n,split in names.items():
            for d in json.loads(z.read('source-data/'+n)):
                msgs=[];prot=set()
                for t in d['dialogue']:
                    prot|=protect_values([a[3] for a in t.get('user_actions',[]) if len(a)>3])
                    prot|=protect_values([a[3] for a in t.get('system_actions',[]) if len(a)>3])
                    prot|=protect_values(list(t.get('belief_state',{}).get('inform slot-values',{}).values()))
                    msgs.append((t['turn_id'],t['user_utterance']))
                yield dict(source='risawoz',source_split=split,dialogue_id=d['dialogue_id'],messages=msgs,protected=prot)


def load_kdconv(raw):
    for domain in ('film','music','travel'):
        for split in ('train','dev','test'):
            data=json.loads((raw/'KdConv/data'/domain/f'{split}.json').read_text(encoding='utf-8'))
            for k,d in enumerate(data):
                prot=protect_values([d.get('name','')]); msgs=[]
                for i,m in enumerate(d['messages']):
                    for a in m.get('attrs',[]):
                        prot|=protect_values([a.get('name','')])
                        if len(a.get('attrvalue',''))<=15: prot|=protect_values([a['attrvalue']])
                    if not m.get('attrs'): msgs.append((i,m['message']))  # chit-chat, not quoting knowledge
                yield dict(source='kdconv',source_split=split,dialogue_id=f'{domain}/{split}/{k}',messages=msgs,protected=prot)


LOADERS=dict(crosswoz=load_crosswoz,risawoz=load_risawoz,kdconv=load_kdconv)


class Reader:
    def __init__(self):
        import jieba
        from pypinyin import Style, lazy_pinyin
        jieba.setLogLevel(60); jieba.initialize()
        from g2pM import G2pM
        self.jieba,self.Style,self.lazy,self.g2pm=jieba,Style,lazy_pinyin,G2pM()

    def boundaries(self,text):
        pos=0; out={0}
        for w in self.jieba.cut(text,HMM=False):
            pos+=len(w); out.add(pos)
        return out

    def readings(self,run):
        """Three readings of the clause: pypinyin with its own phrase segmentation (the draft
        reading), pypinyin over jieba words, and the g2pM polyphone model. Their errors differ
        (pypinyin: 周末了 -> 末了 liao, 多长 -> zhang; g2pM: 不了解 -> le, 行长 -> xing), so every
        position where any two disagree goes to the review with all readings."""
        own=self.lazy(run,style=self.Style.NORMAL,errors='ignore')
        seg=self.lazy(list(self.jieba.cut(run,HMM=False)),style=self.Style.NORMAL,errors='ignore')
        g2p=[x.replace('ü','v').replace('u:','v') for x in self.g2pm(run,tone=False,char_split=False)]
        ok=lambda s:len(s)==len(run) and all(x.isascii() and x.isalpha() for x in s)
        if not (ok(own) and ok(seg) and ok(g2p)): return None,None
        # draft = majority of the three, pypinyin on a three-way tie (it is right more often:
        # g2pM misreads 了解, 出差, 地址, 音乐); 嗯 is typed `en` (rime-ice erases bare `n`)
        draft=[collections.Counter(v).most_common(1)[0][0] if collections.Counter(v).most_common(1)[0][1]>1 else v[0]
               for v in zip(own,seg,g2p)]
        draft=['en' if c=='嗯' and x in ('n','ng') else x for c,x in zip(run,draft)]
        if any(x in ('n','ng','m','hm','hng') for x in draft): return None,None
        return draft,[own,seg,g2p]


def spans(text,protected,reader):
    """All (start, end) clause-final Han spans of 2-12 chars at word boundaries, not on an entity."""
    blocked=set()
    for v in protected:
        for m in re.finditer(re.escape(v),text): blocked.update(range(m.start(),m.end()))
    bounds=reader.boundaries(text); out=[]
    for m in HAN_RUN.finditer(text):
        end=m.end()
        if end<len(text) and not unicodedata.category(text[end]).startswith('P'):
            continue  # the run stops at a digit or Latin word, not at the end of a clause
        for start in range(max(m.start(),end-12),end-1):
            if (start in bounds and end in bounds and text[start] not in LEADING_PARTICLES
                    and not blocked.intersection(range(start,end))):
                out.append((start,end,m.start()))
    return out


def extract(a):
    raw=Path(a.raw); out=Path(a.out)
    if (out/'items.jsonl').exists(): raise SystemExit(f'{out} already has items.jsonl; use a new --out')
    out.mkdir(parents=True,exist_ok=True)
    reader=Reader(); items=[]; stats=collections.defaultdict(collections.Counter)
    for source in a.sources:
        rng=random.Random(f'{a.seed}/{source}')
        dialogues=sorted(LOADERS[source](raw),key=lambda d:(d['source_split'],d['dialogue_id']))
        stats[source]['dialogues']=len(dialogues)
        rng.shuffle(dialogues)
        made=0; seen=set(); same=collections.Counter()
        for d in dialogues:
            if made>=a.per_source: break
            want_bucket=BUCKETS[made%3]; want_ctx=(made//3)%2==0
            msgs=[(i,norm(t)) for i,t in d['messages']]
            msgs=[(i,t) for i,t in msgs if 4<=len(t)<=MAX_CONTEXT+12 and not BAD.search(t)]
            rng.shuffle(msgs); pick=None
            for relax in (0,1,2):  # 0 exact stratum; 1 any context; 2 any length
                for i,t in msgs:
                    cands=[(s,e,r) for s,e,r in spans(t,d['protected'],reader)
                           if (t[max(0,s-MAX_CONTEXT):s],t[s:e]) not in seen and same[t[s:e]]<MAX_SAME_TARGET
                           and (relax>=2 or want_bucket[0]<=e-s<=want_bucket[1]) and (relax>=1 or (s>0)==want_ctx)]
                    if cands: pick=(i,t,rng.choice(cands)); break
                if pick: stats[source][f'relax{relax}']+=1; break
            if not pick: stats[source]['no_span']+=1; continue
            i,t,(s,e,r)=pick
            syl,alt=reader.readings(t[r:e])
            if syl is None: stats[source]['no_reading']+=1; continue
            syl=syl[s-r:]; alt=[x[s-r:] for x in alt]; target=t[s:e]
            conflicts={str(k):dict(zip(('pypinyin','pypinyin_jieba','g2pm'),v))
                       for k,v in enumerate(zip(*alt)) if len(set(v))>1}
            stats[source]['reading_conflict']+=bool(conflicts)
            seen.add((t[max(0,s-MAX_CONTEXT):s],target)); same[target]+=1
            items.append(dict(source=source,source_split=d['source_split'],dialogue_id=d['dialogue_id'],turn=i,
                message=t,span=[s,e],left_context=t[max(0,s-MAX_CONTEXT):s],target=target,syllables=syl,
                reading_conflicts=conflicts,
                keys_full=''.join(syl),keys_abbrev=''.join(x[0] for x in syl) if len(syl)<=ABBREV_MAX else None))
            made+=1
        stats[source]['items']=made
    # dev/blind: alternate inside each source after a seeded shuffle of the dialogues' hashes
    for source in a.sources:
        rows=sorted((it for it in items if it['source']==source),
                    key=lambda it:hashlib.sha256(f'{a.seed}/{it["dialogue_id"]}'.encode()).hexdigest())
        for k,it in enumerate(rows):
            it['split']=('development','blind')[k%2]
            it['id']=f'zh-std-{source}-{hashlib.sha256((source+"/"+it["dialogue_id"]).encode()).hexdigest()[:10]}'
    items.sort(key=lambda it:it['id'])
    assert len({it['id'] for it in items})==len(items)
    with open(out/'items.jsonl','w',encoding='utf-8') as fh:
        for it in items: fh.write(json.dumps(it,ensure_ascii=False)+'\n')
    write_rime_inputs(out,items)
    lengths=collections.Counter(f'{it["source"]}/{it["split"]}/{"ctx" if it["left_context"] else "noctx"}/'
        +next(f'{lo}-{hi}' for lo,hi in BUCKETS if lo<=len(it['target'])<=hi) for it in items)
    manifest=dict(version='ime-standard-zh-v1-draft',seed=a.seed,per_source=a.per_source,raw=str(raw),
        items=len(items),abbrev_items=sum(bool(it['keys_abbrev']) for it in items),
        sources={k:dict(v) for k,v in stats.items()},strata=dict(sorted(lengths.items())),
        python=sys.version.split()[0])
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=1))


def write_rime_inputs(out,items):
    for track in ('full','abbrev'):
        with open(out/f'rime-{track}.jsonl','w',encoding='utf-8') as fh:
            for it in items:
                if it[f'keys_{track}']:
                    fh.write(json.dumps(dict(id=it['id'],keys=it[f'keys_{track}'],n_han=len(it['target'])))+'\n')


def freeze(a):
    draft,review,out=Path(a.draft),Path(a.review),Path(a.out)
    if (out/'items.jsonl').exists(): raise SystemExit(f'{out} already has items.jsonl; use a new --out')
    items=[json.loads(l) for l in open(draft/'items.jsonl',encoding='utf-8') if l.strip()]
    recs={}
    for f in sorted(review.glob('chunk-*.jsonl')):
        for l in open(f,encoding='utf-8'):
            if l.strip(): r=json.loads(l); assert r['id'] not in recs,r['id']; recs[r['id']]=r
    assert set(recs)=={it['id'] for it in items},'every draft item needs exactly one review record'
    assert not [i for i,r in recs.items() if r['decision']=='unsure'],'unsure items need a human decision'
    human=dict(global_={},items={})
    if a.human:
        h=json.loads(Path(a.human).read_text(encoding='utf-8'))
        human=dict(global_=h.get('global',{}),items=h.get('items',{}))
        unknown=set(human['items'])-set(recs); assert not unknown,f'human decisions for unknown ids: {sorted(unknown)[:3]}'
        notes={i:d['note'] for i,d in human['items'].items() if d.get('note','').strip()}
        assert not notes,f'human notes must be folded into the review records first: {notes}'
    shui=human['global_'].get('shui') or 'shui'
    assert shui in ('shui','shei')
    kept=[]; counts=collections.Counter(); overridden=collections.Counter()
    for it in items:
        r=recs[it['id']]; decision=r['decision']
        hd=human['items'].get(it['id'],{}).get('decision')
        if hd in ('keep','drop'):
            if (hd=='drop')!=(decision=='drop'): overridden[f'{decision}->{hd}']+=1
            if hd=='drop': decision='drop'
            elif decision=='drop': decision='keep'
        counts[decision]+=1
        if decision=='drop': continue
        syl=list(r['syllables'] or it['syllables'])
        if shui=='shei':
            syl=['shei' if c=='谁' and s=='shui' else s for c,s in zip(it['target'],syl)]
        assert len(syl)==len(it['target'])
        alts=[x for x in r['alternatives'] if x!=it['target']] if hd!='drop' else []
        assert all(len(x)==len(it['target']) for x in alts)
        reviewed=dict(it,syllables=syl,keys_full=''.join(syl),
                      keys_abbrev=''.join(x[0] for x in syl) if len(syl)<=ABBREV_MAX else None,
                      alternatives=alts,
                      review=dict(decision=decision,note=r['note'],reviewer=r['reviewer'],
                                  human=hd,reading_changed=syl!=it['syllables']))
        kept.append(reviewed)
    out.mkdir(parents=True,exist_ok=True)
    with open(out/'items.jsonl','w',encoding='utf-8') as fh:
        for it in kept: fh.write(json.dumps(it,ensure_ascii=False)+'\n')
    write_rime_inputs(out,kept)
    lengths=collections.Counter(f'{it["source"]}/{it["split"]}/'
        +next(f'{lo}-{hi}' for lo,hi in BUCKETS if lo<=len(it['target'])<=hi) for it in kept)
    manifest=dict(version='ime-standard-zh-v1-r1',draft=str(draft),review=str(review),
        human=str(a.human) if a.human else None,
        labels=dict(ai_expert_reviewed=True,human_spot_review=bool(a.human),
                    native_chinese_review=False,labels_formal_gold=False),
        reading_of_谁=shui,decisions=dict(counts),human_overrides=dict(overridden),items=len(kept),
        abbrev_items=sum(bool(it['keys_abbrev']) for it in kept),
        with_alternatives=sum(bool(it['alternatives']) for it in kept),
        reading_changed=sum(it['review']['reading_changed'] for it in kept),
        strata=dict(sorted(lengths.items())),python=sys.version.split()[0])
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=1))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',choices=['extract','freeze'])
    p.add_argument('--draft',default=str(ROOT/'outputs/benchmarks/ime-standard-zh-v1/draft-r1'))
    p.add_argument('--review',default=str(ROOT/'outputs/benchmarks/ime-standard-zh-v1/expert-review-r1'))
    p.add_argument('--human',help='human-review-r1.json exported from the review page')
    p.add_argument('--raw',default=str(ROOT/'datasets/ime-standard-zh-v1/raw'))
    p.add_argument('--out',default=str(ROOT/'outputs/benchmarks/ime-standard-zh-v1/draft-r1'))
    p.add_argument('--sources',nargs='+',default=['crosswoz','risawoz','kdconv'])
    p.add_argument('--per-source',type=int,default=1000,help='dialogues (= items) per source before review')
    p.add_argument('--seed',type=int,default=20261106)
    a=p.parse_args()
    dict(extract=extract,freeze=freeze)[a.command](a)


if __name__=='__main__':
    main()
