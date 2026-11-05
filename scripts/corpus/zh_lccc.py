"""Clean LCCC-base (Weibo dialogues) into per-turn chat text for the Chinese corpus.

LCCC ships character-tokenised turns ("我 饿 了 。"); spaces are removed except between
two Latin/digit tokens. Each turn becomes one chat sequence, as in the Japanese chat
data (BOS + turn + EOS); the LCCC train/valid/test split is kept, so splits stay
dialogue-disjoint.

Per turn: Traditional -> Simplified (tw2sp, same threshold as stage A), then the stage-A
term list (configs/zh-filter-terms.toml): a hit drops the turn, a lone weak term sends
it to review.jsonl. Turns with fewer than MIN_HAN Chinese characters are dropped.
Train turns of HELDOUT_MIN_CHARS or more that also occur in valid/test are removed
(short stock replies such as 哈哈 or 好的 stay). A train turn seen n times is kept
ceil(sqrt(n)) times (its first occurrences): repeats are compressed but their frequency
order, which the IME prior needs, survives. Writes <out>/{train,validation,test}.jsonl, review.jsonl and manifest.json.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vimeml.corpus.zh_filter import (HAN, Thresholds, counts_as_hit, load_rules, term_hits,  # noqa: E402
                                     to_simplified, traditional_ratio)

SPLITS={'train':'train','valid':'validation','test':'test'}
ALNUM=re.compile(r'^[A-Za-z0-9]+$')
MIN_HAN=2
HELDOUT_MIN_CHARS=10  # shorter turns (哈哈, 谢谢, 好的) are not leakage; keep them in train


def detok(s):
    out=[]
    for t in s.split(' '):
        if not t: continue
        if out and ALNUM.match(t) and ALNUM.match(out[-1][-1]): out.append(' ')
        out.append(t)
    return ''.join(out)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<24),b''): h.update(block)
    return h.hexdigest()


def clean_turn(raw,rules,th):
    text=detok(raw).strip()
    han=HAN.findall(text)
    script='hans'
    if traditional_ratio(han)>th.traditional:
        text=to_simplified(text); script='hant'
    if len(han)<MIN_HAN:
        return text,script,'drop','short',[]
    hits=term_hits(text.lower(),rules)
    found=[t for ts in hits.values() for t in ts]
    if hits and counts_as_hit(hits,rules.weak):
        return text,script,'drop','term_'+'+'.join(sorted(hits)),found
    if hits:
        return text,script,'review','weak_term',found
    return text,script,'keep',None,[]


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source',default=r'D:\BaiduNetdiskDownload\LCCC-base-split')
    p.add_argument('--terms',default=str(ROOT/'configs/zh-filter-terms.toml'))
    p.add_argument('--out',default=str(ROOT/'outputs/zh-corpus/lccc-v2'))
    a=p.parse_args()
    src=Path(a.source); out=Path(a.out)
    if (out/'manifest.json').exists():
        raise SystemExit(f'{out} is complete; use a new --out')
    out.mkdir(parents=True,exist_ok=True)
    rules=load_rules(a.terms); th=Thresholds()
    counts={}; kept={}; review=[]
    for name,split in SPLITS.items():
        dialogues=json.load(open(src/f'LCCC-base_{name}.json',encoding='utf-8'))
        c=collections.Counter(); rows=[]
        for d,turns in enumerate(dialogues):
            for j,raw in enumerate(turns):
                text,script,decision,reason,found=clean_turn(raw,rules,th)
                c[decision]+=1; c[f'script_{script}']+=1
                if reason: c[reason]+=1
                row=dict(text=text,dialogue=d,turn=j,script=script)
                if decision=='keep': rows.append(row)
                elif decision=='review': review.append(dict(row,split=split,hits=found))
        c['dialogues']=len(dialogues); counts[split]=c; kept[split]=rows
        del dialogues
        print(split,dict(c),flush=True)
    heldout={r['text'] for s in ('validation','test') for r in kept[s]}
    rows=[]
    for r in kept['train']:
        if len(r['text'])>=HELDOUT_MIN_CHARS and r['text'] in heldout: counts['train']['heldout_overlap']+=1; continue
        rows.append(r)
    freq=collections.Counter(r['text'] for r in rows); emitted=collections.Counter(); train=[]
    for r in rows:
        if emitted[r['text']]>=math.isqrt(freq[r['text']]-1)+1: counts['train']['duplicate']+=1; continue
        emitted[r['text']]+=1; train.append(r)
    counts['train']['unique']=len(freq)
    top=[dict(text=t,seen=n,kept=emitted[t]) for t,n in freq.most_common(50)]
    kept['train']=train; del rows,freq,emitted
    for split,rows in kept.items():
        with open(out/f'{split}.jsonl','w',encoding='utf-8') as fh:
            for r in rows: fh.write(json.dumps(r,ensure_ascii=False)+'\n')
        counts[split]['written']=len(rows)
        counts[split]['written_chars']=sum(len(r['text']) for r in rows)
    with open(out/'review.jsonl','w',encoding='utf-8') as fh:
        for r in review: fh.write(json.dumps(r,ensure_ascii=False)+'\n')
    manifest=dict(source='LCCC-base (thu-coai), local Baidu Netdisk copy',
                  source_files={f'LCCC-base_{n}.json':sha256(src/f'LCCC-base_{n}.json') for n in SPLITS},
                  terms_sha256=sha256(a.terms),script_sha256=sha256(__file__),
                  filter_sha256=sha256(ROOT/'src/vimeml/corpus/zh_filter.py'),
                  sequence_policy='BOS + one cleaned turn + EOS; LCCC dialogue splits kept',
                  min_han=MIN_HAN,heldout_min_chars=HELDOUT_MIN_CHARS,repeat_policy='ceil(sqrt(n)) copies',top_repeats=top,traditional_threshold=th.traditional,review=len(review),
                  counts={s:dict(c) for s,c in counts.items()})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps(manifest['counts'],ensure_ascii=False,indent=1))


if __name__=='__main__':
    main()
