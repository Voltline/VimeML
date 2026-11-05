"""Stage-A rules for the Chinese web corpus (FineWeb2 cmn_Hani).

Each document gets one decision:

  drop    blocked TLD, tier-1 domain, gambling pinyin in the host, a fake government host
          (x.gov.cn.spam.cn), a content-term hit, too little Chinese text, or mostly repeated lines
  review  tier-2 domain, or a single weak term; kept in full for human review
  keep    everything else

Traditional-script pages are converted to Simplified with OpenCC `tw2sp` (characters
and Taiwan phrasing, e.g. 軟體 -> 软件) and tagged `script='hant'`. Term lists and
domains come from configs/zh-filter-terms.toml; matching runs on the converted text,
lower-cased for Latin letters.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import tomllib


HAN=re.compile(r'[㐀-䶿一-鿿豈-﫿]')


@dataclass(frozen=True)
class Rules:
    pattern: re.Pattern
    category: dict
    weak: frozenset
    tlds: frozenset
    tier1: tuple
    tier2: tuple
    host_terms: tuple=()


def load_rules(path):
    spec=tomllib.loads(Path(path).read_text(encoding='utf-8'))
    category={}
    for cat,terms in spec['categories'].items():
        for t in terms:
            category.setdefault(t.lower(),cat)
    weak=frozenset(t.lower() for t in spec.get('rules',{}).get('weak',[]))
    unknown=weak-set(category)
    if unknown:
        raise ValueError(f'weak terms missing from every category: {sorted(unknown)}')
    terms=sorted(category,key=len,reverse=True)
    domains=spec.get('domains',{})
    return Rules(pattern=re.compile('|'.join(map(re.escape,terms))),category=category,weak=weak,
                 tlds=frozenset(domains.get('block_tlds',[])),tier1=tuple(domains.get('block',[])),
                 tier2=tuple(domains.get('block_tier2',[])),host_terms=tuple(domains.get('host_terms',[])))


def in_domains(host,domains):
    return any(host==d or host.endswith('.'+d) for d in domains)


def term_hits(text,rules):
    """Distinct terms found in `text` (already lower-cased), grouped by category."""
    found={}
    for m in rules.pattern.finditer(text):
        t=m.group(0)
        found.setdefault(rules.category[t],set()).add(t)
    return {c:sorted(ts) for c,ts in found.items()}


def counts_as_hit(hits,weak):
    """A document hits when it contains a non-weak term, or two different terms."""
    found={t for ts in hits.values() for t in ts}
    return bool(found-weak) or len(found)>=2


@lru_cache(maxsize=1)
def _converters():
    import opencc
    return opencc.OpenCC('t2s'),opencc.OpenCC('tw2sp')


@lru_cache(maxsize=1)
def traditional_chars():
    """CJK characters that OpenCC t2s maps to a different character."""
    t2s,_=_converters()
    chars=[chr(c) for r in ((0x3400,0x4dc0),(0x4e00,0xa000),(0xf900,0xfb00)) for c in range(*r)]
    out=t2s.convert(''.join(chars))
    return frozenset(a for a,b in zip(chars,out) if a!=b)


def traditional_ratio(han):
    trad=traditional_chars()
    return sum(c in trad for c in han)/max(1,len(han))


def to_simplified(text):
    return _converters()[1].convert(text)


def repeated_line_fraction(text):
    """Share of non-blank line characters that sit in lines occurring more than once."""
    lines=[l.strip() for l in text.split('\n')]
    lines=[l for l in lines if l]
    total=sum(map(len,lines))
    if not total:
        return 0.0
    counts={}
    for l in lines:
        counts[l]=counts.get(l,0)+1
    return sum(len(l)*n for l,n in counts.items() if n>1)/total


@dataclass(frozen=True)
class Thresholds:
    min_han: int=100
    min_han_share: float=0.4
    max_repeated_lines: float=0.3
    traditional: float=0.08


def classify(text,host,rules,th=Thresholds()):
    """Return the converted text and the stage-A decision for one document."""
    han=HAN.findall(text)
    script='hans'
    if traditional_ratio(han)>th.traditional:
        text=to_simplified(text); script='hant'
    lower=text.lower()
    hits=term_hits(lower,rules)
    drop=[]; review=[]
    tld=host.rsplit('.',1)[-1]
    if tld in rules.tlds: drop.append('tld')
    if in_domains(host,rules.tier1): drop.append('domain_tier1')
    if not host.endswith(('.gov.cn','.edu.cn')) and any(k in host for k in rules.host_terms):
        drop.append('host_term')
    if ('.gov.cn.' in host or '.edu.cn.' in host) and not host.endswith(('.gov.cn','.edu.cn')):
        drop.append('fake_gov')
    if hits:
        if counts_as_hit(hits,rules.weak): drop.extend(f'term_{c}' for c in sorted(hits))
        else: review.append('weak_term')
    if in_domains(host,rules.tier2): review.append('domain_tier2')
    nonspace=sum(not c.isspace() for c in text)
    if len(han)<th.min_han: drop.append('short')
    elif len(han)<th.min_han_share*nonspace: drop.append('low_han')
    if repeated_line_fraction(text)>th.max_repeated_lines: drop.append('repetitive')
    decision='drop' if drop else 'review' if review else 'keep'
    return dict(text=text,script=script,decision=decision,reasons=drop+review,
                hits=[t for ts in hits.values() for t in ts],han=len(han))


SECOND_LEVEL=('com','net','org','gov','edu','ac')


def registrable(host):
    """example.com, example.com.cn: the last two labels, or three under com/net/... + ccTLD."""
    parts=host.split('.')
    if len(parts)>=3 and len(parts[-1])==2 and parts[-2] in SECOND_LEVEL:
        return '.'.join(parts[-3:])
    return '.'.join(parts[-2:])


def random_looking(label):
    """Letters and digits interleaved at least twice (u07bgqr7k, hc28hpjs, kzlue2y).

    Ordinary Chinese site names often carry one number block (lady8844, it168, 51cto,
    3dmgame); those switch between letters and digits only once.
    """
    kinds=[c.isdigit() for c in label if c.isalnum()]
    switches=sum(a!=b for a,b in zip(kinds,kinds[1:]))
    return len(label)>=5 and switches>=2
