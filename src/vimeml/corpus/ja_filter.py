"""Stage-A rules for the Japanese web corpus (FineWeb2 jpn_Jpan, ja-gen plan stage 1.1).

Same shape as the Chinese rules (`zh_filter.py`). Each document gets one decision:

  drop    blocked TLD or domain, a content-term hit, too little Japanese text, too few kana
          (Chinese or kanji-only pages mislabelled as Japanese), or mostly repeated lines
  review  a single weak term or a tier-2 domain. The ja-gen 200M corpus has no review step,
          so these documents are dropped at export, as in the Chinese r2 batch.
  keep    everything else

Japanese characters are hiragana, katakana (with the long-vowel mark) and kanji. Text is not
converted. Term lists and domains come from configs/ja/ja-filter-terms.toml; matching runs on
NFKC-normalized, lower-cased text, so half-width katakana and full-width Latin letters match too.
"""
from dataclasses import dataclass
import re
import unicodedata

from vimeml.corpus.zh_filter import counts_as_hit, in_domains, load_rules, repeated_line_fraction, term_hits

JA = re.compile(r'[ぁ-ゖゝゞァ-ヺー-ヾ㐀-䶿一-鿿豈-﫿々〆ヶ]')
KANA = re.compile(r'[ぁ-ゖゝゞァ-ヺー-ヾ]')

__all__ = ['JA', 'KANA', 'Thresholds', 'classify', 'in_domains', 'load_rules']


@dataclass(frozen=True)
class Thresholds:
    min_ja: int = 100
    min_ja_share: float = 0.4
    min_kana_share: float = 0.15
    max_repeated_lines: float = 0.3


def classify(text, host, rules, th=Thresholds()):
    """Return the stage-A decision for one document (text unchanged)."""
    ja = len(JA.findall(text))
    kana = len(KANA.findall(text))
    hits = term_hits(unicodedata.normalize('NFKC', text).lower(), rules)
    drop, review = [], []
    if host.rsplit('.', 1)[-1] in rules.tlds:
        drop.append('tld')
    if in_domains(host, rules.tier1):
        drop.append('domain_tier1')
    if any(k in host for k in rules.host_terms):
        drop.append('host_term')
    if hits:
        if counts_as_hit(hits, rules.weak):
            drop.extend(f'term_{c}' for c in sorted(hits))
        else:
            review.append('weak_term')
    if in_domains(host, rules.tier2):
        review.append('domain_tier2')
    nonspace = sum(not c.isspace() for c in text)
    if ja < th.min_ja:
        drop.append('short')
    elif ja < th.min_ja_share * nonspace:
        drop.append('low_ja')
    elif kana < th.min_kana_share * ja:
        drop.append('low_kana')
    if repeated_line_fraction(text) > th.max_repeated_lines:
        drop.append('repetitive')
    decision = 'drop' if drop else 'review' if review else 'keep'
    return dict(text=text, script='ja', decision=decision, reasons=drop + review,
                hits=[t for ts in hits.values() for t in ts], han=ja)
