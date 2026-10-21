"""Notation-aware error classes for same-reading kana-kanji conversions.

All candidates in a pool convert the same kana input, so a wrong choice either spells
the same words differently (notation, A) or picks different words (semantic, S).
Conservative A: equal UniDic lemma sequences AND nested kanji sets. Against 210 earlier
hand-tagged V2.1/V4.2 flips this never labels a semantic flip as A (72/72) and recovers
118/138 notation flips; satisfying only one rule is reported as an upper bound ('a').
Requires `fugashi` and `unidic-lite` (requirements-evaluation.txt).
"""
import unicodedata

import fugashi

_TAGGER = None
_SKIP = ('補助記号', '空白')
ERROR_CLASSES = ('correct', 'pool_miss', 'notation', 'notation_upper', 'semantic')


def _tagger():
    global _TAGGER
    if _TAGGER is None:
        _TAGGER = fugashi.Tagger()
    return _TAGGER


def nfkc(text):
    return unicodedata.normalize('NFKC', text)


def kanji(text):
    return {c for c in text if '一' <= c <= '鿿' or c in '々〆ヶ'}


def kana_share(text):
    letters = [c for c in text if c.isalpha()]
    return sum('ぁ' <= c <= 'ヿ' for c in letters) / len(letters) if letters else 0.0


def lemmas(text):
    return [w.feature.lemma or w.surface for w in _tagger()(nfkc(text)) if w.feature.pos1 not in _SKIP]


def relation(choice, answer):
    """'=' exact, 'A' notation (conservative), 'a' notation by one rule only, 'S' semantic."""
    if choice == answer:
        return '='
    same_lemmas = lemmas(choice) == lemmas(answer)
    nested = kanji(choice) <= kanji(answer) or kanji(answer) <= kanji(choice)
    if same_lemmas and nested:
        return 'A'
    return 'a' if same_lemmas or nested else 'S'


def error_class(row, choice):
    """Class of a Top-1 choice and the closest reachable reference ('answer' is None if unreachable)."""
    answers = row['answers']
    if choice in answers:
        return 'correct', choice
    pool = {c['text'] for c in row['candidates']}
    reachable = [a for a in answers if a in pool]
    if not reachable:
        return 'pool_miss', None
    rank, answer = min(('=AaS'.index(relation(choice, a)), a) for a in reachable)
    return {1: 'notation', 2: 'notation_upper', 3: 'semantic'}[rank], answer
