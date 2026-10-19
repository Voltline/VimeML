"""Dual-dictionary labels for mixed-length, literal-aware conversion spans."""
import random
import re
import unicodedata

from .readings import DualReading, KANA, KANJI, RESERVED, hiragana


def normalize_literal(value):
    return hiragana(unicodedata.normalize('NFKC', value))


def literal(value):
    # Input symbols/ASCII/numbers are preserved as typed, not guessed readings.
    return bool(value) and all(not ('一' <= c <= '龯') and
        (c.isspace() or c.isascii() or unicodedata.category(c)[0] in 'PNZS')
        for c in unicodedata.normalize('NFKC', value))


class DomainReading(DualReading):
    def annotate(self, text):
        left, right, cursor = [], [], 0
        self.proper_spans = []
        for w in self.mecab(text):
            start = cursor + len(w.white_space)
            end = start + len(w.surface)
            if text[start:end] != w.surface:
                raise ValueError('MeCab offset mismatch')
            if w.feature.pos2 == '固有名詞':
                self.proper_spans.append((start, end))
            if start > cursor:
                left.append((cursor, start, normalize_literal(text[cursor:start]), True))
            direct = bool(KANA.fullmatch(w.surface)) or literal(w.surface)
            reading = normalize_literal(w.surface) if direct else hiragana(w.feature.kana or '')
            valid = direct or (not w.is_unk and bool(reading))
            # Keep the inherited ambiguity quarantine; do not guess family suffixes.
            if w.surface == '家' and w.feature.kana in ('ヤ', 'ケ'):
                valid = False
            if (len(w.surface) == 1 and KANJI.fullmatch(w.surface) and w.feature.pos1 == '名詞'
                    and ((start and KANJI.fullmatch(text[start-1])) or
                         (end < len(text) and KANJI.fullmatch(text[end])))):
                valid = False
            left.append((start, end, normalize_literal(reading), valid))
            cursor = end
        if cursor < len(text):
            left.append((cursor, len(text), normalize_literal(text[cursor:]), text[cursor:].isspace()))
        for w in self.sudachi.tokenize(text, self.mode):
            start, end, surface = w.begin(), w.end(), w.surface()
            if text[start:end] != surface:
                raise ValueError('Sudachi offset mismatch')
            direct = bool(KANA.fullmatch(surface)) or literal(surface)
            reading = normalize_literal(surface) if direct else hiragana(w.reading_form())
            right.append((start, end, normalize_literal(reading), direct or (not w.is_oov() and bool(reading))))
        return left, right

    def example_near(self, text, anchor, seed):
        if any(x in text for x in RESERVED):
            return None, 'reserved_literal'
        # Bound annotation work on long documents without detaching source offsets.
        lo, hi = max(0, anchor[0] - 64), min(len(text), anchor[1] + 64)
        segment = text[lo:hi]
        left, right = self.annotate(segment)
        right_starts = {t[0]: i for i, t in enumerate(right)}
        right_ends = {t[1]: i for i, t in enumerate(right)}
        right_reading = ''.join(t[2] for t in right)
        reading_offsets, bad = [0], [0]
        for t in right:
            reading_offsets.append(reading_offsets[-1] + len(t[2]))
            bad.append(bad[-1] + (not t[3]))
        rng = random.Random(seed)
        starts = [i for i, t in enumerate(left) if t[0] in right_starts]
        # A bounded random set of common boundaries keeps corpus preparation linear.
        starts = sorted(set(([0] if starts and starts[0] == 0 else []) +
                            rng.sample(starts, min(7, len(starts)))))
        candidates = []
        for i in starts:
            token = left[i]
            start = token[0]
            if start not in right_starts:
                continue
            reading, valid = '', True
            for end_token in left[i:i+48]:
                end = end_token[1]
                if end - start > 48:
                    break
                reading += end_token[2]
                valid &= end_token[3]
                if not valid:
                    break
                if end not in right_ends or not 8 <= end - start <= 48 or not 1 <= len(reading) <= 72:
                    continue
                a, b = right_starts[start], right_ends[end] + 1
                if b <= a or bad[b] != bad[a] or right_reading[reading_offsets[a]:reading_offsets[b]] != reading:
                    continue
                surface = segment[start:end]
                if not KANJI.search(surface) or not surface.strip() or surface != surface.strip():
                    continue
                candidates.append((start+lo, end+lo, reading))
        if not candidates:
            return None, 'no_reliable_extended_span'
        # Equal chance among available broad output lengths; no benchmark targets.
        buckets = {}
        for item in candidates:
            length = item[1] - item[0]
            buckets.setdefault(0 if length <= 16 else 1 if length <= 32 else 2, []).append(item)
        options = buckets[rng.choice(sorted(buckets))]
        start, end, reading = rng.choice(options)
        return {'context': text[:start], 'reading': reading, 'output': text[start:end],
                'span': [start, end], 'mecab_reading': reading, 'sudachi_reading': reading,
                'has_kanji': True, 'orthographic_rule': 'c1-domain-v2',
                'proper_name_present': any(start <= a+lo and b+lo <= end for a,b in self.proper_spans),
                'literal_present': any(literal(c) for c in text[start:end])}, None
