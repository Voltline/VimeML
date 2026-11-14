"""Readings from the Rime dictionary Vime ships (vime_pinyin.dict.yaml).

The dictionary comes from Vime's Scripts/build_rime_dicts.py (pinned public sources:
pinyin-data, phrase-pinyin-data, Unihan kHanyuPinlu, ...); it is generated locally and never
committed. Each line is "word<TAB>space-separated pinyin<TAB>weight"; a word may have
several readings. Only the 8105 characters of 通用规范汉字表 occur, so a text using any other
character cannot be typed in Vime at all.

segment() splits a Han run the way Rime builds a sentence: it maximises the sum over words of
log(weight) - log(1e8) - 13.8 (build_rime_dicts.py), reading each word with its heaviest
reading. common() lists a character's readings holding at least a given share of its weight.
"""
import math

WORD_PENALTY = math.log(1e8) + 13.8


class ReadingTable:
    def __init__(self, path, max_word=8):
        self.words = {}
        body = open(path, encoding="utf-8").read().split("\n...\n", 1)[1]
        for line in body.splitlines():
            parts = line.split("\t")
            if len(parts) < 2 or not parts[0] or line.startswith("#"):
                continue
            weight = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
            self.words.setdefault(parts[0], []).append((parts[1].split(), weight))
        for word, entries in self.words.items():
            entries.sort(key=lambda x: -x[1])
        self.best = {}
        for word, entries in self.words.items():
            reading, weight = entries[0]
            total = sum(w for _, w in entries)
            if len(reading) == len(word) and total > 0:
                self.best[word] = (reading, math.log(total) - WORD_PENALTY)
        self.max_word = max_word
        self.syllables = frozenset(s for entries in self.words.values() for r, _ in entries for s in r)

    def characters(self):
        return {w for w in self.words if len(w) == 1}

    def common(self, char, share):
        entries = [(r[0], w) for r, w in self.words.get(char, ()) if len(r) == 1]
        total = sum(w for _, w in entries)
        if total <= 0:
            return ()
        return tuple(sorted({r for r, w in entries if w >= share * total}))

    def default(self, char):
        entries = self.words.get(char)
        return entries[0][0][0] if entries else None

    def segment(self, run):
        """[(start, end, syllables)] covering run, or None if any character is untypeable."""
        n = len(run)
        score = [-math.inf] * (n + 1)
        back = [None] * (n + 1)
        score[0] = 0.0
        for i in range(n):
            if score[i] == -math.inf:
                continue
            for j in range(i + 1, min(n, i + self.max_word) + 1):
                entry = self.best.get(run[i:j])
                if entry is None:
                    continue
                value = score[i] + entry[1]
                if value > score[j]:
                    score[j], back[j] = value, i
        if back[n] is None and n:
            return None
        out, j = [], n
        while j > 0:
            i = back[j]
            out.append((i, j, self.best[run[i:j]][0]))
            j = i
        return out[::-1]
