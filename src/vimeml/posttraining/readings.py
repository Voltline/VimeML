"""Conservative, original-offset dual dictionary labels for training text."""
import random
import re

KANA = re.compile(r"[ぁ-ゖァ-ヶーゔゕゖ]+\Z")
HIRAGANA = re.compile(r"[ぁ-ゖー]+\Z")
SURFACE = re.compile(r"[一-龯々〆ぁ-ゖァ-ヶー]+\Z")
KANJI = re.compile(r"[一-龯々〆]")
RESERVED = ("<CTX>", "<READ>", "<OUT>", "<s>", "</s>", "<unk>", "<pad>")


def hiragana(value):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in value)


class DualReading:
    def __init__(self):
        import fugashi
        from sudachipy import dictionary, tokenizer
        self.mecab = fugashi.Tagger()
        self.sudachi = dictionary.Dictionary().tokenizer()
        self.mode = tokenizer.Tokenizer.SplitMode.C

    def annotate(self, text):
        """Return aligned tokens; gaps/punctuation remain barriers to spans."""
        left, cursor = [], 0
        for w in self.mecab(text):
            start = cursor + len(w.white_space)
            end = start + len(w.surface)
            if text[start:end] != w.surface or not text[cursor:start].isspace() and start != cursor:
                raise ValueError("MeCab original offset mismatch")
            pos = (w.feature.pos1, w.feature.pos2)
            reading = w.surface if KANA.fullmatch(w.surface) else w.feature.kana
            ok = not w.is_unk and pos[1] != "固有名詞" and bool(reading)
            left.append((start, end, hiragana(reading or ""), ok))
            cursor = end
        right = []
        for w in self.sudachi.tokenize(text, self.mode):
            start, end = w.begin(), w.end()
            if text[start:end] != w.surface():
                raise ValueError("Sudachi original offset mismatch")
            reading = w.surface() if KANA.fullmatch(w.surface()) else w.reading_form()
            ok = not w.is_oov() and w.part_of_speech()[1] != "固有名詞"
            right.append((start, end, hiragana(reading), ok))
        return left, right

    def example(self, text, seed):
        if any(x in text for x in RESERVED):
            return None, "reserved_literal"
        left, right = self.annotate(text)
        # Common original-text boundaries, with at most six morphemes in either analysis.
        other = {}
        for i, a in enumerate(right):
            reading, valid, end = "", True, a[0]
            for b in right[i:i + 6]:
                if b[0] != end:
                    break
                end = b[1]
                reading += b[2]
                valid &= b[3]
                if valid and HIRAGANA.fullmatch(reading):
                    other[(a[0], end)] = reading
        options, reasons = [], set()
        for i, a in enumerate(left):
            reading, valid, end = "", True, a[0]
            for b in left[i:i + 6]:
                if b[0] != end:
                    break
                end = b[1]
                reading += b[2]
                valid &= b[3]
                out = text[a[0]:end]
                if not valid:
                    reasons.add("unknown_or_proper_name")
                    continue
                if not (1 <= len(out) <= 24 and 1 <= len(reading) <= 48):
                    continue
                if not SURFACE.fullmatch(out) or not HIRAGANA.fullmatch(reading):
                    reasons.add("unsupported_surface_or_reading")
                    continue
                if other.get((a[0], end)) != reading:
                    reasons.add("reading_or_boundary_disagreement")
                    continue
                options.append((a[0], end, reading))
        if not options:
            return None, "+".join(sorted(reasons)) or "no_common_span"
        # Equal probability per occupied length bucket, then per available context bucket.
        # This prevents abundant one-morpheme endings from dominating the pool.
        rng = random.Random(seed)
        groups = {}
        for start, end, reading in options:
            key = (0 if len(reading) <= 3 else 1 if len(reading) <= 8 else 2 if len(reading) <= 16 else 3,
                   0 if start == 0 else 1 if start <= 8 else 2 if start <= 24 else 3,
                   bool(KANJI.search(text[start:end])))
            groups.setdefault(key, []).append((start, end, reading))
        key = rng.choice(sorted(groups))
        start, end, reading = rng.choice(groups[key])
        return {"context": text[:start], "reading": reading, "output": text[start:end],
                "span": [start, end], "mecab_reading": reading, "sudachi_reading": reading,
                "has_kanji": bool(KANJI.search(text[start:end])), "stratum": list(key)}, None
