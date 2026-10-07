"""Two-dictionary orthographic kana checks for evaluation preparation only."""

import re
from importlib.metadata import version

KANA = re.compile(r"[ァ-ヺー。、！？「」『』（）・：，,.!?]+")


def katakana(text):
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


class DualReading:
    def __init__(self):
        import fugashi
        from sudachipy import Dictionary, SplitMode

        self.mecab = fugashi.Tagger()
        self.sudachi = Dictionary().create()
        self.mode = SplitMode.C
        self.versions = {
            n: version(n) for n in ("fugashi", "unidic-lite", "sudachipy", "sudachidict-core")
        }

    def analyse(self, text, start=0):
        """Read a suffix in its sentence, rejecting boundaries inside words."""
        left, right, unknown = [], [], []
        offset = 0
        for word in self.mecab(text):
            surface = word.surface
            end = offset + len(surface)
            if offset < start < end:
                unknown.append("boundary_inside_unidic_word")
            if offset < start:
                offset = end
                continue
            offset = end
            if word.is_unk:
                unknown.append(surface)
            if KANA.fullmatch(katakana(surface)):
                left.append(katakana(surface))
            elif word.feature.kana:
                left.append(word.feature.kana)
            else:
                left.append(surface)
                unknown.append(surface)
        for word in self.sudachi.tokenize(text, self.mode):
            surface = word.surface()
            if word.begin() < start < word.end():
                unknown.append("boundary_inside_sudachi_word")
            if word.begin() < start:
                continue
            if KANA.fullmatch(katakana(surface)):
                right.append(katakana(surface))
            else:
                right.append(word.reading_form())
            if word.is_oov():
                unknown.append(surface)
        a, b = "".join(left), "".join(right)
        return {
            "unidic": a,
            "sudachi": b,
            "agreement": a == b,
            "unknown": sorted(set(unknown)),
            "kana_only": bool(KANA.fullmatch(a) and KANA.fullmatch(b)),
        }
