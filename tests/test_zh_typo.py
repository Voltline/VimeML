import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vimeml.posttraining import zh_typo as T  # noqa: E402

CONFIG = {"two_share": 0.15, "weights": {"substitute": 40, "delete": 25, "insert": 20, "transpose": 15},
          "repeat_share": 0.5}


def edits_between(a, b):
    """Smallest number of single-letter edits (Damerau-Levenshtein, adjacent swaps)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


class TypoTest(unittest.TestCase):
    def test_neighbours(self):
        self.assertEqual(set(T.NEIGHBOURS["a"]), set("qwsz"))
        self.assertEqual(set(T.NEIGHBOURS["s"]), set("adwexz"))
        self.assertEqual(set(T.NEIGHBOURS["m"]), set("njk"))
        self.assertEqual(set(T.NEIGHBOURS["p"]), set("ol"))
        for c, near in T.NEIGHBOURS.items():
            self.assertNotIn(c, near)
            for n in near:
                self.assertIn(c, T.NEIGHBOURS[n])

    def test_letter_units(self):
        self.assertEqual(T.letter_units("xi'anshi", ["xi", "an", "shi"]), [0, 0, None, 1, 1, 2, 2, 2])
        self.assertEqual(T.letter_units("zhgr", ["zh", "g", "r"]), [0, 0, 1, 2])
        with self.assertRaises(ValueError):
            T.letter_units("xian", ["xi", "en"])

    def test_typos_stay_in_their_syllables(self):
        units = ["zhong", "guo", "ren", "min", "x", "an"]
        keys = "zhongguorenminx'an"
        rng = random.Random(1)
        kinds, counts = Counter(), Counter()
        cancelled = 0
        for _ in range(4000):
            result = T.add_typos(rng, keys, units, CONFIG)
            if result is None:                   # two edits that undo each other (guo -> uo, zhong -> zhongg)
                cancelled += 1
                continue
            out, done = result
            kinds.update(done)
            counts[len(done)] += 1
            self.assertNotEqual(out.replace("'", ""), keys.replace("'", ""))
            self.assertTrue(out and out[0] != "'" and out[-1] != "'" and "''" not in out)
            self.assertLessEqual(edits_between(keys.replace("'", ""), out.replace("'", "")), len(done))
            self.assertTrue(set(out) <= set("abcdefghijklmnopqrstuvwxyz'"))
        self.assertLess(cancelled, 20)
        self.assertAlmostEqual(counts[2] / 4000, 0.15, delta=0.03)
        total = sum(kinds.values())
        self.assertAlmostEqual(kinds["substitute"] / total, 0.40, delta=0.04)
        self.assertAlmostEqual(kinds["delete"] / total, 0.25, delta=0.04)

    def test_edit_that_does_not_fit_is_redrawn(self):
        only = lambda **w: dict(CONFIG, two_share=0, weights=w)
        self.assertIsNone(T.add_typos(random.Random(0), "a", ["a"], only(transpose=1)))
        self.assertIsNone(T.add_typos(random.Random(0), "a", ["a"], only(delete=1)))      # nothing left
        self.assertEqual(T.add_typos(random.Random(0), "ma", ["ma"], only(transpose=1)), ("am", ["transpose"]))
        out, done = T.add_typos(random.Random(0), "a", ["a"], only(transpose=1, substitute=1))
        self.assertEqual(done, ["substitute"])
        self.assertIn(out, T.NEIGHBOURS["a"])

    def test_deterministic(self):
        units, keys = ["ni", "hao", "ma"], "nihaoma"
        a = [T.add_typos(random.Random(9), keys, units, CONFIG) for _ in range(3)]
        self.assertEqual(a[0], a[1])
        self.assertEqual(a[1], a[2])

    def test_full_pinyin(self):
        syllables = {"xi", "an", "xian", "shi", "si", "zhong", "guo", "a"}
        self.assertTrue(T.full_pinyin("xian", syllables))
        self.assertTrue(T.full_pinyin("xi'an", syllables))
        self.assertTrue(T.full_pinyin("sizhongguo", syllables))
        self.assertFalse(T.full_pinyin("xixn", syllables))
        self.assertFalse(T.full_pinyin("zhonguo", syllables))


if __name__ == "__main__":
    unittest.main()
