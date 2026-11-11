"""ime-predict-zh-v1 decoder: beam search against brute force, healing, key constraints, simulation."""
import itertools
import sys
import unittest
import zlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.benchmarks import predict_zh as P  # noqa: E402

TEXTS = [None, None, "我", "们", "去", "我们", "吃饭", "，"]
BOS, EOS = 0, 1
READINGS = {"我": {"wo"}, "们": {"men"}, "去": {"qu"}, "吃": {"chi"}, "饭": {"fan"}}
UNBOUNDED = 10 ** 6


class FakeLM:
    """Pseudo-random but deterministic next-token log-probabilities over a tiny vocabulary."""
    context_length = 64

    def __init__(self):
        self.vocabulary = P.Vocabulary(TEXTS, "cpu")

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            best = max((t for t in range(2, len(TEXTS)) if text.startswith(TEXTS[t], i)),
                       key=lambda t: len(TEXTS[t]))
            ids.append(best)
            i += len(TEXTS[best])
        return ids

    def context(self, turns, prefix):
        ids = [BOS]
        for turn in turns:
            ids += self.encode(turn) + [EOS, BOS]
        return ids + self.encode(prefix)

    def row(self, sequence):
        seed = zlib.crc32(bytes(str(sequence), "ascii"))
        logits = torch.randn(len(TEXTS), generator=torch.Generator().manual_seed(seed))
        return logits.log_softmax(-1)

    def logprobs(self, sequences):
        return torch.stack([self.row(s) for s in sequences])


def brute_force(lm, turns, prefix, constraint=None):
    """Every token sequence, scored exactly; the decoder with an unbounded beam must match."""
    ids, required = P.healing(lm, lm.context(turns, prefix), prefix)
    max_chars = constraint.max_chars if constraint else P.MAX_CHARS
    han = [t for t in range(len(TEXTS)) if TEXTS[t] and P.all_han(TEXTS[t])]
    best = {}
    for length in range(1, len(required) + max_chars + 1):
        for tokens in itertools.product(han, repeat=length):
            spelled, ok = "", True
            for t in tokens:
                before = spelled
                spelled += TEXTS[t]
                if len(spelled) <= len(required):
                    ok &= required.startswith(spelled)
                else:
                    ok &= spelled.startswith(required)
                    if ok and constraint:
                        done = max(0, len(before) - len(required))
                        ok &= constraint.chars_ok(READINGS, spelled[max(len(before), len(required)):], done)
            new = spelled[len(required):]
            if not ok or not new or len(new) > max_chars:
                continue
            if not (constraint.candidate_length_ok(len(new)) if constraint else P.MIN_CHARS <= len(new)):
                continue
            score = sum(float(lm.row(ids + list(tokens[:k]))[tokens[k]]) for k in range(length))
            best[new] = max(best.get(new, P.NEG), score)
    return P.select(best)


class DecoderTest(unittest.TestCase):
    def assert_same(self, got, want):
        self.assertEqual([t for t, _ in got], [t for t, _ in want])
        for (_, a), (_, b) in zip(got, want):
            self.assertAlmostEqual(a, b, places=4)

    def test_unbounded_beam_equals_brute_force(self):
        lm = FakeLM()
        for prefix in ("", "去", "吃饭，"):
            with self.subTest(prefix=prefix):
                got = P.predict(lm, ["我们去"], prefix, beam=UNBOUNDED)
                self.assertTrue(got)
                self.assert_same(got, brute_force(lm, ["我们去"], prefix))

    def test_narrow_beam_keeps_the_interface(self):
        got = P.predict(FakeLM(), [], "去", beam=2)
        self.assertLessEqual(len(got), P.TOP)
        for text, score in got:
            self.assertTrue(P.all_han(text) and 2 <= len(text) <= 6 and score <= 0)

    def test_healing_takes_back_the_last_chinese_token(self):
        lm = FakeLM()
        ids, required = P.healing(lm, lm.context([], "去我们"), "去我们")
        self.assertEqual((ids, required), ([BOS, 4], "我们"))
        self.assertEqual(P.healing(lm, lm.context([], "吃饭，"), "吃饭，")[1], "")
        self.assert_same(P.predict(lm, [], "去我们", beam=UNBOUNDED), brute_force(lm, [], "去我们"))

    def test_constraints_equal_brute_force(self):
        lm = FakeLM()
        for constraint in (P.Constraint("initials", "wm"), P.Constraint("initials", "cfq"),
                           P.Constraint("syllable", "chi"), P.Constraint("syllable", "wo")):
            for prefix in ("去", "去我"):
                with self.subTest(constraint=constraint, prefix=prefix):
                    got = P.predict(lm, [], prefix, constraint=constraint, readings=READINGS, beam=UNBOUNDED)
                    self.assert_same(got, brute_force(lm, [], prefix, constraint))
                    for text, _ in got:
                        if constraint.kind == "initials":
                            self.assertEqual(len(text), len(constraint.keys))
                        else:
                            self.assertIn(constraint.keys, READINGS[text[0]])

    def test_select_drops_prefix_related_candidates(self):
        chosen = P.select({"我们": -1.0, "我们去": -0.5, "我": -2.0, "去": -3.0, "吃饭": -4.0}, top=5)
        self.assertEqual([t for t, _ in chosen], ["我们去", "去", "吃饭"])


class SimulationTest(unittest.TestCase):
    def test_simulate_counts_keys_shows_and_clicks(self):
        message, costs, ends = "我们去吃饭", [2, 3, 2, 3, 3], [2, 3, 5]
        table = {0: [("我们", -1.0)], 2: [("吃饭", -1.0)], 3: [("吃饭", -0.5)]}
        result = P.simulate(message, costs, ends, lambda p: table.get(p, []), threshold=P.NEG)
        # nothing offered before the first commit: 我们 typed (5), 去 typed (2) because 吃饭 is
        # not a prefix there, 吃饭 clicked (1)
        self.assertEqual(result, {"keys": 8, "baseline_keys": 13, "shows": 2, "clicks": 1})
        high = P.simulate(message, costs, ends, lambda p: table.get(p, []), threshold=0.0)
        self.assertEqual(high, {"keys": 13, "baseline_keys": 13, "shows": 0, "clicks": 0})

    def test_click_takes_the_longest_hit_and_can_end_mid_word(self):
        message, costs, ends = "我们去吃饭", [2, 3, 2, 3, 3], [2, 3, 5]
        table = {2: [("去", -0.5), ("去吃", -1.0)], 4: [("饭饭", -1.0)]}
        result = P.simulate(message, costs, ends, lambda p: table.get(p, []), threshold=P.NEG)
        # 我们 typed (5), 去吃 clicked (1), then 饭 typed to the word end (3)
        self.assertEqual(result, {"keys": 9, "baseline_keys": 13, "shows": 2, "clicks": 1})

    def test_keystrokes_and_boundaries(self):
        costs = P.keystrokes("我们，ok去", lambda run: {"我们": ["wo", "men"], "去": ["qu"]}[run])
        self.assertEqual(costs, [2, 3, 1, 1, 1, 2])
        self.assertEqual(P.boundaries("我们去", lambda m: ["我们", "去"]), [2, 3])
        with self.assertRaises(ValueError):
            P.boundaries("我们去", lambda m: ["我们"])


class FakeProcessor:
    pieces = ["<pad>", "<unk>", "<s>", "</s>", "我", "们", "去", "<0x41>"]

    def get_piece_size(self):
        return len(self.pieces)

    def id_to_piece(self, i):
        return self.pieces[i]

    def is_control(self, i):
        return i in (0, 2, 3)

    def is_unknown(self, i):
        return i == 1

    def is_byte(self, i):
        return i == 7

    def encode(self, text):
        return [self.pieces.index(c) for c in text]


class TinyModel:
    class config:  # noqa: N801 - mirrors model.config
        context_length = 20


class AdapterAndMetricsTest(unittest.TestCase):
    def test_vime_context_joins_turns_and_drops_the_oldest(self):
        lm = P.VimeLM(TinyModel(), FakeProcessor(), "cpu", {"bos": 2, "eos": 3, "pad": 0})
        self.assertIsNone(lm.vocabulary.texts[7])
        self.assertEqual(lm.vocabulary.texts[4], "我")
        self.assertEqual(lm.context(["我们", "去"], ""), [2, 4, 5, 3, 2, 6, 3, 2])
        # limit 20 - 12 = 8 tokens: the oldest turn no longer fits
        self.assertEqual(lm.context(["我们", "去"], "我们"), [2, 6, 3, 2, 4, 5])
        self.assertEqual(len(lm.context([], "我们去" * 5)), 8)

    def test_metrics(self):
        rows = [{"source": "a", "r": 1}, {"source": "a", "r": None}, {"source": "b", "r": 3}]
        out = P.rates(rows, "r")
        self.assertEqual((out["top1"]["hits"], out["top3"]["hits"]), (1, 2))
        self.assertAlmostEqual(out["top3"]["macro"], (0.5 + 1.0) / 2)
        self.assertEqual(P.rank_of_hit([("我们去吃", -1), ("我们", -2)], "我们去"), (2, 2))
        curve = P.coverage_curve([{"id": "x", "confidence": -1, "rank": 1},
                                  {"id": "y", "confidence": -5, "rank": None}], points=(0.5,))
        self.assertEqual(curve["50%"], {"items": 1, "top1": 1.0})

    def test_readings_parser(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for name, body in (("8105.dict.yaml", "---\nname: x\n...\n我\two\t9\n绿\tlv\t5\n绿\tlu\t1\n"),
                               ("41448.dict.yaml", "---\n...\n# c\n𠀀\tqiu\t1\n")):
                Path(tmp, name).write_text(body, encoding="utf-8")
            self.assertEqual(P.load_readings(tmp), {"我": {"wo"}, "绿": {"lv", "lu"}, "𠀀": {"qiu"}})


if __name__ == "__main__":
    unittest.main()
