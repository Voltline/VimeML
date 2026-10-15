"""Only the conversion contract's consequential boundary and mask cases."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.posttraining.data import ConversionCodec, shifted
from vimeml.posttraining.readings import DualReading, hiragana


class ConversionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codec = ConversionCodec(ROOT / "artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model")
        cls.reader = DualReading()

    def test_first_output_and_eos_supervised_once(self):
        ids, first, removed = self.codec.encode("昨日友達と", "はなした", "話した")
        sample = shifted(ids, first)
        self.assertEqual(sample["input_ids"][first - 1], 6)
        self.assertEqual(sample["labels"][first - 1:], self.codec.sp.encode("話した") + [3])
        self.assertTrue(all(x == -100 for x in sample["labels"][:first - 1]))
        self.assertEqual(removed, 0)

    def test_long_context_only_cropped(self):
        ids, first, removed = self.codec.encode("昨日友達と" * 100, "はなした", "話した")
        self.assertGreater(removed, 0)
        self.assertLessEqual(len(ids), 129)
        self.assertEqual(ids[first:], self.codec.sp.encode("話した") + [3])
        with self.assertRaises(ValueError):
            self.codec.encode("<READ>", "あ", "亜")

    def test_real_offsets_spaces_and_particle_orthography(self):
        text = "昨日 友達と話した。私は学校へ行く。"
        left, right = self.reader.annotate(text)
        for tokens in (left, right):
            for start, end, reading, valid in tokens:
                if text[start:end] in ("は", "へ"):
                    self.assertEqual(reading, text[start:end])
                if text[start:end] == "友達":
                    self.assertEqual(start, 3)
        self.assertEqual(hiragana("ガッコウへ"), "がっこうへ")

    def test_span_is_original_inflected_surface(self):
        text = "昨日友達と話した。"
        for seed in range(10):
            ex, reason = self.reader.example(text, seed)
            self.assertIsNone(reason)
            self.assertEqual(ex["output"], text[slice(*ex["span"])])
            self.assertEqual(ex["context"], text[:ex["span"][0]])
            self.assertEqual(ex["mecab_reading"], ex["sudachi_reading"])


if __name__ == "__main__":
    unittest.main()
