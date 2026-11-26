"""Unit tests for the Japanese colloquial paragraph pre-filter and band assignment."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/ja/corpus"))

import ja_colloq  # noqa: E402


class TestJaColloq(unittest.TestCase):
    def test_features_distinguishes_formal_vs_colloquial(self):
        formal = "本規約は、利用者が当サービスを使用する際の条件を定めるものである。"
        colloq = "それめっちゃ分かるよね！今度一緒に食べに行こうよ〜絶対楽しいじゃん（笑）"
        f_formal = ja_colloq.features(formal)
        f_colloq = ja_colloq.features(colloq)
        self.assertIsNotNone(f_formal)
        self.assertIsNotNone(f_colloq)
        self.assertEqual(f_formal[1], 0)
        self.assertEqual(ja_colloq.band(f_formal[2]), "zero")
        self.assertGreaterEqual(f_colloq[1], 3)
        self.assertEqual(ja_colloq.band(f_colloq[2]), "high")

    def test_short_or_kanji_only_rejected(self):
        self.assertIsNone(ja_colloq.features("ありがとう"))  # 5 chars < MIN_JA=6
        self.assertIsNone(ja_colloq.features("株式会社東京都渋谷区"))  # 0% kana

    def test_contractions_detected(self):
        text = "もう忘れちゃったんだよね、早く確認しなきゃ！"
        f = ja_colloq.features(text)
        self.assertIsNotNone(f)
        self.assertGreaterEqual(f[1], 2)
        self.assertIn(ja_colloq.band(f[2]), ("mid", "high"))


if __name__ == "__main__":
    unittest.main()
