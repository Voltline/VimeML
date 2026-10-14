"""V4 contracts: unseen kana, fixed IDs, data isolation and shared sampling."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.tokenizer import v4


class V4TokenizerTests(unittest.TestCase):
    def test_rejects_held_out_paths_before_opening(self):
        for path in ("missing/blind/evaluation_items.json", "missing/test.txt", "missing/test/file"):
            with self.assertRaises(ValueError):
                v4.permitted_path(path)
        with self.assertRaises(ValueError):
            v4.measure_development(None, ROOT / "missing/blind")

    def test_sample_is_train_only_deterministic_and_validated(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temp:
            root = Path(temp)
            path = root / "train.txt"
            lines = [f"第{i}行。日本語🙂" for i in range(300)]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            expected = {"sentences": len(lines), "characters": sum(map(len, lines))}
            first = v4.sample_train(path, root / "a.txt", 101, 42, expected, 8192, lambda *_: None)
            second = v4.sample_train(path, root / "b.txt", 101, 42, expected, 8192, lambda *_: None)
            self.assertEqual(first, second)
            self.assertEqual(first["sample_sentences"], 101)
            self.assertEqual(first["train_sha256"], v4.file_sha(path))
            self.assertEqual((root / "a.txt").read_bytes(), (root / "b.txt").read_bytes())
            self.assertTrue(set((root / "a.txt").read_text(encoding="utf-8").splitlines()) <= set(lines))
            path.write_text("<CTX>予約\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                v4.sample_train(path, root / "bad.txt", 101, 42, expected, 8192, lambda *_: None)
            path.write_text("長" * 10 + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                v4.sample_train(path, root / "bad.txt", 101, 42, expected, 20, lambda *_: None)

    def test_native_unigram_preserves_unseen_kana_and_normalization(self):
        import sentencepiece as spm
        kana = "".join((ROOT / "configs/kana-v4.txt").read_text(encoding="utf-8").splitlines())
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temp:
            root = Path(temp)
            path = root / "train.txt"
            path.write_text("\n".join(f"今日は{i}番の文章です。日本語を学びます。 ABC 123" for i in range(300)), encoding="utf-8")
            prefix = str(root / "tokenizer")
            spm.SentencePieceTrainer.train(input=str(path), model_prefix=prefix, vocab_size=1024,
                hard_vocab_limit=False, model_type="unigram", required_chars=kana,
                byte_fallback=True, normalization_rule_name="identity", add_dummy_prefix=False,
                remove_extra_whitespaces=False, pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                user_defined_symbols=v4.MARKERS, num_threads=2, minloglevel=2)
            model = root / "tokenizer.model"
            processor = spm.SentencePieceProcessor(model_file=str(model))
            report = v4.verify_model(model, processor.get_piece_size(), kana)
            self.assertEqual(report["required_kana_count"], 576)
            self.assertTrue(all(item["passed"] for item in report["kana_checks"]))
            stats = v4.text_statistics(processor, ["日本語🙂", "か\u3099", "ｶﾞ", "  空白  "])
            self.assertEqual(stats["unknown_tokens"], 0)
            self.assertEqual(stats["roundtrip_mismatches"], 0)
            self.assertEqual(stats["mean_characters_retained_in_128_with_bos_eos"], (4 + 2 + 2 + 6) / 4)
            with self.assertRaises(ValueError):
                v4.verify_model(model, 8192, kana)
            # A mismatched frozen inventory must be rejected even if byte fallback can encode it.
            with self.assertRaises(ValueError):
                v4.verify_model(model, processor.get_piece_size(), kana + "漢")


if __name__ == "__main__":
    unittest.main()
