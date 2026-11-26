"""Targeted unit tests for Stage 3 colloquial token encoding and benchmark preparation."""
from collections import Counter
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/ja/corpus"))
sys.path.insert(0, str(ROOT / "scripts/ja/tokenizer"))
sys.path.insert(0, str(ROOT / "scripts/ja/benchmarks"))

import encode_ja_colloq as ejc  # noqa: E402
import prepare_kana_colloq_ja as pkc  # noqa: E402
from vimeml.posttraining.zh_pinyin_data import Part, merge  # noqa: E402
from vimeml.training.joint_ja_kana import Schedule, Store  # noqa: E402


class ToyCharSP:
    """Minimal roundtripping SentencePiece stub for Japanese encoding tests."""

    def __init__(self, chars):
        self.pieces = ["<pad>", "<unk>", "<s>", "</s>", "<CTX>", "<READ>", "<OUT>", "\n"] + list(chars)
        self.p2i = {p: i for i, p in enumerate(self.pieces)}

    def encode(self, text):
        return [self.p2i.get(c, 1) for c in text]

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids if i >= 8)


def fake_analyse(run):
    """Deterministic segmenter stub returning per-character hiragana/kanji readings."""
    table = {
        "今": "キョ", "日": "ウ", "私": "ワタシ", "行": "イ", "食": "タ",
        "見": "ミ", "話": "ハナ", "思": "オモ", "買": "カ", "書": "カ",
        "明": "アシタ", "本": "ホン", "人": "ヒト", "時": "トキ", "気": "キ",
    }
    segs = []
    for i, ch in enumerate(run):
        if "ぁ" <= ch <= "ゖ":
            r = chr(ord(ch) + 0x60)
        elif "ァ" <= ch <= "ヶ" or ch in "ー、。！？「」":
            r = ch
        else:
            r = table.get(ch)
        segs.append((i, i + 1, r, True))
    return segs


class TestJaColloqPipeline(unittest.TestCase):
    def setUp(self):
        chars = sorted(set(
            "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
            "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽゃゅょっー、。！？「」"
            "今日私行食見話思買書明本人時気笑"
        ))
        self.sp = ToyCharSP(chars)
        self.cfg = {
            "label": "ja-colloq-test",
            "plain_fraction": 0.35,
            "kind_weights": {"span": 3, "sentence": 2},
            "plain_minimum_tokens": 4,
            "sample": {
                "target_chars": [4, 30],
                "reading_chars": [4, 64],
                "context_chars": 64,
                "run_chars": [6, 100],
                "end_at_run_end_probability": 0.5,
                "runs_tried": 3,
                "sentences_tried": 3,
            },
        }

    def test_encode_paragraph_and_store_schedule_roundtrip(self):
        texts = [
            "今日は私と行くよね！まじで楽しみなんだよね。",
            "明日また話そうかな、本も買っちゃったし。",
            "今日見に行った人がすごく優しかったじゃん。",
            "思っていたより早く書けちゃったよ。",
            "今日食べるものを買っておくね！",
            "明日の時までに話をまとめなきゃね。",
            "本を見て思いついたんだけどさ、これいいよね。",
            "私が行く前に食べておいたよ。",
            "今日見た話が面白すぎたんだよね。",
            "明日も私と一緒に行くでしょ？",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = root / "parts"
            out = root / "store"
            with mock.patch("kana_samples.analyse", side_effect=fake_analyse):
                for split in ("train", "validation"):
                    part, stats = Part(ejc.KINDS), Counter()
                    exclude = {ejc.line_key(texts[0])} if split == "train" else set()
                    for idx, t in enumerate(texts * (3 if split == "train" else 2)):
                        ejc.encode_paragraph(self.sp, self.cfg, part, stats, f"d{idx}#0", t, split, exclude)
                    part.save(parts / f"{split}-0")
                    info = merge(parts, [f"{split}-0"], out, split, ejc.KINDS)
                    for k in ejc.KINDS:
                        self.assertGreater(info[k]["examples"], 0, f"{split}/{k} empty")
                    if split == "train":
                        self.assertEqual(stats["validation_duplicates"], 3)

            # Verify Store & Schedule from joint_ja_kana read the colloquial store directly
            for k in ejc.KINDS:
                st = Store(out / k, "train")
                self.assertGreater(len(st), 0)
                ids, first = st.get(0)
                if k == "plain":
                    self.assertEqual(first, 1)
                else:
                    self.assertGreater(first, 1)
                    self.assertIn(5, ids)
                    self.assertIn(6, ids)

            sched = Schedule(out, {"plain": 2, "span": 2, "sentence": 1}, seed=7, crop_probability=0.0,
                             extra=[(out, {"plain": 1, "span": 1, "sentence": 1})], ce_only_sources=[1])
            batch = sched[0]
            self.assertEqual(batch["plain"]["input_ids"].shape[0], 3)
            self.assertEqual(batch["span"]["hard"].tolist(), [False, False, True])

    def test_colloquial_benchmark_build_screen_and_freeze(self):
        prefixes = ["あれ、", "ねえ、", "そういえば、", "ちょっと、", "じつは、", "なるほど、", "やっぱり、", "たしか、"]
        subjects = ["今日は私と", "明日あの人と", "今日は本を", "明日は話を", "今日は気ままに"]
        endings = ["行くよね！", "買っちゃったよ。", "食べなきゃね。", "見ておいたじゃん。", "書けちゃったかな。"]
        records = []
        idx = 0
        for p in prefixes:
            for s in subjects:
                e = endings[idx % len(endings)]
                text = f"{p}{s}{e}まじで楽しみなんだよね。"
                records.append({
                    "source": "web_colloq",
                    "document_id": f"doc-{idx:03d}",
                    "group_id": f"web_colloq:doc-{idx:03d}",
                    "text": text,
                    "p_chat": 0.9,
                    "p_spoken": 0.95,
                })
                idx += 1

        pool, stats = pkc.build_pool(records, seed=42, pool_cap_per_role=20, analyser=fake_analyse)
        self.assertGreaterEqual(len(pool), 20)
        self.assertTrue(any(r["provenance"]["colloquial_marker"] for r in pool))

        # Test overlap screening skips own_doc and catches other-doc containment
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            train_jsonl = tdir / "train.jsonl"
            leaked_msg = pool[0]["provenance"]["cleaned_message"]
            own_msg = pool[1]["provenance"]["cleaned_message"]
            own_doc = pool[1]["provenance"]["document_id"]
            train_jsonl.write_text(
                json.dumps({"id": "other-train-doc", "text": leaked_msg}, ensure_ascii=False) + "\n"
                + json.dumps({"id": own_doc, "text": own_msg}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            excluded, audit = pkc.screen_overlap(pool, [train_jsonl])
            self.assertIn(0, excluded)
            self.assertNotIn(1, excluded)

            frozen = tdir / "ime-kana-ja-colloq-v1"
            manifest, m_sha = pkc.freeze_benchmark(pool, excluded, frozen, per_split=4, seed=43)
            self.assertEqual(manifest["splits"], {"development": 4, "test": 4})
            self.assertTrue((frozen / "development/evaluation_items.json").is_file())
            self.assertTrue((frozen / "development/ajimee-input.json").is_file())
            self.assertTrue((frozen / "test/evaluation_items.json").is_file())
            self.assertTrue((frozen / "SHA256SUMS").is_file())
            self.assertEqual(len(m_sha), 64)
            dev_items = json.loads((frozen / "development/evaluation_items.json").read_text("utf-8"))
            test_items = json.loads((frozen / "test/evaluation_items.json").read_text("utf-8"))
            dev_g = {r["provenance"]["group_id"] for r in dev_items}
            test_g = {r["provenance"]["group_id"] for r in test_items}
            self.assertEqual(dev_g & test_g, set())


if __name__ == "__main__":
    unittest.main()
