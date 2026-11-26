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
                    sub_part, sub_stats = Part(ejc.KINDS), Counter()
                    exclude = {ejc.line_key(texts[0])} if split == "train" else set()
                    for idx, t in enumerate(texts * (3 if split == "train" else 5)):
                        in_sub = (idx % 2 == 0)
                        ejc.encode_paragraph(
                            self.sp, self.cfg, part, stats, f"d{idx}#0", t, split, exclude,
                            sub_part=sub_part if in_sub else None,
                            sub_stats=sub_stats if in_sub else None,
                        )
                    part.save(parts / f"{split}-0")
                    sub_part.save(root / "sub_parts" / f"{split}-0")
                    info = merge(parts, [f"{split}-0"], out, split, ejc.KINDS)
                    sub_info = merge(root / "sub_parts", [f"{split}-0"], root / "sub_store", split, ejc.KINDS)
                    for k in ejc.KINDS:
                        self.assertGreater(info[k]["examples"], 0, f"{split}/{k} empty")
                        self.assertGreater(sub_info[k]["examples"], 0, f"{split}/{k} sub empty")
                        self.assertLessEqual(sub_info[k]["examples"], info[k]["examples"])
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

            # Test LLM-confirmed freeze filtering and by_rule_band metadata
            llm_path = tdir / "llm-labels.jsonl"
            kept_lines = []
            for row in pool:
                cid = row["provenance"]["case_id"]
                idx = int(cid.split("-")[1].split("#")[0])
                lbl = 3 if idx % 3 == 0 else (2 if idx % 3 == 1 else 1)
                kind = "chat" if lbl >= 2 else "formal"
                kept_lines.append(json.dumps({"id": cid, "label": lbl, "kind": kind}))
            llm_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
            frozen_llm = tdir / "ime-kana-ja-colloq-llm"
            m_llm, _ = pkc.freeze_benchmark(
                pool, excluded, frozen_llm, per_split=4, seed=44,
                llm_labels=llm_path, include_label2_dialogue=False,
            )
            self.assertTrue(m_llm["llm_confirmed"])
            self.assertEqual(m_llm["label_quality"], "llm_confirmed_d4_d5")
            self.assertGreater(sum(m_llm["by_rule_band"]["development"].values()), 0)
            llm_dev = json.loads((frozen_llm / "development/evaluation_items.json").read_text("utf-8"))
            self.assertTrue(all(r["provenance"]["llm_label"] == 3 for r in llm_dev))

    def test_export_chat_fraction_and_audit_sample(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow not installed in workspace venv")
        import ja_colloq as jc

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            sel = tdir / "select"
            sel.mkdir()
            rows = [
                {"id": f"doc-{i:03d}", "para": 0, "text": f"今日は楽しいね！番号{i}",
                 "p_chat": 0.9 if i % 2 == 0 else 0.75, "p_spoken": 0.85,
                 "split": "train" if i < 24 else "heldout"}
                for i in range(30)
            ]
            pq.write_table(pa.Table.from_pylist(rows), sel / "shard-0.parquet")
            exp_full = tdir / "exp-full"
            exp_half = tdir / "exp-half"
            jc.export(sel, exp_full, chat_min=0.70, chat_fraction=1.0, spoken_units=0)
            jc.export(sel, exp_half, chat_min=0.70, chat_fraction=0.5, spoken_units=0)
            m_full = json.loads((exp_full / "manifest.json").read_text("utf-8"))
            m_half = json.loads((exp_half / "manifest.json").read_text("utf-8"))
            self.assertEqual(m_full["chat_fraction"], 1.0)
            self.assertEqual(m_half["chat_fraction"], 0.5)
            self.assertLess(m_half["counts"]["train_chat_rows"], m_full["counts"]["train_chat_rows"])
            audit_out = tdir / "audit.jsonl"
            jc.audit_sample(exp_full, audit_out, count=10, seed=123)
            audit_rows = [json.loads(l) for l in audit_out.read_text("utf-8").splitlines() if l.strip()]
            self.assertEqual(len(audit_rows), 10)


if __name__ == "__main__":
    unittest.main()
