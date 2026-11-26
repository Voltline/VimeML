"""Targeted unit tests for vimeml-ja-v2-20m Mac Core ML deployment handoff."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/ja/deployment"))

import prepare_ja_v2_20m as pj  # noqa: E402
from vimeml.deployment import zh_v1  # noqa: E402
from vimeml.deployment.ja_search import KanaIndex  # noqa: E402
from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2  # noqa: E402


class ToyJaSP:
    def __init__(self):
        self.pieces = [
            "<pad>", "<unk>", "<s>", "</s>", "<CTX>", "<READ>", "<OUT>", "\n",
            "き", "ょ", "う", "は", "い", "今", "日", "今日", "行", "く", "わ", "た", "し", "私",
            "あ", "し", "た", "の", "明", "駅", "前", "本", "買", "休", "家", "戻", "楽",
            "お", "つ", "か", "れ", "さ", "ま", "で", "て", "ん", "ね", "す", "そ", "ほ", "と", "も", "ろ", "よ",
        ]
        # Keep unique pieces preserving order
        seen, uniq = set(), []
        for p in self.pieces:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        self.pieces = uniq
        self.p2i = {p: i for i, p in enumerate(self.pieces)}

    def bos_id(self):
        return 2

    def pad_id(self):
        return 0

    def get_piece_size(self):
        return len(self.pieces)

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            if text[i:i + 2] in self.p2i:
                out.append(self.p2i[text[i:i + 2]])
                i += 2
            elif text[i] in self.p2i:
                out.append(self.p2i[text[i]])
                i += 1
            else:
                out.append(8)  # deterministic valid token for stub prose
                i += 1
        return out

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids if i >= 8)


class CachedTinyModel(TinyGPTV2):
    """Adds TinyGPTZh13-compatible new_cache / forward_cached for fast CPU handoff tests."""

    def new_cache(self, batch_size=1):
        return {"ids": []}

    @torch.inference_mode()
    def forward_cached(self, input_ids, cache):
        cache["ids"].append(input_ids.long())
        seq = torch.cat(cache["ids"], dim=1)
        return self(seq)[:, -input_ids.shape[1]:, :]


class TestJaDeploymentHandoff(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.sp = ToyJaSP()
        cfg = GPTV2Config(
            vocab_size=self.sp.get_piece_size(),
            context_length=32,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            dropout=0.0,
        )
        self.model = CachedTinyModel(cfg).eval()
        self.index = KanaIndex({
            "きょう": [self.sp.p2i["今日"]],
            "は": [self.sp.p2i["は"]],
            "い": [self.sp.p2i["行"], self.sp.p2i["い"]],
            "く": [self.sp.p2i["く"]],
            "わたし": [self.sp.p2i["私"]],
        })

    def test_zh_v1_releases_accepts_ja_v2_20m(self):
        self.assertIn("vimeml-ja-v2-20m", zh_v1.RELEASES)

    def test_prepare_weights_strips_optimizer_and_verifies_kv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "final.pt"
            deploy = root / "deployment.pt"
            tok_dir = root / "tok"
            tok_dir.mkdir()
            tok_model = tok_dir / "tokenizer.model"
            tok_model.write_bytes(b"fake-spm-bytes")
            ref_dir = root / "fp32-ref"

            payload = {
                "format": "vimeml_tiny_gpt_zh13",
                "architecture": "tiny_gpt_zh13",
                "model": self.model.state_dict(),
                "model_config": self.model.configuration(),
                "step": 30000,
                "signatures": {"train": "sig"},
                "tokenizer_identity": {"tokenizer_model_sha256": pj.sha256(tok_model), "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3}},
                "config": {"arm": "20m-colloq-c40", "architecture": "tiny_gpt_zh13", "stage": "ja-kana"},
                "optimizer": {"state": {"dummy": torch.zeros(4)}},
            }
            torch.save(payload, ckpt)

            def fake_loader(path, tok, device):
                return self.model, self.sp, {"step": 30000}

            res = pj.prepare_weights(
                ckpt, deploy, tok_dir, ref_dir,
                checkpoint_sha256=pj.sha256(ckpt),
                tokenizer_sha256=pj.sha256(tok_model),
                arm="20m-colloq-c40",
                loader=fake_loader,
            )
            self.assertEqual(res["release_name"], "vimeml-ja-v2-20m")
            self.assertFalse(res["optimizer_included"])
            self.assertLessEqual(res["max_abs_kv_vs_stateless"], 3e-4)

            saved_deploy = torch.load(deploy, map_location="cpu", weights_only=True)
            self.assertNotIn("optimizer", saved_deploy)
            self.assertTrue(saved_deploy["deployment_only"])
            self.assertTrue((ref_dir / "logits.npz").is_file())
            ref = json.loads((ref_dir / "reference.json").read_text("utf-8"))
            self.assertEqual(ref["format"], "vimeml_zh_v1_mac_preparation_reference_v1")
            self.assertEqual(ref["release_name"], "vimeml-ja-v2-20m")

    def test_select_fixture_cases_build_fixture_and_package_zip(self):
        fake_manifest = {"split": "development", "sealed": False, "release_id": "rel"}
        fake_rows = [
            {"id": f"dev-{i}", "source": "wrime", "group": f"g{i}",
             "left_context": "わたし" if i % 2 == 0 else "",
             "query": "キョウハイク", "answers": ["今日は行く"]}
            for i in range(6)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("vimeml.benchmarks.kana_gen_ja.load_dev_pool", return_value=(fake_manifest, fake_rows)):
                cases = pj.select_fixture_cases(root, dev_count=4, seed=1)
            self.assertEqual(len(cases), 4 + len(pj.HANDWRITTEN_CASES))

            fix_dir = root / "fixture"
            meta = pj.build_generation_fixture(
                self.model, self.sp, self.index, cases[:4], fix_dir,
                checkpoint_sha256="a" * 64, bundle_manifest_sha256="b" * 64,
                tokenizer_sha256="c" * 64, index_sha256="d" * 64, reading_table_sha256="e" * 64,
                phase="test", beam=4, topk=3, lam=1.6,
            )
            self.assertEqual(meta["format"], "vime_ja_generation_fixture_v1")
            self.assertFalse(meta["test_read"])
            self.assertEqual(meta["cases"], 4)
            lines = (fix_dir / "cases.jsonl").read_text("utf-8").splitlines()
            self.assertEqual(len(lines), 4)
            first = json.loads(lines[0])
            self.assertIn("prompt_ids", first)
            self.assertIn("rescored", first)

            out_zip = root / "ja-handoff.zip"
            pkg = pj.stage_package(out_zip, pj.collect_tree(fix_dir, "outputs/vime-fixtures/ja-test"))
            self.assertEqual(pkg["files"], 3)  # cases.jsonl + meta.json + SHA256SUMS
            with zipfile.ZipFile(out_zip) as z:
                names = set(z.namelist())
                self.assertIn("SHA256SUMS", names)
                self.assertIn("outputs/vime-fixtures/ja-test/cases.jsonl", names)
                self.assertIn("outputs/vime-fixtures/ja-test/meta.json", names)


if __name__ == "__main__":
    unittest.main()
