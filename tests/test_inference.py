"""Legal token sampling, EOS termination and sentence-local context limits."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.infer import JapaneseLM, filtered_logits


class FakeProcessor:
    def encode(self, text, out_type=int):
        return [4 if char == "a" else 5 for char in text]

    def decode(self, ids):
        return "".join("a" if token == 4 else "b" for token in ids if token not in (0, 1, 2, 3))


class FakeModel:
    def __init__(self, logits, context):
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.config = SimpleNamespace(context_length=context)

    def __call__(self, inputs):
        return self.logits.expand(1, inputs.shape[1], -1)


def fake_lm(logits, context=8):
    lm = JapaneseLM.__new__(JapaneseLM)
    lm.device = torch.device("cpu")
    lm.model = FakeModel(logits, context)
    lm.processor = FakeProcessor()
    lm.special = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
    lm.forbidden = (0, 1, 2)
    return lm


class InferenceTests(unittest.TestCase):
    def test_sampling_filters_specials_and_keeps_a_token_even_for_tiny_top_p(self):
        logits = filtered_logits(torch.tensor([100., 90., 80., 7., 6., 5.]), (0, 1, 2), top_k=2, top_p=0.001)
        self.assertEqual(torch.isfinite(logits).nonzero().flatten().tolist(), [3])

    def test_eos_stops_without_changing_prefix(self):
        result = fake_lm([100, 90, 80, 10, 2, 1]).generate("a", temperature=0)
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(result["new_token_ids"], [3])
        self.assertEqual(result["text"], "a")

    def test_context_limit_does_not_slide_or_truncate_prompt(self):
        lm = fake_lm([100, 90, 80, -100, 10, 1], context=3)
        result = lm.generate("a", max_new_tokens=20, temperature=0)
        self.assertEqual(result["stop_reason"], "context_limit")
        self.assertEqual(result["new_token_ids"], [4, 4])
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            lm.generate("aaaa")

    def test_sampling_seed_is_local_and_reproducible(self):
        lm = fake_lm([100, 90, 80, -100, 1, 1], context=16)
        first = lm.generate("a", max_new_tokens=8, top_k=0, top_p=1, seed=42)
        second = lm.generate("a", max_new_tokens=8, top_k=0, top_p=1, seed=42)
        self.assertEqual(first["new_token_ids"], second["new_token_ids"])
        self.assertTrue(all(token in (4, 5) for token in first["new_token_ids"] ))

    def test_foreign_token_manifest_is_rejected_before_model_load(self):
        saved = {"format": "vimeml_tiny_gpt_v1", "config": {"token_dir": "unused"},
                 "signatures": {"tokens": "expected"}}
        with patch("vimeml.training.infer.torch.load", return_value=saved), \
             patch("vimeml.training.infer.file_sha", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
                JapaneseLM("unused.pt", "unused")


if __name__ == "__main__":
    unittest.main()
