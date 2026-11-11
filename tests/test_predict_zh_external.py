"""Step-0 adapters of ime-predict-zh-v1: cached HF scoring and the character 5-gram."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks import predict_zh as P  # noqa: E402
from vimeml.benchmarks.predict_zh_external import CharNgramLM, HFLM, sample_fraction  # noqa: E402

CHARS = "我们明天一起去吃饭看电影好的"


class Tokenizer:
    """One token per character; ids 0-2 are special."""
    all_special_ids = [0, 1, 2]

    def __len__(self):
        return 3 + len(CHARS)

    def decode(self, ids):
        return "".join(CHARS[i - 3] if i >= 3 else "�" for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [CHARS.index(c) + 3 if c in CHARS else 1 for c in text]


def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=24, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return Qwen2ForCausalLM(config)


class Hf(unittest.TestCase):
    def setUp(self):
        self.lm = HFLM(tiny_qwen(), Tokenizer(), "cpu", context_length=24)

    def test_vocabulary(self):
        texts = self.lm.vocabulary.texts
        self.assertEqual(len(texts), 24)
        self.assertEqual(texts[:3], [None] * 3)
        self.assertEqual(texts[3], "我")
        self.assertIsNone(texts[3 + len(CHARS)])  # padded output rows beyond the tokenizer

    def test_cached_logprobs_match_full_forward(self):
        context = self.lm.context(["好的"], "我们明天")
        sequences = [context, context + [5], context + [6, 7], context + [9, 10, 11]]
        for batch in ([sequences[0]], sequences[1:], sequences):
            got = self.lm.logprobs(batch)
            for row, s in enumerate(batch):
                with torch.no_grad():
                    want = self.lm.model(torch.tensor([s])).logits[0, -1].float().log_softmax(-1)
                self.assertTrue(torch.allclose(got[row], want, atol=1e-4), (batch, row))

    def test_context_drops_oldest_turns(self):
        ids = self.lm.context(["好的" * 4, "吃饭"], "我们明天一起")
        self.assertEqual(self.lm.tokenizer.decode(ids), "吃饭�我们明天一起")  # newline is unknown to this tokenizer
        self.assertLessEqual(len(ids), 24 - P.RESERVE)

    def test_decoder_runs(self):
        out = P.predict(self.lm, ["好的"], "我们")
        self.assertTrue(all(P.all_han(t) and 2 <= len(t) <= 6 for t, _ in out))


class Ngram(unittest.TestCase):
    def test_distribution_and_decoder(self):
        docs = ["我们明天一起去吃饭\n好的", "我们一起去看电影", "明天一起去吃饭"] * 2
        lm = CharNgramLM.build(docs, log=lambda *a: None)
        for history in ([], lm.context([], "一起去"), lm.context(["好的"], "我们")):
            self.assertAlmostEqual(float(lm.distribution(history).exp().sum()), 1.0, places=5)
        self.assertEqual(P.predict(lm, [], "一起去")[0][0], "吃饭")
        self.assertIsNone(lm.vocabulary.texts[0])

    def test_sample_fraction(self):
        keys = [str(i) for i in range(20000)]
        share = sum(sample_fraction(k, 0.1) for k in keys) / len(keys)
        self.assertAlmostEqual(share, 0.1, delta=0.01)
        self.assertFalse(any(sample_fraction(k, 0.0) for k in keys))


if __name__ == "__main__":
    unittest.main()
