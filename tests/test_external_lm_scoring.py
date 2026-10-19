"""Check selective vocabulary projection against dense causal scores on CPU."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('external_eval', ROOT / 'scripts/benchmarks/evaluate_external_lm.py')
external = importlib.util.module_from_spec(spec)
spec.loader.exec_module(external)


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        # A boundary-retokenization case: the context token is not a prefix of
        # any joint context+candidate sequence.
        return [20] if text == 'ab' else [7 + ord(c) - ord('a') for c in text]

    def decode(self, ids, **kwargs):
        return ''.join('ab' if i == 20 else chr(i - 7 + ord('a')) for i in ids)

    def convert_ids_to_tokens(self, ident):
        return str(ident)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 9)

    def forward(self, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=(self.embedding(input_ids) * attention_mask[..., None]).cumsum(1))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(bos_token_id=2)
        self.model = Backbone()
        self.lm_head = nn.Linear(9, 32, bias=False)


class ExternalScoringTests(unittest.TestCase):
    def test_chunked_scores_match_full_logits_and_retokenized_boundaries(self):
        torch.manual_seed(71)
        torch.set_num_threads(2)
        model, tokenizer = Model(), Tokenizer()
        candidates = ['c', 'def', 'cdef', 'g', 'hi']
        for context in ('', 'a', 'ab'):
            lm = external.ExternalLM(model, tokenizer, candidate_batch=2, head_chunk=3)
            result = lm.score(context, candidates)
            common = result['common_prefix_tokens_including_bos']
            for candidate, scored in zip(candidates, result['candidates'], strict=True):
                sequence = [2, *tokenizer.encode(context + candidate)]
                inputs = torch.tensor([sequence[:-1]])
                hidden = model.model(inputs, torch.ones_like(inputs)).last_hidden_state
                logits = model.lm_head(hidden)[0].float().log_softmax(-1)
                expected = [float(logits[i - 1, sequence[i]].detach()) for i in range(common, len(sequence))]
                torch.testing.assert_close(torch.tensor(scored['token_log_probabilities']), torch.tensor(expected), atol=1e-6, rtol=1e-6)
                self.assertAlmostEqual(scored['log_probability_sum'], sum(expected), places=5)
                self.assertEqual(scored['scored_tokens'], len(expected))
                self.assertEqual(scored['boundary_retokenized'], context == 'ab')
            full = external.ExternalLM(model, tokenizer, candidate_batch=8, head_chunk=100).score(context, candidates)
            for left, right in zip(result['candidates'], full['candidates'], strict=True):
                self.assertAlmostEqual(left['log_probability_sum'], right['log_probability_sum'], places=5)


if __name__ == '__main__':
    unittest.main()
