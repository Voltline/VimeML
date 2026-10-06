"""Boundary retokenization, teacher forcing, padding and bounded decoding."""
import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.infer import JapaneseLM
from vimeml.training.evaluate_ime import (beam_suggestions, generate_samples, next_batch,
    ranking_metrics, reference_hit, score_candidates, validate_cases)


class Processor:
    mapping = {"": [], "a": [4], "b": [5], "c": [6], "ab": [7], "ac": [4, 6]}
    def encode(self, text, out_type=int):
        return self.mapping[text]

    def decode(self, ids):
        return "".join({4: "a", 5: "b", 6: "c", 7: "ab"}.get(i, "") for i in ids)

    def id_to_piece(self, token):
        return str(token)


class Model:
    def __init__(self, logits, context=8):
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.config = SimpleNamespace(context_length=context)
        self.inputs = []

    def __call__(self, inputs):
        self.inputs.append(inputs.clone())
        return self.logits.expand(*inputs.shape, -1)


def lm_for(logits=None, context=8):
    lm = JapaneseLM.__new__(JapaneseLM)
    lm.device = torch.device("cpu")
    lm.processor = Processor()
    lm.special = dict(pad=0, unk=1, bos=2, eos=3)
    lm.forbidden = (0, 1, 2)
    lm.model = Model(logits or [0.] * 8, context)
    return lm


class IMEEvaluationTests(unittest.TestCase):
    def test_retokenized_boundary_scores_from_shared_prefix_and_has_no_eos(self):
        lm = lm_for()
        result = score_candidates(lm, "a", ["b", "c"])
        self.assertEqual(result["common_prefix_tokens_including_bos"], 1)
        first, second = result["candidates"]
        self.assertEqual(first["token_ids"], [7])
        self.assertTrue(first["boundary_retokenized"])
        self.assertEqual(second["token_ids"], [4, 6])
        self.assertFalse(second["boundary_retokenized"])
        self.assertAlmostEqual(first["log_probability_sum"], -math.log(8), places=6)
        self.assertAlmostEqual(second["log_probability_sum"], -2 * math.log(8), places=6)
        self.assertAlmostEqual(first["log_probability_mean"], second["log_probability_mean"])
        self.assertEqual(lm.model.inputs[0].tolist(), [[2, 0], [2, 4]])
        self.assertNotIn(3, first["token_ids"] + second["token_ids"])

    def test_teacher_forcing_gathers_next_token_and_original_vocab_probabilities(self):
        lm = lm_for([10, 0, 0, 0, 1, 2, 3, 4])
        result = score_candidates(lm, "", ["a", "b"])
        logp = lm.model.logits.log_softmax(-1)
        for row, token in zip(result["candidates"], (4, 5)):
            self.assertAlmostEqual(row["log_probability_sum"], float(logp[token]), places=6)
        self.assertLess(result["candidates"][0]["log_probability_sum"], -8)

    def test_next_batch_gathers_last_valid_position_not_pad(self):
        lm = lm_for()
        class PositionalModel:
            config = SimpleNamespace(context_length=8)
            def __call__(self, inputs):
                return torch.arange(inputs.shape[1]).view(1, -1, 1).expand(*inputs.shape, 8).float()
        lm.model = PositionalModel()
        result = next_batch(lm, [[2, 4], [2, 4, 6]])
        self.assertEqual(result[:, 0].tolist(), [1., 2.])

    def test_context_guard_raises_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            score_candidates(lm_for(context=1), "a", ["b", "c"])

    def test_multigold_rank_and_context_correction(self):
        row = {"group": "one", "acceptable": ["b", "c"],
            "contextual": {"candidates": [{"text": c, "log_probability_sum": p} for c, p in (("a", -3), ("b", -1), ("c", -2))]},
            "context_free": {"candidates": [{"text": c, "log_probability_sum": p} for c, p in (("a", -1), ("b", -2), ("c", -3))]}}
        result = ranking_metrics([row])
        self.assertEqual(result["top1_correct"], 1)
        self.assertEqual(result["corrected_vs_context_free"], 1)
        self.assertEqual(result["all_contexts_correct_groups"], 1)
        self.assertAlmostEqual(result["random_expected_top1_fraction"], 2 / 3)

    def test_batched_greedy_eos_and_local_sample_reproducibility(self):
        lm = lm_for([100, 90, 80, 10, 1, 1, 1, 1])
        greedy, _ = generate_samples(lm, "a", 6, 42, attempts=2)
        self.assertEqual(greedy["new_token_ids"], [3])
        self.assertEqual(greedy["flags"], ["empty"])
        self.assertEqual(lm.model.inputs[0].shape[0], 3)
        lm = lm_for([100, 90, 80, -100, 1, 1, 1, 1])
        a = generate_samples(lm, "a", 3, 42, attempts=2)
        b = generate_samples(lm, "a", 3, 42, attempts=2)
        self.assertEqual(a, b)

    def test_beams_nonempty_unique_bounded_and_forbidden_specials(self):
        lm = lm_for([100, 90, 80, 10, 5, 4, 3, 2], context=3)
        result = beam_suggestions(lm, "a", max_tokens=6, width=4, count=3)
        self.assertTrue(result)
        self.assertEqual(len({item["comparison_text"] for item in result}), len(result))
        for item in result:
            self.assertTrue(item["comparison_text"])
            self.assertLessEqual(len(item["new_token_ids"]), 2)
            self.assertTrue(set(item["new_token_ids"]).isdisjoint({0, 1, 2}))
        self.assertTrue(any(inputs.shape[0] > 1 for inputs in lm.model.inputs))

    def test_reference_coverage_does_not_accept_invalid_unicode(self):
        self.assertTrue(reference_hit([{"comparison_text": "abc", "flags": []}], ["ab"]))
        self.assertFalse(reference_hit([{"comparison_text": "abc", "flags": ["replacement_character"]}], ["ab"]))

    def test_frozen_case_schema(self):
        cases = json.loads((ROOT / "configs/ime-eval-v1.json").read_text(encoding="utf-8"))
        validate_cases(cases)
        self.assertEqual(len(cases["ranking"]), 32)
        self.assertEqual(len(cases["suggestions"]), 14)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
