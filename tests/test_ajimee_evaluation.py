"""Real-pool metrics, no-gold inference, stable ties and all-case fallbacks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.ajimee import prepare
from vimeml.benchmarks.evaluate_ajimee import eligibility, load_export, metrics, min_cer, rerank


class AJIMEEvaluationTests(unittest.TestCase):
    def test_multiple_references_missing_gold_and_empty_candidates_keep_denominator(self):
        rows = [
            {"answers": ["a", "b"], "orders": {"azookey": ["x", "b"], "lm": ["b", "x"]}, "fallbacks": {}},
            {"answers": ["a"], "orders": {"azookey": ["a", "x"], "lm": ["x", "a"]}, "fallbacks": {}},
            {"answers": ["a"], "orders": {"azookey": ["x"], "lm": ["x"]}, "fallbacks": {}},
            {"answers": ["a"], "orders": {"azookey": [], "lm": []}, "fallbacks": {"lm": "empty_candidates"}},
        ]
        result = metrics(rows, "lm")
        self.assertEqual(result["top1_correct"], 1)
        self.assertEqual(result["top1_accuracy"], .25)
        self.assertEqual(result["candidate_pool_covered"], 2)
        self.assertEqual(result["covered_top1_accuracy"], .5)
        self.assertEqual(result["mean_min_cer"], .75)
        self.assertEqual(result["corrected_vs_azookey"], 1)
        self.assertEqual(result["regressed_vs_azookey"], 1)
        self.assertEqual(result["fallback_cases"], 1)

    def test_cer_uses_best_reference_and_its_length_without_width_normalization(self):
        self.assertEqual(min_cer(["abc", "a"], "ab"), 1 / 3)
        self.assertEqual(min_cer(["A"], "\uff21"), 1)
        self.assertEqual(min_cer(["abc"], ""), 1)

    def test_joint_encoding_window_limit_checks_every_candidate(self):
        processor = SimpleNamespace(encode=lambda text, out_type: list(text), decode=lambda ids: "".join(ids))
        lm = SimpleNamespace(processor=processor, model=SimpleNamespace(config=SimpleNamespace(context_length=4)))
        self.assertIsNone(eligibility(lm, "a", ["bbb"]))
        self.assertEqual(eligibility(lm, "a", ["b", "bbbb"]), "context_length")
        self.assertEqual(eligibility(lm, "aaaa", ["b"]), "context_length")

    def test_inference_never_receives_references_ties_preserve_original_and_fallback_is_whole_case(self):
        rows = [{"id": "a", "left_context": "ctx", "answers": ["GOLD_ABSENT"],
                 "orders": {"azookey": ["x", "y"]}, "fallbacks": {}}]
        calls = []
        def score(lm, context, texts):
            calls.append((context, list(texts)))
            return {"candidates": [{"text": text, "log_probability_sum": -1., "log_probability_mean": -1.} for text in texts]}
        with patch('vimeml.benchmarks.evaluate_ajimee.eligibility', side_effect=["context_length", None]), \
             patch('vimeml.training.evaluate_ime.score_candidates', side_effect=score):
            rerank(None, rows)
        self.assertEqual(calls, [("", ["x", "y"])])
        self.assertEqual(rows[0]["orders"]["lm_context_sum"], ["x", "y"])
        self.assertEqual(rows[0]["orders"]["lm_no_context_sum"], ["x", "y"])
        self.assertEqual(rows[0]["fallbacks"]["lm_context_sum"], "context_length")

    def test_export_alignment_and_rank_are_verified(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            directory = Path(folder)
            source = directory / 'source.json'
            source.write_text(json.dumps([{"index": 1, "input": "a", "context_text": "", "expected_output": ["x", "y"]}]), encoding='utf-8')
            prepared = directory / 'prepared'
            prepare(source, prepared)
            exported = prepared / 'ajimee-results'
            exported.mkdir()
            (exported / 'ajimee-input.json').write_bytes((prepared / 'ajimee-input.json').read_bytes())
            for name, text in [('converter-version.txt', 'a' * 40), ('dictionary-versions.txt', ' ' + 'b' * 40 + ' Dictionary\n'), ('swift-version.txt', 'Swift')]:
                (exported / name).write_text(text, encoding='utf-8')
            raw = {"n_best": 20, "items": [{"query": "a", "left_context": "", "right_context": "", "answers": ["x", "y"], "max_rank": 0,
                                                 "outputs": [{"text": "y", "score": -1.}]}]}
            result = exported / 'azookey-candidates.json'
            result.write_text(json.dumps(raw), encoding='utf-8')
            self.assertEqual(len(load_export(prepared)[2]), 1)
            raw['items'][0]['left_context'] = 'future'
            result.write_text(json.dumps(raw), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'misaligned'):
                load_export(prepared)
            raw['items'][0]['left_context'] = ''
            raw['items'][0]['max_rank'] = -1
            result.write_text(json.dumps(raw), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'rank inconsistent'):
                load_export(prepared)


if __name__ == '__main__':
    unittest.main()
