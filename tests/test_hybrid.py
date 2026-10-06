"""Hybrid rank semantics, frozen-policy provenance and development independence."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.benchmarks.ajimee import DEVELOPMENT_FORMAT
from vimeml.benchmarks.hybrid import (FORMULA, add_hybrid, calibrate, check_independence,
    diagnostics, hybrid_order, load_cache, prepare_development, validate_policy)


def rows_for_test():
    scores = [{"text": 'a', "log_probability_sum": -4., "token_ids": [4], "pieces": ['a'],
               "token_log_probabilities": [-4.], "scored_tokens": 1, "boundary_retokenized": False},
              {"text": 'b', "log_probability_sum": -1., "token_ids": [5], "pieces": ['b'],
               "token_log_probabilities": [-1.], "scored_tokens": 1, "boundary_retokenized": False}]
    return [{"id": 'dev:1', "query": 'reading', "left_context": 'ctx', "answers": ['b'],
             "candidates": [{"text": 'a', "score": -1.}, {"text": 'b', "score": -2.}],
             "orders": {"azookey": ['a', 'b'], "lm_context_sum": ['b', 'a'], "lm_no_context_sum": ['b', 'a']},
             "lm_scores": {"contextual": {"candidates": scores}}, "fallbacks": {}}]


def cache_for_test():
    return {"benchmark_manifest": {"format": DEVELOPMENT_FORMAT, "role": 'development',
             "labels_reviewed": True, "source_sha256": 'dev-source', "excluded_source_sha256": 'ajimee-source'},
            "fingerprint": {"checkpoint": 'one', "dictionary": 'one'},
            "files_sha256": {"scores.jsonl": 'dev-scores'}}


class HybridTests(unittest.TestCase):
    def test_zero_preserves_exact_original_even_if_scores_tied_or_out_of_order(self):
        candidates = [{"text": 'b', "score": -2.}, {"text": 'a', "score": -1.}]
        self.assertEqual(hybrid_order(candidates, [], 0)[0], ['b', 'a'])
        add_hybrid(rows_for_test(), 0)

    def test_positive_lambda_changes_rank_and_exact_tie_keeps_original(self):
        row = rows_for_test()[0]
        candidates = row['candidates']
        scores = row['lm_scores']['contextual']['candidates']
        self.assertEqual(hybrid_order(candidates, scores, 1/3)[0], ['a', 'b'])
        order, values = hybrid_order(candidates, scores, 1)
        self.assertEqual(order, ['b', 'a'])
        self.assertEqual(values, [-5., -3.])
        for weight in (-1., math.inf, math.nan):
            with self.assertRaises(ValueError):
                hybrid_order(candidates, scores, weight)

    def test_missing_gold_does_not_change_candidate_pool_or_inference_order(self):
        rows = rows_for_test()
        rows[0]['answers'] = ['GOLD_NOT_IN_POOL']
        add_hybrid(rows, 1)
        self.assertEqual(rows[0]['orders']['hybrid'], ['b', 'a'])
        self.assertEqual({c['text'] for c in rows[0]['candidates']}, {'a', 'b'})

    def test_whole_case_fallback_and_zero_do_not_require_lm_scores(self):
        rows = rows_for_test()
        rows[0]['lm_scores'] = {}
        rows[0]['fallbacks']['lm_context_sum'] = 'context_length'
        add_hybrid(rows, 1)
        self.assertEqual(rows[0]['orders']['hybrid'], ['a', 'b'])
        self.assertEqual(rows[0]['fallbacks']['hybrid'], 'context_length')
        add_hybrid(rows, 0)
        self.assertNotIn('hybrid', rows[0]['fallbacks'])

    def test_tuning_rejects_ajimee_and_unreviewed_labels(self):
        cache = cache_for_test()
        cache['benchmark_manifest']['role'] = 'evaluation'
        with self.assertRaisesRegex(ValueError, 'AJIMEE cannot be tuned'):
            calibrate(cache, rows_for_test(), [0., 1.])
        cache = cache_for_test()
        cache['benchmark_manifest']['labels_reviewed'] = False
        with self.assertRaisesRegex(ValueError, 'need review'):
            calibrate(cache, rows_for_test(), [0., 1.])

    def test_calibration_selection_is_predeclared_and_policy_rejects_data_or_version_overlap(self):
        cache = cache_for_test()
        policy = calibrate(cache, rows_for_test(), [0., .5, 1.])
        self.assertEqual(policy['lambda'], .5)
        self.assertEqual(policy['formula'], FORMULA)
        other = copy.deepcopy(cache)
        other['benchmark_manifest']['source_sha256'] = 'test-source'
        rows = rows_for_test()
        rows[0].update(query='other-reading', left_context='different-context')
        self.assertEqual(validate_policy(policy, other, rows), .5)
        other['fingerprint']['checkpoint'] = 'two'
        with self.assertRaisesRegex(ValueError, 'differs'):
            validate_policy(policy, other, rows)
        with self.assertRaisesRegex(ValueError, 'different data'):
            validate_policy(policy, cache, rows)
        other = copy.deepcopy(cache)
        other['benchmark_manifest']['source_sha256'] = 'test-source'
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_policy(policy, other, rows_for_test())

    def test_normalized_overlap_rejected_and_prepare_checks_before_writing(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            check_independence([{'query': '\uff21', 'left_context': '', 'answers': ['x']}],
                               [{'query': 'A', 'left_context': '', 'answers': ['y']}])
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            p = Path(folder)
            source = [{"index": 'new', "input": 'r', "context_text": '', "expected_output": ['x']}]
            (p/'source.json').write_text(json.dumps(source), encoding='utf-8')
            (p/'excluded.json').write_text(json.dumps(source), encoding='utf-8')
            with self.assertRaises(ValueError):
                prepare_development(p/'source.json', p/'out', p/'excluded.json')
            self.assertFalse((p/'out').exists())

    def test_diagnostics_are_observations_and_do_not_change_order(self):
        row = rows_for_test()[0]
        row['lm_scores']['contextual']['candidates'][1].update(token_ids=[4,5], pieces=['a','<0x90>'])
        row['orders']['hybrid'] = ['b','a']
        result = diagnostics(row)
        self.assertEqual(result['strict_token_prefix_pairs'], [{'shorter': 'a', 'longer': 'b'}])
        self.assertEqual(result['byte_fallback_candidates'], [{'text': 'b', 'byte_tokens': 1}])
        self.assertEqual(row['orders']['hybrid'], ['b','a'])


if __name__ == '__main__':
    unittest.main()
