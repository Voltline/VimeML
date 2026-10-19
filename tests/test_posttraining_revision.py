"""Consequential controls for domain labels and the explicit monitor-only policy."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vimeml.posttraining.readings_v2 import DomainReading, normalize_literal
from vimeml.posttraining.runtime import initial_state, apply_epoch_result


class RevisionTests(unittest.TestCase):
    def test_known_name_and_literals_are_reliably_aligned(self):
        reader = DomainReading()
        for text in ('札幌へ旅行します。', '今日は名古屋へ行きます！１２'):
            left, right = reader.annotate(text)
            self.assertTrue(all(t[3] for t in left))
            self.assertTrue(all(t[3] for t in right))
            self.assertEqual(''.join(t[2] for t in left), ''.join(t[2] for t in right))
            self.assertIn('。' if text.endswith('。') else '!', ''.join(t[2] for t in left))
        self.assertEqual(normalize_literal('１２月！'), '12月!')

    def test_extended_span_preserves_original_surface(self):
        reader = DomainReading()
        text = '今日は札幌への旅行を計画しています。明日出発します。'
        for seed in range(4):
            ex, reason = reader.example_near(text, [5, 9], seed)
            self.assertIsNone(reason)
            self.assertEqual(ex['output'], text[slice(*ex['span'])])
            self.assertEqual(ex['context'], text[:ex['span'][0]])
            self.assertEqual(ex['mecab_reading'], ex['sudachi_reading'])

    def test_monitor_policy_records_degradation_without_vetoing_selection(self):
        state = initial_state()
        settings = {'baseline_lm_bpc':3.0, 'lm_relative_degradation':.01,
                    'lm_degradation_patience':2, 'early_stopping_relative_delta':.002,
                    'early_stopping_patience':3, 'lm_retention_policy':'monitor'}
        for step, loss in ((1, 1.0), (2, .9)):
            state['step'] = step
            roles, qualified = apply_epoch_result(state, loss, 3.3, 500, settings)
            self.assertFalse(qualified)
            self.assertIn('best-rerank', roles)
        self.assertEqual(state['forgetting_count'], 2)
        self.assertIsNone(state['stop_reason'])


if __name__ == '__main__':
    unittest.main()
