"""V4.2 committed resume state and tail-batch exposure, without model/corpus training."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.training.joint_v42 import initial_state, consumed_examples, validate_state


def plan():
    # Two epochs of 10 and 9 windows with batch 4: 3 updates each, smaller tail batches.
    return {'settings': {'updates_per_epoch': [3, 3], 'windows_per_epoch': [10, 9], 'lm_batch': 4}}


def commit(state, epoch, batch, step, examples):
    state.update(epoch=epoch, epoch_batch=batch, step=step,
                 lm_cursor={'epoch': epoch, 'batch': batch, 'offset': 0})
    state['totals']['lm']['examples'] = examples


class V42TrainingTests(unittest.TestCase):
    def test_tail_batch_exposure(self):
        p, state = plan(), initial_state()
        self.assertEqual(consumed_examples(state, p), 0)
        commit(state, 0, 2, 2, 8)
        self.assertEqual(consumed_examples(state, p), 8)
        commit(state, 0, 3, 3, 10)
        self.assertEqual(consumed_examples(state, p), 10)
        commit(state, 1, 1, 4, 14)
        self.assertEqual(consumed_examples(state, p), 14)
        commit(state, 2, 0, 6, 19)
        self.assertEqual(consumed_examples(state, p), 19)

    def test_pending_validation_and_completed_cursors(self):
        p, state = plan(), initial_state()
        validate_state(state, p)
        commit(state, 0, 3, 3, 10)
        validate_state(state, p)  # Saved before full validation.
        state['epoch_results'] = [{'epoch': 1}]
        validate_state(state, p)  # Saved after validation, before advancing.
        commit(state, 1, 0, 3, 10)
        validate_state(state, p)
        commit(state, 1, 2, 5, 18)
        validate_state(state, p)
        broken = copy.deepcopy(state)
        broken['totals']['lm']['examples'] += 1
        with self.assertRaises(ValueError):
            validate_state(broken, p)
        broken = copy.deepcopy(state)
        broken['step'] += 1
        with self.assertRaises(ValueError):
            validate_state(broken, p)
        commit(state, 2, 0, 6, 19)
        state['epoch_results'].append({'epoch': 2})
        validate_state(state, p)
        with self.assertRaises(ValueError):
            commit(state, 2, 1, 7, 23)
            validate_state(state, p)


if __name__ == '__main__':
    unittest.main()
