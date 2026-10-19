"""V4.1 quotas and committed resume state, without model/corpus training."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.training.v41_data import conversion_quota
from vimeml.training.joint_v41 import initial_state, validate_state, combine_conversion
from vimeml.posttraining.streams import CommittedStream, loader_for
from test_posttraining_runtime import ToyDataset


class V41TrainingTests(unittest.TestCase):
    def test_exact_conversion_epoch_and_mid_batch_resume(self):
        def make():
            return loader_for(ToyDataset(17), 8, workers=0, seed=1219827974, multiplier=2)
        stream = CommittedStream(make())
        for epoch in range(3):
            sizes = [conversion_quota(b, 5, 17) for b in range(5)]
            self.assertEqual(sum(sizes), 17)
            consumed = []
            for b, size in enumerate(sizes):
                before = stream.state_dict()
                expected = stream.next_group(size)
                resumed = CommittedStream(make(), before)
                actual = resumed.next_group(size)
                self.assertEqual(stream.state_dict(), resumed.state_dict())
                self.assertEqual([x['input_ids'].tolist() for x in expected],
                                 [x['input_ids'].tolist() for x in actual])
                combined=combine_conversion(expected)[0]
                originals=[(row[:int(n)].tolist(),labels[:int(n)].tolist())
                           for part in expected for row,labels,n in
                           zip(part['input_ids'],part['labels'],part['lengths'])]
                joined=[(row[:int(n)].tolist(),labels[:int(n)].tolist())
                        for row,labels,n in zip(combined['input_ids'],combined['labels'],combined['lengths'])]
                self.assertEqual(originals,joined)
                consumed.extend(int(row[0]) for part in expected for row in part['input_ids'])
            self.assertEqual(sorted(consumed), list(range(7, 24)))

    def test_pending_validation_and_completed_cursors(self):
        plan = {'settings': {'updates_per_epoch': [5, 6]}, 'conversion_examples': 17}
        state = initial_state()
        validate_state(state, plan)
        state.update(step=5, epoch_batch=5, lm_cursor={'epoch': 0, 'batch': 5, 'offset': 0})
        state['totals']['conversion']['examples'] = 17
        validate_state(state, plan)  # Saved before full validation.
        state['epoch_results'] = [{'epoch': 1}]
        validate_state(state, plan)  # Saved after validation, before advancing.
        state.update(epoch=1, epoch_batch=0, lm_cursor={'epoch': 1, 'batch': 0, 'offset': 0})
        validate_state(state, plan)
        state.update(epoch_batch=3, step=8, lm_cursor={'epoch': 1, 'batch': 3, 'offset': 0})
        state['totals']['conversion']['examples'] = 25
        validate_state(state, plan)
        broken = copy.deepcopy(state)
        broken['totals']['conversion']['examples'] += 1
        with self.assertRaises(ValueError):
            validate_state(broken, plan)
        state.update(epoch=2, epoch_batch=0, step=11, lm_cursor={'epoch': 2, 'batch': 0, 'offset': 0})
        state['epoch_results'].append({'epoch': 2})
        state['totals']['conversion']['examples'] = 34
        validate_state(state, plan)


if __name__ == '__main__':
    unittest.main()
