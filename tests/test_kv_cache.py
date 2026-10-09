"""Cache offset, rectangular causality, branching and trace generalization."""
import unittest
import numpy as np
import torch
from vimeml.deployment.kv_cache import Runtime, empty_cache, trace
from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2


class CacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(716)
        self.model = TinyGPTV2(GPTV2Config(vocab_size=128, context_length=32, d_model=32,
            n_heads=4, n_layers=2, d_ff=64)).eval()
        self.graph = trace(self.model)

    def call(self, ids, past=0, state=None, last=False):
        state = tuple(torch.from_numpy(a) for a in empty_cache(self.model.config)) if state is None else state
        result = self.graph(torch.tensor([ids], dtype=torch.int32), torch.tensor([past], dtype=torch.int32),
            *state)
        return (result[0][:, -1:] if last else result[0], *result[1:])

    @torch.inference_mode()
    def test_branch_snapshot_is_immutable_and_stale_tail_is_masked(self):
        prefix = [2, 9, 17]
        _, k, v = self.call(prefix)
        saved = (k.clone(), v.clone())
        for suffix in ([25], [18, 19, 20], [4, 5]):
            actual, _, _ = self.call(suffix, len(prefix), (k, v))
            torch.testing.assert_close(actual, self.model(torch.tensor([prefix+list(suffix)]))[:, len(prefix):], atol=3e-5, rtol=3e-4)
        self.assertTrue(torch.equal(k, saved[0]) and torch.equal(v, saved[1]))
        dirty = (k.clone(), v.clone())
        for a in dirty: a[:, :, :, 3:, :] = 999
        actual, _, _ = self.call([25], 3, dirty)
        expected, _, _ = self.call([25], 3, (k, v))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @torch.inference_mode()
    def test_chunking_capacity_padding_and_last_row_output(self):
        ids = [2] + [i % 120 + 4 for i in range(31)]
        expected = self.model(torch.tensor([ids]))
        for chunks in ([1]*32, [16, 16], [1, 7, 24], [32]):
            start, state, parts = 0, None, []
            for count in chunks:
                logits, k, v = self.call(ids[start:start+count], start, state)
                parts.append(logits); state = (k, v); start += count
            torch.testing.assert_close(torch.cat(parts, dim=1), expected, atol=3e-5, rtol=3e-4)
        last, _, _ = self.call(ids, last=True)
        torch.testing.assert_close(last, expected[:, -1:], atol=3e-5, rtol=3e-4)
        plain, _, _ = self.call(ids[:5])
        padded, _, _ = self.call(ids[:5]+[0]*27)
        torch.testing.assert_close(plain, padded[:, :5], atol=3e-5, rtol=3e-4)

    def test_runtime_rejects_invalid_offsets_and_owns_returned_snapshots(self):
        runtime = Runtime.__new__(Runtime)
        runtime.config = self.model.config
        runtime.calls = runtime.processed_tokens = 0
        buffers = empty_cache(runtime.config)
        class FakeModel:
            def predict(inner, inputs):
                for value in buffers: value.fill(runtime.calls + 1)
                return {"logits": np.ones((1, inputs["input_ids"].size, 128), np.float32),
                        "new_key_cache": buffers[0], "new_value_cache": buffers[1]}
        runtime.model = FakeModel()
        _, first = runtime.step([2, 7])
        _, second = runtime.step([9], first, 2)
        self.assertTrue(all(np.all(a == 1) for a in first))
        self.assertTrue(all(np.all(a == 2) for a in second))
        for ids, state, past in [([], None, 0), ([128], None, 0), ([2], None, 1),
                                ([2], first, -1), ([2, 3], first, 31), ([2], (), 0),
                                ([[2], [3]], None, 0), ([1.5], None, 0), ([2**32+2], None, 0)]:
            with self.subTest(ids=ids, past=past), self.assertRaises(ValueError):
                runtime.step(ids, state, past)
