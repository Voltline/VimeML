"""Shape-generic search backend (random weights) and the pinyin generation scorer."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks import pinyin_gen_zh as pg  # noqa: E402
from vimeml.deployment import zh_v1_batch as zb  # noqa: E402
from vimeml.deployment.zh_torch_backend import ModelBatchBackend  # noqa: E402
from vimeml.training.model_zh13 import GPTZh13Config, TinyGPTZh13  # noqa: E402
from vimeml.training.model_zh200m import GPTZh200mConfig, TinyGPTZh200m  # noqa: E402


def check(test, model, sequences, steps):
    backend = ModelBatchBackend(model)
    states = []
    for s in sequences:
        logits, state = backend.prefill(s)
        test.assertLess((logits - model(torch.tensor([s]))[0, -1]).abs().max().item(), 1e-4)
        states.append(state)
    for step in steps:
        logits, states = backend.decode(step, states)
        sequences = [s + [t] for s, t in zip(sequences, step)]
        for r, s in enumerate(sequences):
            test.assertLess((logits[r] - model(torch.tensor([s]))[0, -1]).abs().max().item(), 1e-4)


class BackendTest(unittest.TestCase):
    def test_v1_rows_at_different_positions(self):
        torch.manual_seed(1116)
        model = TinyGPTZh13(GPTZh13Config()).float().eval()
        check(self, model, [[2, 4, 100, 200], [2], [2, 9, 8, 7, 6, 5, 4]], [[300, 301, 302], [7, 8, 9]])

    def test_same_search_as_the_v1_reference_backend(self):
        torch.manual_seed(1116)
        model = TinyGPTZh13(GPTZh13Config()).float().eval()
        ours, ref = ModelBatchBackend(model), zb.TorchBatchBackend(model)
        a, sa = ours.prefill([2, 10, 11])
        b, sb = ref.prefill([2, 10, 11])
        self.assertLess((a - b).abs().max().item(), 1e-4)
        a, _ = ours.decode([12, 13], [sa, sa])
        b, _ = ref.decode([12, 13], [sb, sb])
        self.assertLess((a - b).abs().max().item(), 1e-4)

    def test_200m_shape(self):
        torch.manual_seed(1116)
        model = TinyGPTZh200m(GPTZh200mConfig(vocab_size=12288, context_length=128, d_model=1024, n_heads=16,
                                              n_layers=15, d_ff=2560)).float().eval()
        check(self, model, [[2, 4, 5], [2, 4, 6, 7, 8]], [[100, 200]])


class ScoreTest(unittest.TestCase):
    def rows(self, hits):
        return [{"id": f"s{i // 2}", "input": str(i), "kind": "full", "text": "对", "gen": ["对" if h else "错", "x"],
                 "genScores": [-1.0, -2.0], "prior": [0.0, 5.0], "rows": 1} for i, h in enumerate(hits)]

    def test_rescoring_and_sign_flip(self):
        base, new = self.rows([0] * 40), self.rows([1] * 40)
        self.assertEqual(pg.table(new, 0)["all"]["top1"], 100.0)
        self.assertEqual(pg.table(new, 1.6)["all"]["rescored_top1"], 0.0)   # prior moves "x" first
        s = pg.sign_flip(base, new, 0, 0)
        self.assertEqual((s["gained"], s["lost"]), (40, 0))
        self.assertLess(s["p"], 1e-3)
        self.assertEqual(pg.sign_flip(new, new, 0, 0)["p"], 1.0)


if __name__ == "__main__":
    unittest.main()
