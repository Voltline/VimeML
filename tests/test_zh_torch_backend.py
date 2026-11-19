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
from vimeml.deployment.zh_torch_graph import GraphBackend  # noqa: E402
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

    def test_rows_from_several_calls_in_any_order(self):
        torch.manual_seed(1118)
        model = TinyGPTZh13(GPTZh13Config()).float().eval()
        backend = ModelBatchBackend(model)
        seqs = [[2, 4, 100], [2, 9, 8, 7, 6], [2]]
        states = [backend.prefill(s)[1] for s in seqs]
        _, first = backend.decode([50, 51, 52], states)            # one call, three rows
        _, second = backend.decode([60], [states[1]])               # another call
        mixed = [first[2], second[0], first[0], first[2], states[0]]
        tokens = [70, 71, 72, 73, 74]
        prefixes = [seqs[2] + [52], seqs[1] + [60], seqs[0] + [50], seqs[2] + [52], seqs[0]]
        logits, out = backend.decode(tokens, mixed)
        for r, s in enumerate(prefixes):
            self.assertLess((logits[r] - model(torch.tensor([s + [tokens[r]]]))[0, -1]).abs().max().item(), 1e-4)
        logits, _ = backend.decode([80, 81], [out[1], out[4]])
        for r, s in ((0, prefixes[1] + [71, 80]), (1, prefixes[4] + [74, 81])):
            self.assertLess((logits[r] - model(torch.tensor([s]))[0, -1]).abs().max().item(), 1e-4)

    def test_200m_shape(self):
        torch.manual_seed(1116)
        model = TinyGPTZh200m(GPTZh200mConfig(vocab_size=12288, context_length=128, d_model=1024, n_heads=16,
                                              n_layers=15, d_ff=2560)).float().eval()
        check(self, model, [[2, 4, 5], [2, 4, 6, 7, 8]], [[100, 200]])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA graphs need CUDA")
class GraphBackendTest(unittest.TestCase):
    def test_rows_from_several_calls_and_widths(self):
        torch.manual_seed(1119)
        model = TinyGPTZh13(GPTZh13Config()).float().eval()
        reference = TinyGPTZh13(GPTZh13Config()).float().eval()
        reference.load_state_dict(model.state_dict())
        backend = GraphBackend(model, rows=4, widths=(8, 16))

        def expect(logits, prefixes):
            for r, s in enumerate(prefixes):
                want = reference(torch.tensor([s]))[0, -1]
                self.assertLess((logits[r].cpu() - want).abs().max().item(), 1e-4)

        seqs = [[2, 4, 100], [2, 9, 8, 7, 6, 5, 4], [2]]
        states = []
        for s in seqs:
            logits, state = backend.prefill(s)
            expect(logits[None], [s])
            states.append(state)
        logits, first = backend.decode([50, 51, 52], states)        # width 8
        expect(logits, [s + [t] for s, t in zip(seqs, [50, 51, 52])])
        logits, second = backend.decode([60, 61], [first[1], first[0]])
        expect(logits, [seqs[1] + [51, 60], seqs[0] + [50, 61]])
        mixed = [second[0], first[2], states[1], second[1]]          # rows from four sources, any order
        prefixes = [seqs[1] + [51, 60], seqs[2] + [52], seqs[1], seqs[0] + [50, 61]]
        logits, out = backend.decode([70, 71, 72, 73], mixed)        # width 16 (a row of length 9)
        expect(logits, [s + [t] for s, t in zip(prefixes, [70, 71, 72, 73])])
        with torch.no_grad():
            tail = prefixes[0] + [70]
            for t in range(80, 87):                                   # past both widths: the full context
                logits, out = backend.decode([t], [out[0]])
                tail = tail + [t]
            expect(logits, [tail])


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


class MergeTest(unittest.TestCase):
    def test_shards_join_in_set_order(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frozen = tmp / "frozen"
            frozen.mkdir()
            records = [{"id": f"r{i}", "split": "development", "full_input": f"k{i}", "words": ["a", "b"],
                        "abbreviations": [{"input": f"m{i}", "abbreviated_word_indices": [0]}]} for i in range(5)]
            pg.write_jsonl(frozen / "development.jsonl", records)
            pg.write_jsonl(frozen / "development-contexts.jsonl", [{"id": r["id"], "infield": ""} for r in records])
            meta = {"checkpoint_sha256": "x", "step": 1, "mode": "task", "beam": 16, "topk": 10, "context": True,
                    "prior": None, "readings_sha256": "r", "set_manifest_sha256": "m", "limit": 0, "seconds": 1.0}
            for i in range(2):
                d = tmp / f"s{i}"
                d.mkdir()
                pg.write_jsonl(d / "rows.jsonl", [{"id": r["id"], "kind": k, "input": r[f]}
                                                  for r in records[i::2]
                                                  for k, f in (("full", "full_input"), ("mixed", "full_input"))])
                (d / "meta.json").write_text(json.dumps({**meta, "shard": [i, 2]}), "utf-8")
            info = pg.merge([tmp / "s1", tmp / "s0"], frozen, tmp / "out")
            rows = pg.read_jsonl(tmp / "out" / "rows.jsonl")
            self.assertEqual([(r["id"], r["kind"]) for r in rows],
                             [(f"r{i}", k) for i in range(5) for k in ("full", "mixed")])
            self.assertEqual((info["inputs"], info["shards"]), (10, 2))
            with self.assertRaises(ValueError):
                pg.merge([tmp / "s0"], frozen, tmp / "bad")


if __name__ == "__main__":
    unittest.main()
