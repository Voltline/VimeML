"""Fixed V5 parameter contract, KV path, scoring, checkpoint identity and resume checks."""
import copy
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import sentencepiece as spm
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.model_v5 import GPTV5Config, TinyGPTV5, PARAMETERS, LAYER_PARAMETERS, CACHE_SHAPE
from vimeml.training.infer_v4 import JapaneseLMV4
from vimeml.training.infer_v5 import JapaneseLMV5, score_candidates_cached
from vimeml.training.evaluate_ime import score_candidates
from vimeml.training.data import file_sha
from vimeml.training.train import optimizer_for
from vimeml.training import joint_v5
from vimeml.posttraining.infer import fp32_evaluation

TOKENIZER = ROOT / "artifacts/tokenizers/v4-a/ja-unigram-8192"
CONFIG = tomllib.loads((ROOT / "configs/train-v5.toml").read_text(encoding="utf-8"))


class V5ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(1219)
        cls.model = TinyGPTV5().eval()

    def test_exact_parameter_contract_and_rejects_architecture_changes(self):
        model = self.model
        self.assertEqual(model.parameter_count(), PARAMETERS)
        self.assertEqual(model.token_embedding.weight.numel(), 8192 * 512)
        self.assertEqual(model.position_embedding.weight.numel(), 128 * 512)
        for block in model.blocks:
            self.assertEqual(sum(p.numel() for p in block.parameters()), LAYER_PARAMETERS)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        self.assertEqual(GPTV5Config(**CONFIG["model"]), GPTV5Config())
        for key, value in (("d_model", 384), ("n_layers", 7), ("n_heads", 6), ("d_ff", 960), ("vocab_size", 16384)):
            with self.assertRaises(ValueError):
                GPTV5Config(**{key: value})

    def test_causality_padding_and_loss(self):
        ids = torch.tensor([[2, 10, 20, 30]])
        padded = torch.tensor([[2, 10, 20, 30, 0, 0, 0, 0]])
        labels = torch.tensor([[10, 20, 30, 3, -100, -100, -100, -100]])
        with torch.no_grad():
            short, full = self.model(ids), self.model(padded)
            torch.testing.assert_close(short, full[:, :4], atol=3e-4, rtol=3e-4)
            actual = self.model(padded, labels)
            expected = F.cross_entropy(full.reshape(-1, 8192), labels.reshape(-1), reduction="sum", ignore_index=-100)
            torch.testing.assert_close(actual["loss_sum"], expected)

    def test_incremental_chunks_and_rollback(self):
        ids = torch.randint(7, 8192, (1, 128), generator=torch.Generator().manual_seed(512))
        cache = self.model.new_cache()
        self.assertEqual(tuple(cache.key.shape), CACHE_SHAPE)
        with torch.no_grad():
            full = self.model(ids)
            start = 0
            for size in (1, 7, 3, 20, 97):
                actual = self.model.forward_cached(ids[:, start:start + size], cache)
                torch.testing.assert_close(actual, full[:, start:start + size], atol=3e-4, rtol=3e-4)
                start += size
        with self.assertRaises(ValueError):
            self.model.forward_cached(torch.tensor([[10]]), cache)
        branch = self.model.new_cache()
        self.model.forward_cached(ids[:, :7], branch)
        saved = branch.clone()
        first = self.model.forward_cached(torch.tensor([[100, 101, 102]]), branch)
        branch.restore(saved)
        self.model.forward_cached(torch.tensor([[200, 201]]), branch)
        branch.restore(saved)
        again = self.model.forward_cached(torch.tensor([[100, 101, 102]]), branch)
        torch.testing.assert_close(first, again, atol=0, rtol=0)

    def test_cached_and_full_candidate_scores_agree(self):
        lm = JapaneseLMV5.__new__(JapaneseLMV5)
        lm.model, lm.device = self.model, torch.device("cpu")
        lm.processor = spm.SentencePieceProcessor(model_file=str(TOKENIZER / "tokenizer.model"))
        lm.special = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
        lm.forbidden = (0, 1, 2, 4, 5, 6)
        for context, candidates in [("昨日友達と", ["話した", "離した", "はなした"]), ("", ["ありがとう", "有難う"])]:
            original = score_candidates(lm, context, candidates)
            cached = score_candidates_cached(lm, context, candidates)
            for left, right in zip(original["candidates"], cached["candidates"], strict=True):
                self.assertEqual(left["token_ids"], right["token_ids"])
                self.assertAlmostEqual(left["log_probability_sum"], right["log_probability_sum"], delta=3e-4)
        with self.assertRaises(ValueError):
            lm.prefix_ids("<CTX>今日は")


def fake_plan():
    tokens = ROOT / CONFIG["token_dir"] / "manifest.json"
    return {"signatures": {"dataset": CONFIG["dataset_sha256"], "tokens": file_sha(tokens), "code": "test"},
            "settings": {"max_steps": 10, "warmup_steps": 1}}


class V5TrainingContractTests(unittest.TestCase):
    def test_plan_rejects_wrong_stage_architecture_or_size(self):
        for change in ({"stage": "v4.3"}, {"architecture": "tiny_gpt_v4"},
                       {"model": {**CONFIG["model"], "d_model": 384}}):
            with self.assertRaises(ValueError):
                joint_v5.build_plan({**copy.deepcopy(CONFIG), **change}, ROOT, verify_files=False)

    def test_update_checkpoint_inference_and_resume_identity(self):
        torch.manual_seed(5)
        settings = {**CONFIG["training"], "device": "cpu", "precision": "fp32"}
        model = TinyGPTV5()
        optimizer = optimizer_for(model, settings, torch.device("cpu"))
        ids = torch.randint(7, 8192, (4, 33), generator=torch.Generator().manual_seed(6))
        batch = {"input_ids": ids[:, :-1], "labels": ids[:, 1:], "lengths": torch.full((4,), 32)}
        model.train()
        losses = [joint_v5.update_lm(model, optimizer, [batch], settings, torch.device("cpu"), "fp32")["loss"]
                  for _ in range(3)]
        self.assertLess(losses[-1], losses[0])
        with fp32_evaluation(model):
            self.assertEqual(tuple(model(ids).shape), (4, 33, 8192))
        self.assertTrue(model.training)

        state = joint_v5.initial_state()
        state["step"] = 3
        state["training_state"] = None
        plan = fake_plan()
        payload = joint_v5.checkpoint_payload(model, optimizer, state, CONFIG, plan, "fp32",
                                              {"mode": "random", "seed": 1})
        payload["training_state"]["step"] = 3
        self.assertEqual(payload["format"], "vimeml_tiny_gpt_v5")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "epoch-test.pt"
            torch.save(payload, path)
            lm = JapaneseLMV5(path, TOKENIZER)
            self.assertEqual(lm.metadata["architecture"], "tiny_gpt_v5")
            self.assertEqual(lm.model.parameter_count(), PARAMETERS)
            with self.assertRaises(ValueError):
                JapaneseLMV4(path, TOKENIZER)
            saved = torch.load(path, map_location="cpu", weights_only=True)
        fresh = TinyGPTV5()
        joint_v5.restore_checkpoint(copy.deepcopy(saved), fresh, optimizer_for(fresh, settings, torch.device("cpu")),
                                    CONFIG, plan["signatures"], "fp32")
        for name, value in model.state_dict().items():
            torch.testing.assert_close(fresh.state_dict()[name], value, atol=0, rtol=0)
        changed = copy.deepcopy(CONFIG)
        changed["training"]["learning_rate"] = 1e-3
        with self.assertRaises(ValueError):
            joint_v5.restore_checkpoint(copy.deepcopy(saved), TinyGPTV5(), optimizer_for(fresh, settings, torch.device("cpu")),
                                        changed, plan["signatures"], "fp32")


if __name__ == "__main__":
    unittest.main()
