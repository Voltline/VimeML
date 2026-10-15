"""Consequential C1 checks: task normalization, real V4 weights, streams and resume.

CPU unit checks only; no corpus training, CUDA trial, benchmark run or network.
"""
import copy
import gc
import json
from pathlib import Path
import random
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vimeml.posttraining.data import ConversionCodec, TOKENIZER_SHA
from vimeml.posttraining.infer import score_candidates, evaluate_snapshot
from vimeml.posttraining.runtime import (update, initialize_v4, initial_state, apply_epoch_result,
                                         restore_checkpoint, validate_task_checkpoint)
from vimeml.posttraining.streams import loader_for, CommittedStream, TaskCollator
from vimeml.posttraining.train import build_plan, checkpoint_payload, validate_resume_state, run
from vimeml.training.model_v4 import TinyGPTV4
from vimeml.training.train import rng_state, restore_rng


class ToyDataset:
    split, context_length, epoch_indexed = "train", 128, False
    def __init__(self, count=17):
        self.count = count
    def __len__(self):
        return self.count
    def __getitem__(self, index):
        n = 1 + index % 5
        return {"input_ids": [index + 7] * n, "labels": [index + 7] * n}
    def window_lengths(self, indices):
        return np.array([1 + i % 5 for i in indices])
    def close(self):
        pass


class TinyConversion:
    def __init__(self, split):
        self.split, self.count = split, 3 if split == "train" else 2
        self.offsets = np.arange(self.count + 1) * 8
        self.manifest = {"splits": {"validation": {"supervised_tokens_full_reading": 4}}}
    def __len__(self):
        return self.count
    def __getitem__(self, index):
        return {"input_ids": [2, 4, 8, 5, 9, 6, 10], "labels": [-100] * 5 + [10, 3]}
    def close(self):
        pass


class ToyLM(nn.Module):
    def __init__(self, vocab=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 5)
        self.head = nn.Linear(5, vocab, bias=False)
    def forward(self, ids, labels=None):
        logits = self.head(self.embedding(ids))
        if labels is None:
            return logits
        valid = labels != -100
        return {"loss_sum": F.cross_entropy(logits[valid], labels[valid], reduction="sum"), "token_count": valid.sum()}


def settings():
    return {"conversion_microbatch": 1, "lm_microbatch": 2, "conversion_weight": .7,
            "lm_weight": .3, "grad_clip": 1000000.0,
            "early_stopping_relative_delta": .002, "early_stopping_patience": 3,
            "baseline_lm_bpc": 3.0277770232964114, "lm_relative_degradation": .01,
            "lm_degradation_patience": 2}


class C1RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = tomllib.loads((ROOT / "configs/posttrain-v4-c1.toml").read_text(encoding="utf-8"))

    def test_masked_task_token_means_equal_unsplit_objective(self):
        torch.manual_seed(13)
        first, second = ToyLM(), ToyLM()
        second.load_state_dict(first.state_dict())
        collate = TaskCollator()
        conv = collate([{"input_ids": [7, 8, 9], "labels": [-100, -100, 9]},
                        {"input_ids": [10, 11], "labels": [11, 12]},
                        {"input_ids": [12, 13, 14, 15], "labels": [-100, 13, 14, 15]}])
        lm = collate([{"input_ids": [7, 8], "labels": [8, 9]},
                      {"input_ids": [10, 11, 12, 13], "labels": [11, 12, 13, 14]}])
        one = torch.optim.SGD(first.parameters(), lr=.01)
        two = torch.optim.SGD(second.parameters(), lr=.01)
        result = update(first, one, [conv], [lm], settings(), torch.device("cpu"), "fp32")
        two.zero_grad()
        loss = .7 * second(conv["input_ids"], conv["labels"])["loss_sum"] / 6
        loss += .3 * second(lm["input_ids"], lm["labels"])["loss_sum"] / 6
        loss.backward()
        two.step()
        self.assertEqual(result["supervised_tokens"], {"conversion": 6, "lm": 6})
        self.assertGreater(result["input_tokens"]["conversion"], 6)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)

    def test_lm_tail_exact_size_and_resume_after_partial_next_epoch(self):
        def make():
            return loader_for(ToyDataset(5), 4, 0, 11, multiplier=2)
        stream = CommittedStream(make())
        stream.next_group(4)
        group = stream.next_group(4)
        self.assertEqual(sum(len(b["lengths"]) for b in group), 4)
        self.assertEqual(stream.state_dict(), {"epoch": 1, "batch": 0, "offset": 3})
        resumed = CommittedStream(make(), stream.state_dict())
        a, b = stream.next_group(4), resumed.next_group(4)
        self.assertEqual([x["input_ids"].tolist() for x in a], [x["input_ids"].tolist() for x in b])
        self.assertEqual(stream.state_dict(), resumed.state_dict())

    def test_prefetch_does_not_advance_committed_cursor(self):
        loader = loader_for(ToyDataset(), 4, 1, 47, multiplier=2)
        stream = CommittedStream(loader)
        stream.next_group(4)
        cursor = stream.state_dict()
        resumed_loader = loader_for(ToyDataset(), 4, 0, 47, multiplier=2)
        resumed = CommittedStream(resumed_loader, cursor)
        for _ in range(4):
            a, b = stream.next_group(4), resumed.next_group(4)
            self.assertEqual([x["input_ids"].tolist() for x in a], [x["input_ids"].tolist() for x in b])
        del stream, loader
        gc.collect()

    def test_frozen_v4_source_weights_and_fresh_optimizer(self):
        model = TinyGPTV4()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        c = self.config
        identity = {"tokenizer_model_sha256": TOKENIZER_SHA, "vocab_size": 8192,
                    "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3}}
        info = initialize_v4(model, ROOT / c["initialization"]["checkpoint"], c["initialization"]["sha256"],
            c["initialization"]["step"], identity, c["identities"]["tokens"], c["identities"]["windows"])
        source = torch.load(ROOT / c["initialization"]["checkpoint"], map_location="cpu", weights_only=True)
        self.assertTrue(all(torch.equal(value, source["model"][key]) for key, value in model.state_dict().items()))
        self.assertFalse(optimizer.state)
        self.assertEqual(info["source_step"], 584000)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_real_v4_optimizer_rng_checkpoint_resume_next_update(self):
        torch.manual_seed(17)
        random.seed(17)
        model = TinyGPTV4()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batch = TaskCollator()([{"input_ids": [2, 4, 8, 5, 9, 6, 10],
                                "labels": [-100, -100, -100, -100, -100, 10, 3]}])
        update(model, optimizer, [batch], [batch], settings(), torch.device("cpu"), "fp32")
        state = initial_state()
        state.update(step=1, conversion_cursor=1, lm_cursor={"epoch": 0, "batch": 1, "offset": 0})
        plan = {"signatures": {"fixture": "unit_only"}, "settings": {"max_steps": 3, "warmup_steps": 1}}
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temporary:
            path = Path(temporary) / "unit-checkpoint.pt"
            torch.save(checkpoint_payload(model, optimizer, state, self.config, plan, "fp32", {"source_step": 584000}), path)
            expected_rng = (random.random(), torch.rand(4))
            update(model, optimizer, [batch], [batch], settings(), torch.device("cpu"), "fp32")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            restored_model = TinyGPTV4()
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
            recovered, rng, _ = restore_checkpoint(saved, restored_model, restored_optimizer, self.config, plan["signatures"], "fp32")
            restore_rng(rng)
            self.assertEqual(random.random(), expected_rng[0])
            torch.testing.assert_close(torch.rand(4), expected_rng[1], atol=0, rtol=0)
            update(restored_model, restored_optimizer, [batch], [batch], settings(), torch.device("cpu"), "fp32")
            self.assertEqual(recovered["lm_cursor"], state["lm_cursor"])
            for a, b in zip(model.parameters(), restored_model.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            altered = copy.deepcopy(saved)
            altered["posttraining"]["stage"] = "v4-b"
            with self.assertRaises(ValueError):
                validate_task_checkpoint(altered)
            with self.assertRaises(ValueError):
                restore_checkpoint(saved, restored_model, restored_optimizer, self.config, {"fixture": "changed"}, "fp32")

    def test_plateau_is_cumulative_and_lm_guard_controls_selection(self):
        state, s = initial_state(), settings()
        for step, loss in enumerate([1.0, .9991, .9985, .9978], 1):
            state["step"] = step
            apply_epoch_result(state, loss, 3.03, 500, s)
        self.assertEqual(state["plateau_count"], 0)
        original = state["best_rerank_key"][:]
        for step in (5, 6):
            state["step"] = step
            roles, allowed = apply_epoch_result(state, .99 - step * .001, 3.2, 700, s)
            self.assertFalse(allowed)
            self.assertNotIn("best-rerank", roles)
        self.assertEqual(state["best_rerank_key"], original)
        self.assertEqual(state["stop_reason"], "lm_retention_guard")

    def test_pending_epoch_validation_state_is_resumable(self):
        state = initial_state()
        plan = {"conversion_examples": 5, "settings": {"epochs": 2, "updates_per_epoch": 2,
                "max_steps": 4, "conversion_batch": 4, "lm_batch": 4}}
        state.update(step=2, conversion_cursor=2)
        state["totals"]["conversion"]["examples"] = 5
        state["totals"]["lm"]["examples"] = 8
        validate_resume_state(state, plan)
        state["epoch_results"] = [{"epoch": 1}]
        validate_resume_state(state, plan)
        state.update(epoch=1, conversion_cursor=0)
        validate_resume_state(state, plan)
        state["lm_cursor"]["offset"] = -1
        with self.assertRaises(ValueError):
            CommittedStream(loader_for(ToyDataset(), 4, 0, 42), state["lm_cursor"])

    def test_out_sum_excludes_eos_and_overflow_falls_back_whole_case(self):
        codec = ConversionCodec(ROOT / self.config["tokenizer_dir"] / "tokenizer.model")
        model = ToyLM(vocab=8192).eval()
        result = score_candidates(model, codec, "昨日友達と", "はなした", ["話した", "離した"])
        expected = []
        for output in ("話した", "離した"):
            ids, first, _ = codec.encode("昨日友達と", "はなした", output)
            logits = model(torch.tensor([ids[:-1]]))[0].detach().log_softmax(-1)
            expected.append(sum(float(logits[i - 1, ids[i]]) for i in range(first, len(ids) - 1)))
        torch.testing.assert_close(torch.tensor(result["scores"]), torch.tensor(expected), atol=1e-5, rtol=1e-6)
        with patch.object(model, "forward", side_effect=AssertionError("Must not partially score overflow case")):
            result = score_candidates(model, codec, "", "はなした", ["話した", "話した" * 300])
        self.assertEqual(result["fallback"], "structured_context_overflow")
        self.assertIsNone(result["scores"])

    def test_reading_comes_only_from_query_and_fallback_preserves_azoo_order(self):
        row = {"id": "fixture", "query": "ハナシタ", "left_context": "昨日", "answers": ["離した"],
               "candidates": [{"text": "離した"}, {"text": "話した"}], "orders": {"azookey": ["話した", "離した"]}}
        with patch("vimeml.posttraining.infer.score_candidates", return_value={"scores": None, "fallback": "structured_context_overflow"}) as score:
            result = evaluate_snapshot(None, None, [row])
        self.assertEqual(score.call_args.args[3], "はなした")
        self.assertEqual(score.call_args.args[4], ["話した", "離した"])
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["fallbacks"], 1)

    def test_production_plan_and_pretraining_guard_remain_separate(self):
        plan = build_plan(self.config)
        self.assertEqual(plan["settings"]["max_steps"], 133176)
        self.assertEqual(plan["settings"]["warmup_steps"], 1332)
        self.assertEqual(plan["parameters"], 15070848)
        from vimeml.training import model_factory_v4
        from vimeml.training.train import validate_config
        base = tomllib.loads((ROOT / "configs/train-v4-b12cosine.toml").read_text(encoding="utf-8"))
        base["initialization"] = {"checkpoint": self.config["initialization"]["checkpoint"]}
        with self.assertRaisesRegex(ValueError, "starts from scratch"):
            validate_config(base)

    def test_trainer_recovers_interrupted_epoch_validation_exactly(self):
        """Actual run loop/V4/AdamW, tiny CPU fixture streams, no corpus training."""
        plan = build_plan(self.config)
        plan = copy.deepcopy(plan)
        plan["settings"].update(device="cpu", precision="fp32", num_workers=0, conversion_batch=2, lm_batch=2,
            conversion_microbatch=2, lm_microbatch=2, epochs=2, updates_per_epoch=2, max_steps=4,
            warmup_steps=1, log_every=1, eval_every=2, checkpoint_every=2, validation_batch=2,
            baseline_lm_bpc=100.0)
        plan.update(conversion_examples=3, validation_examples=2, lm_validation_targets=3, lm_validation_characters=20)
        def lm_data(*args, **kwargs):
            return ToyDataset(5 if args[2] == "train" else 2)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temporary:
            base = Path(temporary)
            config = copy.deepcopy(self.config)
            config["runtime"]["compile_backbone"] = False
            config["tracking"]["enabled"] = False
            def configure(name):
                local = copy.deepcopy(config)
                local["output_dir"] = str(base / name)
                local["log_dir"] = str(base / (name + "-tb"))
                return local
            interrupted = configure("interrupted")
            with patch("vimeml.posttraining.train.build_plan", return_value=plan), \
                 patch("vimeml.posttraining.train.ConversionDataset", side_effect=lambda path, split: TinyConversion(split)), \
                 patch("vimeml.posttraining.train.PrefixCropWindowDataset", side_effect=lm_data), \
                 patch("vimeml.posttraining.train.SentenceWindowDataset", side_effect=lm_data), \
                 patch("vimeml.posttraining.train.read_snapshot", return_value=[]), \
                 patch("vimeml.posttraining.train.evaluate_snapshot", return_value={"correct": 1, "total": 2, "fallbacks": 0}) as evaluation:
                full = configure("full")
                expected = run(full)
                evaluation.side_effect = RuntimeError("injected interruption during epoch validation")
                with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                    run(interrupted)
                saved = torch.load(base / "interrupted/last.pt", map_location="cpu", weights_only=True)
                self.assertEqual(saved["training_state"]["conversion_cursor"], 2)
                self.assertEqual(saved["training_state"]["epoch_results"], [])
                evaluation.side_effect = None
                actual = run(interrupted, resume=True)
            reference = torch.load(base / "full/last.pt", map_location="cpu", weights_only=True)
            recovered = torch.load(base / "interrupted/last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(actual["status"], "complete")
            self.assertEqual(actual["step"], 4)
            self.assertEqual(actual["lm_cursor"], expected["lm_cursor"])
            for key in reference["model"]:
                torch.testing.assert_close(reference["model"][key], recovered["model"][key], atol=0, rtol=0)
            self.assertFalse(json.loads((base / "interrupted/failure.json").read_text(encoding="utf-8"))["active"])


if __name__ == "__main__":
    unittest.main()
