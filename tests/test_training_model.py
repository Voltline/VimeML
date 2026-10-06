"""Causal masking, loss semantics, actual learning and exact committed resume."""
import contextlib
import copy
import io
import json
import signal
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training import train
from vimeml.training.data import file_sha, prepare_indexes, write_json
from vimeml.training.model import GPTConfig, TinyGPT


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.config = GPTConfig(vocab_size=16, context_length=8, d_model=32,
                                n_heads=4, n_layers=2, d_ff=64)

    def test_causality_and_right_padding_do_not_change_valid_logits(self):
        model = TinyGPT(self.config).eval()
        prefix = torch.tensor([[2, 5, 6]])
        with torch.no_grad():
            short = model(prefix)
            changed_future = model(torch.tensor([[2, 5, 6, 9, 10]]))
            padded = model(torch.tensor([[2, 5, 6, 0, 0, 0, 0, 0]]))
        torch.testing.assert_close(short, changed_future[:, :3], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(short, padded[:, :3], rtol=1e-5, atol=1e-6)

    def test_loss_matches_already_shifted_targets_and_ignores_pad_positions(self):
        model = TinyGPT(self.config).eval()
        inputs = torch.tensor([[2, 5, 6, 0, 0], [2, 9, 0, 0, 0]])
        labels = torch.tensor([[5, 6, 3, -100, -100], [9, 3, -100, -100, -100]])
        logits = model(inputs)
        expected = F.cross_entropy(logits.reshape(-1, 16), labels.reshape(-1), ignore_index=-100, reduction="sum")
        actual = model(inputs, labels)
        torch.testing.assert_close(actual["loss_sum"], expected)
        self.assertEqual(int(actual["token_count"]), 5)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_tiny_synthetic_batch_can_be_learned(self):
        model = TinyGPT(self.config)
        inputs = torch.tensor([[2, 5, 6]] * 4)
        labels = torch.tensor([[5, 6, 3]] * 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        initial = float(model(inputs, labels)["loss_sum"].detach())
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            (model(inputs, labels)["loss_sum"] / 12).backward()
            optimizer.step()
        final = float(model(inputs, labels)["loss_sum"].detach())
        self.assertLess(final, initial * 0.2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_model_bf16_cuda_forward_backward_is_finite(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 unsupported")
        model = TinyGPT().cuda()
        inputs = torch.randint(4, 16384, (8, 32), device="cuda")
        labels = torch.randint(4, 16384, (8, 32), device="cuda")
        labels[:, 24:] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(inputs, labels)["loss_sum"] / 192
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all())
        self.assertEqual(model.parameter_count(), 7386624)


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "outputs")
        self.root = Path(self.temporary.name)
        token_dir = self.root / "tokens"
        token_dir.mkdir()
        sequences = [[2, 5, 6, 3], [2, 7, 8, 9, 3], [2, 10, 3],
                     [2, 5, 6, 7, 8, 9, 10, 11, 12, 3], [2, 11, 12, 3]]
        offsets = [0]
        for sequence in sequences:
            offsets.append(offsets[-1] + len(sequence))
        hashes, splits = {}, {}
        for split in ("train", "validation", "test"):
            contents = {"tokens.bin": struct.pack("<" + "H" * offsets[-1], *sum(sequences, [])),
                        "offsets.bin": struct.pack("<" + "Q" * len(offsets), *offsets),
                        "sources.bin": bytes(i % 2 for i in range(len(sequences)))}
            for suffix, data in contents.items():
                path = token_dir / f"{split}.{suffix}"
                path.write_bytes(data)
                hashes[path.name] = file_sha(path)
            splits[split] = {"sentences": len(sequences), "stored_tokens": offsets[-1],
                             "prediction_pairs": offsets[-1] - len(sequences),
                             "max_sequence_tokens": max(map(len, sequences))}
        write_json(token_dir / "manifest.json", {"status": "complete",
            "format": "vimeml_sentence_tokens_v1", "token_dtype": "uint16_le", "offset_dtype": "uint64_le",
            "offset_unit": "tokens", "vocab_size": 16, "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
            "source_ids": {"fineweb": 0, "tatoeba": 1}, "splits": splits, "output_sha256": hashes})
        prepare_indexes(token_dir, self.root / "indexes", 8)
        self.config = {"token_dir": str(token_dir), "index_dir": str(self.root / "indexes"),
            "model": {"vocab_size": 16, "context_length": 8, "d_model": 16, "n_heads": 2,
                      "n_layers": 1, "d_ff": 32, "dropout": 0.1},
            "training": {"device": "cpu", "precision": "fp32", "seed": 42, "cpu_threads": 2,
                "batch_size": 2, "num_workers": 0, "bucket_multiplier": 2, "gradient_accumulation": 2,
                "max_steps": 6, "learning_rate": 0.001, "min_learning_rate": 0.0001, "warmup_steps": 1,
                "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
                "log_every": 2, "eval_every": 3, "eval_batches": 2, "checkpoint_every": 3,
                "benchmark_warmup_steps": 1}}

    def tearDown(self):
        self.temporary.cleanup()

    def configuration(self, name):
        config = copy.deepcopy(self.config)
        config["output_dir"] = str(self.root / name)
        config["log_dir"] = str(self.root / f"{name}-events")
        return config

    def test_trainer_resume_matches_uninterrupted_with_dropout_and_epoch_boundaries(self):
        reference, interrupted = self.configuration("reference"), self.configuration("interrupted")
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(reference)
            original_rate = train.learning_rate

            def stop_at_second_update(step, settings):
                if step == 2:
                    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                return original_rate(step, settings)

            with patch.object(train, "learning_rate", side_effect=stop_at_second_update):
                stopped = train.run(interrupted)
            self.assertEqual(stopped["status"], "interrupted")
            self.assertEqual(stopped["step"], 2)
            resumed = train.run(interrupted, resume=True)
            self.assertEqual(resumed["status"], "complete")
        left = torch.load(Path(reference["output_dir"]) / "last.pt", weights_only=True)
        right = torch.load(Path(interrupted["output_dir"]) / "last.pt", weights_only=True)
        self.assertEqual(left["total_tokens"], right["total_tokens"])
        self.assertEqual((left["epoch"], left["batch_cursor"]), (right["epoch"], right["batch_cursor"]))
        for name in left["model"]:
            torch.testing.assert_close(left["model"][name], right["model"][name], rtol=0, atol=0)
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(interrupted["log_dir"]).Reload()
        self.assertIn("loss/train", events.Tags()["scalars"])
        self.assertIn("loss/validation", events.Tags()["scalars"])
        self.assertTrue((Path(interrupted["output_dir"]) / "best.pt").exists())
        changed = copy.deepcopy(interrupted)
        changed["training"]["batch_size"] += 1
        with self.assertRaisesRegex(ValueError, "same configuration"):
            train.run(changed, resume=True)

    def test_dry_run_creates_no_training_artifacts(self):
        config = self.configuration("plan")
        with contextlib.redirect_stdout(io.StringIO()):
            train.run(config, dry_run=True)
        self.assertFalse(Path(config["output_dir"]).exists())
        self.assertFalse(Path(config["log_dir"]).exists())

    def test_exact_epoch_and_full_validation_for_last_and_earlier_best(self):
        config = self.configuration("epoch")
        config["training"].update(epochs=1, max_steps=3, gradient_accumulation=1,
            eval_every=1, validate_full_at_end=True)
        original_evaluate = train.evaluate
        sampled_losses = iter((10.0, 9.0, 8.0, 9.0))

        def controlled_validation(model, loader, device, precision):
            result = original_evaluate(model, loader, device, precision)
            if len(loader.batch_sampler.sampler) == 4:
                result["loss"] = next(sampled_losses)
            return result

        with contextlib.redirect_stdout(io.StringIO()), patch.object(train, "evaluate", side_effect=controlled_validation):
            result = train.run(config)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_data_passes"], 1.0)
        self.assertEqual(result["total_trained_windows"], 6)
        self.assertEqual(result["total_trained_tokens"], 21)
        self.assertEqual(result["full_validation"]["best_checkpoint_step"], 2)
        for checkpoint in ("last", "best"):
            self.assertEqual(result["full_validation"][checkpoint]["prediction_pairs"], 21)
        saved = torch.load(Path(config["output_dir"]) / "last.pt", weights_only=True)
        self.assertEqual(saved["step"], 3)
        # An interrupted final evaluation can be completed without another update.
        (Path(config["output_dir"]) / "full-validation.json").unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            recovered = train.run(config, resume=True)
        self.assertEqual(recovered["step"], 3)
        self.assertEqual(recovered["total_trained_tokens"], 21)

    def test_epoch_schedule_rejects_incomplete_pass(self):
        settings = self.config["training"].copy()
        settings.update(epochs=1, gradient_accumulation=1, max_steps=2)
        with self.assertRaisesRegex(ValueError, "Epoch mode"):
            train.validate_epoch_schedule(settings, windows=6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp_fused_optimizer_and_spawn_workers_in_fixture_run(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 unsupported")
        config = self.configuration("cuda-fixture")
        config["training"].update(device="cuda", precision="bf16", num_workers=2,
                                  max_steps=2, gradient_accumulation=1)
        with contextlib.redirect_stdout(io.StringIO()):
            result = train.run(config)
        self.assertEqual(result["status"], "complete")
        self.assertGreater(result["benchmark_effective_tokens_per_second"], 0)
        self.assertGreater(result["peak_allocated_mib"], 0)
        saved = torch.load(Path(config["output_dir"]) / "last.pt", weights_only=True, map_location="cpu")
        self.assertEqual(saved["precision"], "bf16")
        self.assertTrue(all(torch.isfinite(value).all() for value in saved["model"].values()))


if __name__ == "__main__":
    unittest.main()
