"""V4 extension binds the source checkpoint and keeps the learning rate continuous."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.continuation_v4 import configure, validate_resume
from vimeml.training.train import learning_rate


class V4ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.source = tomllib.loads((ROOT / "configs/train-v4-b.toml").read_text(encoding="utf-8"))
        self.source["training"]["epochs"] = 4
        self.source["training"].pop("early_stopping_mode", None)
        settings = {**self.source["training"], "max_steps": 196940}
        rate = learning_rate(140000, settings)
        self.config = copy.deepcopy(self.source)
        self.config["training"].update(epochs=8, early_stopping_mode="plateau")
        self.config["continuation"] = {"source_step": 140000, "source_max_steps": 196940,
            "source_learning_rate": rate, "source_train_sha256": "old-trainer",
            "source_checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest()}
        self.before = {"tokens": "frozen-tokens", "windows": "frozen-windows",
                       "training_code": {"train.py": "old-trainer", "model_v4.py": "fixed-model"}}
        self.after = {**self.before, "training_code": {"train.py": "new-trainer",
            "model_v4.py": "fixed-model", "continuation_v4.py": "extension-helper"}}
        self.saved = {"format": "vimeml_tiny_gpt_v4", "precision": "bf16", "step": 140000,
            "scheduler": {"max_steps": 196940}, "config": self.source, "signatures": self.before,
            "optimizer": {"param_groups": [{"lr": rate}, {"lr": rate}]}}

    def validate(self, saved=None, config=None, signatures=None):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temp:
            path = Path(temp) / "last.pt"
            path.write_bytes(b"checkpoint")
            validate_resume(saved or self.saved, config or self.config,
                            signatures or self.after, "bf16", path)

    def test_explicit_extension_accepts_only_intended_changes(self):
        self.validate()

    def test_chained_extension_keeps_plateau_policy_and_binds_previous_helper(self):
        saved = copy.deepcopy(self.saved)
        saved["config"] = copy.deepcopy(self.config)
        saved["signatures"] = copy.deepcopy(self.after)
        config = copy.deepcopy(self.config)
        config["training"]["epochs"] = 12
        config["continuation"]["source_train_sha256"] = "new-trainer"
        config["continuation"]["source_continuation_sha256"] = "extension-helper"
        signatures = copy.deepcopy(self.after)
        signatures["training_code"]["continuation_v4.py"] = "chained-extension-helper"
        self.validate(saved, config, signatures)
        config["continuation"]["source_continuation_sha256"] = "wrong-helper"
        with self.assertRaises(ValueError): self.validate(saved, config, signatures)

    def test_learning_rate_is_continuous_and_decays_to_floor(self):
        settings = {**self.config["training"], "max_steps": 393880}
        configure(self.config, settings)
        anchor = self.config["continuation"]["source_learning_rate"]
        self.assertAlmostEqual(learning_rate(140000, settings), anchor, places=15)
        rates = [learning_rate(i, settings) for i in range(140000, 393881, 1000)]
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
        self.assertAlmostEqual(learning_rate(393880, settings), .00006, places=15)

    def test_rejects_data_model_batch_checkpoint_cursor_and_lr_changes(self):
        for kind in ("data", "model", "batch", "checkpoint", "cursor", "lr"):
            with self.subTest(kind=kind):
                saved, config, signatures = copy.deepcopy((self.saved, self.config, self.after))
                if kind == "data": signatures["tokens"] = "other-data"
                if kind == "model": signatures["training_code"]["model_v4.py"] = "other-model"
                if kind == "batch": config["training"]["batch_size"] = 256
                if kind == "checkpoint": config["continuation"]["source_checkpoint_sha256"] = "wrong"
                if kind == "cursor": saved["step"] += 1
                if kind == "lr": saved["optimizer"]["param_groups"][0]["lr"] *= 2
                with self.assertRaises(ValueError): self.validate(saved, config, signatures)


if __name__ == "__main__":
    unittest.main()
