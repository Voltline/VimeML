"""W&B is optional; no test authenticates or uploads to an external service."""
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training import train as trainer
spec = importlib.util.spec_from_file_location("train_wandb_entry", ROOT / "scripts/training/train_wandb.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class WandbEntryTests(unittest.TestCase):
    def test_dry_run_never_initializes_wandb(self):
        fake = SimpleNamespace(init=Mock(side_effect=AssertionError("Network init forbidden")))
        with patch.dict(sys.modules, {"wandb": fake}), patch.object(trainer, "run") as run:
            with contextlib.redirect_stdout(io.StringIO()):
                entry.main(["--dry-run"])
            fake.init.assert_not_called()
            self.assertTrue(run.call_args.kwargs["dry_run"])

    def test_tensorboard_sync_initializes_and_forwards_resume_without_training_changes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temporary:
            root = Path(temporary)
            output = root / "artifacts/models/fixture"
            output.mkdir(parents=True)
            (output / "last.pt").write_bytes(b"fixture")
            config = root / "config.toml"
            config.write_text('output_dir="artifacts/models/fixture"\nlog_dir="runs/fixture"\n'
                              '[model]\nvocab_size=16\n[training]\nmax_steps=2\n', encoding="utf-8")
            cloud = Mock(id="example", url="https://example.invalid/tracking/runs/example")
            context = Mock()
            context.__enter__ = Mock(return_value=cloud)
            context.__exit__ = Mock(return_value=False)
            fake = SimpleNamespace(init=Mock(return_value=context), Settings=Mock(return_value={}))
            with patch.object(entry, "ROOT", root), patch.dict(sys.modules, {"wandb": fake}), \
                    patch.object(trainer, "run", return_value={"status": "interrupted"}) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    entry.main(["--config", str(config), "--resume", "--project", "test-project"])
            options = fake.init.call_args.kwargs
            self.assertTrue(options["sync_tensorboard"])
            self.assertFalse(options["save_code"])
            self.assertEqual(options["project"], "test-project")
            self.assertNotIn("token_dir", options["config"])
            self.assertTrue(run.call_args.kwargs["resume"])
            cloud.summary.update.assert_called_once()
            context.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
