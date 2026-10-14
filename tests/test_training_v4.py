"""V4-only two-split trainer checkpoint identity and exact committed resume."""
import contextlib
import copy
import io
import signal
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.model_factory_v4 import register_v4
from vimeml.training.model_v4 import GPTV4Config
from vimeml.training import train
from vimeml.training.data import file_sha, prepare_indexes, write_json


class V4TrainerTests(unittest.TestCase):
    def test_two_split_synthetic_resume_is_exact_and_uses_v4_format(self):
        register_v4()
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temp:
            root = Path(temp)
            token_dir, index_dir = root / "tokens", root / "indexes"
            token_dir.mkdir()
            sequences = [[2, 7, 8, 3], [2, 9, 10, 11, 3], [2, 12, 3], [2, *([13] * 140), 3], [2, 14, 15, 3]]
            offsets = [0]
            for sequence in sequences:
                offsets.append(offsets[-1] + len(sequence))
            hashes, splits = {}, {}
            for split in ("train", "validation"):
                data = {"tokens.bin": struct.pack(f"<{offsets[-1]}H", *sum(sequences, [])),
                        "offsets.bin": struct.pack(f"<{len(offsets)}Q", *offsets),
                        "sources.bin": bytes(i % 2 for i in range(len(sequences)))}
                for suffix, contents in data.items():
                    path = token_dir / f"{split}.{suffix}"
                    path.write_bytes(contents)
                    hashes[path.name] = file_sha(path)
                splits[split] = {"sentences": 5, "stored_tokens": offsets[-1],
                                 "prediction_pairs": offsets[-1] - 5, "max_sequence_tokens": 142}
            write_json(token_dir / "manifest.json", {"status": "complete", "format": "vimeml_sentence_tokens_v1",
                "token_dtype": "uint16_le", "offset_dtype": "uint64_le", "offset_unit": "tokens",
                "vocab_size": 8192, "tokenizer_model_sha256": "synthetic-test-only", "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
                "source_ids": {"fineweb": 0, "tatoeba": 1}, "splits": splits, "output_sha256": hashes})
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_indexes(token_dir, index_dir, 128, splits=("train", "validation"))
            model_config = __import__('dataclasses').asdict(GPTV4Config())
            base = {"architecture": "tiny_gpt_v4", "token_dir": str(token_dir), "index_dir": str(index_dir),
                "model": model_config, "training": {"device": "cpu", "precision": "fp32", "seed": 42,
                    "cpu_threads": 2, "batch_size": 2, "num_workers": 0, "bucket_multiplier": 2,
                    "gradient_accumulation": 2, "max_steps": 3, "learning_rate": .0006, "min_learning_rate": .00006,
                    "warmup_steps": 1, "weight_decay": .1, "beta1": .9, "beta2": .95, "grad_clip": 1.0,
                    "log_every": 1, "eval_every": 2, "eval_batches": 1, "checkpoint_every": 2,
                    "benchmark_warmup_steps": 0, "prefix_crop_probability": .3, "prefix_crop_min_remaining_tokens": 8}}
            configs = []
            for name in ("reference", "resumed"):
                config = copy.deepcopy(base)
                config.update(output_dir=str(root / name), log_dir=str(root / f"{name}-events"))
                configs.append(config)
            with contextlib.redirect_stdout(io.StringIO()):
                train.run(configs[0])
                original_rate = train.learning_rate
                def stop(step, settings):
                    if step == 2:
                        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                    return original_rate(step, settings)
                with patch.object(train, "learning_rate", side_effect=stop):
                    halted = train.run(configs[1])
                self.assertEqual(halted["status"], "interrupted")
                self.assertEqual(halted["step"], 2)
                train.run(configs[1], resume=True)
            left = torch.load(root / "reference/last.pt", map_location="cpu", weights_only=True)
            right = torch.load(root / "resumed/last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(right["format"], "vimeml_tiny_gpt_v4")
            self.assertEqual((left["total_tokens"], left["epoch"], left["batch_cursor"]),
                             (right["total_tokens"], right["epoch"], right["batch_cursor"]))
            for name in left["model"]:
                torch.testing.assert_close(left["model"][name], right["model"][name], atol=0, rtol=0)
            self.assertFalse((token_dir / "test.tokens.bin").exists())


if __name__ == "__main__":
    unittest.main()
