"""One focused check of real V2 crops, persistent workers and checkpoint resume."""

import contextlib
import copy
import io
import json
import signal
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from vimeml.training import train
from vimeml.training.data import SentenceWindowDataset, make_loader, prepare_indexes, write_json
from vimeml.training.data_v2 import PrefixCropWindowDataset


def snapshots(loader):
    return [{key: value.clone() for key, value in batch.items()} for batch in loader]


def identical(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.keys() == b.keys()
        assert all(torch.equal(a[key], b[key]) for key in a)


def fixture_resume(output):
    token_dir, index_dir = output / "fixture-tokens", output / "fixture-indexes"
    token_dir.mkdir()
    sequences = [
        [2, *[4 + i % 60 for i in range(length)], 3] for length in (6, 8, 14, 15, 16, 17, 30, 41)
    ]
    offsets = np.cumsum([0, *map(len, sequences)], dtype=np.uint64)
    splits = {}
    for split in ("train", "validation", "test"):
        np.asarray([token for sequence in sequences for token in sequence], dtype="<u2").tofile(
            token_dir / f"{split}.tokens.bin"
        )
        offsets.astype("<u8").tofile(token_dir / f"{split}.offsets.bin")
        (token_dir / f"{split}.sources.bin").write_bytes(bytes(len(sequences)))
        splits[split] = {
            "sentences": len(sequences),
            "stored_tokens": int(offsets[-1]),
            "prediction_pairs": int(offsets[-1]) - len(sequences),
            "max_sequence_tokens": max(map(len, sequences)),
        }
    write_json(
        token_dir / "manifest.json",
        {
            "status": "complete",
            "format": "vimeml_sentence_tokens_v1",
            "token_dtype": "uint16_le",
            "offset_dtype": "uint64_le",
            "offset_unit": "tokens",
            "vocab_size": 64,
            "special_ids": dict(pad=0, unk=1, bos=2, eos=3),
            "source_ids": {"fineweb": 0},
            "splits": splits,
            "output_sha256": {},
        },
    )
    index = prepare_indexes(token_dir, index_dir, 16, verification="metadata")
    base = {
        "architecture": "tiny_gpt_v2",
        "token_dir": str(token_dir),
        "index_dir": str(index_dir),
        "model": dict(
            vocab_size=64,
            context_length=16,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            dropout=0.1,
        ),
        "verification": {"mode": "metadata"},
        "training": dict(
            device="cpu",
            precision="fp32",
            seed=42,
            cpu_threads=2,
            batch_size=4,
            num_workers=0,
            bucket_multiplier=4,
            gradient_accumulation=1,
            epochs=2,
            max_steps=0,
            learning_rate=0.001,
            min_learning_rate=0.0001,
            warmup_steps=1,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
            grad_clip=1.0,
            log_every=999,
            eval_every=999,
            eval_batches=1,
            checkpoint_every=999,
            benchmark_warmup_steps=0,
            validate_full_at_end=True,
            prefix_crop_probability=0.30,
            prefix_crop_min_remaining_tokens=8,
        ),
    }
    configs = []
    for name in ("reference", "resumed"):
        config = copy.deepcopy(base)
        config.update(output_dir=str(output / name), log_dir=str(output / f"{name}-events"))
        configs.append(config)
    stop_step = (index["splits"]["train"]["windows"] + 3) // 4 + 1
    original_rate = train.learning_rate

    def stop_in_second_epoch(step, settings):
        if step == stop_step:
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        return original_rate(step, settings)

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        reference = train.run(configs[0])
        with patch.object(train, "learning_rate", side_effect=stop_in_second_epoch):
            interrupted = train.run(configs[1])
        assert interrupted["status"] == "interrupted"
        resumed = train.run(configs[1], resume=True)
    (output / "fixture-training.log").write_text(stream.getvalue(), encoding="utf-8")
    left = torch.load(output / "reference/last.pt", weights_only=True)
    right = torch.load(output / "resumed/last.pt", weights_only=True)
    for key in (
        "step",
        "epoch",
        "batch_cursor",
        "total_tokens",
        "total_windows",
        "total_dropped_tokens",
        "total_cropped_windows",
    ):
        assert left[key] == right[key]
    for name in left["model"]:
        torch.testing.assert_close(left["model"][name], right["model"][name], rtol=0, atol=0)
    assert reference["completed_data_passes"] == resumed["completed_data_passes"] == 2.0
    assert resumed["prefix_crop_dropped_tokens"] > 0
    assert resumed["uncropped_prediction_pairs_seen"] == 2 * splits["train"]["prediction_pairs"]
    assert (
        resumed["full_validation"]["last"]["prediction_pairs"]
        == splits["validation"]["prediction_pairs"]
    )
    return {
        "exact_checkpoint_resume": True,
        "dropout": 0.1,
        "epochs": 2,
        "updates": resumed["step"],
        "interrupted_step": interrupted["step"],
        "trained_tokens": resumed["total_trained_tokens"],
        "dropped_tokens": resumed["prefix_crop_dropped_tokens"],
        "full_epoch_accounting": "passed",
        "validation_unaugmented": True,
    }


def main():
    torch.set_num_threads(2)
    output = ROOT / "outputs/model-checks/v2-prefix-crop"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output for a new acceptance run.")
    output.mkdir(parents=True, exist_ok=True)
    token_dir, index_dir = (
        ROOT / "artifacts/token-data/corpus-v2-16k",
        ROOT / "artifacts/training-data/corpus-v2-c128",
    )
    original = SentenceWindowDataset(token_dir, index_dir)
    cropped = PrefixCropWindowDataset(token_dir, index_dir)
    forced = PrefixCropWindowDataset(token_dir, index_dir, probability=1.0)
    worker_loader = None
    try:
        indices = list(range(4096))
        indices.extend(sorted({int(cropped.first[0]), int(cropped.first[0]) + 1, len(cropped) - 1}))
        counts = {}
        for epoch in (0, 1):
            cropped.set_epoch(epoch)
            lengths = cropped.window_lengths(indices)
            eligible = applied = dropped = 0
            offsets = []
            for index, length in zip(indices, lengths):
                before, after = original[index], cropped[(epoch, index)]
                offset = after["crop_offset"]
                assert len(after["input_ids"]) == int(length)
                assert after["input_ids"] == before["input_ids"][offset:]
                assert after["labels"] == before["labels"][offset:]
                assert after == cropped[(epoch, index)]
                if offset:
                    assert before["window_start"] == 0 and len(after["labels"]) >= 8
                    assert after["input_ids"][0] != 2
                if before["window_start"] > 0 or len(before["labels"]) <= 8:
                    assert offset == 0
                if before["window_start"] == 0 and len(before["labels"]) > 8:
                    eligible += 1
                    force = forced[(epoch, index)]
                    assert force["crop_offset"] > 0 and len(force["labels"]) >= 8
                offsets.append(offset)
                applied += offset > 0
                dropped += offset
            counts[str(epoch)] = {
                "samples": len(indices),
                "eligible": eligible,
                "cropped": applied,
                "crop_rate_among_eligible": applied / eligible,
                "dropped_tokens": dropped,
            }
            if epoch == 0:
                first_offsets = offsets
            else:
                assert first_offsets != offsets
        selected = list(range(256))
        worker_loader = make_loader(
            cropped,
            batch_size=32,
            num_workers=2,
            indices=selected,
            shuffle=False,
            bucket_multiplier=4,
        )
        for epoch in (0, 1):
            worker_loader.batch_sampler.set_epoch(epoch)
            actual = snapshots(worker_loader)
            reference_loader = make_loader(
                cropped,
                batch_size=32,
                num_workers=0,
                indices=selected,
                shuffle=False,
                bucket_multiplier=4,
            )
            reference_loader.batch_sampler.set_epoch(epoch)
            reference = snapshots(reference_loader)
            identical(actual, reference)
            resume_loader = make_loader(
                cropped,
                batch_size=32,
                num_workers=0,
                indices=selected,
                shuffle=False,
                bucket_multiplier=4,
                start_batch=3,
            )
            resume_loader.batch_sampler.set_epoch(epoch, start_batch=3)
            identical(actual[3:], snapshots(resume_loader))
            del reference_loader, resume_loader
        for split in ("validation", "test"):
            unchanged = SentenceWindowDataset(token_dir, index_dir, split)
            try:
                sample = unchanged[0]
                assert "crop_offset" not in sample
            finally:
                unchanged.close()
        report = {
            "status": "passed",
            "probability": 0.30,
            "min_remaining_tokens": 8,
            "real_corpus_samples": counts,
            "scalar_vector_lengths_match": True,
            "same_epoch_reproducible": True,
            "different_epochs_change_crops": True,
            "persistent_worker_epochs": [0, 1],
            "worker_counts_compared": [0, 2],
            "prefetched_resume_batches_exact": True,
            "short_and_continuation_windows_unchanged": True,
            "validation_test_unchanged": True,
            "fixture_training": fixture_resume(output),
            "formal_training_started": False,
        }
        write_json(output / "report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        del worker_loader
        original.close()
        cropped.close()
        forced.close()


if __name__ == "__main__":
    main()
