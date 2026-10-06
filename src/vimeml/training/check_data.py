"""Prepare compact window indexes and manually check the actual data path."""
import argparse
import itertools
import json
import random
import time
import tomllib
from pathlib import Path

import numpy as np

from vimeml.training.data import (
    IGNORE_INDEX, SPLITS, SentenceWindowDataset, collate_arrays,
    make_loader, prepare_indexes, write_json,
)

ROOT = Path(__file__).resolve().parents[3]


def check_batch(batch, dataset):
    """Independently compare each batch row to its original complete sentence."""
    inputs, labels, mask = batch["input_ids"], batch["labels"], batch["attention_mask"]
    if inputs.dtype != np.int64 or labels.dtype != np.int64 or mask.dtype != np.bool_:
        raise ValueError("Incorrect tensor dtypes.")
    if inputs.ndim != 2 or inputs.shape != labels.shape or inputs.shape != mask.shape:
        raise ValueError("Incorrect batch shapes.")
    if not 1 <= inputs.shape[1] <= dataset.context_length:
        raise ValueError("Batch exceeds model context.")
    dataset._open()
    pad = dataset._store.manifest["special_ids"]["pad"]
    if np.any(labels[~mask] != IGNORE_INDEX) or np.any(inputs[~mask] != pad):
        raise ValueError("Padding would contribute to loss.")
    for row, length in enumerate(batch["lengths"]):
        sentence = int(batch["sentence_index"][row])
        start = int(batch["window_start"][row])
        sequence = dataset._store[sentence]
        expected_length = min(dataset.context_length, len(sequence) - 1 - start)
        if int(length) != expected_length or start % dataset.context_length:
            raise ValueError("Incorrect window boundary.")
        if not np.array_equal(mask[row], np.arange(inputs.shape[1]) < length):
            raise ValueError("Invalid attention mask.")
        if (not np.array_equal(inputs[row, :length], sequence[start:start + length])
                or not np.array_equal(labels[row, :length], sequence[start + 1:start + length + 1])):
            raise ValueError("Incorrect next-token shift or sentence boundary crossing.")
        if int(batch["source_id"][row]) != int(dataset._sources[sentence]):
            raise ValueError("Incorrect source metadata.")
    return int(mask.sum())


def check_core(dataset, seed):
    rng = random.Random(seed)
    indices = {0, len(dataset) - 1}
    indices.update(rng.randrange(len(dataset)) for _ in range(32))
    # Explicitly exercise every window of a deterministic selection of long sentences.
    for position in range(min(12, len(dataset.sentences))):
        indices.update(range(int(dataset.first[position]), int(dataset.end[position])))
    try:
        samples = [dataset[i] for i in sorted(indices)]
        batch = collate_arrays(samples, pad_id=dataset._store.manifest["special_ids"]["pad"])
        return {"checked_windows": len(samples), "checked_prediction_pairs": check_batch(batch, dataset)}
    finally:
        dataset.close()


def check_torch(dataset, config):
    import torch
    loader = make_loader(dataset, config["batch_size"], config["num_workers"], config["seed"])
    pairs = batches = positions = rows = 0
    begin = time.perf_counter()
    iterator = iter(loader)
    try:
        for batch in itertools.islice(iterator, config["check_batches"]):
            arrays = {key: value.numpy() for key, value in batch.items()}
            pairs += check_batch(arrays, dataset)
            positions += batch["input_ids"].numel()
            rows += batch["input_ids"].shape[0]
            batches += 1
        if batches == 0:
            raise ValueError("Empty DataLoader.")
    finally:
        # Releasing the owning DataLoader shuts down its persistent spawn workers.
        del iterator, loader
        dataset.close()
    elapsed = time.perf_counter() - begin
    return {"batches": batches, "windows": rows, "prediction_pairs": pairs,
            "elapsed_seconds_including_worker_startup": elapsed,
            "padding_fraction": 1 - pairs / positions,
            "prediction_pairs_per_second_including_worker_startup": pairs / elapsed,
            "workers": config["num_workers"]}


def check_gpu():
    import torch
    import torch.nn.functional as functional
    result = {"torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available()}
    if result["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        # Exercise forward/backward kernels; no model, checkpoint or training job.
        values = torch.randn(16, 32, device="cuda", requires_grad=True)
        weight = torch.randn(64, 32, device="cuda", requires_grad=True)
        targets = torch.arange(16, device="cuda")
        loss = functional.cross_entropy(functional.linear(values, weight), targets)
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or not torch.isfinite(weight.grad).all():
            raise ValueError("CUDA forward/backward produced non-finite values.")
        result.update(device=properties.name, total_memory_mib=properties.total_memory / 1024**2,
                      compute_capability=list(torch.cuda.get_device_capability(0)),
                      forward_backward_check="passed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/loader.toml")
    parser.add_argument("--prepare-only", action="store_true", help="Index and NumPy checks only; PyTorch not required.")
    parser.add_argument("--workers", type=int, help="Override DataLoader CPU process count.")
    args = parser.parse_args(argv)
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    if args.workers is not None:
        config["num_workers"] = args.workers
    if (config["context_length"] < 8 or config["context_length"] % 8 or
            config["batch_size"] < 1 or config["num_workers"] < 0 or config["check_batches"] < 1):
        parser.error("Use context_length divisible by 8, positive batch/check counts, nonnegative workers.")
    if not args.prepare_only:
        try:
            import torch
        except ImportError as error:
            parser.error(f"Install requirements.txt; see docs/training.md for the CUDA 12.8 build: {error}")
    token_dir = ROOT / config["token_dir"]
    index_dir = ROOT / config["index_dir"]
    manifest = prepare_indexes(token_dir, index_dir, config["context_length"])
    report = {"mode": "prepare_only" if args.prepare_only else "pytorch_check",
              "config": config, "index_splits": manifest["splits"], "core_checks": {}}
    for split in SPLITS:
        dataset = SentenceWindowDataset(token_dir, index_dir, split)
        report["core_checks"][split] = check_core(dataset, config["seed"])
        print(f"{split}: {len(dataset):,} windows; core check passed", flush=True)
    if not args.prepare_only:
        report["pytorch_checks"] = {}
        # Test remains structurally checked above, never used for tuning/early stopping.
        for split in ("train", "validation"):
            dataset = SentenceWindowDataset(token_dir, index_dir, split)
            report["pytorch_checks"][split] = check_torch(dataset, config)
            print(f"{split}: PyTorch batches passed", flush=True)
        report["environment"] = check_gpu()
    report["status"] = "passed"
    path = index_dir / ("prepare-check.json" if args.prepare_only else "loader-check.json")
    write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
