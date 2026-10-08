"""Real-data V2.1 throughput comparison; model updates are discarded, never checkpointed."""

import argparse
import collections
import gc
import json
import math
import statistics
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from vimeml.training.data import make_loader, write_json
from vimeml.training.data_v2 import PrefixCropWindowDataset
from vimeml.training.model_factory import model_from_checkpoint
from vimeml.training.runtime_v2 import compile_backbone
from vimeml.training.train import learning_rate, optimizer_for, worker_init


def build_loader(config, batch_size, workers, cuda):
    settings = config["training"]
    if config.get("data_mixture"):
        from vimeml.training.data_mixture import mixture_dataset
        dataset = mixture_dataset(config, ROOT)
    else:
        dataset = PrefixCropWindowDataset(
            ROOT / config["token_dir"],
            ROOT / config["index_dir"],
            seed=settings["seed"],
            probability=settings["prefix_crop_probability"],
            min_remaining_tokens=settings["prefix_crop_min_remaining_tokens"],
        )
    loader = make_loader(
        dataset,
        batch_size,
        workers,
        settings["seed"],
        pin_memory=cuda,
        bucket_multiplier=settings["bucket_multiplier"],
        worker_init_fn=worker_init,
    )
    loader.batch_sampler.set_epoch(settings.get("data_epoch_offset", 0))
    return dataset, loader


def compilation_counts():
    # Diagnostic counters are optional and vary across PyTorch versions.
    try:
        from torch._dynamo.utils import counters
    except ImportError:
        return {}
    return {
        f"{group}/{key}": value
        for group, values in counters.items()
        for key, value in values.items()
        if group in {"stats", "frames", "inductor", "graph_break"}
        and isinstance(value, (int, float))
    }


def measure(config, saved, batch_size, workers, warmup, steps, variant, plan_only):
    settings = {**config["training"]}
    dataset, loader = build_loader(config, batch_size, workers, not plan_only)
    settings["max_steps"] = settings["epochs"] * math.ceil(len(dataset) / batch_size)
    model = optimizer = iterator = loss = inputs = labels = None
    device = torch.device("cuda")
    if not plan_only:
        torch.manual_seed(settings["seed"])
        model = model_from_checkpoint(saved).to(device).train()
        optimizer = optimizer_for(model, settings, device)
        if variant == "compiled":
            compile_backbone(model)
        torch.cuda.reset_peak_memory_stats(device)
    before_counts = compilation_counts()
    records = []
    shapes = collections.Counter()
    try:
        for index in range(warmup + steps):
            started = time.perf_counter()
            if iterator is None:
                iterator = iter(loader)
            batch = next(iterator)
            data_seconds = time.perf_counter() - started
            tokens = int(batch["lengths"].sum())
            positions = batch["input_ids"].numel()
            shape = list(batch["input_ids"].shape)
            shapes["x".join(map(str, shape))] += 1
            if not plan_only:
                rate = learning_rate(index + 1, settings)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                inputs = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    denominator = (dataset.manifest["epochs"][dataset.epoch]["effective_tokens"] /
                                   math.ceil(len(dataset) / batch_size)
                                   if settings.get("loss_token_normalization") == "epoch_mean_tokens" else tokens)
                    loss = model(inputs, labels)["loss_sum"] / denominator
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings["grad_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                torch.cuda.synchronize(device)
                if not math.isfinite(float(loss)):
                    raise FloatingPointError("Non-finite benchmark loss.")
            records.append(
                {
                    "step": index + 1,
                    "warmup": index < warmup,
                    "shape": shape,
                    "tokens": tokens,
                    "positions": positions,
                    "data_seconds": data_seconds,
                    "update_seconds": time.perf_counter() - started,
                }
            )
            if not plan_only and ((index + 1) % 16 == 0 or index == 0):
                print(
                    f"{variant} batch={batch_size} step={index + 1}/{warmup + steps} "
                    f"seconds={records[-1]['update_seconds']:.3f}",
                    flush=True,
                )
        measured = records[warmup:]
        seconds = sum(row["update_seconds"] for row in measured)
        counts = compilation_counts()
        result = {
            "variant": variant,
            "batch_size": batch_size,
            "workers": workers,
            "warmup_steps": warmup,
            "measured_steps": steps,
            "warmup_seconds": sum(row["update_seconds"] for row in records[:warmup]),
            "measured_seconds": seconds,
            "shape_counts": dict(shapes),
            "padding_fraction": 1
            - sum(row["tokens"] for row in measured)
            / sum(row["positions"] for row in measured),
            "p50_update_ms": statistics.median(
                row["update_seconds"] for row in measured
            )
            * 1000,
            "p95_update_ms": sorted(row["update_seconds"] for row in measured)[
                math.ceil(0.95 * steps) - 1
            ]
            * 1000,
            "compilation_counter_delta": {
                key: value - before_counts.get(key, 0)
                for key, value in counts.items()
                if value != before_counts.get(key, 0)
            },
            "records": records,
        }
        if not plan_only:
            result.update(
                {
                    "loader_wait_fraction": sum(row["data_seconds"] for row in measured)
                    / seconds,
                    "effective_tokens_per_second": sum(
                        row["tokens"] for row in measured
                    )
                    / seconds,
                    "samples_per_second": sum(row["shape"][0] for row in measured)
                    / seconds,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                    / 1024**2,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved(device)
                    / 1024**2,
                }
            )
        else:
            result["gpu_throughput_measured"] = False
            result["cpu_data_fraction"] = (
                sum(row["data_seconds"] for row in measured) / seconds
            )
        return result
    finally:
        iterator = loader = optimizer = model = loss = inputs = labels = None
        dataset.close()
        if not plan_only:
            gc.collect()
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train-v21.toml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--variants", nargs="+", choices=("eager", "compiled"), default=["eager", "compiled"])
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Read real batches on CPU; no model load or GPU updates.",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory.")
    if args.steps < 1 or args.warmup < 1 or any(size < 1 for size in args.batch_sizes):
        parser.error("Batch sizes, steps and warmup must be positive.")
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    settings = config["training"]
    workers = settings["num_workers"] if args.workers is None else args.workers
    if (
        workers < 0
        or settings["precision"] != "bf16"
        or settings["gradient_accumulation"] != 1
    ):
        parser.error(
            "This benchmark expects nonnegative workers, BF16 and accumulation=1."
        )
    torch.set_num_threads(settings["cpu_threads"])
    saved = None
    if not args.plan_only:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            parser.error(
                "CUDA BF16 is required for measurements; use --plan-only on CPU."
            )
        saved = torch.load(
            ROOT / config["initialization"]["checkpoint"],
            map_location="cpu",
            weights_only=True,
        )
    args.output.mkdir(parents=True)
    report = {
        "format": "vimeml_v3_real_data_performance_v1" if config.get("data_mixture") else "vimeml_v21_real_data_performance_v1",
        "status": "running",
        "torch_version": str(torch.__version__),
        "gpu": None if args.plan_only else torch.cuda.get_device_name(),
        "config": str(args.config),
        "data_epoch": settings.get("data_epoch_offset", 0),
        "formal_training_started": False,
        "saved_model_or_optimizer": False,
        "plan_only": args.plan_only,
        "cache_policy": "Existing disk caches retained; first-use and warmup are reported separately.",
        "timing_scope": "Real loader, prefix crop, H2D, forward/backward, clipping, fused AdamW and synchronization; no epoch evaluation/checkpoint/W&B.",
        "interpretation": "A short prefix of the real epoch, not full-epoch or thermal steady-state performance; measured-region recompilation is not discarded.",
        "results": [],
    }
    try:
        for size in args.batch_sizes:
            variants = ("loader_only",) if args.plan_only else args.variants
            for variant in variants:
                try:
                    item = measure(
                        config,
                        saved,
                        size,
                        workers,
                        args.warmup,
                        args.steps,
                        variant,
                        args.plan_only,
                    )
                except torch.cuda.OutOfMemoryError:
                    item = {
                        "variant": variant,
                        "batch_size": size,
                        "status": "unavailable_cuda_oom",
                    }
                report["results"].append(item)
                write_json(args.output / "report.json", report)
            if not args.plan_only and "eager" in args.variants and "compiled" in args.variants:
                eager, compiled = report["results"][-2:]
                if any(
                    item.get("status") == "unavailable_cuda_oom"
                    for item in (eager, compiled)
                ):
                    continue
                if [(r["shape"], r["tokens"]) for r in eager["records"]] != [
                    (r["shape"], r["tokens"]) for r in compiled["records"]
                ]:
                    raise ValueError("Eager/compiled batch sequence summaries differ.")
                compiled["speedup_vs_same_batch_eager"] = (
                    eager["measured_seconds"] / compiled["measured_seconds"]
                )
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                **report,
                "results": [
                    {k: v for k, v in item.items() if k != "records"}
                    for item in report["results"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
