"""Manual Tiny GPT training: bounded batches, token-weighted loss and checkpoints."""
import argparse
import json
import math
import random
import signal
import time
import tomllib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from vimeml.training.data import SentenceWindowDataset, file_sha, make_loader, prepare_indexes, write_json
from vimeml.training.model import GPTConfig, TinyGPT

ROOT = Path(__file__).resolve().parents[3]


def learning_rate(step, settings):
    """step is one-based optimizer update; linear warmup followed by cosine."""
    warmup, maximum = settings["warmup_steps"], settings["max_steps"]
    high, low = settings["learning_rate"], settings["min_learning_rate"]
    if warmup and step <= warmup:
        return high * step / warmup
    fraction = min(1.0, max(0.0, (step - warmup) / max(1, maximum - warmup)))
    return low + (high - low) * (1 + math.cos(math.pi * fraction)) / 2


def worker_init(worker_id):
    # Ctrl+C goes to the parent, which checkpoints after its current update.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)


def amp_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type,
                         dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def resolve_precision(device, requested):
    if requested == "auto":
        return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            "fp16" if device.type == "cuda" else "fp32")
    if requested not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be auto/fp32/bf16/fp16.")
    if device.type != "cuda" and requested != "fp32":
        raise ValueError("CPU checks use fp32; choose precision=fp32 or auto.")
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16; choose precision=auto or fp16.")
    return requested


def optimizer_for(model, settings, device):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": settings["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=settings["learning_rate"], betas=(settings["beta1"], settings["beta2"]),
        fused=device.type == "cuda")


def atomic_checkpoint(path, state):
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@torch.inference_mode()
def evaluate(model, loader, device, precision):
    model.eval()
    loss_sum = total_tokens = 0
    try:
        for batch in loader:
            inputs = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with amp_context(device, precision):
                result = model(inputs, labels)
            loss_sum += float(result["loss_sum"])
            total_tokens += int(batch["lengths"].sum())
        if not total_tokens or not math.isfinite(loss_sum):
            raise ValueError("Empty or non-finite validation result.")
        return {"loss": loss_sum / total_tokens, "prediction_pairs": total_tokens,
                "perplexity": math.exp(min(50, loss_sum / total_tokens))}
    finally:
        model.train()


def validate_config(config):
    model = GPTConfig(**config["model"])
    settings = config["training"]
    positive = ("cpu_threads", "batch_size", "bucket_multiplier", "gradient_accumulation",
                "max_steps", "log_every", "eval_every", "eval_batches", "checkpoint_every")
    if any(settings[name] < 1 for name in positive) or settings["num_workers"] < 0:
        raise ValueError("Invalid training batch/worker/interval configuration.")
    if model.context_length % 8 or not 0 <= settings["warmup_steps"] <= settings["max_steps"]:
        raise ValueError("Context must be divisible by 8; warmup must fit training steps.")
    if (not 0 < settings["min_learning_rate"] <= settings["learning_rate"] or
            settings["weight_decay"] < 0 or settings["grad_clip"] <= 0 or
            not 0 <= settings["beta1"] < 1 or not 0 <= settings["beta2"] < 1 or
            not 0 <= settings["benchmark_warmup_steps"] < settings["max_steps"]):
        raise ValueError("Invalid optimizer or benchmark configuration.")
    return model, settings


def validate_epoch_schedule(settings, windows):
    """An explicit epoch promise must match a complete, loss-preserving pass."""
    batches = math.ceil(windows / settings["batch_size"])
    epochs = settings.get("epochs")
    if epochs is not None:
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        # Current epoch mode ends on a committed update without borrowing data
        # from the next epoch to fill a partial accumulation group.
        if settings["gradient_accumulation"] != 1 or settings["max_steps"] != epochs * batches:
            raise ValueError("Epoch mode requires gradient_accumulation=1 and max_steps=epochs*batches_per_epoch.")
    return batches


def run(config, resume=False, dry_run=False):
    model_config, settings = validate_config(config)
    token_dir, index_dir = ROOT / config["token_dir"], ROOT / config["index_dir"]
    output, log_dir = ROOT / config["output_dir"], ROOT / config["log_dir"]
    token_manifest = json.loads((token_dir / "manifest.json").read_text(encoding="utf-8"))
    index_manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if (token_manifest["vocab_size"] != model_config.vocab_size or
            index_manifest["context_length"] != model_config.context_length):
        raise ValueError("Model vocabulary/context must match token data/window index.")
    signatures = {"tokens": file_sha(token_dir / "manifest.json"),
                  "windows": file_sha(index_dir / "manifest.json"),
                  "training_code": {name: file_sha(Path(__file__).with_name(name))
                                    for name in ("train.py", "model.py", "data.py")}}
    if token_manifest.get("status") != "complete" or index_manifest.get("status") != "complete":
        raise ValueError("Complete token store and window index required.")
    if signatures["tokens"] != index_manifest["token_manifest_sha256"]:
        raise ValueError("Window index belongs to a different token dataset.")
    batches_per_epoch = validate_epoch_schedule(settings, index_manifest["splits"]["train"]["windows"])
    device = torch.device(settings["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Initial trainer supports CPU or CUDA.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; verify installation before training.")
    precision = resolve_precision(device, settings["precision"])
    torch.set_num_threads(settings["cpu_threads"])
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings["seed"])
    model = TinyGPT(model_config)
    plan = {"model": model.configuration(), "parameters": model.parameter_count(),
            "device": str(device), "precision": precision, "max_steps": settings["max_steps"],
            "training_prediction_pairs_per_epoch": token_manifest["splits"]["train"]["prediction_pairs"],
            "training_batches_per_epoch": batches_per_epoch,
            "planned_epochs": settings.get("epochs"),
            "output": str(output), "tensorboard": str(log_dir),
            "validation": "Fixed random subset, token-weighted loss; test unused.",
            "corpus_quality_mode": token_manifest.get("corpus_quality_mode"),
            "full_validation_at_end": settings.get("validate_full_at_end", False),
            "purpose": "First full-corpus baseline." if settings.get("epochs") else
                       "Short pipeline/throughput run; not a deployment-ready model."}
    if dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    last_path = output / "last.pt"
    if resume:
        if not last_path.exists():
            raise ValueError("No last.pt checkpoint to resume.")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is nonempty; use --resume or a new output directory.")
    elif log_dir.exists() and any(log_dir.iterdir()):
        raise ValueError("TensorBoard directory is nonempty; use a new log directory.")
    prepare_indexes(token_dir, index_dir, model_config.context_length)
    model.to(device)
    optimizer = optimizer_for(model, settings, device)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    step = epoch = batch_cursor = total_tokens = total_windows = 0
    best_val = math.inf
    benchmark_tokens = benchmark_steps = 0
    benchmark_seconds = 0.0
    restored_rng = None
    if resume:
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["signatures"] != signatures or saved["precision"] != precision:
            raise ValueError("Resume requires the same configuration, precision and dataset signatures.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step, epoch, batch_cursor = saved["step"], saved["epoch"], saved["batch_cursor"]
        total_tokens, best_val = saved["total_tokens"], saved["best_val"]
        total_windows = saved["total_windows"]
        benchmark_tokens, benchmark_steps, benchmark_seconds = (
            saved["benchmark_tokens"], saved["benchmark_steps"], saved["benchmark_seconds"])
        restored_rng = saved["rng"]
        if step >= settings["max_steps"]:
            summary_path = output / "summary.json"
            final_eval_complete = ((output / "full-validation.json").exists() and
                summary_path.exists() and json.loads(summary_path.read_text(encoding="utf-8")).get("status") == "complete")
            if not settings.get("validate_full_at_end", False) or final_eval_complete:
                print(f"Already complete at step {step}; no training started.")
                return plan
            print("Training updates complete; resuming unfinished full validation.", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "plan.json", plan)
    environment = {"torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                   "device": str(device), "precision": precision}
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
    write_json(output / "environment.json", environment)
    train_data = SentenceWindowDataset(token_dir, index_dir, "train")
    val_data = SentenceWindowDataset(token_dir, index_dir, "validation")
    train_loader = make_loader(train_data, settings["batch_size"], settings["num_workers"],
        settings["seed"], pin_memory=device.type == "cuda", bucket_multiplier=settings["bucket_multiplier"],
        start_batch=batch_cursor, worker_init_fn=worker_init)
    train_loader.batch_sampler.set_epoch(epoch, batch_cursor)
    val_size = min(len(val_data), settings["batch_size"] * settings["eval_batches"])
    val_indices = random.Random(settings["seed"] + 1_000_019).sample(range(len(val_data)), val_size)
    val_loader = make_loader(val_data, settings["batch_size"], settings["num_workers"],
        settings["seed"], shuffle=False, pin_memory=device.type == "cuda",
        bucket_multiplier=settings["bucket_multiplier"], indices=val_indices, worker_init_fn=worker_init)
    writer = SummaryWriter(str(log_dir), purge_step=step + 1 if resume else None, flush_secs=5)
    writer.add_text("run/config", json.dumps(config, ensure_ascii=False, indent=2), step)
    stop_requested = False
    previous_signal = signal.getsignal(signal.SIGINT)

    def stop_handler(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print("Stop requested; finishing the current update, then saving checkpoint.", flush=True)

    signal.signal(signal.SIGINT, stop_handler)
    iterator = full_loader = None
    start_time = time.perf_counter()
    latest_validation = None

    def save_checkpoint(path):
        atomic_checkpoint(path, {"format": "vimeml_tiny_gpt_v1", "model": model.state_dict(),
            "model_config": model.configuration(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config": config, "signatures": signatures,
            "precision": precision, "step": step, "epoch": epoch, "batch_cursor": batch_cursor,
            "total_tokens": total_tokens, "total_windows": total_windows,
            "best_val": best_val, "rng": rng_state(),
            "benchmark_tokens": benchmark_tokens, "benchmark_steps": benchmark_steps,
            "benchmark_seconds": benchmark_seconds})

    def validation():
        result = evaluate(model, val_loader, device, precision)
        writer.add_scalar("loss/validation", result["loss"], step)
        writer.add_scalar("perplexity/validation", result["perplexity"], step)
        writer.flush()
        print(f"[validation] step={step} loss={result['loss']:.4f} tokens={result['prediction_pairs']:,}", flush=True)
        return result

    interval_loss = interval_tokens = interval_positions = 0
    interval_seconds = interval_data_seconds = 0.0
    try:
        if not resume:
            latest_validation = validation()
            best_val = latest_validation["loss"]
            save_checkpoint(last_path)  # Initial state can resume even before step 1.
            save_checkpoint(output / "best.pt")
        if restored_rng is not None:
            restore_rng(restored_rng)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        print(f"Training {plan['parameters']:,} parameters; {precision}; max_steps={settings['max_steps']}", flush=True)
        while step < settings["max_steps"] and not stop_requested:
            update_start = time.perf_counter()
            microbatches = []
            for _ in range(settings["gradient_accumulation"]):
                if iterator is None:
                    iterator = iter(train_loader)
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    batch_cursor = 0
                    train_loader.batch_sampler.set_epoch(epoch)
                    iterator = iter(train_loader)
                    batch = next(iterator)
                microbatches.append(batch)
                batch_cursor += 1
            data_seconds = time.perf_counter() - update_start
            token_count = sum(int(batch["lengths"].sum()) for batch in microbatches)
            positions = sum(batch["input_ids"].numel() for batch in microbatches)
            rate = learning_rate(step + 1, settings)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            loss_sums = []
            for batch in microbatches:
                inputs = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with amp_context(device, precision):
                    result = model(inputs, labels)
                    loss = result["loss_sum"] / token_count
                loss_sums.append(result["loss_sum"].detach())
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                raise FloatingPointError("FP16 update was skipped; resume last committed checkpoint or choose bf16.")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            loss_sum = float(torch.stack(loss_sums).sum())
            if not math.isfinite(loss_sum):
                raise FloatingPointError("Non-finite training loss.")
            seconds = time.perf_counter() - update_start
            step += 1
            total_tokens += token_count
            total_windows += sum(batch["input_ids"].shape[0] for batch in microbatches)
            interval_loss += loss_sum
            interval_tokens += token_count
            interval_positions += positions
            interval_seconds += seconds
            interval_data_seconds += data_seconds
            if step > settings["benchmark_warmup_steps"]:
                benchmark_tokens += token_count
                benchmark_steps += 1
                benchmark_seconds += seconds
            if step == 1 or step % settings["log_every"] == 0 or step == settings["max_steps"] or stop_requested:
                metrics = {"loss/train": interval_loss / interval_tokens,
                    "optimizer/learning_rate": rate, "optimizer/grad_norm": float(grad_norm),
                    "throughput/effective_tokens_per_second": interval_tokens / interval_seconds,
                    "data/padding_fraction": 1 - interval_tokens / interval_positions,
                    "data/wait_fraction": interval_data_seconds / interval_seconds,
                    "progress/trained_tokens": total_tokens, "progress/percent": 100 * step / settings["max_steps"]}
                if device.type == "cuda":
                    metrics["memory/peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
                    metrics["memory/peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2
                for tag, value in metrics.items():
                    writer.add_scalar(tag, value, step)
                elapsed = time.perf_counter() - start_time
                progress = {"status": "training", "step": step, "max_steps": settings["max_steps"],
                    "epoch": epoch, "batch_cursor": batch_cursor, "total_tokens": total_tokens,
                    "elapsed_seconds_this_session": elapsed, "metrics": metrics}
                write_json(output / "progress.json", progress)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(progress, ensure_ascii=False) + "\n")
                print(f"step {step}/{settings['max_steps']} loss={metrics['loss/train']:.4f} "
                      f"tokens/s={metrics['throughput/effective_tokens_per_second']:,.0f} "
                      f"padding={metrics['data/padding_fraction']:.1%} elapsed={elapsed:.1f}s", flush=True)
                interval_loss = interval_tokens = interval_positions = 0
                interval_seconds = interval_data_seconds = 0.0
            if step % settings["eval_every"] == 0 or step == settings["max_steps"]:
                latest_validation = validation()
                if latest_validation["loss"] < best_val:
                    best_val = latest_validation["loss"]
                    save_checkpoint(output / "best.pt")
            if step % settings["checkpoint_every"] == 0 or step == settings["max_steps"]:
                save_checkpoint(last_path)
        save_checkpoint(last_path)
        status = "complete" if step >= settings["max_steps"] else "interrupted"
        if status == "complete" and settings.get("epochs"):
            expected_tokens = settings["epochs"] * token_manifest["splits"]["train"]["prediction_pairs"]
            expected_windows = settings["epochs"] * index_manifest["splits"]["train"]["windows"]
            if total_tokens != expected_tokens or total_windows != expected_windows:
                raise ValueError("Training did not cover the exact promised windows/prediction pairs.")
        full_validation = None
        if status == "complete" and settings.get("validate_full_at_end", False):
            print("[validation_full] Evaluating all validation windows for last/best checkpoints...", flush=True)
            write_json(output / "progress.json", {"status": "evaluating_full_validation", "step": step,
                       "max_steps": settings["max_steps"], "total_tokens": total_tokens})
            # Release sampled validation workers before creating the full loader.
            val_loader = None
            full_loader = make_loader(val_data, settings["batch_size"], settings["num_workers"],
                settings["seed"], shuffle=False, pin_memory=device.type == "cuda",
                bucket_multiplier=settings["bucket_multiplier"], worker_init_fn=worker_init)
            last_full = evaluate(model, full_loader, device, precision)
            writer.add_scalar("loss/validation_full_last", last_full["loss"], step)
            best_saved = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            best_step = best_saved["step"]
            if best_step == step:
                best_full = dict(last_full)
            else:
                model.load_state_dict(best_saved["model"])
                best_full = evaluate(model, full_loader, device, precision)
            del best_saved
            expected_validation = token_manifest["splits"]["validation"]["prediction_pairs"]
            if any(result["prediction_pairs"] != expected_validation for result in (last_full, best_full)):
                raise ValueError("Full validation did not cover every validation prediction pair.")
            writer.add_scalar("loss/validation_full_best", best_full["loss"], step)
            writer.flush()
            full_validation = {"last": last_full, "best": best_full, "best_checkpoint_step": best_step,
                               "best_selection": "Lowest fixed-subset validation loss during training."}
            write_json(output / "full-validation.json", full_validation)
            print(f"[validation_full] last={last_full['loss']:.4f} best={best_full['loss']:.4f} "
                  f"tokens={expected_validation:,}", flush=True)
        speed = benchmark_tokens / benchmark_seconds if benchmark_seconds else None
        summary = {"status": status, "step": step, "parameters": model.parameter_count(),
            "total_trained_tokens": total_tokens, "best_validation_loss": best_val,
            "total_trained_windows": total_windows,
            "completed_data_passes": total_tokens / token_manifest["splits"]["train"]["prediction_pairs"],
            "last_validation": latest_validation, "benchmark_steps": benchmark_steps,
            "full_validation": full_validation,
            "benchmark_effective_tokens_per_second": speed,
            "benchmark_seconds_excluding_validation_checkpoints_and_first_warmup_steps": benchmark_seconds,
            "estimated_one_epoch_training_hours_excluding_validation_checkpoints":
                token_manifest["splits"]["train"]["prediction_pairs"] / speed / 3600 if speed else None,
            "estimate_note": "Short-run measurement; length mix, GPU power/temperature and validation/checkpoint overhead affect full-run time.",
            "elapsed_seconds_this_session": time.perf_counter() - start_time,
            "checkpoint": str(last_path), "tensorboard": str(log_dir)}
        if device.type == "cuda":
            summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
            summary["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", summary)
        write_json(output / "manifest.json", {"status": status, "format": "vimeml_tiny_gpt_run_v1",
                   "plan": plan, "signatures": signatures, "step": step})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Summary: {output / 'summary.json'}")
        return summary
    except BaseException as error:
        write_json(output / "progress.json", {"status": "failed", "step": step,
                   "error": str(error), "resume": "Resume the last committed last.pt checkpoint."})
        raise
    finally:
        signal.signal(signal.SIGINT, previous_signal)
        writer.close()
        # Persistent iterators are owned by their loaders; delete both owners.
        del iterator, train_loader, val_loader, full_loader
        train_data.close()
        val_data.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train-smoke.toml")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only; no training or output files.")
    parser.add_argument("--resume", action="store_true", help="Resume last.pt with identical configuration.")
    args = parser.parse_args(argv)
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    run(config, resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
