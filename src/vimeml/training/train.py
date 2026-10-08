"""Tiny GPT training: bounded batches, token-weighted loss and checkpoints."""
import argparse
import json
import math
import random
import signal
import subprocess
import sys
import time
import tomllib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from vimeml.training.data import SentenceWindowDataset, file_sha, make_loader, prepare_indexes, write_json
from vimeml.training.model_factory import checkpoint_format, configuration_for, create_model
from vimeml.training.data_v2 import PrefixCropWindowDataset

ROOT = Path(__file__).resolve().parents[3]


def learning_rate(step, settings):
    """One-based update; optional stable phase before the cosine decay."""
    warmup, maximum = settings["warmup_steps"], settings["max_steps"]
    high, low = settings["learning_rate"], settings["min_learning_rate"]
    if warmup and step <= warmup:
        return high * step / warmup
    decay_start = (settings.get("stable_steps", 0)
                   if settings.get("learning_rate_schedule", "cosine") == "stable_decay"
                   else warmup)
    fraction = min(1.0, max(0.0, (step - decay_start) / max(1, maximum - decay_start)))
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
def evaluate(model, loader, device, precision, characters=None):
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
        metrics = {"loss": loss_sum / total_tokens, "nll_sum": loss_sum,
                   "prediction_pairs": total_tokens,
                   "perplexity": math.exp(min(50, loss_sum / total_tokens))}
        if characters is not None:
            if characters < 1:
                raise ValueError("Validation characters must be positive.")
            metrics.update(characters=characters, bpc=loss_sum / (characters * math.log(2)),
                           bpc_policy="All prediction targets including EOS; original Unicode characters excluding BOS/EOS.")
        return metrics
    finally:
        model.train()


def validate_config(config):
    model = configuration_for(config.get("architecture", "tiny_gpt_v1"), config["model"])
    settings = dict(config["training"])
    normalization = settings.get("loss_token_normalization", "batch_tokens")
    if normalization not in {"batch_tokens", "epoch_mean_tokens"} or (
            normalization == "epoch_mean_tokens" and not config.get("data_mixture")):
        raise ValueError("Epoch-mean loss normalization requires a frozen token mixture.")
    crop_probability = settings.get("prefix_crop_probability", 0.0)
    crop_minimum = settings.get("prefix_crop_min_remaining_tokens", 8)
    if (not math.isfinite(crop_probability) or not 0 <= crop_probability <= 1 or
            type(crop_minimum) is not int or crop_minimum < 1):
        raise ValueError("Invalid prefix crop probability or minimum token count.")
    if crop_probability and config.get("architecture", "tiny_gpt_v1") != "tiny_gpt_v2":
        raise ValueError("Prefix crops are only enabled for tiny_gpt_v2.")
    if settings.get("early_stopping_patience", 0) < 0 or settings.get("early_stopping_min_delta", 0) < 0:
        raise ValueError("Invalid epoch degradation stopping configuration.")
    if type(settings.get('data_epoch_offset', 0)) is not int or settings.get('data_epoch_offset', 0) < 0:
        raise ValueError('data_epoch_offset must be a nonnegative integer.')
    if config.get('runtime', {}).get('compile_backbone', False) and config.get('architecture') != 'tiny_gpt_v2':
        raise ValueError('Backbone compilation is only supported for V2.')
    automatic_steps = settings.get("max_steps") == 0 and settings.get("epochs", 0) > 0
    schedule = settings.get("learning_rate_schedule", "cosine")
    stable_steps = settings.get("stable_steps", 0)
    if schedule not in {"cosine", "stable_decay"} or type(stable_steps) is not int:
        raise ValueError("Invalid learning rate schedule.")
    if (stable_steps < 0 or (schedule == "stable_decay" and
            (stable_steps < settings["warmup_steps"] or
             (not automatic_steps and stable_steps >= settings["max_steps"])))):
        raise ValueError("Stable phase must leave a nonempty decay interval after warmup.")
    baseline_bpc = config.get("initialization", {}).get("baseline_validation_bpc")
    if baseline_bpc is not None and (not math.isfinite(baseline_bpc) or baseline_bpc <= 0):
        raise ValueError("Initialization baseline BPC must be finite and positive.")
    if config.get("initialization", {}).get("restore_optimizer", False) and not config["initialization"].get("checkpoint"):
        raise ValueError("Optimizer continuation requires an initialization checkpoint.")
    positive = ("cpu_threads", "batch_size", "bucket_multiplier", "gradient_accumulation",
                "log_every", "eval_every", "eval_batches", "checkpoint_every")
    if any(settings[name] < 1 for name in positive) or settings["num_workers"] < 0 or (not automatic_steps and settings["max_steps"] < 1):
        raise ValueError("Invalid training batch/worker/interval configuration.")
    if model.context_length % 8 or settings["warmup_steps"] < 0 or (not automatic_steps and settings["warmup_steps"] > settings["max_steps"]):
        raise ValueError("Context must be divisible by 8; warmup must fit training steps.")
    if (not 0 < settings["min_learning_rate"] <= settings["learning_rate"] or
            settings["weight_decay"] < 0 or settings["grad_clip"] <= 0 or
            not 0 <= settings["beta1"] < 1 or not 0 <= settings["beta2"] < 1 or
            settings["benchmark_warmup_steps"] < 0 or
            (not automatic_steps and settings["benchmark_warmup_steps"] >= settings["max_steps"])):
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


def tokenizer_identity_from_manifest(manifest):
    """Read vocabulary identity from current or frozen legacy token manifests."""
    digest = manifest.get("tokenizer_model_sha256")
    if digest is None:
        candidates = [value for name, value in manifest.get("input_sha256", {}).items()
                      if name.replace("\\", "/").rsplit("/", 1)[-1] == "tokenizer.model"]
        if len(candidates) != 1:
            raise ValueError("Token manifest must identify exactly one tokenizer model.")
        digest = candidates[0]
    return {"tokenizer_model_sha256": digest,
            "vocab_size": manifest["vocab_size"], "special_ids": manifest["special_ids"]}


def run(config, resume=False, dry_run=False):
    model_config, settings = validate_config(config)
    architecture = config.get("architecture", "tiny_gpt_v1")
    token_dir, index_dir = ROOT / config["token_dir"], ROOT / config["index_dir"]
    output, log_dir = ROOT / config["output_dir"], ROOT / config["log_dir"]
    token_manifest = json.loads((token_dir / "manifest.json").read_text(encoding="utf-8"))
    tokenizer_identity = tokenizer_identity_from_manifest(token_manifest)
    index_manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if (token_manifest["vocab_size"] != model_config.vocab_size or
            index_manifest["context_length"] != model_config.context_length):
        raise ValueError("Model vocabulary/context must match token data/window index.")
    code_names = ["train.py", "model.py", "data.py", "model_factory.py"]
    mixture = None
    epoch_batches = None
    if config.get("data_mixture"):
        if architecture != "tiny_gpt_v2" or settings.get("data_epoch_offset", 0) != 0:
            raise ValueError("Mixture epochs require V2 and data_epoch_offset=0.")
        from vimeml.training.data_mixture import mixture_dataset
        prepared = mixture_dataset(config, ROOT)
        mixture = prepared.manifest
        prepared.close()
        if settings.get("epochs") != len(mixture["epochs"]) or settings["gradient_accumulation"] != 1:
            raise ValueError("Training epochs must match the frozen mixture, with accumulation=1.")
        code_names.append("data_mixture.py")
        epoch_batches = [math.ceil(record["windows"] / settings["batch_size"]) for record in mixture["epochs"]]
    if architecture == "tiny_gpt_v2":
        code_names.extend(("model_v2.py", "data_v2.py"))
        if settings.get("validate_full_each_epoch", False):
            code_names.append("evaluation_v2.py")
        if config.get('runtime', {}).get('compile_backbone', False) or config.get('initialization', {}).get('checkpoint'):
            code_names.append('runtime_v2.py')
    signatures = {"tokens": file_sha(token_dir / "manifest.json"),
                  "windows": file_sha(index_dir / "manifest.json"),
                  "training_code": {name: file_sha(Path(__file__).with_name(name))
                                    for name in code_names}}
    if mixture:
        signatures["mixture"] = file_sha(ROOT / config["data_mixture"]["directory"] / "manifest.json")
    if token_manifest.get("status") != "complete" or index_manifest.get("status") != "complete":
        raise ValueError("Complete token store and window index required.")
    if signatures["tokens"] != index_manifest["token_manifest_sha256"]:
        raise ValueError("Window index belongs to a different token dataset.")
    windows_per_epoch = mixture["epochs"][0]["windows"] if mixture else index_manifest["splits"]["train"]["windows"]
    if settings["max_steps"] == 0:
        settings["max_steps"] = sum(epoch_batches) if mixture else settings["epochs"] * math.ceil(windows_per_epoch / settings["batch_size"])
        _, settings = validate_config({**config, "training": settings})
    if mixture:
        if settings["max_steps"] != sum(epoch_batches):
            raise ValueError("Mixture max_steps must cover the exact frozen epochs.")
        batches_per_epoch = epoch_batches[0]
    else:
        batches_per_epoch = validate_epoch_schedule(settings, windows_per_epoch)
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
    model = create_model(architecture, model_config)
    plan = {"architecture": architecture, "model": model.configuration(), "parameters": model.parameter_count(),
            "device": str(device), "precision": precision, "max_steps": settings["max_steps"],
            "training_prediction_pairs_per_epoch": mixture["epochs"][0]["effective_tokens"] if mixture else token_manifest["splits"]["train"]["prediction_pairs"],
            "data_mixture": mixture,
            "training_batches_per_epoch": batches_per_epoch,
            "planned_epochs": settings.get("epochs"),
            "initialization": config.get('initialization'),
            "runtime": config.get('runtime'),
            "data_epoch_offset": settings.get('data_epoch_offset', 0),
            "prefix_crop": {"probability": settings.get("prefix_crop_probability", 0.0),
                "min_remaining_tokens": settings.get("prefix_crop_min_remaining_tokens", 8),
                "determinism": "seed, zero-based epoch and sample index; epoch-tagged worker inputs"},
            "output": str(output), "tensorboard": str(log_dir),
            "validation": "Fixed random sentence-complete subset plus full validation each epoch; BPC includes EOS; test unused."
                if settings.get('validate_full_each_epoch', False) else "Fixed random subset, token-weighted loss; test unused.",
            "full_validation_each_epoch": settings.get('validate_full_each_epoch', False),
            "corpus_quality_mode": token_manifest.get("corpus_quality_mode"),
            "full_validation_at_end": settings.get("validate_full_at_end", False),
            "purpose": "Continued pretraining with compatible AdamW state." if config.get('initialization', {}).get('restore_optimizer') else
                       "Continued pretraining from frozen weights with a fresh optimizer." if config.get('initialization') else
                       "Full-corpus experiment." if settings.get("epochs") else
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
    prepare_indexes(token_dir, index_dir, model_config.context_length,
                    verification=config.get("verification", {}).get("mode", "sha256"))
    model.to(device)
    optimizer = optimizer_for(model, settings, device)
    initialization = None
    if config.get('initialization', {}).get('checkpoint') and not resume:
        from vimeml.training.runtime_v2 import initialize_weights
        initialization = initialize_weights(model, ROOT / config['initialization']['checkpoint'], signatures,
            optimizer=optimizer if config['initialization'].get('restore_optimizer', False) else None,
            precision=precision, allow_new_data=config['initialization'].get('allow_new_data', False),
            tokenizer_identity=tokenizer_identity)
        print(f"Initializing from step {initialization['source_step']}; {initialization['optimizer']}.", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    step = epoch = batch_cursor = total_tokens = total_windows = 0
    total_dropped_tokens = total_cropped_windows = 0
    total_chat_tokens = 0
    chat_source_ids = [value for name, value in token_manifest["source_ids"].items()
                       if name in ("real-persona-chat", "mrmp-chat")]
    best_val = math.inf
    best_epoch_bpc = config.get('initialization', {}).get('baseline_validation_bpc', math.inf)
    degrading_epochs = 0
    evaluated_epochs = []
    early_stopped = False
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
        total_dropped_tokens = saved.get("total_dropped_tokens", 0)
        total_cropped_windows = saved.get("total_cropped_windows", 0)
        total_chat_tokens = saved.get("total_chat_tokens", 0)
        if mixture:
            batches_per_epoch = epoch_batches[epoch]
        best_epoch_bpc = saved.get("best_epoch_bpc", math.inf)
        degrading_epochs = saved.get("degrading_epochs", 0)
        evaluated_epochs = saved.get("evaluated_epochs", [])
        early_stopped = saved.get("early_stopped", False)
        initialization = saved.get('initialization')
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
    environment['initialization'] = initialization
    if config.get('runtime', {}).get('compile_backbone', False):
        from vimeml.training.runtime_v2 import compile_backbone
        environment['runtime_optimization'] = compile_backbone(model)
    source_provenance = ROOT / config.get('source_provenance', 'source-provenance.json')
    if source_provenance.exists():
        environment['source_provenance'] = json.loads(source_provenance.read_text(encoding='utf-8'))
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
    write_json(output / "environment.json", environment)
    if mixture:
        train_data = mixture_dataset(config, ROOT)
    elif settings.get("prefix_crop_probability", 0.0):
        train_data = PrefixCropWindowDataset(token_dir, index_dir, "train", seed=settings["seed"],
            probability=settings["prefix_crop_probability"],
            min_remaining_tokens=settings.get("prefix_crop_min_remaining_tokens", 8))
    else:
        train_data = SentenceWindowDataset(token_dir, index_dir, "train")
    val_data = SentenceWindowDataset(token_dir, index_dir, "validation")
    train_loader = make_loader(train_data, settings["batch_size"], settings["num_workers"],
        settings["seed"], pin_memory=device.type == "cuda", bucket_multiplier=settings["bucket_multiplier"],
        start_batch=batch_cursor, worker_init_fn=worker_init)
    train_loader.batch_sampler.set_epoch(epoch + settings.get('data_epoch_offset', 0), batch_cursor)
    val_size = min(len(val_data), settings["batch_size"] * settings["eval_batches"])
    val_indices = random.Random(settings["seed"] + 1_000_019).sample(range(len(val_data)), val_size)
    val_characters = None
    if settings.get("validate_full_each_epoch", False):
        from vimeml.training.evaluation_v2 import validation_subset
        val_indices, val_characters = validation_subset(val_data, val_indices, ROOT / config['evaluation']['tokenizer_dir'])
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
    domain_loaders = {}
    start_time = time.perf_counter()
    latest_validation = None

    def save_checkpoint(path):
        atomic_checkpoint(path, {"format": checkpoint_format(architecture), "architecture": architecture,
            "model": model.state_dict(),
            "model_config": model.configuration(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config": config, "signatures": signatures,
            "precision": precision, "step": step, "epoch": epoch, "batch_cursor": batch_cursor,
            "source_provenance": environment.get('source_provenance'),
            "initialization": initialization,
            "tokenizer_identity": tokenizer_identity,
            "total_chat_tokens": total_chat_tokens,
            "total_tokens": total_tokens, "total_windows": total_windows,
            "total_dropped_tokens": total_dropped_tokens, "total_cropped_windows": total_cropped_windows,
            "best_val": best_val, "rng": rng_state(),
            "best_epoch_bpc": best_epoch_bpc, "degrading_epochs": degrading_epochs,
            "evaluated_epochs": evaluated_epochs, "early_stopped": early_stopped,
            "benchmark_tokens": benchmark_tokens, "benchmark_steps": benchmark_steps,
            "benchmark_seconds": benchmark_seconds,
            "scheduler": {"name": settings.get("learning_rate_schedule", "cosine"), "step": step, "max_steps": settings["max_steps"],
                "stable_steps": settings.get("stable_steps", 0),
                "warmup_steps": settings["warmup_steps"], "learning_rate": settings["learning_rate"],
                "min_learning_rate": settings["min_learning_rate"]}})

    def validation():
        result = evaluate(model, val_loader, device, precision, val_characters)
        writer.add_scalar("loss/validation", result["loss"], step)
        writer.add_scalar("perplexity/validation", result["perplexity"], step)
        if 'bpc' in result:
            writer.add_scalar("bpc/validation_subset", result['bpc'], step)
        writer.flush()
        print(f"[validation] step={step} loss={result['loss']:.4f} tokens={result['prediction_pairs']:,}", flush=True)
        return result

    def evaluate_epoch(number):
        nonlocal full_loader, best_epoch_bpc, degrading_epochs, early_stopped
        from vimeml.training.evaluation_v2 import epoch_ime
        print(f"[epoch {number}] Full validation and frozen IME evaluations...", flush=True)
        save_checkpoint(last_path)
        if full_loader is None:
            full_loader = make_loader(val_data, settings["batch_size"], settings["num_workers"],
                settings["seed"], shuffle=False, pin_memory=device.type == "cuda",
                bucket_multiplier=settings["bucket_multiplier"], worker_init_fn=worker_init)
        result = evaluate(model, full_loader, device, precision, token_manifest['splits']['validation']['characters'])
        if result['prediction_pairs'] != token_manifest['splits']['validation']['prediction_pairs']:
            raise ValueError('Incomplete epoch validation coverage.')
        improved = result['bpc'] < best_epoch_bpc
        degrading_epochs = (degrading_epochs + 1 if result['bpc'] > best_epoch_bpc +
                            settings.get('early_stopping_min_delta', .01) else 0)
        if improved:
            best_epoch_bpc = result['bpc']
        ime = epoch_ime(model, ROOT / config['evaluation']['tokenizer_dir'],
                       config['evaluation']['benchmarks'], ROOT, output, number)
        for name, value in result.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f'epoch_validation/{name}', value, step)
        for benchmark, values in ime.items():
            for name, value in values.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'ime/{benchmark}/{name}', value, step)
        writer.add_scalar('progress/completed_epochs', number, step)
        writer.flush()
        patience = settings.get('early_stopping_patience', 0)
        early_stopped = bool(patience and degrading_epochs >= patience)
        domains = evaluate_domains()
        report = {'epoch': number, 'step': step, 'validation': result, 'domains': domains, 'ime': ime,
                  'best_epoch_bpc': best_epoch_bpc, 'degrading_epochs': degrading_epochs,
                  'early_stopped': early_stopped}
        epoch_output = output / 'epoch-evaluation' / f'epoch-{number}'
        epoch_output.mkdir(parents=True, exist_ok=True)
        write_json(epoch_output / 'summary.json', report)
        evaluated_epochs.append(number)
        save_checkpoint(output / f'epoch-{number}.pt')
        if improved:
            save_checkpoint(output / 'best-epoch.pt')
        save_checkpoint(last_path)
        print(f"[epoch {number}] BPC={result['bpc']:.6f} IME={ime}", flush=True)
        return result

    def evaluate_domains():
        reports = {}
        for name, paths in config.get('evaluation', {}).get('domains', {}).items():
            if name not in domain_loaders:
                data = SentenceWindowDataset(ROOT / paths['token_dir'], ROOT / paths['index_dir'], 'validation')
                loader = make_loader(data, settings['batch_size'], settings['num_workers'], settings['seed'],
                    shuffle=False, pin_memory=device.type == 'cuda', bucket_multiplier=settings['bucket_multiplier'], worker_init_fn=worker_init)
                metadata = json.loads((ROOT / paths['token_dir'] / 'manifest.json').read_text(encoding='utf-8'))['splits']['validation']
                domain_loaders[name] = (data, loader, metadata)
            data, loader, metadata = domain_loaders[name]
            reports[name] = evaluate(model, loader, device, precision, metadata['characters'])
            if reports[name]['prediction_pairs'] != metadata['prediction_pairs']:
                raise ValueError(f'Incomplete {name} domain validation')
            for key, value in reports[name].items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'domain/{name}/{key}', value, step)
        writer.flush()
        return reports

    interval_loss = interval_tokens = interval_positions = 0
    interval_windows = interval_dropped_tokens = interval_cropped_windows = 0
    interval_chat_tokens = 0
    interval_seconds = interval_data_seconds = 0.0
    try:
        if not resume:
            latest_validation = validation()
            if config.get('evaluation', {}).get('domains'):
                write_json(output / 'initial-domain-validation.json', evaluate_domains())
            best_val = latest_validation["loss"]
            save_checkpoint(last_path)  # Initial state can resume even before step 1.
            save_checkpoint(output / "best.pt")
            if math.isfinite(best_epoch_bpc):
                save_checkpoint(output / "best-epoch.pt")
        if restored_rng is not None:
            restore_rng(restored_rng)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        print(f"Training {plan['parameters']:,} parameters; {precision}; max_steps={settings['max_steps']}", flush=True)
        if settings.get('validate_full_each_epoch', False) and batch_cursor == batches_per_epoch:
            if epoch + 1 not in evaluated_epochs:
                evaluate_epoch(epoch + 1)
        while step < settings["max_steps"] and not stop_requested and not early_stopped:
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
                    if mixture:
                        batches_per_epoch = epoch_batches[epoch]
                    train_loader.batch_sampler.set_epoch(epoch + settings.get('data_epoch_offset', 0))
                    iterator = iter(train_loader)
                    batch = next(iterator)
                microbatches.append(batch)
                batch_cursor += 1
            data_seconds = time.perf_counter() - update_start
            token_count = sum(int(batch["lengths"].sum()) for batch in microbatches)
            chat_tokens = sum(int(batch['lengths'][torch.isin(batch['source_id'], torch.tensor(chat_source_ids, dtype=torch.int64))].sum())
                              for batch in microbatches) if chat_source_ids else 0
            dropped_tokens = sum(int(batch["crop_offset"].sum()) for batch in microbatches if "crop_offset" in batch)
            cropped_windows = sum(int((batch["crop_offset"] > 0).sum()) for batch in microbatches if "crop_offset" in batch)
            window_count = sum(batch["input_ids"].shape[0] for batch in microbatches)
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
                    denominator = (mixture["epochs"][epoch]["effective_tokens"] / batches_per_epoch
                                   if settings.get("loss_token_normalization") == "epoch_mean_tokens" else token_count)
                    loss = result["loss_sum"] / denominator
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
            total_chat_tokens += chat_tokens
            interval_chat_tokens += chat_tokens
            total_windows += window_count
            total_dropped_tokens += dropped_tokens
            total_cropped_windows += cropped_windows
            interval_loss += loss_sum
            interval_tokens += token_count
            interval_positions += positions
            interval_windows += window_count
            interval_dropped_tokens += dropped_tokens
            interval_cropped_windows += cropped_windows
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
                    "throughput/samples_per_second": interval_windows / interval_seconds,
                    "data/non_padding_tokens_per_update": token_count,
                    "data/prefix_crop_fraction": interval_cropped_windows / interval_windows,
                    "data/prefix_crop_dropped_tokens": interval_dropped_tokens,
                    "data/padding_fraction": 1 - interval_tokens / interval_positions,
                    "data/wait_fraction": interval_data_seconds / interval_seconds,
                    "timing/update_seconds": seconds,
                    "data/chat_token_fraction": interval_chat_tokens / interval_tokens,
                    "data/chat_token_fraction_cumulative": total_chat_tokens / total_tokens,
                    "progress/epoch_fraction": epoch + batch_cursor / batches_per_epoch,
                    "progress/trained_tokens": total_tokens, "progress/percent": 100 * step / settings["max_steps"]}
                if device.type == "cuda":
                    metrics["memory/peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
                    metrics["memory/peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2
                for tag, value in metrics.items():
                    writer.add_scalar(tag, value, step)
                elapsed = time.perf_counter() - start_time
                progress = {"status": "training", "step": step, "max_steps": settings["max_steps"],
                    "epoch": epoch, "batch_cursor": batch_cursor, "total_tokens": total_tokens,
                    "total_chat_tokens": total_chat_tokens,
                    "total_dropped_tokens": total_dropped_tokens, "total_cropped_windows": total_cropped_windows,
                    "elapsed_seconds_this_session": elapsed, "metrics": metrics}
                write_json(output / "progress.json", progress)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(progress, ensure_ascii=False) + "\n")
                print(f"step {step}/{settings['max_steps']} loss={metrics['loss/train']:.4f} "
                      f"tokens/s={metrics['throughput/effective_tokens_per_second']:,.0f} "
                      f"padding={metrics['data/padding_fraction']:.1%} elapsed={elapsed:.1f}s", flush=True)
                interval_loss = interval_tokens = interval_positions = 0
                interval_windows = interval_dropped_tokens = interval_cropped_windows = 0
                interval_chat_tokens = 0
                interval_seconds = interval_data_seconds = 0.0
            if step % settings["eval_every"] == 0 or step == settings["max_steps"]:
                latest_validation = validation()
                if latest_validation["loss"] < best_val:
                    best_val = latest_validation["loss"]
                    save_checkpoint(output / "best.pt")
            if step % settings["checkpoint_every"] == 0 or step == settings["max_steps"]:
                save_checkpoint(last_path)
            if settings.get('validate_full_each_epoch', False) and batch_cursor == batches_per_epoch:
                completed_epoch = epoch + 1
                if completed_epoch not in evaluated_epochs:
                    evaluate_epoch(completed_epoch)
        save_checkpoint(last_path)
        status = "early_stopped" if early_stopped else ("complete" if step >= settings["max_steps"] else "interrupted")
        if status == "complete" and settings.get("epochs"):
            expected_tokens = sum(r["uncropped_tokens"] for r in mixture["epochs"]) if mixture else settings["epochs"] * token_manifest["splits"]["train"]["prediction_pairs"]
            expected_windows = sum(r["windows"] for r in mixture["epochs"]) if mixture else settings["epochs"] * index_manifest["splits"]["train"]["windows"]
            if total_tokens + total_dropped_tokens != expected_tokens or total_windows != expected_windows:
                raise ValueError("Training did not cover the exact promised windows/prediction pairs.")
            if mixture and (total_chat_tokens != sum(r["chat_tokens"] for r in mixture["epochs"]) or
                            total_tokens != sum(r["effective_tokens"] for r in mixture["epochs"])):
                raise ValueError("Effective token coverage/chat quota differs from the frozen mixture.")
        full_validation = None
        if status in {'complete', 'early_stopped'} and settings.get("validate_full_at_end", False):
            print("[validation_full] Evaluating all validation windows for last/best checkpoints...", flush=True)
            write_json(output / "progress.json", {"status": "evaluating_full_validation", "step": step,
                       "max_steps": settings["max_steps"], "total_tokens": total_tokens})
            # Release sampled validation workers before creating the full loader.
            val_loader = None
            if full_loader is None:
                full_loader = make_loader(val_data, settings["batch_size"], settings["num_workers"],
                    settings["seed"], shuffle=False, pin_memory=device.type == "cuda",
                    bucket_multiplier=settings["bucket_multiplier"], worker_init_fn=worker_init)
            val_char_count = token_manifest['splits']['validation'].get('characters')
            epoch_report = output / 'epoch-evaluation' / f'epoch-{epoch + 1}' / 'summary.json'
            if settings.get('validate_full_each_epoch', False) and epoch_report.exists():
                cached_epoch = json.loads(epoch_report.read_text(encoding='utf-8'))
                last_full = cached_epoch['validation'] if cached_epoch['step'] == step else None
            else:
                last_full = None
            if last_full is None:
                last_full = evaluate(model, full_loader, device, precision, val_char_count)
            writer.add_scalar("loss/validation_full_last", last_full["loss"], step)
            best_saved = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            best_step = best_saved["step"]
            if best_step == step:
                best_full = dict(last_full)
            else:
                model.load_state_dict(best_saved["model"])
                best_full = evaluate(model, full_loader, device, precision, val_char_count)
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
            "total_chat_tokens": total_chat_tokens,
            "chat_token_fraction": total_chat_tokens / total_tokens if total_tokens else None,
            "completed_data_passes": epoch + batch_cursor / batches_per_epoch,
            "effective_token_passes": total_tokens / (mixture["epochs"][0]["effective_tokens"]
                if mixture else token_manifest["splits"]["train"]["prediction_pairs"]),
            "effective_token_passes_basis": "First frozen cropped mixture epoch" if mixture else "Uncropped token-store train split",
            "prefix_crop_dropped_tokens": total_dropped_tokens,
            "prefix_cropped_windows": total_cropped_windows,
            "uncropped_prediction_pairs_seen": total_tokens + total_dropped_tokens,
            "last_validation": latest_validation, "benchmark_steps": benchmark_steps,
            "full_validation": full_validation,
            "evaluated_epochs": evaluated_epochs, "best_epoch_bpc": best_epoch_bpc if math.isfinite(best_epoch_bpc) else None,
            "early_stopped": early_stopped,
            "benchmark_effective_tokens_per_second": speed,
            "benchmark_seconds_excluding_validation_checkpoints_and_first_warmup_steps": benchmark_seconds,
            "estimated_one_epoch_training_hours_excluding_validation_checkpoints":
                plan['training_prediction_pairs_per_epoch'] / speed / 3600 if speed else None,
            "estimate_note": "Short-run measurement; length mix, GPU power/temperature and validation/checkpoint overhead affect full-run time.",
            "elapsed_seconds_this_session": time.perf_counter() - start_time,
            "checkpoint": str(last_path), "tensorboard": str(log_dir)}
        if device.type == "cuda":
            summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
            summary["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", summary)
        write_json(output / "manifest.json", {"status": status,
                   "format": "vimeml_tiny_gpt_run_v2" if architecture == "tiny_gpt_v2" else "vimeml_tiny_gpt_run_v1",
                   "architecture": architecture,
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
        # Release iterators and loaders before closing the memory maps.
        iterator = train_loader = val_loader = full_loader = None
        train_data.close()
        val_data.close()
        for data, loader, _ in domain_loaders.values():
            del loader
            data.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train-smoke.toml")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only; no training or output files.")
    parser.add_argument("--resume", action="store_true", help="Resume last.pt with identical configuration.")
    args = parser.parse_args(argv)
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    tracking = config.get("tracking", {})
    if tracking.get("enabled", False):
        command = [sys.executable, "-X", "utf8", "-u",
                   str(ROOT / "scripts/training/train_wandb.py"), "--config", str(args.config.resolve())]
        if args.resume:
            command.append("--resume")
        if args.dry_run:
            command.append("--dry-run")
        if tracking.get("mode", "online") == "offline":
            command.append("--offline")
        subprocess.run(command, check=True, cwd=ROOT)
        return
    run(config, resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
