"""Dedicated V4-C1 full-parameter conversion + LM replay training."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time

import torch

from vimeml.posttraining.data import ConversionDataset, ConversionCodec, TOKENIZER_SHA
from vimeml.posttraining.infer import read_snapshot, fp32_evaluation, evaluate_snapshot
from vimeml.posttraining.runtime import (STAGE, TASK_CONTRACT, initialize_v4, update,
    initial_state, apply_epoch_result, restore_checkpoint)
from vimeml.posttraining.streams import ConversionWindows, CommittedStream, loader_for
from vimeml.training.data import SentenceWindowDataset, file_sha, write_json
from vimeml.training.data_v2 import PrefixCropWindowDataset
from vimeml.training.model_v4 import TinyGPTV4, GPTV4Config
from vimeml.training.train import (optimizer_for, learning_rate, atomic_checkpoint,
                                  rng_state, restore_rng, resolve_precision)

ROOT = Path(__file__).resolve().parents[3]
MODEL_FORMAT = "vimeml_tiny_gpt_v4"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_signature(root=ROOT):
    paths = sorted((root / "src/vimeml/posttraining").glob("*.py"))
    paths += [root / f"src/vimeml/training/{name}.py" for name in (
        "train", "data", "data_v2", "model_v2", "model_v4", "runtime_v2", "model_factory", "model_factory_v4")]
    paths += [root / "src/vimeml/tokenizer/store.py"]
    paths += [root / "scripts/posttraining/train.py"]
    hashes = {p.relative_to(root).as_posix(): file_sha(p) for p in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def build_plan(config, root=ROOT, verify_files=False):
    """CPU-only identity/config preflight; no model, optimizer, output or W&B."""
    root = Path(root)
    if config.get("architecture") != "tiny_gpt_v4" or config.get("stage") != STAGE:
        raise ValueError("Use the dedicated V4-C1 configuration")
    model = GPTV4Config(**config["model"])
    settings = dict(config["training"])
    if settings.get("lm_retention_policy", "enforce") not in ("enforce", "monitor"):
        raise ValueError("Unknown LM retention policy")
    positive_ints = ("epochs", "conversion_batch", "lm_batch", "conversion_microbatch", "lm_microbatch",
                     "bucket_multiplier", "cpu_threads", "log_every", "eval_every", "checkpoint_every",
                     "validation_batch", "validation_subset", "early_stopping_patience", "lm_degradation_patience")
    if any(type(settings.get(k)) is not int or settings[k] < 1 for k in positive_ints):
        raise ValueError("Task sizes, intervals, epochs and patience must be positive integers")
    if type(settings["num_workers"]) is not int or settings["num_workers"] < 0:
        raise ValueError("Invalid worker count")
    if type(settings.get("validation_workers", 0)) is not int or settings.get("validation_workers", 0) < 0:
        raise ValueError("Invalid validation worker count")
    if type(settings["seed"]) is not int or not 0 <= settings["seed"] < 2**63:
        raise ValueError("Seed must be a nonnegative int64")
    scalars = ("conversion_weight", "lm_weight", "learning_rate", "min_learning_rate", "warmup_ratio",
               "weight_decay", "grad_clip", "beta1", "beta2", "early_stopping_relative_delta",
               "baseline_lm_bpc", "lm_relative_degradation", "prefix_crop_probability")
    if any(not math.isfinite(settings[k]) for k in scalars):
        raise ValueError("Training scalars must be finite")
    if (not 0 < settings["conversion_weight"] < 1 or not 0 < settings["lm_weight"] < 1
            or abs(settings["conversion_weight"] + settings["lm_weight"] - 1) > 1e-9
            or not 0 < settings["min_learning_rate"] <= settings["learning_rate"]
            or not 0 <= settings["warmup_ratio"] < 1 or settings["weight_decay"] < 0
            or settings["grad_clip"] <= 0 or not 0 <= settings["beta1"] < 1 or not 0 <= settings["beta2"] < 1
            or not 0 <= settings["early_stopping_relative_delta"] < 1
            or settings["baseline_lm_bpc"] <= 0 or not 0 <= settings["lm_relative_degradation"] < 1
            or not 0 <= settings["prefix_crop_probability"] <= 1):
        raise ValueError("Invalid loss, schedule, optimizer or stopping settings")
    if settings["precision"] not in ("bf16", "fp32") or settings["device"] not in ("cuda", "cpu"):
        raise ValueError("C1 supports BF16 CUDA or explicit FP32; no implicit precision fallback")
    if settings["device"] == "cpu" and settings["precision"] != "fp32":
        raise ValueError("CPU requires FP32")
    conversion_dir = root / config["conversion_dir"]
    token_dir, index_dir = root / config["token_dir"], root / config["index_dir"]
    tokenizer = root / config["tokenizer_dir"] / "tokenizer.model"
    files = {"conversion": conversion_dir / "manifest.json", "acceptance": conversion_dir / "acceptance.json",
             "tokens": token_dir / "manifest.json", "windows": index_dir / "manifest.json", "tokenizer": tokenizer}
    signatures = {name: file_sha(path) for name, path in files.items()}
    if signatures != config["identities"] or signatures["tokenizer"] != TOKENIZER_SHA:
        raise ValueError("Frozen data/tokenizer manifest identity mismatch")
    conversion, acceptance = read_json(files["conversion"]), read_json(files["acceptance"])
    tokens, windows = read_json(files["tokens"]), read_json(files["windows"])
    if (conversion.get("status") != "complete" or conversion.get("format") != "vimeml_conversion_v1"
            or acceptance.get("status") != "pass" or acceptance["manifest_sha256"] != signatures["conversion"]
            or tokens.get("status") != "complete" or windows.get("status") != "complete"
            or tokens["tokenizer_model_sha256"] != TOKENIZER_SHA or tokens["vocab_size"] != 8192
            or windows["context_length"] != 128 or windows["token_manifest_sha256"] != signatures["tokens"]
            or conversion["contract"]["tokenizer_sha256"] != TOKENIZER_SHA):
        raise ValueError("Incomplete/incompatible conversion or LM data")
    # Verify exactly the binary files consumed by the GPU trainer. Audit JSONL,
    # raw TXT and dictionary binaries do not need uploading or rescanning.
    required = []
    for split in ("train", "validation"):
        count = conversion["splits"][split]["examples"]
        stored = conversion["splits"][split]["stored_tokens"]
        for suffix, size in (("tokens.bin", stored * 2), ("offsets.bin", (count + 1) * 8),
                             ("out-start.bin", count), ("reading-dropout.bin", count)):
            name = f"{split}.{suffix}"
            path = conversion_dir / name
            if path.stat().st_size != size:
                raise ValueError(f"Conversion file size mismatch: {name}")
            required.append((path, conversion["output_sha256"][name]))
        for suffix in ("tokens.bin", "offsets.bin", "sources.bin"):
            name = f"{split}.{suffix}"
            path = token_dir / name
            if path.stat().st_size != tokens["splits"][split]["file_bytes"][suffix]:
                raise ValueError(f"LM file size mismatch: {name}")
            required.append((path, tokens["output_sha256"][name]))
        path = index_dir / f"{split}.windows.npz"
        if path.stat().st_size != windows["splits"][split]["index_bytes"]:
            raise ValueError("Window index size mismatch")
        required.append((path, windows["splits"][split]["index_sha256"]))
    if verify_files:
        for path, expected in required:
            if file_sha(path) != expected:
                raise ValueError(f"Training binary changed: {path.name}")
    rows = read_snapshot("standard-development", root)
    from vimeml.posttraining.infer import TRACKS
    signatures["standard-development"] = TRACKS["standard-development"][1]
    signatures["code"], hashes = code_signature(root)
    count = conversion["splits"]["train"]["examples"]
    settings["updates_per_epoch"] = math.ceil(count / settings["conversion_batch"])
    settings["max_steps"] = settings["epochs"] * settings["updates_per_epoch"]
    settings["warmup_steps"] = max(1, math.ceil(settings["max_steps"] * settings["warmup_ratio"])) if settings["warmup_ratio"] else 0
    for name in ("conversion", "lm"):
        if settings[f"{name}_microbatch"] > settings[f"{name}_batch"]:
            raise ValueError("Microbatch exceeds its logical task batch")
    source = root / config["initialization"]["checkpoint"]
    if not source.is_file():
        raise ValueError("Frozen V4-B initialization checkpoint is missing")
    if verify_files and file_sha(source) != config["initialization"]["sha256"]:
        raise ValueError("Source checkpoint identity mismatch")
    if not isinstance(config.get("runtime", {}).get("compile_backbone"), bool):
        raise ValueError("Explicit compile_backbone boolean required")
    if config["runtime"]["compile_backbone"] and settings["device"] != "cuda":
        raise ValueError("Compilation setting is for the CUDA training environment")
    return {"stage": STAGE, "parameters": 15_070_848, "settings": settings,
            "signatures": signatures, "code_sha256": hashes, "task_contract": TASK_CONTRACT,
            "conversion_examples": count, "validation_examples": conversion["splits"]["validation"]["examples"],
            "lm_validation_targets": tokens["splits"]["validation"]["prediction_pairs"],
            "lm_validation_characters": tokens["splits"]["validation"]["characters"],
            "standard_cases": len(rows), "required_files": [str(p.relative_to(root)) for p, _ in required],
            "verification": "full consumed binary SHA256" if verify_files else "pinned manifests plus binary sizes; no model/CUDA/training"}


def validate(model, loader, device, expected_tokens=None, characters=None):
    total, nll, inputs, examples = 0, 0.0, 0, 0
    # Caller owns fp32_evaluation so one epoch shares that context with IME.
    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        result = model(batch["input_ids"].to(device, non_blocking=True), labels)
        count = int((batch["labels"] != -100).sum())
        if int(result["token_count"]) != count:
            raise ValueError("Model/dataset supervised token counts disagree")
        total += count
        nll += float(result["loss_sum"])
        inputs += int(batch["lengths"].sum())
        examples += len(batch["lengths"])
    if not total or not math.isfinite(nll) or expected_tokens is not None and total != expected_tokens:
        raise ValueError("Empty/non-finite/incomplete full validation")
    report = {"loss": nll / total, "nll_sum": nll, "supervised_tokens": total,
              "input_tokens": inputs, "examples": examples, "precision": "fp32"}
    if characters is not None:
        report.update(characters=characters, bpc=nll / (characters * math.log(2)))
    return report


def checkpoint_payload(model, optimizer, state, config, plan, precision, initialization):
    return {"format": MODEL_FORMAT, "architecture": "tiny_gpt_v4", "model": model.state_dict(),
            "model_config": model.configuration(), "optimizer": optimizer.state_dict(),
            "config": config, "signatures": plan["signatures"], "precision": precision,
            "step": state["step"], "epoch": state["epoch"], "training_state": copy.deepcopy(state),
            "tokenizer_identity": {"tokenizer_model_sha256": TOKENIZER_SHA, "vocab_size": 8192,
                                   "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3}},
            "posttraining": {"stage": STAGE, "task_contract": TASK_CONTRACT},
            "initialization": initialization, "rng": rng_state(),
            "scheduler": {"step": state["step"], "max_steps": plan["settings"]["max_steps"],
                          "warmup_steps": plan["settings"]["warmup_steps"], "name": "cosine"}}


def progress(state, plan, status="running", latest=None):
    done = state["epoch"] + state["conversion_cursor"] / plan["settings"]["updates_per_epoch"]
    seconds = state["training_seconds"]
    rate = state["step"] / seconds if seconds else None
    return {"status": status, "step": state["step"], "max_steps": plan["settings"]["max_steps"],
            "completed_conversion_epochs": done, "conversion_cursor": state["conversion_cursor"],
            "lm_cursor": state["lm_cursor"], "totals": state["totals"],
            "plateau_count": state["plateau_count"], "forgetting_count": state["forgetting_count"],
            "stop_reason": state["stop_reason"], "training_seconds": seconds,
            "training_only_eta_seconds": (plan["settings"]["max_steps"] - state["step"]) / rate if rate else None,
            "latest_update": latest, "full_validation": state["epoch_results"][-1] if state["epoch_results"] else None}


def terminal_report(state, plan, output, status):
    report = progress(state, plan, status)
    report.update(best_conversion_loss=state["best_conversion"], best_lm_bpc=state["best_lm"],
        best_rerank_key=state["best_rerank_key"], checkpoints={name: str(output / f"{name}.pt")
        for name in ("best", "best-rerank", "best-conversion", "best-lm", "last") if (output / f"{name}.pt").exists()})
    return report


def validate_resume_state(state, plan):
    settings = plan["settings"]
    e, c, step = state["epoch"], state["conversion_cursor"], state["step"]
    if (any(type(x) is not int or x < 0 for x in (e, c, step)) or e > settings["epochs"]
            or c > settings["updates_per_epoch"] or step != e * settings["updates_per_epoch"] + c
            or step > settings["max_steps"]
            or state["totals"]["conversion"]["examples"] != e * plan["conversion_examples"] + min(c * settings["conversion_batch"], plan["conversion_examples"])
            or state["totals"]["lm"]["examples"] != step * settings["lm_batch"]):
        raise ValueError("Checkpoint counters are not a committed two-stream update")
    results = state["epoch_results"]
    expected_epochs = list(range(1, e + 1))
    if c == settings["updates_per_epoch"] and len(results) == e + 1:
        expected_epochs.append(e + 1)
    if [x["epoch"] for x in results] != expected_epochs:
        raise ValueError("Checkpoint has inconsistent pending epoch validation")


def run(config, resume=False, root=ROOT):
    from torch.utils.tensorboard import SummaryWriter
    root = Path(root)
    output, log_dir = root / config["output_dir"], root / config["log_dir"]
    last = output / "last.pt"
    if not resume and (output.exists() and any(output.iterdir()) or log_dir.exists() and any(log_dir.iterdir())):
        raise ValueError("Fresh C1 run requires empty output and TensorBoard directories")
    if resume and not last.is_file():
        raise ValueError("No C1 last.pt to resume")
    print("Verifying frozen C1/LM/source identities...", flush=True)
    plan = build_plan(config, root, verify_files=True)
    settings = plan["settings"]
    torch.set_num_threads(settings["cpu_threads"])
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    device = torch.device(settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA configuration requires a GPU instance")
    precision = resolve_precision(device, settings["precision"])
    torch.set_float32_matmul_precision("high")
    model = TinyGPTV4(GPTV4Config(**config["model"]))
    identity = {"tokenizer_model_sha256": TOKENIZER_SHA, "vocab_size": 8192,
                "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3}}
    # Load source weights before moving model / creating the fresh optimizer.
    initialization = None
    if not resume:
        initialization = initialize_v4(model, root / config["initialization"]["checkpoint"],
            config["initialization"]["sha256"], config["initialization"]["step"], identity,
            plan["signatures"]["tokens"], plan["signatures"]["windows"])
    model.to(device)
    optimizer = optimizer_for(model, settings, device)
    state, restored = initial_state(), None
    if resume:
        saved = torch.load(last, map_location="cpu", weights_only=True)
        state, restored, initialization = restore_checkpoint(saved, model, optimizer, config, plan["signatures"], precision)
        validate_resume_state(state, plan)
        if (output / "failure.json").exists():
            old_failure = read_json(output / "failure.json")
            write_json(output / "failure.json", {**old_failure, "active": False,
                "recovered_by_resume_from_step": state["step"]})
        if state["stop_reason"] or state["epoch"] >= settings["epochs"]:
            print("C1 run is already complete/early-stopped; no updates started", flush=True)
            report = terminal_report(state, plan, output, "early_stopped" if state["stop_reason"] else "complete")
            write_json(output / "summary.json", report)
            write_json(output / "progress.json", report)
            return report
        del saved
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "plan.json", plan)
    environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "precision": precision,
                   "device": str(device), "initialization": initialization, "validation_precision": "primitive_FP32"}
    if config["runtime"]["compile_backbone"]:
        from vimeml.training.runtime_v2 import compile_backbone
        environment["compilation"] = compile_backbone(model)
        environment["compilation"]["checkpoint_inference"] = "Plain TinyGPTV4"
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
    write_json(output / "environment.json", environment)
    conv_train = ConversionWindows(ConversionDataset(root / config["conversion_dir"], "train"))
    conv_val = ConversionWindows(ConversionDataset(root / config["conversion_dir"], "validation"))
    lm_train = PrefixCropWindowDataset(root / config["token_dir"], root / config["index_dir"], "train",
        seed=settings["seed"] + 17, probability=settings["prefix_crop_probability"], min_remaining_tokens=8)
    lm_val = SentenceWindowDataset(root / config["token_dir"], root / config["index_dir"], "validation")
    kwargs = {"workers": settings["num_workers"], "multiplier": settings["bucket_multiplier"], "pin_memory": device.type == "cuda"}
    conv_loader = loader_for(conv_train, settings["conversion_batch"], seed=settings["seed"], **kwargs)
    lm_loader = loader_for(lm_train, settings["lm_batch"], seed=settings["seed"] + 17, **kwargs)
    val_kwargs = {**kwargs, "workers": settings.get("validation_workers", 0)}
    conv_validation = loader_for(conv_val, settings["validation_batch"], seed=settings["seed"], shuffle=False, **val_kwargs)
    lm_validation = loader_for(lm_val, settings["validation_batch"], seed=settings["seed"], shuffle=False, **val_kwargs)
    subset_indices = sorted(random.Random(settings["seed"] + 1000019).sample(range(len(conv_val)), min(len(conv_val), settings["validation_subset"])))
    subset_loader = loader_for(conv_val, settings["validation_batch"], seed=settings["seed"], indices=subset_indices, shuffle=False, **val_kwargs)
    lm_stream = CommittedStream(lm_loader, state["lm_cursor"])
    conv_stream = CommittedStream(conv_loader, {"epoch": state["epoch"], "batch": state["conversion_cursor"], "offset": 0})
    codec = ConversionCodec(root / config["tokenizer_dir"] / "tokenizer.model")
    standard = read_snapshot("standard-development", root)
    writer = SummaryWriter(str(log_dir), purge_step=state["step"] + 1 if resume else None, flush_secs=5)
    requested = False
    old_handlers = {}
    def request_stop(signum, frame):
        nonlocal requested
        if requested:
            raise KeyboardInterrupt("Second interrupt; preserve last committed checkpoint")
        requested = True
        print("Stopping after current committed update/validation, then saving C1 cursors", flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_stop)
    def save(name="last"):
        atomic_checkpoint(output / f"{name}.pt", checkpoint_payload(model, optimizer, state, config, plan, precision, initialization))
    def evaluate_conversion(loader, expected=None):
        with fp32_evaluation(model):
            return validate(model, loader, device, expected)
    metrics_file = (output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
    if restored is not None:
        restore_rng(restored)
    write_json(output / "progress.json", progress(state, plan, "initializing" if not resume else "running"))
    terminal_status = None
    try:
        if not resume:
            initial = evaluate_conversion(conv_validation, conv_val.data.manifest["splits"]["validation"]["supervised_tokens_full_reading"])
            state["conversion_reference"] = state["best_conversion"] = initial["loss"]
            write_json(output / "initial-validation.json", {"conversion": initial, "lm_bpc": settings["baseline_lm_bpc"],
                "lm_bpc_origin": "reused full FP32 result of SHA256-pinned V4-B source; no second pretraining run"})
            save()
            save("best-conversion")
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        while True:
            if state["conversion_cursor"] == settings["updates_per_epoch"]:
                number = state["epoch"] + 1
                if not any(r["epoch"] == number for r in state["epoch_results"]):
                    # A crash in any validation restarts this pending epoch from
                    # the checkpoint before counters/selection were changed.
                    save()
                    write_json(output / "progress.json", progress(state, plan, "validating"))
                    epoch_output = output / "epoch-evaluation" / f"epoch-{number}"
                    epoch_output.mkdir(parents=True, exist_ok=True)
                    print(f"[epoch {number}] full conversion, full LM FP32 and frozen Standard", flush=True)
                    before_eval = rng_state()
                    with fp32_evaluation(model):
                        cv = validate(model, conv_validation, device,
                            conv_val.data.manifest["splits"]["validation"]["supervised_tokens_full_reading"])
                        lv = validate(model, lm_validation, device, plan["lm_validation_targets"], plan["lm_validation_characters"])
                        ime = evaluate_snapshot(model, codec, standard, epoch_output / "standard-development")
                    restore_rng(before_eval)
                    roles, qualified = apply_epoch_result(state, cv["loss"], lv["bpc"], ime["correct"], settings)
                    report = {"epoch": number, "step": state["step"], "conversion": cv, "lm": lv, "ime": ime,
                              "lm_retention_qualified": qualified, "stop_reason": state["stop_reason"],
                              "plateau_count": state["plateau_count"], "forgetting_count": state["forgetting_count"]}
                    state["epoch_results"].append(report)
                    write_json(epoch_output / "summary.json", report)
                    write_json(output / "full-validation.json", report)
                    for tag, value in (("conversion/full_validation_loss", cv["loss"]), ("lm/full_validation_bpc", lv["bpc"]),
                                       ("ime/standard_correct", ime["correct"]), ("ime/fallbacks", ime["fallbacks"]),
                                       ("stopping/plateau_count", state["plateau_count"]), ("stopping/forgetting_count", state["forgetting_count"])):
                        writer.add_scalar(tag, value, state["step"])
                    for role in roles:
                        save(role)
                    save(f"epoch-{number}")
                    save()
                    print(f"[epoch {number}] conversion={cv['loss']:.6f} LM BPC={lv['bpc']:.6f} IME={ime['correct']}/{ime['total']}", flush=True)
                state["epoch"] += 1
                state["conversion_cursor"] = 0
                save()
                conv_stream = CommittedStream(conv_loader, {"epoch": state["epoch"], "batch": 0, "offset": 0})
            if state["stop_reason"] or state["step"] >= settings["max_steps"] or requested:
                break
            began = time.perf_counter()
            conv_batch = conv_stream.next_batch()
            replay = lm_stream.next_group(settings["lm_batch"])
            rate = learning_rate(state["step"] + 1, settings)
            for group in optimizer.param_groups:
                group["lr"] = rate
            result = update(model, optimizer, [conv_batch], replay, settings, device, precision)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            # Commit BOTH task cursors only after successful AdamW completion.
            state["step"] += 1
            state["conversion_cursor"] += 1
            state["lm_cursor"] = lm_stream.state_dict()
            seconds = time.perf_counter() - began
            state["training_seconds"] += seconds
            for task in ("conversion", "lm"):
                for field in state["totals"][task]:
                    state["totals"][task][field] += result[field][task]
            if state["step"] % settings["log_every"] == 0:
                latest = {"step": state["step"], "learning_rate": rate, "seconds": seconds, **result}
                metrics_file.write(json.dumps(latest) + "\n")
                for task in ("conversion", "lm"):
                    writer.add_scalar(f"{task}/train_loss", result["losses"][task], state["step"])
                    writer.add_scalar(f"{task}/supervised_tokens", state["totals"][task]["supervised_tokens"], state["step"])
                    writer.add_scalar(f"{task}/input_tokens", state["totals"][task]["input_tokens"], state["step"])
                    writer.add_scalar(f"{task}/supervised_tokens_per_second", result["supervised_tokens"][task] / seconds, state["step"])
                writer.add_scalar("optimizer/learning_rate", rate, state["step"])
                writer.add_scalar("optimizer/grad_norm", result["grad_norm"], state["step"])
                write_json(output / "progress.json", progress(state, plan, latest=latest))
                print(f"step={state['step']}/{settings['max_steps']} conv={result['losses']['conversion']:.4f} lm={result['losses']['lm']:.4f} seconds={seconds:.3f}", flush=True)
            if state["step"] % settings["checkpoint_every"] == 0:
                save()
            if state["step"] % settings["eval_every"] == 0 and state["conversion_cursor"] != settings["updates_per_epoch"]:
                subset = evaluate_conversion(subset_loader)
                writer.add_scalar("conversion/fixed_subset_loss", subset["loss"], state["step"])
                write_json(output / "subset-validation.json", {"step": state["step"], **subset})
                print(f"[fixed subset] step={state['step']} conversion={subset['loss']:.6f}", flush=True)
        terminal_status = "early_stopped" if state["stop_reason"] else (
            "complete" if state["step"] >= settings["max_steps"] else "interrupted")
        save()
        summary = terminal_report(state, plan, output, terminal_status)
        if device.type == "cuda":
            summary["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / 2**20
            summary["peak_reserved_MiB"] = torch.cuda.max_memory_reserved(device) / 2**20
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", summary)
        return summary
    except BaseException as error:
        # Never overwrite last.pt with an update/AdamW call that may be partial.
        write_json(output / "failure.json", {"status": "failed", "active": True, "error_type": type(error).__name__,
            "error": str(error), "last_committed_in_memory_step": state["step"],
            "safe_resume_checkpoint": str(last), "checkpoint_not_overwritten": True})
        raise
    finally:
        metrics_file.close()
        writer.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for data in (conv_train.data, conv_val.data, lm_train, lm_val):
            data.close()
