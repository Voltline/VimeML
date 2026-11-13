"""zh-200m pretraining arms (docs/zh-CN/training-zh-200m-plan.md, 200M Chinese desktop model).

The vimeml-tiny-zh-v1 LM/chat recipe on the zh-200m stores, at ten times the
parameters and, for the default arms, four times the context. Each arm has its
own frozen mixture from prepare_zh200m.py; the arm fixes vocabulary, context
length, layer count and seed. Every update covers the same 65,536 supervised
tokens: 128 windows of 512 tokens at context 512, 512 windows of 128 tokens at
context 128. Milestone k (k = 1..20) is the first update at or after k/20 of
max_steps: full FP32 web/chat validation and memorization-probe BPC.
Checkpoints are kept at milestones 1 and 4 (1/20 and 1/5 of the budget) and at
the end (final.pt).
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from vimeml.training.data import SentenceWindowDataset, file_sha, write_json
from vimeml.training.model_zh200m import SHAPES, GPTZh200mConfig, TinyGPTZh200m
from vimeml.training.train import (optimizer_for, learning_rate, atomic_checkpoint, rng_state,
                                   restore_rng, resolve_precision, amp_context)
from vimeml.training.zh200m_data import ARMS, training_dataset
from vimeml.posttraining.streams import CommittedStream, loader_for, microbatches
from vimeml.posttraining.train import validate

ROOT = Path(__file__).resolve().parents[3]
STAGE = "zh-200m"
ARCHITECTURE = "tiny_gpt_zh200m"
FORMAT = "vimeml_tiny_gpt_zh200m"
SPECIAL = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
MILESTONES = 20
KEEP_MILESTONES = (1, 4)
TOKENS_PER_UPDATE = 65_536
TASK_CONTRACT = {
    "objective": "single_task_lm_token_normalized",
    "tasks": ["web_lm", "chat_lm"],
    "normalization": "total supervised tokens per update",
    "lm": "ordinary_unmarked_LM_all_valid_targets",
    "ime_validation": "none_in_training; prediction and rerank benchmarks scored after training",
    "tokens_per_update": TOKENS_PER_UPDATE,
}


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def milestone_steps(total):
    return [math.ceil(k * total / MILESTONES) for k in range(1, MILESTONES + 1)]


def build_plan(config, root=ROOT, verify_files=True):
    root = Path(root)
    if (config.get("stage") != STAGE or config.get("architecture") != ARCHITECTURE
            or config.get("initialization") != {"mode": "random"}):
        raise ValueError("zh-200m requires explicit random initialization; no source checkpoint")
    arm = config.get("arm")
    if arm not in ARMS:
        raise ValueError("Unknown zh-200m arm")
    vocab, context, pool, epochs, seed = ARMS[arm]
    model = GPTZh200mConfig(**config["model"])
    s = dict(config["training"])
    if ((model.vocab_size, model.context_length, model.n_layers, s["seed"])
            != (vocab, context, SHAPES[(vocab, context)][0], seed)):
        raise ValueError(f"{arm} configuration differs from its registered arm")
    if s["lm_batch"] * model.context_length != TOKENS_PER_UPDATE:
        raise ValueError(f"zh-200m fixes {TOKENS_PER_UPDATE} tokens per update; "
                         f"{s['lm_batch']} windows x {model.context_length} tokens does not match")
    for key in ("lm_batch", "lm_microbatch", "num_workers", "cpu_threads", "log_every", "eval_every",
                "checkpoint_every", "validation_batch", "bucket_multiplier", "warmup_steps"):
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(f"Invalid training integer: {key}")
    for key in ("smoke_updates", "smoke_validation_windows"):
        if type(s.get(key, 0)) is not int or s.get(key, 0) < 0:
            raise ValueError(f"Invalid smoke setting: {key}")
    smoke = s.get("smoke_updates", 0) > 0
    if s.get("smoke_validation_windows", 0) and not smoke:
        raise ValueError("Validation limits are for smoke runs only")
    if smoke and s["smoke_updates"] < MILESTONES:
        raise ValueError("Smoke runs need at least 20 updates to reach every milestone")
    for key in ("learning_rate", "min_learning_rate", "beta1", "beta2", "weight_decay", "grad_clip"):
        if not math.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f"Invalid training scalar: {key}")
    compiled = config["runtime"]["compile_backbone"]
    if not ((s["device"] == "cuda" and s["precision"] == "bf16" and compiled)
            or (s["device"] == "cpu" and s["precision"] == "fp32" and not compiled and smoke)):
        raise ValueError("Formal training is CUDA/BF16/compiled; only smoke runs are CPU/FP32")
    if (not 0 < s["min_learning_rate"] <= s["learning_rate"] or s["beta1"] >= 1 or s["beta2"] >= 1
            or s["prefix_crop_probability"] != 0.30 or s["lm_microbatch"] > s["lm_batch"]):
        raise ValueError("zh-200m optimizer/crop/microbatch contract mismatch")
    mix_dir = root / config["data_mixture"]["directory"]
    mix = read(mix_dir / "manifest.json")
    request = mix["request"]
    if (mix["status"] != "complete" or request["arm"] != arm or request["vocab_size"] != vocab
            or request["context_length"] != context
            or request["pool"] != pool or request["epochs"] != epochs or len(mix["epochs"]) != epochs
            or request["seed"] != seed or request["crop_probability"] != s["prefix_crop_probability"]):
        raise ValueError("Frozen mixture differs from the training configuration")
    if file_sha(mix_dir / "manifest.json") != config["dataset_sha256"]:
        raise ValueError("zh-200m dataset manifest changed")
    required = {mix_dir / "manifest.json": config["dataset_sha256"],
                mix_dir / "request.json": file_sha(mix_dir / "request.json"),
                mix_dir / mix["probe"]["file"]: mix["probe"]["sha256"]}
    for record in mix["epochs"]:
        required[mix_dir / record["file"]] = record["sha256"]
    for store in (request["web"], request["chat"]):
        folder, index = root / store["token_dir"], root / store["index_dir"]
        manifest, windows = read(folder / "manifest.json"), read(index / "manifest.json")
        if (file_sha(folder / "manifest.json") != store["token_manifest_sha256"]
                or file_sha(index / "manifest.json") != store["window_manifest_sha256"]
                or manifest["tokenizer_model_sha256"] != request["tokenizer_model_sha256"]
                or manifest["vocab_size"] != vocab or manifest["special_ids"] != SPECIAL):
            raise ValueError(f"Changed {store['name']} store or tokenizer")
        for name, digest in manifest["output_sha256"].items():
            required[folder / name] = digest
        for split in ("train", "validation"):
            required[index / f"{split}.windows.npz"] = windows["splits"][split]["index_sha256"]
        required[folder / "manifest.json"] = store["token_manifest_sha256"]
        required[index / "manifest.json"] = store["window_manifest_sha256"]
    code = {p.relative_to(ROOT).as_posix(): file_sha(p) for p in sorted((ROOT / "src/vimeml").rglob("*.py"))}
    code["scripts/training/train_zh200m.py"] = file_sha(ROOT / "scripts/training/train_zh200m.py")
    signatures = {"dataset": config["dataset_sha256"], "tokens": request["web"]["token_manifest_sha256"],
                  "windows": request["web"]["window_manifest_sha256"],
                  "code": hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()}
    if verify_files:
        for path, digest in required.items():
            if file_sha(path) != digest:
                raise ValueError("Changed training dependency: " + str(path))
    s["epochs"] = epochs
    s["updates_per_epoch"] = [math.ceil(r["windows"] / s["lm_batch"]) for r in mix["epochs"]]
    s["windows_per_epoch"] = [r["windows"] for r in mix["epochs"]]
    s["max_steps"] = sum(s["updates_per_epoch"])
    s["milestone_steps"] = milestone_steps(s["smoke_updates"] if smoke else s["max_steps"])
    web_manifest = read(root / request["web"]["token_dir"] / "manifest.json")
    identity = {"tokenizer_model_sha256": request["tokenizer_model_sha256"], "vocab_size": vocab,
                "special_ids": SPECIAL, "reserved_ids": web_manifest["reserved_ids"]}
    return {"stage": STAGE, "arm": arm, "parameters": SHAPES[(vocab, context)][1], "settings": s,
            "signatures": signatures,
            "code_sha256": code, "tokenizer_identity": identity, "request": request, "probe": mix["probe"],
            "budget": {"web_characters_total": mix["web_characters_total"], "chat_passes": mix["chat_passes"],
                       "effective_tokens": sum(r["effective_tokens"] for r in mix["epochs"])},
            "required_files": {Path(p).resolve().as_posix(): h for p, h in required.items()},
            "task_contract": {**TASK_CONTRACT, "arm": arm, "pool": pool, "epochs": epochs,
                              "context_length": context, "lm_batch": s["lm_batch"]}}


def initial_state():
    return {"epoch": 0, "epoch_batch": 0, "step": 0, "lm_cursor": {"epoch": 0, "batch": 0, "offset": 0},
            "milestones": [], "training_seconds": 0.,
            "totals": {"lm": {k: 0 for k in ("examples", "input_tokens", "supervised_tokens", "padded_positions")}}}


def consumed_examples(state, plan):
    s = plan["settings"]
    e, b = state["epoch"], state["epoch_batch"]
    total = sum(s["windows_per_epoch"][:e])
    if e < len(s["updates_per_epoch"]) and b:
        total += s["windows_per_epoch"][e] if b == s["updates_per_epoch"][e] else b * s["lm_batch"]
    return total


def due_milestones(step, plan):
    return [k + 1 for k, point in enumerate(plan["settings"]["milestone_steps"]) if point <= step]


def validate_state(state, plan):
    e, b = state["epoch"], state["epoch_batch"]
    updates = plan["settings"]["updates_per_epoch"]
    if not 0 <= e <= len(updates) or b < 0 or (e == len(updates) and b) or (e < len(updates) and b > updates[e]):
        raise ValueError("Invalid committed LM epoch cursor")
    if state["step"] != sum(updates[:e]) + b:
        raise ValueError("Step does not match committed epoch updates")
    if state["totals"]["lm"]["examples"] != consumed_examples(state, plan):
        raise ValueError("LM exposure does not match committed updates")
    if state["lm_cursor"] != {"epoch": e, "batch": b, "offset": 0}:
        raise ValueError("LM stream cursor mismatch")
    done, due = [r["milestone"] for r in state["milestones"]], due_milestones(state["step"], plan)
    # A crash between the committed update and its validation leaves the last one pending.
    if done != due and done != due[:-1]:
        raise ValueError("Invalid completed/pending milestone history")


def progress(state, plan, status="running", latest=None):
    seconds = state["training_seconds"]
    rate = state["step"] / seconds if seconds else None
    return {"status": status, "arm": plan["arm"], "step": state["step"], "max_steps": plan["settings"]["max_steps"],
            "epoch": state["epoch"], "lm_cursor": state["lm_cursor"], "totals": state["totals"],
            "training_seconds": seconds, "latest_update": latest,
            "training_only_eta_seconds": (plan["settings"]["max_steps"] - state["step"]) / rate if rate else None,
            "latest_milestone": state["milestones"][-1] if state["milestones"] else None}


def update_lm(model, optimizer, batches, s, device, precision):
    """One atomic logical LM update; never average unequal microbatch means."""
    denominator = sum(int((b["labels"] != -100).sum()) for b in batches)
    if denominator <= 0:
        raise ValueError("The LM task needs supervised targets")
    optimizer.zero_grad(set_to_none=True)
    inputs = sum(int(b["lengths"].sum()) for b in batches)
    examples = sum(len(b["lengths"]) for b in batches)
    positions = 0
    pieces = []
    for batch in microbatches(batches, s["lm_microbatch"]):
        ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with amp_context(device, precision):
            result = model(ids, labels)
            loss = result["loss_sum"] / denominator
        pieces.append(result["loss_sum"].detach())
        loss.backward()
        positions += ids.numel()
    value = float(torch.stack(pieces).sum())
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite LM training loss")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip"], error_if_nonfinite=True))
    optimizer.step()
    return {"loss": value / denominator, "supervised_tokens": denominator, "input_tokens": inputs,
            "examples": examples, "padded_positions": positions, "grad_norm": grad_norm}


def checkpoint_payload(model, optimizer, state, config, plan, precision, initialization):
    return {"format": FORMAT, "architecture": ARCHITECTURE, "model": model.state_dict(),
            "model_config": model.configuration(), "optimizer": optimizer.state_dict(),
            "config": config, "signatures": plan["signatures"], "precision": precision,
            "step": state["step"], "epoch": state["epoch"], "training_state": copy.deepcopy(state),
            "tokenizer_identity": plan["tokenizer_identity"],
            "pretraining": {"stage": STAGE, "task_contract": plan["task_contract"]},
            "initialization": initialization, "rng": rng_state(),
            "scheduler": {"step": state["step"], "max_steps": plan["settings"]["max_steps"],
                          "warmup_steps": plan["settings"]["warmup_steps"], "name": "cosine"}}


def restore_checkpoint(saved, model, optimizer, config, plan, precision):
    if (saved.get("format") != FORMAT or saved.get("architecture") != ARCHITECTURE
            or saved.get("model_config") != model.configuration()
            or saved.get("pretraining", {}).get("task_contract") != plan["task_contract"]
            or saved.get("tokenizer_identity") != plan["tokenizer_identity"]):
        raise ValueError("Explicit compatible zh-200m checkpoint required")
    if saved["config"] != config or saved["signatures"] != plan["signatures"] or saved["precision"] != precision:
        raise ValueError("Resume requires identical config, code, data and precision")
    if saved["step"] != saved["training_state"]["step"] or saved["epoch"] != saved["training_state"]["epoch"]:
        raise ValueError("Checkpoint header and committed state disagree")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    return copy.deepcopy(saved["training_state"]), saved["rng"], saved["initialization"]


def run(config, resume=False, root=ROOT):
    from torch.utils.tensorboard import SummaryWriter
    from vimeml.posttraining.infer import fp32_evaluation
    root = Path(root)
    output, log_dir = root / config["output_dir"], root / config["log_dir"]
    if not resume and ((output.exists() and any(output.iterdir())) or (log_dir.exists() and any(log_dir.iterdir()))):
        raise ValueError("A fresh arm requires empty output and log directories")
    print("Verifying the frozen arm mixture, token stores and maps...", flush=True)
    plan = build_plan(config, root, verify_files=True)
    s = plan["settings"]
    torch.set_num_threads(s["cpu_threads"])
    random.seed(s["seed"])
    torch.manual_seed(s["seed"])
    device = torch.device(s["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("A GPU is required for formal training")
    precision = resolve_precision(device, s["precision"])
    torch.set_float32_matmul_precision("high")
    model = TinyGPTZh200m(GPTZh200mConfig(**config["model"])).to(device)
    optimizer = optimizer_for(model, s, device)
    state = initial_state()
    initialization = {"mode": "random", "seed": s["seed"], "source_checkpoint": None, "optimizer": "fresh_AdamW"}
    restored = None
    if resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        state, restored, initialization = restore_checkpoint(saved, model, optimizer, config, plan, precision)
        validate_state(state, plan)
        del saved
        if state["epoch"] == s["epochs"] and len(state["milestones"]) == MILESTONES:
            print("Arm is already complete", flush=True)
            return progress(state, plan, "complete")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "plan.json", plan)
    if config["runtime"]["compile_backbone"]:
        from vimeml.training.runtime_v2 import compile_backbone
        compilation = compile_backbone(model)
    else:
        compilation = None
    environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device": str(device),
                   "precision": precision, "parameters": model.parameter_count(), "initialization": initialization,
                   "compilation": compilation, "validation_precision": "primitive_FP32",
                   "smoke_updates": s.get("smoke_updates", 0)}
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
    write_json(output / "environment.json", environment)
    request = plan["request"]
    lm_train = training_dataset(root / config["data_mixture"]["directory"], root)
    web, chat = request["web"], request["chat"]
    web_val = SentenceWindowDataset(root / web["token_dir"], root / web["index_dir"], "validation")
    chat_val = SentenceWindowDataset(root / chat["token_dir"], root / chat["index_dir"], "validation")
    web_train = SentenceWindowDataset(root / web["token_dir"], root / web["index_dir"], "train")
    probe_ids = np.load(root / config["data_mixture"]["directory"] / plan["probe"]["file"], allow_pickle=False)
    options = {"workers": s["num_workers"], "multiplier": s["bucket_multiplier"], "pin_memory": device.type == "cuda"}
    lm_loader = loader_for(lm_train, s["lm_batch"], seed=s["seed"], max_width=model.config.context_length,
                           **options)
    limit = s.get("smoke_validation_windows", 0)

    def val_loader(data, indices=None):
        if limit:
            indices = list(range(len(data)))[:limit] if indices is None else list(indices[:limit])
        return loader_for(data, s["validation_batch"], workers=0, seed=s["seed"], shuffle=False,
                          indices=indices, pin_memory=device.type == "cuda",
                          max_width=model.config.context_length)
    web_stats = read(root / web["token_dir"] / "manifest.json")["splits"]["validation"]
    chat_stats = read(root / chat["token_dir"] / "manifest.json")["splits"]["validation"]
    full = {"web": (val_loader(web_val), web_stats["prediction_pairs"], web_stats["characters"]),
            "chat": (val_loader(chat_val), chat_stats["prediction_pairs"], chat_stats["characters"]),
            "probe": (val_loader(web_train, [int(i) for i in probe_ids]), plan["probe"]["prediction_pairs"],
                      plan["probe"]["characters"])}
    subset = sorted(random.Random(s["seed"] + 1000019).sample(range(len(web_val)), min(4096, len(web_val))))
    subset_loader = loader_for(web_val, s["validation_batch"], workers=0, seed=s["seed"], indices=subset,
                               shuffle=False, pin_memory=device.type == "cuda")
    writer = SummaryWriter(str(log_dir), purge_step=state["step"] + 1 if resume else None, flush_secs=5)
    metrics = (output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
    requested = False
    handlers = {}

    def request_stop(signum, frame):
        nonlocal requested
        if requested:
            raise KeyboardInterrupt("Second interrupt; keep last committed checkpoint")
        requested = True
        print("Saving after the current committed update/validation", flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_stop)

    def save(name="last"):
        payload = checkpoint_payload(model, optimizer, state, config, plan, precision, initialization)
        payload["training_regime"] = "random_initialized_zh_lm_chat_pretraining"
        atomic_checkpoint(output / f"{name}.pt", payload)

    def milestone(k):
        save()
        write_json(output / "progress.json", progress(state, plan, "validating"))
        print(f"[milestone {k}/{MILESTONES}] full web, chat and probe validation", flush=True)
        before = rng_state()
        report = {"milestone": k, "step": state["step"], "epoch_position": state["epoch"] + (
            state["epoch_batch"] / s["updates_per_epoch"][state["epoch"]] if state["epoch"] < s["epochs"] else 0),
            "supervised_tokens": state["totals"]["lm"]["supervised_tokens"]}
        with fp32_evaluation(model):
            for name, (loader, pairs, characters) in full.items():
                if limit:
                    report[name] = validate(model, loader, device)
                else:
                    report[name] = validate(model, loader, device, pairs, characters)
        restore_rng(before)
        state["milestones"].append(report)
        write_json(output / "milestones.json", state["milestones"])
        for name in full:
            value = report[name].get("bpc", report[name]["loss"])
            writer.add_scalar(f"{name}/full_validation_{'bpc' if 'bpc' in report[name] else 'loss'}",
                              value, state["step"])
        if k in KEEP_MILESTONES:
            save(f"milestone-{k:02d}")
        save()
        print(f"[milestone {k}] " + " ".join(f"{n}={report[n].get('bpc', report[n]['loss']):.6f}" for n in full),
              flush=True)

    def pending():
        for k in due_milestones(state["step"], plan):
            if k > len(state["milestones"]):
                milestone(k)
    if restored is not None:
        restore_rng(restored)
    write_json(output / "progress.json", progress(state, plan, "running" if resume else "initializing"))
    smoke = s.get("smoke_updates", 0)
    try:
        if not resume:
            save()
        pending()
        while state["epoch"] < s["epochs"] and not requested:
            e = state["epoch"]
            lm_stream = CommittedStream(lm_loader, state["lm_cursor"])
            updates = s["updates_per_epoch"][e]
            model.train()
            while state["epoch_batch"] < updates and not requested:
                began = time.perf_counter()
                lm_batch = lm_stream.next_batch()
                rate = learning_rate(state["step"] + 1, s)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                result = update_lm(model, optimizer, [lm_batch], s, device, precision)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                state["step"] += 1
                state["epoch_batch"] += 1
                state["lm_cursor"] = lm_stream.state_dict()
                seconds = time.perf_counter() - began
                state["training_seconds"] += seconds
                for field in state["totals"]["lm"]:
                    state["totals"]["lm"][field] += result[field]
                if state["step"] % s["log_every"] == 0:
                    latest = {"step": state["step"], "learning_rate": rate, "seconds": seconds, **result}
                    metrics.write(json.dumps(latest) + "\n")
                    writer.add_scalar("lm/train_loss", result["loss"], state["step"])
                    writer.add_scalar("lm/supervised_tokens", state["totals"]["lm"]["supervised_tokens"], state["step"])
                    writer.add_scalar("optimizer/learning_rate", rate, state["step"])
                    writer.add_scalar("optimizer/grad_norm", result["grad_norm"], state["step"])
                    write_json(output / "progress.json", progress(state, plan, latest=latest))
                    print(f"step={state['step']}/{s['max_steps']} epoch={e + 1}/{s['epochs']} "
                          f"lm={result['loss']:.4f} seconds={seconds:.3f}", flush=True)
                if state["step"] % s["checkpoint_every"] == 0:
                    save()
                if state["step"] % s["eval_every"] == 0:
                    with fp32_evaluation(model):
                        subset_result = validate(model, subset_loader, device)
                    writer.add_scalar("web/fixed_subset_loss", subset_result["loss"], state["step"])
                    write_json(output / "subset-validation.json", {"step": state["step"], **subset_result})
                pending()
                model.train()
                if smoke and state["step"] >= smoke:
                    break
            if (smoke and state["step"] >= smoke) or requested:
                break
            state["epoch"] += 1
            state["epoch_batch"] = 0
            state["lm_cursor"] = {"epoch": state["epoch"], "batch": 0, "offset": 0}
            validate_state(state, plan)
            save()
        if smoke and state["step"] >= smoke:
            terminal = "smoke_complete"
        else:
            terminal = "complete" if state["epoch"] == s["epochs"] else "interrupted"
        if terminal == "complete" and len(state["milestones"]) != MILESTONES:
            raise ValueError("Training ended without all 20 milestones")
        save()
        if terminal in ("complete", "smoke_complete"):
            save("final")
        result = progress(state, plan, terminal)
        result["milestones"] = state["milestones"]
        result["checkpoints"] = sorted(p.name for p in output.glob("*.pt"))
        if device.type == "cuda":
            result["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / 2**20
            result["peak_reserved_MiB"] = torch.cuda.max_memory_reserved(device) / 2**20
        write_json(output / "summary.json", result)
        write_json(output / "progress.json", result)
        return result
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "active": True, "error_type": type(error).__name__,
                                             "error": str(error), "last_committed_in_memory_step": state["step"],
                                             "safe_resume_checkpoint": str(output / "last.pt")})
        raise
    finally:
        writer.close()
        metrics.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        lm_train.close()
        for data in (web_val, chat_val, web_train):
            data.close()
