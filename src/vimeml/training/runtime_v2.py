"""Optional training compilation and frozen-weight initialization for V2."""

import json
from types import MethodType
from pathlib import Path

import torch
from torch.nn import functional as F

from vimeml.training.model_v2 import RMSNorm


def rmsnorm_operations(hidden, weight, eps):
    normalized = hidden.float()
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    return normalized.to(hidden.dtype) * weight


def fuse_rmsnorm(model):
    """Compile only pure norm operations, leaving variable-length logits eager."""
    compiled = torch.compile(rmsnorm_operations, fullgraph=True, dynamic=True)

    def forward(module, hidden):
        return compiled(hidden, module.weight, module.eps)

    count = 0
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.forward = MethodType(forward, module)
            count += 1
    if not count:
        raise ValueError("RMSNorm fusion requires a V2 model.")
    return {
        "implementation": "torch.compile RMSNorm only",
        "norm_modules": count,
        "dynamic_shapes": True,
        "state_dict_unchanged": True,
        "checkpoint_inference": "Plain primitive RMSNorm; training fusion is optional.",
    }


def compile_backbone(model):
    """Compile the full hidden-state stack, keeping variable-size token loss eager."""

    def hidden_stack(input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = model.dropout(
            model.token_embedding(input_ids) + model.position_embedding(positions)
        )
        for block in model.blocks:
            hidden = block(hidden)
        return model.final_norm(hidden)

    compiled = torch.compile(hidden_stack, fullgraph=True, dynamic=True)

    def forward(module, input_ids, labels=None):
        if (
            input_ids.ndim != 2
            or not 1 <= input_ids.shape[1] <= module.config.context_length
        ):
            raise ValueError("Expected [batch, time] within the model context.")
        hidden = compiled(input_ids)
        if labels is None:
            return module.lm_head(hidden)
        if labels.shape != input_ids.shape:
            raise ValueError("Labels must match input shape.")
        valid = labels != -100
        logits = module.lm_head(hidden[valid])
        return {
            "loss_sum": F.cross_entropy(logits, labels[valid], reduction="sum"),
            "token_count": valid.sum(),
        }

    model.forward = MethodType(forward, model)
    return {
        "implementation": "torch.compile full hidden stack; eager valid-token head and CE",
        "dynamic_shapes": True,
        "state_dict_unchanged": True,
        "checkpoint_inference": "Plain TinyGPTV2",
    }


def initialize_weights(model, checkpoint, signatures, optimizer=None, precision=None,
                       allow_new_data=False, tokenizer_identity=None):
    """Start a separate run; optionally carry compatible AdamW moments forward."""
    path = Path(checkpoint)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved.get("format") != "vimeml_tiny_gpt_v2"
        or saved["model_config"] != model.configuration()
    ):
        raise ValueError("Initialization requires the same V2 model configuration.")
    data_changed = any(
        saved["signatures"][name] != signatures[name] for name in ("tokens", "windows")
    )
    tokenizer_check = None
    if data_changed:
        if not allow_new_data or optimizer is not None or tokenizer_identity is None:
            raise ValueError("New-data initialization requires explicit permission, matching tokenizer and a fresh optimizer.")
        source_identity = saved.get("tokenizer_identity")
        if source_identity is None:
            root = Path(__file__).resolve().parents[3]
            source_manifest = root / saved["config"]["token_dir"] / "manifest.json"
            source_tokens = json.loads(source_manifest.read_text(encoding="utf-8"))
            source_hash = source_tokens.get("tokenizer_model_sha256")
            if source_hash is None:
                candidates = [value for name, value in source_tokens.get("input_sha256", {}).items()
                              if name.replace("\\", "/").endswith("/tokenizer.model")]
                if len(candidates) != 1:
                    raise ValueError("Source tokenizer identity is unavailable in the frozen manifest.")
                source_hash = candidates[0]
            source_identity = {"tokenizer_model_sha256": source_hash,
                               "vocab_size": source_tokens["vocab_size"], "special_ids": source_tokens["special_ids"]}
        if source_identity != tokenizer_identity:
            raise ValueError("Source and target tokenizer identities differ.")
        tokenizer_check = "Frozen tokenizer identity matched; new corpus allowed; optimizer reset"
    if optimizer is not None:
        if saved.get("precision") != precision or not saved.get("optimizer", {}).get(
            "state"
        ):
            raise ValueError(
                "Optimizer continuation requires matching precision and nonempty state."
            )
        source_groups = saved["optimizer"]["param_groups"]
        if len(source_groups) != len(optimizer.param_groups):
            raise ValueError("Optimizer parameter groups differ.")
        for source, target in zip(source_groups, optimizer.param_groups):
            if (
                len(source["params"]) != len(target["params"])
                or source["betas"] != target["betas"]
                or source["weight_decay"] != target["weight_decay"]
                or source["eps"] != target["eps"]
                or source.get("amsgrad", False) != target.get("amsgrad", False)
            ):
                raise ValueError(
                    "Optimizer continuation requires compatible AdamW groups."
                )
    model.load_state_dict(saved["model"])
    if optimizer is not None:
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(saved["optimizer"])
        for group, rate in zip(optimizer.param_groups, learning_rates):
            group["lr"] = rate
    details = {
        "checkpoint": str(path.resolve()),
        "source_step": saved["step"],
        "source_trained_windows": saved["total_windows"],
        "source_epoch": saved["epoch"],
        "source_batch_cursor": saved["batch_cursor"],
        "source_signatures": saved["signatures"],
        "new_training_data": data_changed,
        "tokenizer_compatibility": tokenizer_check,
        "file_bytes": path.stat().st_size,
        "optimizer": (
            "restored AdamW moments and parameter step counters; new local scheduler and data cursors"
            if optimizer is not None
            else "fresh AdamW; source optimizer, scheduler and cursors are not restored"
        ),
        "optimizer_state_count": len(saved["optimizer"]["state"])
        if optimizer is not None
        else 0,
        "verification": "Model configuration and small dataset manifest signatures; no repeated checkpoint SHA256",
    }
    del saved
    return details
