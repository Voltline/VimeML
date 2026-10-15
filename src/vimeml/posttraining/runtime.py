"""V4-only initialization, task identity, normalized updates and stop policy."""
import copy
import math
from pathlib import Path

import torch

from vimeml.posttraining.data import TOKENIZER_SHA
from vimeml.posttraining.infer import SCORE_CONTRACT, READING_CONTRACT
from vimeml.posttraining.streams import microbatches
from vimeml.training.data import file_sha
from vimeml.training.model_v4 import GPTV4Config
from vimeml.training.train import amp_context

STAGE = "v4-c1"
TASK_CONTRACT = {
    "format": "BOS_CTX_context_READ_reading_OUT_surface_EOS_v1",
    "labels": "already_shifted_OUT_and_EOS_only_PAD_minus100",
    "lm": "ordinary_unmarked_LM_all_valid_targets",
    "normalization": "independent_task_total_supervised_tokens_per_update",
    "score": SCORE_CONTRACT, "reading": READING_CONTRACT,
    "context_length": 128, "tokenizer_sha256": TOKENIZER_SHA,
    "marker_ids": {"<CTX>": 4, "<READ>": 5, "<OUT>": 6},
}


def validate_task_checkpoint(saved):
    if (saved.get("format") != "vimeml_tiny_gpt_v4" or saved.get("architecture") != "tiny_gpt_v4"
            or saved.get("model_config") != GPTV4Config().__dict__
            or saved.get("posttraining", {}).get("stage") != STAGE
            or saved["posttraining"].get("task_contract") != TASK_CONTRACT):
        raise ValueError("Explicit compatible V4-C1 checkpoint required")
    expected = {"tokenizer_model_sha256": TOKENIZER_SHA, "vocab_size": 8192,
                "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3}}
    if saved.get("tokenizer_identity") != expected:
        raise ValueError("Conditional checkpoint tokenizer identity mismatch")


def initialize_v4(model, source, expected_sha, expected_step, tokenizer_identity, token_signature, window_signature):
    """Cross-stage initialization loads weights only, never AdamW/RNG/cursors."""
    if file_sha(source) != expected_sha:
        raise ValueError("Source checkpoint SHA256 mismatch")
    saved = torch.load(source, map_location="cpu", weights_only=True)
    if (saved.get("format") != "vimeml_tiny_gpt_v4" or saved.get("architecture") != "tiny_gpt_v4"
            or saved.get("posttraining") or saved.get("model_config") != model.configuration()
            or saved.get("step") != expected_step or saved.get("tokenizer_identity") != tokenizer_identity
            or saved.get("signatures", {}).get("tokens") != token_signature
            or saved.get("signatures", {}).get("windows") != window_signature):
        raise ValueError("Source must be the frozen matching V4-B LM checkpoint")
    model.load_state_dict(saved["model"], strict=True)
    return {"checkpoint": str(Path(source)), "checkpoint_sha256": expected_sha,
            "source_step": saved["step"], "source_total_tokens": saved["total_tokens"],
            "source_epoch": saved["epoch"], "optimizer": "fresh_AdamW_no_source_state_restored"}


def update(model, optimizer, conversion, lm, settings, device, precision):
    """One atomic logical update; never average unequal microbatch means."""
    denominators = {name: sum(int((b["labels"] != -100).sum()) for b in batches)
                    for name, batches in (("conversion", conversion), ("lm", lm))}
    if min(denominators.values()) <= 0:
        raise ValueError("Each task needs supervised targets")
    optimizer.zero_grad(set_to_none=True)
    losses, inputs, examples, positions = {}, {}, {}, {}
    for name, batches in (("conversion", conversion), ("lm", lm)):
        pieces = []
        inputs[name] = sum(int(b["lengths"].sum()) for b in batches)
        examples[name] = sum(len(b["lengths"]) for b in batches)
        positions[name] = 0
        for batch in microbatches(batches, settings[f"{name}_microbatch"]):
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with amp_context(device, precision):
                result = model(ids, labels)
                loss = result["loss_sum"] * (settings[f"{name}_weight"] / denominators[name])
            pieces.append(result["loss_sum"].detach())
            # Non-finite gradients are rejected BEFORE optimizer.step by clip.
            loss.backward()
            positions[name] += ids.numel()
        value = float(torch.stack(pieces).sum())
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite {name} training loss")
        losses[name] = value / denominators[name]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True))
    optimizer.step()
    return {"losses": losses, "supervised_tokens": denominators, "input_tokens": inputs,
            "examples": examples, "padded_positions": positions, "grad_norm": grad_norm,
            "weighted_loss": sum(losses[n] * settings[f"{n}_weight"] for n in losses)}


def initial_state():
    return {"step": 0, "epoch": 0, "conversion_cursor": 0,
            "lm_cursor": {"epoch": 0, "batch": 0, "offset": 0},
            "epoch_results": [], "conversion_reference": None, "plateau_count": 0,
            "forgetting_count": 0, "best_conversion": None, "best_lm": None,
            "best_rerank_key": None, "stop_reason": None,
            "totals": {name: {"examples": 0, "input_tokens": 0, "supervised_tokens": 0, "padded_positions": 0}
                       for name in ("conversion", "lm")}, "training_seconds": 0.0}


def apply_epoch_result(state, conversion_loss, lm_bpc, ime_correct, settings):
    """Returns checkpoint roles; caller applies exactly once per completed epoch."""
    if not all(math.isfinite(x) and x > 0 for x in (conversion_loss, lm_bpc)):
        raise ValueError("Invalid full validation metrics")
    roles = []
    if state["best_conversion"] is None or conversion_loss < state["best_conversion"]:
        state["best_conversion"] = conversion_loss
        roles.append("best-conversion")
    if state["best_lm"] is None or lm_bpc < state["best_lm"]:
        state["best_lm"] = lm_bpc
        roles.append("best-lm")
    reference = state["conversion_reference"]
    if reference is None or conversion_loss < reference * (1 - settings["early_stopping_relative_delta"]):
        state["conversion_reference"], state["plateau_count"] = conversion_loss, 0
    else:
        state["plateau_count"] += 1
    allowed = lm_bpc <= settings["baseline_lm_bpc"] * (1 + settings["lm_relative_degradation"])
    state["forgetting_count"] = 0 if allowed else state["forgetting_count"] + 1
    key = [int(ime_correct), -conversion_loss, -state["step"]]
    if allowed and (state["best_rerank_key"] is None or key > state["best_rerank_key"]):
        state["best_rerank_key"] = key
        roles += ["best-rerank", "best"]
    if state["forgetting_count"] >= settings["lm_degradation_patience"]:
        state["stop_reason"] = "lm_retention_guard"
    elif state["plateau_count"] >= settings["early_stopping_patience"]:
        state["stop_reason"] = "conversion_plateau"
    return roles, allowed


def restore_checkpoint(saved, model, optimizer, config, signatures, precision):
    validate_task_checkpoint(saved)
    if saved["config"] != config or saved["signatures"] != signatures or saved["precision"] != precision:
        raise ValueError("C1 resume requires identical config, code, data, task and precision")
    if saved["step"] != saved["training_state"]["step"] or saved["epoch"] != saved["training_state"]["epoch"]:
        raise ValueError("Checkpoint header and committed state disagree")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    return copy.deepcopy(saved["training_state"]), saved["rng"], saved["initialization"]
