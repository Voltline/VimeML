"""Explicit, checkpoint-bound V4 budget extension; ordinary resume stays strict."""

import copy
import math

from vimeml.training.data import file_sha


def configure(config, settings):
    receipt = config["continuation"]
    start, previous = receipt["source_step"], receipt["source_max_steps"]
    rate = receipt["source_learning_rate"]
    if (config.get("architecture") != "tiny_gpt_v4" or
            type(start) is not int or type(previous) is not int or
            not 0 < start <= previous or not math.isfinite(rate) or
            not settings["min_learning_rate"] <= rate <= settings["learning_rate"] or
            settings.get("learning_rate_schedule", "cosine") != "cosine" or
            not settings.get("epochs") or settings["gradient_accumulation"] != 1):
        raise ValueError("Invalid V4 continuation schedule.")
    if settings["max_steps"] and settings["max_steps"] <= previous:
        raise ValueError("Continuation must extend the previous training budget.")
    settings["continuation_start_step"] = start
    settings["continuation_start_learning_rate"] = rate


def validate_resume(saved, config, signatures, precision, checkpoint):
    """Allow only more epochs, an explicit plateau policy, and the trainer update."""
    receipt = config["continuation"]
    if (saved.get("format") != "vimeml_tiny_gpt_v4" or
            saved["precision"] != precision or saved["step"] != receipt["source_step"] or
            saved["scheduler"]["max_steps"] != receipt["source_max_steps"] or
            file_sha(checkpoint) != receipt["source_checkpoint_sha256"]):
        raise ValueError("V4 continuation does not match its source checkpoint.")
    normalized = copy.deepcopy(config)
    normalized.pop("continuation")
    expected = copy.deepcopy(saved["config"])
    expected.pop("continuation", None)
    current = normalized["training"]
    previous = expected["training"]
    if current["epochs"] <= previous["epochs"]:
        raise ValueError("V4 continuation must increase epochs.")
    current["epochs"] = previous["epochs"]
    if current.get("early_stopping_mode") != "plateau":
        raise ValueError("The explicit continuation policy must be plateau.")
    if "early_stopping_mode" not in previous:
        current.pop("early_stopping_mode")
    if normalized != expected:
        raise ValueError("Only epochs and plateau stopping may change in a V4 extension.")
    before, after = saved["signatures"], signatures
    if {k: v for k, v in before.items() if k != "training_code"} != {
            k: v for k, v in after.items() if k != "training_code"}:
        raise ValueError("V4 continuation requires identical data signatures.")
    old_code, new_code = before["training_code"], after["training_code"]
    old_helper = old_code.get("continuation_v4.py")
    if (old_code["train.py"] != receipt["source_train_sha256"] or
            (old_helper is not None and old_helper != receipt.get("source_continuation_sha256")) or
            set(new_code) != set(old_code) | {"continuation_v4.py"} or
            any(new_code[k] != v for k, v in old_code.items()
                if k not in {"train.py", "continuation_v4.py"})):
        raise ValueError("Unexpected model/runtime/data code change during extension.")
    if any(not math.isclose(group["lr"], receipt["source_learning_rate"],
                            rel_tol=0, abs_tol=1e-15)
           for group in saved["optimizer"]["param_groups"]):
        raise ValueError("Continuation learning rate must match the saved optimizer.")
    if saved.get("early_stopped"):
        raise ValueError("An early-stopped experiment requires a separate decision.")
