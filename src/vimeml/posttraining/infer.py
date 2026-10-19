"""Conditional OUT-sum scoring on explicitly allowlisted nonblind snapshots."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import MethodType
import unicodedata

import torch
from torch.nn import functional as F

from vimeml.posttraining.data import ConversionCodec
from vimeml.posttraining.readings import hiragana
from vimeml.training.data import file_sha, write_json
from vimeml.training.model_v4 import TinyGPTV4

ROOT = Path(__file__).resolve().parents[3]
SCORE_CONTRACT = "full_vocabulary_OUT_logp_sum_excluding_EOS_v1"
READING_CONTRACT = "NFKC_then_katakana_to_hiragana_preserve_punctuation_v1"
# Fixed paths and hashes: no directory discovery, arbitrary split or blind mode.
TRACKS = {
    "standard-development": (728, "bf8fc3d6d0bf7323ff6d196a13d555f2f6d099bbf946adf434827e450b687a06"),
    "legacy-regression": (500, "dc2b9430c5676b95dae5c83156741bac35ca2d4c33ebff520b1394c224668e66"),
    "ajimee": (200, "2c18523afc3c23f7341b096f72dd7d84e81a5a2e143d822fbf78512d008fef85"),
    "historical-development": (137, "cd356733e6d6b743ad61cf8cc3f28303de27a84a5238a204615319897f71c8b5"),
}


def read_snapshot(track, root=ROOT):
    if track not in TRACKS:
        raise ValueError("Only explicitly frozen development/regression tracks are allowed")
    count, expected = TRACKS[track]
    if not expected:
        raise ValueError("Unpinned snapshot")
    path = Path(root) / f"outputs/evaluation-v4-20261015/inputs/{track}.json"
    if file_sha(path) != expected:
        raise ValueError(f"Frozen snapshot changed: {track}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != count:
        raise ValueError("Frozen snapshot count mismatch")
    return rows


@contextmanager
def fp32_evaluation(model):
    """Evaluate primitive FP32 operations, avoiding a second compiled graph.

    Weights stay FP32 under BF16 AMP training. Never clone a second CUDA model.
    """
    training, forward = model.training, model.forward
    tf32 = torch.backends.cuda.matmul.allow_tf32
    model.eval()
    model.forward = MethodType(TinyGPTV4.forward, model)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode(), torch.autocast(next(model.parameters()).device.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        model.forward = forward
        model.train(training)


def normalize_reading(query):
    return hiragana(unicodedata.normalize("NFKC", query))


def azookey_order(row):
    pool = [x["text"] for x in row["candidates"]]
    order = row.get("orders", {}).get("azookey", pool)
    if not pool or len(set(pool)) != len(pool) or len(order) != len(pool) or set(order) != set(pool):
        raise ValueError("Frozen candidate order/pool invalid")
    return order


@torch.inference_mode()
def score_candidates(model, codec, context, reading, candidates):
    """Whole-case fallback on overflow; never crop context during evaluation."""
    encoded = []
    for candidate in candidates:
        try:
            item = codec.encode(context, reading, candidate)
        except ValueError as error:
            return {"scores": None, "fallback": "invalid_structured_field", "detail": str(error)}
        if item is None or item[2]:
            return {"scores": None, "fallback": "structured_context_overflow"}
        ids, first, _ = item
        encoded.append((ids, first))
    if not encoded:
        raise ValueError("Empty candidate pool")
    device = next(model.parameters()).device
    width = max(len(ids) - 1 for ids, _ in encoded)
    inputs = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    rows, positions, targets = [], [], []
    for row, (ids, first) in enumerate(encoded):
        inputs[row, :len(ids) - 1] = torch.tensor(ids[:-1], device=device)
        # EOS is a training target; the fixed inference score excludes it.
        for position in range(first, len(ids) - 1):
            rows.append(row)
            positions.append(position - 1)
            targets.append(ids[position])
    logits = model(inputs)
    row_ids = torch.tensor(rows, device=device)
    selected = logits[row_ids, torch.tensor(positions, device=device)]
    logp = -F.cross_entropy(selected.float(), torch.tensor(targets, device=device), reduction="none")
    scores = torch.zeros(len(encoded), device=device).scatter_add_(0, row_ids, logp)
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Non-finite conditional candidate score")
    return {"scores": scores.cpu().tolist(), "fallback": None}


def evaluate_snapshot(model, codec, rows, destination=None, condition="context_reading"):
    if condition not in {"context_reading", "empty_reading", "empty_context"}:
        raise ValueError("Unknown diagnostic condition")
    counts, by_source, records = {"total": 0, "correct": 0, "azookey_correct": 0, "corrected": 0, "damaged": 0, "fallbacks": 0}, {}, []
    for row in rows:
        order = azookey_order(row)
        # The user's query alone supplies READ. Answers are used only AFTER
        # scoring for correctness; never synthesize reading from surface/gold.
        reading = normalize_reading(row["query"])
        context = row["left_context"]
        actual_reading = "" if condition == "empty_reading" else reading
        actual_context = "" if condition == "empty_context" else context
        result = score_candidates(model, codec, actual_context, actual_reading, order)
        ranked = order if result["scores"] is None else [order[i] for i in sorted(
            range(len(order)), key=lambda i: (-result["scores"][i], i))]
        correct, baseline = ranked[0] in row["answers"], order[0] in row["answers"]
        source = row.get("source", "unspecified")
        sub = by_source.setdefault(source, {"total": 0, "correct": 0})
        sub["total"] += 1
        sub["correct"] += int(correct)
        counts["total"] += 1
        counts["correct"] += int(correct)
        counts["azookey_correct"] += int(baseline)
        counts["corrected"] += int(correct and not baseline)
        counts["damaged"] += int(baseline and not correct)
        counts["fallbacks"] += int(result["fallback"] is not None)
        records.append({"id": row["id"], "original_query": row["query"], "normalized_reading": reading,
                        "condition": condition, "context": actual_context, "answers": row["answers"],
                        "candidate_order": order, "scores": result["scores"], "ranked": ranked,
                        "correct": correct, "azookey_correct": baseline, "fallback": result["fallback"]})
    counts["net_improvement"] = counts["corrected"] - counts["damaged"]
    counts["top1"] = counts["correct"] / counts["total"]
    report = {**counts, "by_source": by_source, "condition": condition, "precision": "fp32",
              "score_contract": SCORE_CONTRACT, "reading_contract": READING_CONTRACT,
              "overflow_policy": "whole-case original AzooKey order; full denominator; no context crop"}
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "metrics.json", report)
        with (destination / "scores.jsonl").open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return report


def load_conditional_checkpoint(path, tokenizer, device="cpu"):
    from vimeml.training import model_factory_v4
    from vimeml.training.model_factory import model_from_checkpoint
    from vimeml.posttraining.runtime import TASK_CONTRACT, validate_task_checkpoint
    checkpoint_path = Path(path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_task_checkpoint(saved)
    if saved["posttraining"]["task_contract"] != TASK_CONTRACT:
        raise ValueError("Incompatible conditional scoring contract")
    for name, path in (("conversion", ROOT / saved["config"]["conversion_dir"] / "manifest.json"),
                       ("tokens", ROOT / saved["config"]["token_dir"] / "manifest.json")):
        if file_sha(path) != saved["signatures"][name]:
            raise ValueError(f"Conditional checkpoint {name} identity mismatch")
    codec = ConversionCodec(tokenizer)
    model = model_from_checkpoint(saved).to(device).eval()
    return model, codec, {"step": saved["step"], "stage": saved["posttraining"]["stage"],
                          "checkpoint_sha256": file_sha(checkpoint_path), "source": saved["initialization"]}
