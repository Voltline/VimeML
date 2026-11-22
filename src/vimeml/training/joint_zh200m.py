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

The same trainer runs the zh-800m model (stage "zh-800m", docs/zh-CN/training-zh-800m-plan.md):
its own shape, its three-web-store mixture and 2048 windows per update; everything
else (loop, validation, milestones, checkpoints, data parallelism) is shared. Stage "zh-800m-arch2"
(docs/training-zh-800m-arch2-plan.md) is the arch2 model on the same data at 8192 or 16384 windows per
update; its training rows are packed into 2048-token rows of per-segment causal attention (packed_update).

Joint pinyin pretraining (optional [pinyin] section, the zh-800m deviation of 2026-11-17): the
arm's mixture, seed, schedule and budget stay unchanged, and every update adds pinyin.mix samples
from build_zh_pinyin.py stores (step-indexed, joint_zh_pinyin.Schedule), as the 12K joint-pinyin
200M did. The loss is "per_kind": (lm_batch / batch) * mean LM loss + sum_k (mix[k] / batch) *
mean loss of k. Each rank packs its rows of every task together: sorted by length, longest
first, split where the saved padding outweighs the cost of another pass, into microbatches
of at most lm_microbatch * context_length padded positions (the memory bound of
lm_microbatch full-length windows), each row's tokens weighted by its task's weight / that
task's global supervised tokens. The update is the same sum as one forward per task; only the
float summation order differs. Pinyin validation (loss, teacher-forced exact) runs at every
milestone and every eval_every.

With data parallelism every rank runs a share of the LM validation batches (whole batches,
the same batches as one device, FP64 sums), and the periodic last.pt is written by a
background thread from a CPU copy; milestone, epoch, final and stop checkpoints stay
synchronous, and last.pt is always replaced atomically.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import threading
import time

import numpy as np
import torch

from vimeml.training.data import SentenceWindowDataset, file_sha, write_json
from vimeml.training import model_zh800m, model_zh800m_arch2, zh200m_data, zh800m_data
from vimeml.training.model_zh200m import SHAPES, GPTZh200mConfig, TinyGPTZh200m
from vimeml.training.train import (optimizer_for, learning_rate, atomic_checkpoint, rng_state,
                                   restore_rng, resolve_precision, amp_context)
from vimeml.training.zh200m_data import ARMS
from vimeml.posttraining.streams import CommittedStream, loader_for, microbatches
from vimeml.training.distributed import Context, NullWriter, setup as setup_parallel

ROOT = Path(__file__).resolve().parents[3]
STAGE = "zh-200m"
ARCHITECTURE = "tiny_gpt_zh200m"
FORMAT = "vimeml_tiny_gpt_zh200m"
SPECIAL = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
MILESTONES = 20
KEEP_MILESTONES = (1, 4)
# Every arm draws the same number of windows per update, matching the vimeml-tiny-zh-v1
# batch: windows hold whole sentences and are far shorter than the context, so the padded
# capacity is not the batch size. 512 windows is about 17k supervised tokens at c512.
WINDOWS_PER_UPDATE = 512
TASK_CONTRACT = {
    "objective": "single_task_lm_token_normalized",
    "tasks": ["web_lm", "chat_lm"],
    "normalization": "total supervised tokens per update",
    "lm": "ordinary_unmarked_LM_all_valid_targets",
    "ime_validation": "none_in_training; prediction and rerank benchmarks scored after training",
    "windows_per_update": WINDOWS_PER_UPDATE,
}


class Family:
    def __init__(self, stage, architecture, fmt, config_class, model_class, shapes, data, windows, tags):
        """windows: the LM windows per update, or a tuple of the allowed ones."""
        self.stage, self.architecture, self.format = stage, architecture, fmt
        self.config_class, self.model_class, self.shapes = config_class, model_class, shapes
        self.data, self.windows, self.tags = data, windows, tags
        self.allowed = windows if isinstance(windows, tuple) else (windows,)

    def arm(self, name):
        """(vocab, context, pool or None, epochs, seed) of a registered arm."""
        if name not in self.data.ARMS:
            raise ValueError(f"Unknown {self.stage} arm")
        record = self.data.ARMS[name]
        return record if len(record) == 5 else (record[0], record[1], None, record[2], record[3])

    def stores(self, request):
        return list(request.get("web_stores", [request["web"]])) + [request["chat"]]


FAMILIES = {
    STAGE: Family(STAGE, ARCHITECTURE, FORMAT, GPTZh200mConfig, TinyGPTZh200m, SHAPES, zh200m_data,
                  WINDOWS_PER_UPDATE, ["zh-200m"]),
    # 800M: 2048 windows (~80k supervised tokens) per update; 512 would mean ~5M updates
    # over ~100B tokens, too small a batch for the model and for data parallelism.
    "zh-800m": Family("zh-800m", "tiny_gpt_zh800m", "vimeml_tiny_gpt_zh800m", model_zh800m.GPTZh800mConfig,
                      model_zh800m.TinyGPTZh800m, model_zh800m.SHAPES, zh800m_data, 2048, ["zh-800m"]),
    # arch2 (docs/training-zh-800m-arch2-plan.md): the zh-800m data at k = 4 or 8 times arch1's 2048 windows,
    # training rows packed to 2048 tokens with per-segment causal attention (model_zh800m_arch2).
    "zh-800m-arch2": Family("zh-800m-arch2", "tiny_gpt_zh800m_arch2", "vimeml_tiny_gpt_zh800m_arch2",
                            model_zh800m_arch2.GPTZh800mArch2Config, model_zh800m_arch2.TinyGPTZh800mArch2,
                            {(12_288, 512): (model_zh800m_arch2.LAYERS, model_zh800m_arch2.PARAMETERS)}, zh800m_data,
                            (8192, 16384), ["zh-800m", "arch2"]),
}


def family_of(config):
    family = FAMILIES.get(config.get("stage"))
    if family is None:
        raise ValueError(f"Unknown stage {config.get('stage')!r}: zh-200m/zh-800m require their own stage "
                         "and explicit random initialization")
    return family


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def milestone_steps(total):
    return [math.ceil(k * total / MILESTONES) for k in range(1, MILESTONES + 1)]


def build_plan(config, root=ROOT, verify_files=True):
    root = Path(root)
    family = family_of(config)
    if (config.get("architecture") != family.architecture
            or config.get("initialization") != {"mode": "random"}):
        raise ValueError(f"{family.stage} requires explicit random initialization; no source checkpoint")
    arm = config.get("arm")
    vocab, context, pool, epochs, seed = family.arm(arm)
    model = family.config_class(**config["model"])
    s = dict(config["training"])
    if ((model.vocab_size, model.context_length, model.n_layers, s["seed"])
            != (vocab, context, family.shapes[(vocab, context)][0], seed)):
        raise ValueError(f"{arm} configuration differs from its registered arm")
    if s["lm_batch"] not in family.allowed:
        raise ValueError(f"{family.stage} takes {family.windows} windows per update; "
                         f"{s['lm_batch']} does not match")
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
        raise ValueError(f"{family.stage} optimizer/crop/microbatch contract mismatch")
    mix_dir = root / config["data_mixture"]["directory"]
    mix = read(mix_dir / "manifest.json")
    request = mix["request"]
    if (mix["status"] != "complete" or mix["format"] != family.data.FORMAT or request["arm"] != arm
            or request["vocab_size"] != vocab or request["context_length"] != context
            or request.get("pool") != pool or request["epochs"] != epochs or len(mix["epochs"]) != epochs
            or request["seed"] != seed or request["crop_probability"] != s["prefix_crop_probability"]):
        raise ValueError("Frozen mixture differs from the training configuration")
    if file_sha(mix_dir / "manifest.json") != config["dataset_sha256"]:
        raise ValueError(f"{family.stage} dataset manifest changed")
    required = {mix_dir / "manifest.json": config["dataset_sha256"],
                mix_dir / "request.json": file_sha(mix_dir / "request.json"),
                mix_dir / mix["probe"]["file"]: mix["probe"]["sha256"]}
    for record in mix["epochs"]:
        required[mix_dir / record["file"]] = record["sha256"]
    for store in family.stores(request):
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
    plan = {"stage": family.stage, "arm": arm, "parameters": family.shapes[(vocab, context)][1], "settings": s,
            "signatures": signatures,
            "code_sha256": code, "tokenizer_identity": identity, "request": request, "probe": mix["probe"],
            "budget": {"web_characters_total": mix["web_characters_total"], "chat_passes": mix["chat_passes"],
                       "effective_tokens": sum(r["effective_tokens"] for r in mix["epochs"])},
            "required_files": {Path(p).resolve().as_posix(): h for p, h in required.items()},
            "task_contract": {**TASK_CONTRACT, "windows_per_update": s["lm_batch"], "arm": arm, "pool": pool,
                              "epochs": epochs,
                              "context_length": context, "lm_batch": s["lm_batch"]}}
    if "pinyin" in config:
        add_pinyin(plan, config["pinyin"], root, verify_files)
    return plan


PINYIN_KINDS = ("full", "mixed", "initials", "error")


def add_pinyin(plan, p, root, verify_files):
    """Extend an arm's plan with the pinyin stores and their per-update mix."""
    if set(p) - {"directory", "manifest_sha256", "mix", "validation_examples"}:
        raise ValueError("Unknown [pinyin] keys")
    mix = {k: int(p["mix"].get(k, 0)) for k in PINYIN_KINDS}
    if set(p["mix"]) - set(PINYIN_KINDS) or any(v < 0 for v in mix.values()) or not sum(mix.values()):
        raise ValueError("Invalid pinyin mix")
    directory = root / p["directory"]
    manifest = read(directory / "manifest.json")
    if (file_sha(directory / "manifest.json") != p["manifest_sha256"] or manifest.get("format") != "vimeml_zh_pinyin_v1"
            or manifest.get("status") != "complete"
            or manifest["tokenizer_sha256"] != plan["tokenizer_identity"]["tokenizer_model_sha256"]):
        raise ValueError("Pinyin stores changed or do not match the arm tokenizer")
    required = {directory / "manifest.json": p["manifest_sha256"]}
    for stores in manifest["stores"].values():
        for kind in PINYIN_KINDS:
            for name, digest in stores[kind]["files"].items():
                required[directory / kind / name] = digest
    if verify_files:
        for path, digest in required.items():
            if file_sha(path) != digest:
                raise ValueError("Changed training dependency: " + str(path))
    examples = int(p.get("validation_examples", 4096))
    if examples < 1:
        raise ValueError("Pinyin validation needs a positive size")
    s = plan["settings"]
    batch = s["lm_batch"] + sum(mix.values())
    train = {k: manifest["stores"]["train"][k]["examples"] for k in PINYIN_KINDS}
    plan["signatures"]["pinyin"] = p["manifest_sha256"]
    plan["required_files"].update({Path(q).resolve().as_posix(): h for q, h in required.items()})
    plan["pinyin"] = {"directory": p["directory"], "mix": mix, "batch": batch,
                      "weights": {"lm": s["lm_batch"] / batch, **{k: v / batch for k, v in mix.items()}},
                      "train_examples": train, "validation_examples": examples,
                      "passes": {k: s["max_steps"] * mix[k] / train[k] for k in PINYIN_KINDS if mix[k]}}
    plan["task_contract"] = {**plan["task_contract"], "objective": "joint_lm_and_pinyin_per_kind",
                             "tasks": ["web_lm", "chat_lm", *PINYIN_KINDS],
                             "normalization": "per_kind: batch share times mean loss per task",
                             "pinyin": {"mix": mix, "manifest_sha256": p["manifest_sha256"]}}


# What one more forward/backward pass costs, in padded positions: on the zh-800m smokes a pass
# costs about as much as 8k positions of compute (launch-bound kernels of 27 layers, DDP hooks,
# the eager loss), so splitting rows into more passes only pays when it saves more padding.
PASS_COST = 8192


def segments(widths, budget, pass_cost=PASS_COST):
    """Split rows sorted by padded width (descending) into contiguous segments at width changes,
    minimizing padded positions + pass_cost * passes; each segment is padded to its first width
    and cut into passes of budget // width rows. Returns the segment start indices."""
    starts = [i for i in range(len(widths)) if i == 0 or widths[i] != widths[i - 1]] + [len(widths)]
    levels = len(starts) - 1
    best, choice = [0.0] * (levels + 1), [levels] * levels
    for a in range(levels - 1, -1, -1):
        width = widths[starts[a]]
        rows_per_pass = budget // width
        best[a] = math.inf
        for b in range(a + 1, levels + 1):
            count = starts[b] - starts[a]
            cost = count * width + pass_cost * math.ceil(count / rows_per_pass) + best[b]
            if cost < best[a]:
                best[a], choice[a] = cost, b
    out, a = [], 0
    while a < levels:
        out.append(starts[a])
        a = choice[a]
    return out


def packed_rows(parts, budget, pass_cost=PASS_COST):
    """[(kind index, collated batch), ...] -> [(input_ids, labels, row kind), ...] on the CPU.

    Rows of every batch are sorted by length (longest first; ties by kind, then row), split
    into segments of similar width (segments(): padding traded against passes), and each
    segment is cut greedily into microbatches with rows * padded width <= budget, the width
    being the microbatch's first (longest) row's length rounded up to 8. Padding is id 0 /
    label -100, as the collators pad."""
    rows = sorted(((int(length), k, b, r) for k, b in parts for r, length in enumerate(b["lengths"].tolist())),
                  key=lambda row: (-row[0], row[1], row[3]))
    widths = [math.ceil(row[0] / 8) * 8 for row in rows]
    if widths and widths[0] > budget:
        raise ValueError("A row longer than the microbatch budget")
    cuts = set(segments(widths, budget, pass_cost))
    groups, current, width = [], [], 0
    for i, row in enumerate(rows):
        if current and (i in cuts or (len(current) + 1) * width > budget):
            groups.append((current, width))
            current = []
        if not current:
            width = widths[i]
        current.append(row)
    if current:
        groups.append((current, width))
    out = []
    for group, width in groups:
        ids = torch.zeros(len(group), width, dtype=torch.long)
        labels = torch.full((len(group), width), -100, dtype=torch.long)
        kind = torch.tensor([row[1] for row in group], dtype=torch.long)
        by_batch = {}
        for j, (_, _, batch, r) in enumerate(group):
            entry = by_batch.setdefault(id(batch), (batch, [], []))
            entry[1].append(j)
            entry[2].append(r)
        for batch, slots, source in by_batch.values():
            span = min(width, batch["input_ids"].shape[1])
            index = torch.tensor(source, dtype=torch.long)
            slot = torch.tensor(slots, dtype=torch.long)
            ids[slot, :span] = batch["input_ids"][index, :span].long()
            labels[slot, :span] = batch["labels"][index, :span].long()
        out.append((ids, labels, kind))
    return out


def update_joint(model, optimizer, batches, weights, s, device, precision, ctx=None, net=None):
    """One update over {"lm": LM batch, kind: pinyin batch, ...}, per_kind normalized.

    This rank's rows of every task (after the data-parallel share) are packed by packed_rows
    into microbatches of at most lm_microbatch * context_length padded positions; the pinyin
    tasks never raise the memory bound set for the LM windows. Each token's loss is weighted
    by weights[kind] * world / (the task's global supervised tokens), so the gradient is that
    of one forward per task. padded_positions counts the positions of every task, including
    the empty row added to a one-row microbatch."""
    ctx, net = ctx or Context(), net or model
    tokens = {k: int((b["labels"] != -100).sum()) for k, b in batches.items()}
    lm = batches["lm"]
    if tokens["lm"] <= 0:
        raise ValueError("The LM task needs supervised targets")
    optimizer.zero_grad(set_to_none=True)
    kinds = [k for k in batches if tokens[k]]
    parts = []
    for kind in kinds:
        part = ctx.share(batches[kind]) if ctx.active else batches[kind]
        if part is not None:
            parts.append((kinds.index(kind), part))
    scale = torch.tensor([weights[k] * ctx.world / tokens[k] for k in kinds], dtype=torch.float32)
    budget = s["lm_microbatch"] * model.config.context_length
    if getattr(model, "packed_training", False):
        return packed_update(model, optimizer, batches, kinds, tokens, parts, scale, budget, s, device, precision,
                             ctx, net)
    work = packed_rows(parts, budget)
    totals = torch.zeros(len(kinds), dtype=torch.float32, device=device)
    positions = 0
    for i, (ids, labels, row_kind) in enumerate(work):
        if len(row_kind) == 1 and 2 * ids.shape[1] <= budget:
            # torch.compile specializes a dimension of size 1 (a recompile, then a second graph):
            # an empty row (no labels, no loss, no gradient) keeps every microbatch at >= 2 rows.
            ids = torch.cat([ids, torch.zeros_like(ids)])
            labels = torch.cat([labels, torch.full_like(labels, -100)])
            row_kind = row_kind.repeat(2)
        ids = ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        row_kind = row_kind.to(device, non_blocking=True)
        token_weights = scale.to(device)[row_kind][:, None].expand(labels.shape)
        with ctx.no_sync(net, i == len(work) - 1):
            with amp_context(device, precision):
                out = net(ids, labels, token_weights=token_weights)
            out["loss_sum"].backward()
        token_kind = row_kind[:, None].expand(labels.shape)[labels != -100]
        totals += torch.bincount(token_kind, weights=out["token_losses"].float(), minlength=len(kinds))
        positions += ids.numel()
    sums = totals.tolist()
    if ctx.active:
        *sums, positions = ctx.sum(sums + [positions], device)
        positions = int(positions)
    sums = dict(zip(kinds, sums))
    if not all(math.isfinite(v) for v in sums.values()):
        raise FloatingPointError("Non-finite joint training loss")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip"], error_if_nonfinite=True))
    optimizer.step()
    return {"loss": sums["lm"] / tokens["lm"], "supervised_tokens": tokens["lm"],
            "input_tokens": int(lm["lengths"].sum()), "examples": len(lm["lengths"]),
            "padded_positions": positions, "grad_norm": grad_norm,
            "pinyin": {k: {"loss": sums[k] / tokens[k], "supervised_tokens": tokens[k],
                           "examples": len(batches[k]["lengths"])} for k in kinds if k != "lm"}}


def segment_microbatches(parts, length, limit, rows_per_pass):
    """Packed training rows of this rank (arch2): every row of every task is one segment, packed first-fit
    decreasing into rows of `length` tokens (model_zh800m_arch2.pack), then cut into passes of at most
    rows_per_pass rows. Returns [(input_ids, labels, segments, position kind), ...]; padding positions have
    label -100 and kind 0. A one-row pass gets an empty second row (segment -1, no labels), as packed_rows'
    microbatches do, so compiled shapes never specialize on one row."""
    sequences, kinds = [], []
    for k, batch in parts:
        for r, n in enumerate(batch["lengths"].tolist()):
            sequences.append((batch["input_ids"][r, :n].long(), batch["labels"][r, :n].long()))
            kinds.append(k)
    ids, labels, segments, _ = model_zh800m_arch2.pack(sequences, length, limit)
    kind = torch.tensor(kinds, dtype=torch.long)[segments.clamp(min=0)]
    out = []
    for a in range(0, len(ids), rows_per_pass):
        part = [t[a:a + rows_per_pass] for t in (ids, labels, segments, kind)]
        if len(part[0]) == 1:
            pads = (0, -100, -1, 0)
            part = [torch.cat([t, torch.full_like(t, pad)]) for t, pad in zip(part, pads)]
        out.append(tuple(part))
    return out


def packed_update(model, optimizer, batches, kinds, tokens, parts, scale, budget, s, device, precision, ctx, net):
    """update_joint for a packing model: same per-token weights and normalization, rows packed into segments."""
    length = model.config.pack_length
    packing = time.perf_counter()
    work = segment_microbatches(parts, length, model.config.context_length, max(1, budget // length))
    packing = time.perf_counter() - packing
    totals = torch.zeros(len(kinds), dtype=torch.float32, device=device)
    positions = 0
    for i, (ids, labels, segments, kind) in enumerate(work):
        ids, labels = ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        segments, kind = segments.to(device, non_blocking=True), kind.to(device, non_blocking=True)
        with ctx.no_sync(net, i == len(work) - 1):
            with amp_context(device, precision):
                out = net(ids, labels, token_weights=scale.to(device)[kind], segments=segments)
            out["loss_sum"].backward()
        totals += torch.bincount(kind[labels != -100], weights=out["token_losses"].float(), minlength=len(kinds))
        positions += ids.numel()
    local_positions = positions
    sums = totals.tolist()
    if ctx.active:
        *sums, positions = ctx.sum(sums + [positions], device)
        positions = int(positions)
    sums = dict(zip(kinds, sums))
    if not all(math.isfinite(v) for v in sums.values()):
        raise FloatingPointError("Non-finite joint training loss")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip"], error_if_nonfinite=True))
    optimizer.step()
    lm = batches["lm"]
    return {"loss": sums["lm"] / tokens["lm"], "supervised_tokens": tokens["lm"],
            "input_tokens": int(lm["lengths"].sum()), "examples": len(lm["lengths"]),
            "padded_positions": positions, "grad_norm": grad_norm,
            "pinyin": {k: {"loss": sums[k] / tokens[k], "supervised_tokens": tokens[k],
                           "examples": len(batches[k]["lengths"])} for k in kinds if k != "lm"},
            "timing": {"pack": packing, "passes": len(work), "padded": local_positions}}


def shared_validate(model, loader, device, ctx, expected_tokens=None, characters=None):
    """posttraining.train.validate over this rank's share of the loader's batches (batch i on
    rank i % world), summed over ranks: the same batches and per-batch sums as one device."""
    # The next batch is read and collated on a thread while the GPU runs this one, and the per-batch sums
    # stay on the GPU until the end: no host sync per batch. The sums are the same values, added in order.
    mine = [indices for i, indices in enumerate(loader.batch_sampler) if i % ctx.world == ctx.rank]

    def read(indices):
        batch = loader.collate_fn([loader.dataset[j] for j in indices])
        if torch.device(device).type == "cuda":
            batch = {k: v.pin_memory() if torch.is_tensor(v) else v for k, v in batch.items()}
        return batch
    sums, counts, total, inputs, examples = [], [], 0, 0, 0
    with ThreadPoolExecutor(1) as pool:
        upcoming = pool.submit(read, mine[0]) if mine else None
        for n in range(len(mine)):
            batch = upcoming.result()
            upcoming = pool.submit(read, mine[n + 1]) if n + 1 < len(mine) else None
            labels = batch["labels"].to(device, non_blocking=True)
            result = model(batch["input_ids"].to(device, non_blocking=True), labels)
            sums.append(result["loss_sum"].detach())
            counts.append((result["token_count"], int((batch["labels"] != -100).sum())))
            inputs += int(batch["lengths"].sum())
            examples += len(batch["lengths"])
    nll = 0.0
    for loss_sum, (counted, count) in zip(sums, counts):
        if int(counted) != count:
            raise ValueError("Model/dataset supervised token counts disagree")
        total += count
        nll += float(loss_sum)
    if ctx.active:
        nll, total, inputs, examples = ctx.sum([nll, total, inputs, examples], device)
        total, inputs, examples = int(total), int(inputs), int(examples)
    if not total or not math.isfinite(nll) or expected_tokens is not None and total != expected_tokens:
        raise ValueError("Empty/non-finite/incomplete full validation")
    report = {"loss": nll / total, "nll_sum": nll, "supervised_tokens": total,
              "input_tokens": inputs, "examples": examples, "precision": "fp32"}
    if characters is not None:
        report.update(characters=characters, bpc=nll / (characters * math.log(2)))
    return report


def on_cpu(value):
    """A copy of a checkpoint payload with every tensor copied to the CPU."""
    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: on_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(on_cpu(v) for v in value)
    return value


class BackgroundSaver:
    """Writes one checkpoint at a time on a thread; wait() before anything else saves or exits.
    The thread is not a daemon, so a pending write finishes before the process ends."""
    def __init__(self):
        self.thread, self.error = None, None

    def submit(self, path, payload):
        self.wait()
        payload = on_cpu(payload)

        def write():
            try:
                atomic_checkpoint(path, payload)
            except BaseException as error:      # reported by the next wait()
                self.error = error
        self.thread = threading.Thread(target=write, name="checkpoint-writer")
        self.thread.start()

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            error, self.error = self.error, None
            raise error


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


def update_lm(model, optimizer, batches, s, device, precision, ctx=None, net=None):
    """One atomic logical LM update; never average unequal microbatch means.

    With data parallelism (ctx, net = the DDP module) each rank runs its share of the global
    batch; the denominator stays the global supervised-token count."""
    ctx, net = ctx or Context(), net or model
    denominator = sum(int((b["labels"] != -100).sum()) for b in batches)
    if denominator <= 0:
        raise ValueError("The LM task needs supervised targets")
    optimizer.zero_grad(set_to_none=True)
    inputs = sum(int(b["lengths"].sum()) for b in batches)
    examples = sum(len(b["lengths"]) for b in batches)
    if ctx.active:
        batches = [part for part in (ctx.share(b) for b in batches) if part is not None]
    parts = list(microbatches(batches, s["lm_microbatch"]))
    positions = 0
    pieces = []
    for i, batch in enumerate(parts):
        ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with ctx.no_sync(net, i == len(parts) - 1):
            with amp_context(device, precision):
                result = net(ids, labels)
                loss = result["loss_sum"] * (ctx.world / denominator)
            loss.backward()
        pieces.append(result["loss_sum"].detach())
        positions += ids.numel()
    value = float(torch.stack(pieces).sum())
    if ctx.active:
        value, positions = ctx.sum([value, positions], device)
        positions = int(positions)
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite LM training loss")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip"], error_if_nonfinite=True))
    optimizer.step()
    return {"loss": value / denominator, "supervised_tokens": denominator, "input_tokens": inputs,
            "examples": examples, "padded_positions": positions, "grad_norm": grad_norm}


def checkpoint_payload(model, optimizer, state, config, plan, precision, initialization):
    family = family_of(config)
    return {"format": family.format, "architecture": family.architecture, "model": model.state_dict(),
            "model_config": model.configuration(), "optimizer": optimizer.state_dict(),
            "config": config, "signatures": plan["signatures"], "precision": precision,
            "step": state["step"], "epoch": state["epoch"], "training_state": copy.deepcopy(state),
            "tokenizer_identity": plan["tokenizer_identity"],
            "pretraining": {"stage": family.stage, "task_contract": plan["task_contract"]},
            "initialization": initialization, "rng": rng_state(),
            "scheduler": {"step": state["step"], "max_steps": plan["settings"]["max_steps"],
                          "warmup_steps": plan["settings"]["warmup_steps"], "name": "cosine"}}


def restore_checkpoint(saved, model, optimizer, config, plan, precision):
    family = family_of(config)
    if (saved.get("format") != family.format or saved.get("architecture") != family.architecture
            or saved.get("model_config") != model.configuration()
            or saved.get("pretraining", {}).get("task_contract") != plan["task_contract"]
            or saved.get("tokenizer_identity") != plan["tokenizer_identity"]):
        raise ValueError(f"Explicit compatible {family.stage} checkpoint required")
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
    ctx = setup_parallel(config["training"]["device"])
    output, log_dir = root / config["output_dir"], root / config["log_dir"]
    if not resume and ((output.exists() and any(output.iterdir())) or (log_dir.exists() and any(log_dir.iterdir()))):
        raise ValueError("A fresh arm requires empty output and log directories")
    ctx.barrier()                       # every rank has seen the empty directories before rank 0 writes
    if ctx.main:
        print("Verifying the frozen arm mixture, token stores and maps...", flush=True)
    plan = build_plan(config, root, verify_files=ctx.main)
    ctx.barrier()

    def put(path, value):
        if ctx.main:
            write_json(path, value)
    s = plan["settings"]
    torch.set_num_threads(s["cpu_threads"])
    random.seed(s["seed"])
    torch.manual_seed(s["seed"])
    device = ctx.device(s["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("A GPU is required for formal training")
    precision = resolve_precision(device, s["precision"])
    torch.set_float32_matmul_precision("high")
    family = family_of(config)
    model = family.model_class(family.config_class(**config["model"])).to(device)
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
    if ctx.main:
        output.mkdir(parents=True, exist_ok=True)
    put(output / "config.json", config)
    put(output / "plan.json", plan)
    if config["runtime"]["compile_backbone"]:
        if getattr(model, "packed_training", False):
            from vimeml.training.model_zh800m_arch2 import compile_backbone
        else:
            from vimeml.training.runtime_v2 import compile_backbone
        compilation = compile_backbone(model)
    else:
        compilation = None
    net = ctx.wrap(model, device)
    environment = {"data_parallel_ranks": ctx.world, "torch": str(torch.__version__), "cuda": torch.version.cuda, "device": str(device),
                   "precision": precision, "parameters": model.parameter_count(), "initialization": initialization,
                   "compilation": compilation, "validation_precision": "primitive_FP32",
                   "smoke_updates": s.get("smoke_updates", 0)}
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
    put(output / "environment.json", environment)
    request = plan["request"]
    lm_train = family.data.training_dataset(root / config["data_mixture"]["directory"], root)
    web, chat = request["web"], request["chat"]
    web_val = SentenceWindowDataset(root / web["token_dir"], root / web["index_dir"], "validation")
    chat_val = SentenceWindowDataset(root / chat["token_dir"], root / chat["index_dir"], "validation")
    web_train = SentenceWindowDataset(root / web["token_dir"], root / web["index_dir"], "train")
    probe_ids = np.load(root / config["data_mixture"]["directory"] / plan["probe"]["file"], allow_pickle=False)
    options = {"workers": s["num_workers"], "multiplier": s["bucket_multiplier"], "pin_memory": device.type == "cuda"}
    lm_loader = loader_for(lm_train, s["lm_batch"], seed=s["seed"], max_width=model.config.context_length,
                           **options)
    limit = s.get("smoke_validation_windows", 0)
    joint = plan.get("pinyin")
    pinyin_stream = pinyin_sets = None
    if joint:
        from torch.utils.data import DataLoader
        from vimeml.training import joint_zh_pinyin as JP
        state["totals"].setdefault("pinyin", {k: {"examples": 0, "supervised_tokens": 0, "loss_sum": 0.0}
                                              for k in joint["mix"] if joint["mix"][k]})
        pinyin_dir = root / joint["directory"]
        schedule = JP.Schedule(pinyin_dir, {"plain": 0, **joint["mix"]}, s["seed"], s["prefix_crop_probability"])
        end = s.get("smoke_updates", 0) or s["max_steps"]
        extra = {"num_workers": s["num_workers"], "pin_memory": device.type == "cuda"}
        if s["num_workers"]:
            extra.update(multiprocessing_context="spawn", prefetch_factor=4, persistent_workers=True)
        pinyin_stream = iter(DataLoader(schedule, batch_size=None, sampler=range(state["step"], end), **extra))
        pinyin_sets = {}
        for kind in PINYIN_KINDS:
            store = JP.Store(pinyin_dir / kind, "validation")
            count = min(len(store), joint["validation_examples"])
            chosen = np.random.default_rng([s["seed"], 11, PINYIN_KINDS.index(kind)]).choice(len(store), count,
                                                                                             replace=False)
            pinyin_sets[kind] = (store, sorted(int(i) for i in chosen))

    def pinyin_validation(step, tag):
        with fp32_evaluation(model):
            report = JP.evaluate(model, pinyin_sets, s["validation_batch"], device, limit)
        for kind, r in report.items():
            writer.add_scalar(f"{kind}/{tag}_loss", r["loss"], step)
            writer.add_scalar(f"{kind}/{tag}_tf_exact", r["tf_exact"], step)
        print(f"[pinyin {tag} step {step}] " + " ".join(
            f"{k}={r['loss']:.4f}/{r['tf_exact']:.3f}" for k, r in report.items()), flush=True)
        return report

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
                               shuffle=False, pin_memory=device.type == "cuda",
                               max_width=model.config.context_length)
    if ctx.main:
        writer = SummaryWriter(str(log_dir), purge_step=state["step"] + 1 if resume else None, flush_secs=5)
        metrics = (output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
    else:
        writer, metrics = NullWriter(), open(os.devnull, "w", encoding="utf-8")
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

    saver = BackgroundSaver()

    def save(name="last", background=False):
        if not ctx.main:
            return
        saver.wait()
        payload = checkpoint_payload(model, optimizer, state, config, plan, precision, initialization)
        payload["training_regime"] = ("random_initialized_zh_joint_lm_chat_pinyin_pretraining" if joint
                                      else "random_initialized_zh_lm_chat_pretraining")
        if background:
            saver.submit(output / f"{name}.pt", payload)
        else:
            atomic_checkpoint(output / f"{name}.pt", payload)

    def milestone(k):
        began = time.perf_counter()
        save()
        put(output / "progress.json", progress(state, plan, "validating"))
        report = ctx.broadcast(measure(k))      # every rank validates its share; rank 0 adds pinyin
        state["milestones"].append(report)
        if not ctx.main:
            return
        write_json(output / "milestones.json", state["milestones"])
        for name in full:
            value = report[name].get("bpc", report[name]["loss"])
            writer.add_scalar(f"{name}/full_validation_{'bpc' if 'bpc' in report[name] else 'loss'}",
                              value, state["step"])
        if k in KEEP_MILESTONES:
            save(f"milestone-{k:02d}")
        save()
        print(f"[milestone {k}] " + " ".join(f"{n}={report[n].get('bpc', report[n]['loss']):.6f}" for n in full)
              + f" validation={report['validation_seconds']:.0f}s total={time.perf_counter() - began:.0f}s", flush=True)

    def measure(k):
        if ctx.main:
            print(f"[milestone {k}/{MILESTONES}] full web, chat and probe validation", flush=True)
        before, began = rng_state(), time.perf_counter()
        report = {"milestone": k, "step": state["step"], "epoch_position": state["epoch"] + (
            state["epoch_batch"] / s["updates_per_epoch"][state["epoch"]] if state["epoch"] < s["epochs"] else 0),
            "supervised_tokens": state["totals"]["lm"]["supervised_tokens"]}
        with fp32_evaluation(model):
            for name, (loader, pairs, characters) in full.items():
                if limit:
                    report[name] = shared_validate(model, loader, device, ctx)
                else:
                    report[name] = shared_validate(model, loader, device, ctx, pairs, characters)
        report["lm_validation_seconds"] = round(time.perf_counter() - began, 1)
        if joint and ctx.main:
            report["pinyin"] = pinyin_validation(state["step"], "milestone")
        report["validation_seconds"] = round(time.perf_counter() - began, 1)
        restore_rng(before)
        return report

    def pending():
        for k in due_milestones(state["step"], plan):
            if k > len(state["milestones"]):
                milestone(k)
    if restored is not None:
        restore_rng(restored)
    put(output / "progress.json", progress(state, plan, "running" if resume else "initializing"))
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
                fetched = time.perf_counter()
                rate = learning_rate(state["step"] + 1, s)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                pinyin = None
                if joint:
                    extra = next(pinyin_stream)
                    fetched_pinyin = time.perf_counter()
                    result = update_joint(model, optimizer, {"lm": lm_batch, **extra},
                                          joint["weights"], s, device, precision, ctx, net)
                    pinyin = result.pop("pinyin")
                    for kind, r in pinyin.items():
                        totals = state["totals"]["pinyin"][kind]
                        totals["examples"] += r["examples"]
                        totals["supervised_tokens"] += r["supervised_tokens"]
                        totals["loss_sum"] += r["loss"] * r["supervised_tokens"]
                else:
                    result = update_lm(model, optimizer, [lm_batch], s, device, precision, ctx, net)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                state["step"] += 1
                state["epoch_batch"] += 1
                state["lm_cursor"] = lm_stream.state_dict()
                seconds = time.perf_counter() - began
                state["training_seconds"] += seconds
                # Step breakdown on every rank (GPU idle until the first forward: data, then packing on the CPU;
                # "gpu" is the rest, including waits for slower ranks in the gradient all-reduce).
                timing = result.pop("timing", {})
                timing.update(lm_data=fetched - began, step=seconds)
                if joint:
                    timing["pinyin_data"] = fetched_pinyin - fetched
                timing["gpu"] = seconds - timing["lm_data"] - timing.get("pinyin_data", 0) - timing.get("pack", 0)
                requested = ctx.any(requested, device)      # every rank stops after the same update
                for field in state["totals"]["lm"]:
                    state["totals"]["lm"][field] += result[field]
                if state["step"] % s["log_every"] == 0:
                    # every rank's breakdown at log steps (one small all-reduce, same steps on all ranks);
                    # packed stages also report each rank's pass count and padded positions, which is what
                    # the balanced share in distributed.Context controls.
                    fields = ["lm_data", "pinyin_data", "pack", "gpu"]
                    fields += [name for name in ("passes", "padded") if name in timing]
                    fields += ["step"]
                    slots = [0.0] * (len(fields) * ctx.world)
                    for f, name in enumerate(fields):
                        slots[ctx.rank * len(fields) + f] = float(timing.get(name, 0.0))
                    slots = ctx.sum(slots, device)
                    timing["ranks"] = {name: [round(slots[r * len(fields) + f], 3) for r in range(ctx.world)]
                                       for f, name in enumerate(fields)}
                    if compilation is not None:
                        from torch._dynamo.utils import counters
                        timing["compiled_graphs"] = int(counters["stats"]["unique_graphs"])
                if state["step"] % s["log_every"] == 0 and ctx.main:
                    latest = {"step": state["step"], "learning_rate": rate, "seconds": seconds, **result,
                              "timing": {k: round(v, 4) if isinstance(v, float) else v for k, v in timing.items()}}
                    if pinyin is not None:
                        latest["pinyin"] = pinyin
                        for kind, r in pinyin.items():
                            writer.add_scalar(f"{kind}/train_loss", r["loss"], state["step"])
                    metrics.write(json.dumps(latest) + "\n")
                    writer.add_scalar("lm/train_loss", result["loss"], state["step"])
                    writer.add_scalar("lm/supervised_tokens", state["totals"]["lm"]["supervised_tokens"], state["step"])
                    writer.add_scalar("optimizer/learning_rate", rate, state["step"])
                    writer.add_scalar("optimizer/grad_norm", result["grad_norm"], state["step"])
                    put(output / "progress.json", progress(state, plan, latest=latest))
                    print(f"step={state['step']}/{s['max_steps']} epoch={e + 1}/{s['epochs']} "
                          f"lm={result['loss']:.4f} seconds={seconds:.3f}", flush=True)
                if state["step"] % s["checkpoint_every"] == 0:
                    blocked = time.perf_counter()
                    save(background=True)       # waits for the previous write, then copies to the CPU
                    if ctx.main:
                        print(f"[checkpoint step {state['step']}] blocked={time.perf_counter() - blocked:.1f}s", flush=True)
                if state["step"] % s["eval_every"] == 0:
                    evaluated = time.perf_counter()
                    with fp32_evaluation(model):
                        subset_result = shared_validate(model, subset_loader, device, ctx)
                    writer.add_scalar("web/fixed_subset_loss", subset_result["loss"], state["step"])
                    if joint and ctx.main:
                        before = rng_state()
                        subset_result["pinyin"] = pinyin_validation(state["step"], "validation")
                        restore_rng(before)
                    subset_result["seconds"] = round(time.perf_counter() - evaluated, 1)
                    put(output / "subset-validation.json", {"step": state["step"], **subset_result})
                    if ctx.main:
                        print(f"[subset step {state['step']}] loss={subset_result['loss']:.4f} "
                              f"seconds={subset_result['seconds']:.0f}", flush=True)
                pending()
                net.train()
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
        if compilation is not None:
            from torch._dynamo.utils import counters
            result["compiled_graphs"] = int(counters["stats"]["unique_graphs"])
        if device.type == "cuda":
            result["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / 2**20
            result["peak_reserved_MiB"] = torch.cuda.max_memory_reserved(device) / 2**20
        put(output / "summary.json", result)
        put(output / "progress.json", result)
        return result
    except BaseException as error:
        try:
            saver.wait()            # a pending last.pt finishes (or reports its own error) first
        except BaseException as pending_error:
            print(f"Background checkpoint failed: {pending_error!r}", flush=True)
        put(output / "failure.json", {"status": "failed", "active": True, "error_type": type(error).__name__,
                                             "error": str(error), "last_committed_in_memory_step": state["step"],
                                             "safe_resume_checkpoint": str(output / "last.pt")})
        raise
    finally:
        writer.close()
        metrics.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        lm_train.close()
        ctx.close()
        for data in (web_val, chat_val, web_train):
            data.close()
