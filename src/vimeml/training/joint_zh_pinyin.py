"""Stage 1 of the zh three-task model: continue the 12K 20M on pinyin-conditioned data.

Starts from a finished vimeml-tiny-zh-v1 checkpoint (weights only, fresh AdamW) and trains
on the five stores of build_zh_pinyin.py. Every update takes a fixed number of samples
from each store (data.mix, e.g. plain 256 / full 77 / mixed 77 / initials 77 / error 25).
Store k reads position p = step * mix[k] + j of an endless stream made of seeded
permutations, one per pass, so the batch of any step is a pure function of (seed, step)
and resuming needs nothing but the step.

plain windows keep the original 0.30 prefix crop (first windows only, at least 8 tokens
left) and supervise every target; pinyin samples supervise output + EOS only.
normalization "tokens": loss = sum of token losses / supervised tokens in the update
(every token equal, plain dominates); "per_kind": sum_k (mix[k] / batch) * mean loss of k.

Validation (FP32) at step 0 and every eval_every: loss per store, and for pinyin stores
tf_exact = share of samples whose every output token (and EOS) is the argmax under
teacher forcing. Checkpoints: last.pt every checkpoint_every, a weights-only
step-XXXXXX.pt every keep_every, final.pt at the end.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from vimeml.training.data import file_sha, write_json
from vimeml.training.model_zh13 import SHAPES, GPTZh13Config, TinyGPTZh13
from vimeml.training.train import optimizer_for, learning_rate, atomic_checkpoint, resolve_precision, amp_context
from vimeml.posttraining.streams import TaskCollator

ROOT = Path(__file__).resolve().parents[3]
STAGE = "zh-pinyin-s1"
ARCHITECTURE = "tiny_gpt_zh13"
FORMAT = "vimeml_tiny_gpt_zh13"
DATA_FORMAT = "vimeml_zh_pinyin_v1"
STORES = ("plain", "full", "mixed", "initials", "error")
CROP_MINIMUM = 8
SOURCE_FILES = ("src/vimeml/training/joint_zh_pinyin.py", "src/vimeml/training/model_zh13.py",
                "src/vimeml/training/model_v2.py", "src/vimeml/training/train.py",
                "src/vimeml/training/runtime_v2.py", "src/vimeml/posttraining/streams.py",
                "scripts/training/train_zh_pinyin.py")


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


class Store:
    def __init__(self, directory, split):
        self.directory, self.split = Path(directory), split
        self.ids = np.memmap(self.directory / f"{split}.tokens.bin", mode="r", dtype="<u2")
        self.offsets = np.fromfile(self.directory / f"{split}.offsets.bin", dtype="<u8")
        self.first = np.fromfile(self.directory / f"{split}.first.bin", dtype="u1")
        if len(self.offsets) != len(self.first) + 1 or int(self.offsets[-1]) != len(self.ids):
            raise ValueError(f"{directory}/{split} storage size mismatch")

    def __len__(self):
        return len(self.first)

    def get(self, index):
        return self.ids[int(self.offsets[index]):int(self.offsets[index + 1])].tolist(), int(self.first[index])


def shifted(ids, first, crop=0):
    """first == 1: LM window (every target); otherwise output + EOS only."""
    if first == 1:
        ids = ids[crop:]
        return {"input_ids": ids[:-1], "labels": ids[1:]}
    return {"input_ids": ids[:-1], "labels": [-100] * (first - 1) + ids[first:]}


class Schedule:
    """dataset[step] -> {store: collated batch}; picklable for spawn workers."""

    def __init__(self, directory, mix, seed, crop_probability):
        self.directory, self.mix, self.seed, self.crop_probability = Path(directory), dict(mix), seed, crop_probability
        self._open()

    def _open(self):
        self.stores = {k: Store(self.directory / k, "train") for k in STORES if self.mix[k]}
        self.cache = {}
        self.collate = TaskCollator(128)

    def __getstate__(self):
        return {"directory": self.directory, "mix": self.mix, "seed": self.seed, "crop_probability": self.crop_probability}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def permutation(self, kind, epoch):
        key = (kind, epoch)
        if key not in self.cache:
            self.cache = {k: v for k, v in self.cache.items() if k[0] != kind}
            n = len(self.stores[kind])
            self.cache[key] = np.random.default_rng([self.seed, STORES.index(kind), epoch]).permutation(n)
        return self.cache[key]

    def indices(self, kind, step):
        n, count = len(self.stores[kind]), self.mix[kind]
        out = []
        for p in range(step * count, (step + 1) * count):
            out.append(int(self.permutation(kind, p // n)[p % n]))
        return out

    def __len__(self):
        return 1 << 40

    def __getitem__(self, step):
        rng = np.random.default_rng([self.seed, 1 << 20, step])
        batches = {}
        for kind, store in self.stores.items():
            samples = []
            for index in self.indices(kind, step):
                ids, first = store.get(index)
                crop = 0
                if (first == 1 and ids[0] == 2 and len(ids) - 1 - CROP_MINIMUM >= 1
                        and rng.random() < self.crop_probability):
                    crop = 1 + int(rng.integers(len(ids) - 1 - CROP_MINIMUM))
                samples.append(shifted(ids, first, crop))
            batches[kind] = self.collate(samples)
        return batches

    def epochs(self, step):
        return {k: step * self.mix[k] / len(s) for k, s in self.stores.items()}


def build_plan(config, root=ROOT, verify_files=True):
    root = Path(root)
    if config.get("stage") != STAGE or config.get("architecture") != ARCHITECTURE:
        raise ValueError("Not a zh-pinyin-s1 configuration")
    model = GPTZh13Config(**config["model"])
    if model.vocab_size != 12288:
        raise ValueError("Stage 1 continues the 12K model")
    s = dict(config["training"])
    for key in ("max_steps", "warmup_steps", "num_workers", "cpu_threads", "log_every", "eval_every",
                "checkpoint_every", "keep_every", "validation_batch", "validation_plain_examples"):
        if type(s[key]) is not int or s[key] < (0 if key == "warmup_steps" else 1):
            raise ValueError(f"Invalid training integer: {key}")
    if s["normalization"] not in ("tokens", "per_kind"):
        raise ValueError("normalization is tokens or per_kind")
    smoke = s.get("smoke_updates", 0) > 0
    compiled = config["runtime"]["compile_backbone"]
    if not ((s["device"] == "cuda" and s["precision"] == "bf16" and compiled)
            or (s["device"] == "cpu" and s["precision"] == "fp32" and not compiled and smoke)):
        raise ValueError("Formal training is CUDA/BF16/compiled; only smoke runs are CPU/FP32")
    if not 0 < s["min_learning_rate"] <= s["learning_rate"]:
        raise ValueError("Invalid learning rates")
    mix = {k: int(config["data"]["mix"].get(k, 0)) for k in STORES}
    if any(v < 0 for v in mix.values()) or not sum(mix.values()):
        raise ValueError("Invalid mix")
    init = config["initialization"]
    if init.get("mode") != "checkpoint":
        raise ValueError("Stage 1 starts from a checkpoint")
    data_dir = root / config["data"]["directory"]
    manifest = read(data_dir / "manifest.json")
    if manifest.get("format") != DATA_FORMAT or manifest.get("status") != "complete":
        raise ValueError("Incomplete or unsupported pinyin data")
    required = {data_dir / "manifest.json": config["data"]["manifest_sha256"],
                root / init["checkpoint"]: init["checkpoint_sha256"]}
    for split, stores in manifest["stores"].items():
        for kind, record in stores.items():
            for name, digest in record["files"].items():
                required[data_dir / kind / name] = digest
    if file_sha(data_dir / "manifest.json") != config["data"]["manifest_sha256"]:
        raise ValueError("Pinyin data manifest changed")
    if verify_files:
        for path, digest in required.items():
            if file_sha(path) != digest:
                raise ValueError("Changed training dependency: " + str(path))
    code = {name: file_sha(ROOT / name) for name in SOURCE_FILES}
    signatures = {"data": config["data"]["manifest_sha256"], "source": init["checkpoint_sha256"],
                  "code": hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()}
    batch = sum(mix.values())
    train = {k: manifest["stores"]["train"][k]["examples"] for k in STORES}
    passes = {k: s["max_steps"] * mix[k] / train[k] for k in STORES if mix[k]}
    return {"stage": STAGE, "parameters": SHAPES[12288][1], "settings": s, "mix": mix, "batch": batch,
            "signatures": signatures, "code_sha256": code, "train_examples": train, "passes": passes,
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "required_files": {Path(p).resolve().as_posix(): h for p, h in required.items()}}


def initial_state():
    return {"step": 0, "training_seconds": 0.0, "evaluations": [],
            "totals": {k: {"examples": 0, "supervised_tokens": 0, "loss_sum": 0.0} for k in STORES}}


def update(model, optimizer, batches, plan, device, precision):
    s = plan["settings"]
    tokens = {k: int((b["labels"] != -100).sum()) for k, b in batches.items()}
    total = sum(tokens.values())
    if total <= 0:
        raise ValueError("No supervised targets")
    optimizer.zero_grad(set_to_none=True)
    sums = {}
    for kind, batch in batches.items():
        if not tokens[kind]:
            continue
        ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with amp_context(device, precision):
            loss_sum = model(ids, labels)["loss_sum"]
            if s["normalization"] == "tokens":
                loss = loss_sum / total
            else:
                loss = loss_sum * (plan["mix"][kind] / plan["batch"]) / tokens[kind]
        loss.backward()
        sums[kind] = float(loss_sum.detach())
    if not all(math.isfinite(v) for v in sums.values()):
        raise FloatingPointError("Non-finite training loss")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip"], error_if_nonfinite=True))
    optimizer.step()
    return {"loss": sum(sums.values()) / total, "grad_norm": grad_norm, "supervised_tokens": total,
            "kinds": {k: {"loss": sums[k] / tokens[k], "supervised_tokens": tokens[k],
                          "examples": len(batches[k]["lengths"])} for k in sums}}


def validation_sets(directory, plan, seed):
    out = {}
    for kind in STORES:
        store = Store(Path(directory) / kind, "validation")
        indices = range(len(store))
        if kind == "plain" and len(store) > plan["settings"]["validation_plain_examples"]:
            rng = np.random.default_rng([seed, 7])
            indices = sorted(rng.choice(len(store), plan["settings"]["validation_plain_examples"], replace=False))
        out[kind] = (store, [int(i) for i in indices])
    return out


@torch.no_grad()
def evaluate(model, sets, batch_size, device, limit=0):
    collate, report = TaskCollator(128), {}
    for kind, (store, indices) in sets.items():
        if limit:
            indices = indices[:limit]
        nll, count, exact, examples = 0.0, 0, 0, 0
        for start in range(0, len(indices), batch_size):
            samples = [shifted(*store.get(i)) for i in indices[start:start + batch_size]]
            batch = collate(samples)
            ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            logits = model(ids).float()
            valid = labels != -100
            losses = torch.nn.functional.cross_entropy(logits[valid], labels[valid], reduction="sum")
            nll += float(losses)
            count += int(valid.sum())
            hit = (logits.argmax(-1) == labels) | ~valid
            exact += int(hit.all(dim=1).sum())
            examples += len(samples)
        report[kind] = {"loss": nll / count, "supervised_tokens": count, "examples": examples}
        if kind != "plain":
            report[kind]["tf_exact"] = exact / examples
    return report


def load_source(model, path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("architecture") != ARCHITECTURE or saved.get("model_config") != model.configuration():
        raise ValueError("Source checkpoint is not the 12K vimeml-tiny-zh-v1 shape")
    model.load_state_dict(saved["model"], strict=True)
    return {"step": saved.get("step"), "format": saved.get("format"),
            "pretraining": saved.get("pretraining", {}).get("stage")}


def checkpoint_payload(model, optimizer, state, config, plan, precision, source):
    return {"format": FORMAT, "architecture": ARCHITECTURE, "model": model.state_dict(),
            "model_config": model.configuration(), "optimizer": optimizer.state_dict(), "config": config,
            "signatures": plan["signatures"], "precision": precision, "step": state["step"],
            "training_state": copy.deepcopy(state), "posttraining": {"stage": STAGE, "mix": plan["mix"]},
            "initialization": {"mode": "checkpoint", "source": source, "optimizer": "fresh_AdamW"},
            "tokenizer_identity": {"tokenizer_model_sha256": plan["tokenizer_sha256"], "vocab_size": 12288},
            "scheduler": {"step": state["step"], "max_steps": plan["settings"]["max_steps"], "name": "cosine"}}


def run(config, resume=False, root=ROOT, log=print):
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from vimeml.posttraining.infer import fp32_evaluation
    root = Path(root)
    output, log_dir = root / config["output_dir"], root / config["log_dir"]
    if not resume and ((output.exists() and any(output.iterdir())) or (log_dir.exists() and any(log_dir.iterdir()))):
        raise ValueError("A fresh run requires empty output and log directories")
    log("Verifying the source checkpoint and pinyin data...")
    plan = build_plan(config, root, verify_files=True)
    s = plan["settings"]
    torch.set_num_threads(s["cpu_threads"])
    torch.manual_seed(s["seed"])
    device = torch.device(s["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("A GPU is required for formal training")
    precision = resolve_precision(device, s["precision"])
    torch.set_float32_matmul_precision("high")
    model = TinyGPTZh13(GPTZh13Config(**config["model"]))
    source = load_source(model, root / config["initialization"]["checkpoint"])
    model.to(device)
    optimizer = optimizer_for(model, s, device)
    state = initial_state()
    if resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if (saved.get("format") != FORMAT or saved["config"] != config or saved["signatures"] != plan["signatures"]
                or saved["precision"] != precision or saved["step"] != saved["training_state"]["step"]):
            raise ValueError("Resume requires identical config, code, data and precision")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        state = copy.deepcopy(saved["training_state"])
        del saved
    smoke = s.get("smoke_updates", 0)
    end = smoke or s["max_steps"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "plan.json", plan)
    compilation = None
    if config["runtime"]["compile_backbone"]:
        from vimeml.training.runtime_v2 import compile_backbone
        compilation = compile_backbone(model)
    environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device": str(device),
                   "precision": precision, "parameters": model.parameter_count(), "source": source,
                   "compilation": compilation, "smoke_updates": smoke}
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
    write_json(output / "environment.json", environment)
    data_dir = root / config["data"]["directory"]
    schedule = Schedule(data_dir, plan["mix"], s["seed"], s["prefix_crop_probability"])
    sets = validation_sets(data_dir, plan, s["seed"])
    limit = s.get("smoke_validation_examples", 0)
    writer = SummaryWriter(str(log_dir), purge_step=state["step"] + 1 if resume else None, flush_secs=5)
    metrics = (output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
    requested = False
    handlers = {}

    def request_stop(signum, frame):
        nonlocal requested
        if requested:
            raise KeyboardInterrupt("Second interrupt; keep last committed checkpoint")
        requested = True
        log("Saving after the current update")
    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_stop)

    def save(name="last"):
        atomic_checkpoint(output / f"{name}.pt", checkpoint_payload(model, optimizer, state, config, plan,
                                                                    precision, source))

    def keep():
        atomic_checkpoint(output / f"step-{state['step']:06d}.pt",
                          {"format": FORMAT, "architecture": ARCHITECTURE, "model": model.state_dict(),
                           "model_config": model.configuration(), "step": state["step"],
                           "posttraining": {"stage": STAGE, "mix": plan["mix"]}, "signatures": plan["signatures"]})

    def evaluation():
        if any(r["step"] == state["step"] for r in state["evaluations"]):
            return
        with fp32_evaluation(model):
            report = evaluate(model, sets, s["validation_batch"], device, limit)
        report = {"step": state["step"], **report}
        state["evaluations"].append(report)
        write_json(output / "evaluations.json", state["evaluations"])
        for kind in STORES:
            writer.add_scalar(f"{kind}/validation_loss", report[kind]["loss"], state["step"])
            if "tf_exact" in report[kind]:
                writer.add_scalar(f"{kind}/validation_tf_exact", report[kind]["tf_exact"], state["step"])
        log(f"[eval step {state['step']}] " + " ".join(
            f"{k}={report[k]['loss']:.4f}" + (f"/{report[k]['tf_exact']:.3f}" if "tf_exact" in report[k] else "")
            for k in STORES))
        model.train()

    options = {"num_workers": s["num_workers"], "pin_memory": device.type == "cuda"}
    if s["num_workers"]:
        options.update(multiprocessing_context="spawn", prefetch_factor=4, persistent_workers=True)
    try:
        evaluation()
        if not resume:
            save()
        loader = DataLoader(schedule, batch_size=None, sampler=range(state["step"], end), **options)
        model.train()
        for batches in loader:
            if requested:
                break
            began = time.perf_counter()
            rate = learning_rate(state["step"] + 1, s)
            for group in optimizer.param_groups:
                group["lr"] = rate
            result = update(model, optimizer, batches, plan, device, precision)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            state["step"] += 1
            seconds = time.perf_counter() - began
            state["training_seconds"] += seconds
            for kind, r in result["kinds"].items():
                totals = state["totals"][kind]
                totals["examples"] += r["examples"]
                totals["supervised_tokens"] += r["supervised_tokens"]
                totals["loss_sum"] += r["loss"] * r["supervised_tokens"]
            if state["step"] % s["log_every"] == 0:
                latest = {"step": state["step"], "learning_rate": rate, "seconds": seconds, **result}
                metrics.write(json.dumps(latest) + "\n")
                writer.add_scalar("train/loss", result["loss"], state["step"])
                for kind, r in result["kinds"].items():
                    writer.add_scalar(f"{kind}/train_loss", r["loss"], state["step"])
                writer.add_scalar("optimizer/learning_rate", rate, state["step"])
                writer.add_scalar("optimizer/grad_norm", result["grad_norm"], state["step"])
                rate_per_s = state["step"] / state["training_seconds"]
                write_json(output / "progress.json", {
                    "status": "running", "step": state["step"], "max_steps": end, "latest": latest,
                    "passes": schedule.epochs(state["step"]),
                    "training_only_eta_seconds": (end - state["step"]) / rate_per_s if rate_per_s else None})
                log(f"step={state['step']}/{end} loss={result['loss']:.4f} " + " ".join(
                    f"{k}={r['loss']:.3f}" for k, r in result["kinds"].items()) + f" seconds={seconds:.3f}")
            if state["step"] % s["eval_every"] == 0 or state["step"] == end:
                evaluation()
            if state["step"] % s["keep_every"] == 0 or state["step"] == end:
                keep()
            if state["step"] % s["checkpoint_every"] == 0:
                save()
        terminal = ("smoke_complete" if smoke else "complete") if state["step"] == end else "interrupted"
        save()
        if terminal != "interrupted":
            save("final")
        result = {"status": terminal, "step": state["step"], "max_steps": end, "totals": state["totals"],
                  "training_seconds": state["training_seconds"], "evaluations": state["evaluations"],
                  "checkpoints": sorted(p.name for p in output.glob("*.pt"))}
        if device.type == "cuda":
            result["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / 2**20
        write_json(output / "summary.json", result)
        write_json(output / "progress.json", result)
        return result
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                                              "error": str(error), "last_committed_step": state["step"],
                                              "safe_resume_checkpoint": str(output / "last.pt")})
        raise
    finally:
        writer.close()
        metrics.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
