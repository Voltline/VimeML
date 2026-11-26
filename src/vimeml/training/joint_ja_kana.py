"""ja-gen stage 2: train the 20M and 200M on the ja-kana-v1-12k store (encode_kana.py).

One trainer for every ja-gen run:
  pretraining from scratch ([initialization] mode = "random"): the 20M plain LM
      (plain + chat), the 200M joint reading pretraining (web + plain + chat + reading kinds);
  continuation (mode = "checkpoint", weights only, fresh AdamW): the 20M reading arm and its
      LM control.
architecture "tiny_gpt_zh13" (12K/d384/L9, about 20M) or "tiny_gpt_zh200m" (12K/L15/c128,
193.6M); checkpoints carry the zh13 / zh200m formats, so the zh export, KD-teacher and
evaluation loaders read them.

Data: every update takes data.mix[k] samples of each kind k. Kind k reads position
p = step * mix[k] + j of an endless stream; pass e = p // n is a seeded bijection of the n
samples (a Feistel network with cycle walking: no permutation array, so the 600M-window web
store costs no memory), so the batch of any step is a pure function of (seed, step) and a
resume needs only the step. With data parallelism (torchrun) rank r reads the positions
j = r, r + world, ... of every kind; the per-kind supervised-token counts are summed over
the ranks before the loss is scaled, so the gradient is the one-device gradient.

Plain kinds (first = 1: plain, chat, web) supervise every target with the 0.30 prefix crop
of first windows; reading kinds supervise output + EOS only. normalization "tokens": loss =
sum of token losses / supervised tokens of the update; "per_kind": sum_k (mix[k] / batch) x
mean token loss of k.

Validation (FP32, rank 0) at step 0, every eval_every and at the end: loss per kind on at most
validation_examples samples per kind (seeded choice), and teacher-forced exact match for
reading kinds. Checkpoints: last.pt every checkpoint_every (rank 0), weights-only
step-XXXXXXX.pt every keep_every, final.pt at the end.
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
from vimeml.training.distributed import Context, NullWriter, setup as setup_parallel
from vimeml.training.model_zh13 import GPTZh13Config, TinyGPTZh13
from vimeml.training import model_zh200m
from vimeml.training.train import optimizer_for, learning_rate, atomic_checkpoint, resolve_precision, amp_context
from vimeml.posttraining.streams import TaskCollator

ROOT = Path(__file__).resolve().parents[3]
STAGE = "ja-kana"
DATA_FORMAT = "vimeml_ja_kana_v1"
KINDS = ("plain", "chat", "web", "span", "sentence", "web_span", "web_sentence")
PLAIN_KINDS = ("plain", "chat", "web")
SPECIAL = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
CROP_MINIMUM = 8
FAMILIES = {"tiny_gpt_zh13": (GPTZh13Config, TinyGPTZh13, "vimeml_tiny_gpt_zh13"),
            "tiny_gpt_zh200m": (model_zh200m.GPTZh200mConfig, model_zh200m.TinyGPTZh200m, "vimeml_tiny_gpt_zh200m")}
SOURCE_FILES = ("src/vimeml/training/joint_ja_kana.py", "src/vimeml/training/model_zh13.py",
                "src/vimeml/training/model_zh200m.py", "src/vimeml/training/model_v2.py",
                "src/vimeml/training/train.py", "src/vimeml/training/runtime_v2.py",
                "src/vimeml/training/distributed.py", "src/vimeml/posttraining/streams.py",
                "scripts/ja/training/train_ja_kana.py")


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


class Permutation:
    """A seeded bijection of range(n): a 4-round Feistel network on 2 * half bits, cycle-walked into n."""

    def __init__(self, n, seed):
        self.n = int(n)
        self.half = max(1, ((self.n - 1).bit_length() + 1) // 2)
        self.mask = np.uint64((1 << self.half) - 1)
        self.keys = [np.uint64(k) for k in np.random.default_rng(seed).integers(0, 1 << 63, size=4)]

    def _round(self, x):
        h = x * np.uint64(0x9E3779B97F4A7C15)
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
        return h

    def _feistel(self, x):
        shift = np.uint64(self.half)
        left, right = x >> shift, x & self.mask
        for k in self.keys:
            left, right = right, left ^ (self._round(right ^ k) & self.mask)
        return (left << shift) | right

    def __call__(self, positions):
        x = np.asarray(positions, dtype=np.uint64)
        with np.errstate(over="ignore"):
            out = self._feistel(x)
            outside = out >= np.uint64(self.n)
            while outside.any():
                out[outside] = self._feistel(out[outside])
                outside = out >= np.uint64(self.n)
        return out.astype(np.int64)


class Store:
    def __init__(self, directory, split):
        self.directory, self.split = Path(directory), split
        self.ids = np.memmap(self.directory / f"{split}.tokens.bin", mode="r", dtype="<u2")
        self.offsets = np.memmap(self.directory / f"{split}.offsets.bin", mode="r", dtype="<u8")
        self.first = np.memmap(self.directory / f"{split}.first.bin", mode="r", dtype="u1")
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
    """dataset[step] -> {kind: collated batch of this rank's rows}; picklable for spawn workers."""

    def __init__(self, directory, mix, seed, crop_probability, rank=0, world=1):
        self.directory, self.mix, self.seed = Path(directory), dict(mix), seed
        self.crop_probability, self.rank, self.world = crop_probability, rank, world
        self._open()

    def _open(self):
        self.stores = {k: Store(self.directory / k, "train") for k in KINDS if self.mix.get(k, 0)}
        self.permutations = {}
        self.collate = TaskCollator(128)

    def __getstate__(self):
        return {k: getattr(self, k) for k in ("directory", "mix", "seed", "crop_probability", "rank", "world")}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def permutation(self, kind, epoch):
        key = (kind, epoch)
        if key not in self.permutations:
            self.permutations = {k: v for k, v in self.permutations.items() if k[0] != kind}
            self.permutations[key] = Permutation(len(self.stores[kind]), [self.seed, KINDS.index(kind), epoch])
        return self.permutations[key]

    def indices(self, kind, step):
        n, count = len(self.stores[kind]), self.mix[kind]
        positions = np.arange(step * count + self.rank, (step + 1) * count, self.world, dtype=np.int64)
        out = np.empty(len(positions), dtype=np.int64)
        epochs = positions // n
        for epoch in np.unique(epochs):
            chosen = epochs == epoch
            out[chosen] = self.permutation(kind, int(epoch))(positions[chosen] % n)
        return out.tolist()

    def __len__(self):
        return 1 << 40

    def __getitem__(self, step):
        rng = np.random.default_rng([self.seed, 1 << 20, step, self.rank])
        out = {}
        for kind, store in self.stores.items():
            group = []
            for index in self.indices(kind, step):
                ids, first = store.get(index)
                crop = 0
                if (first == 1 and ids[0] == 2 and len(ids) - 1 - CROP_MINIMUM >= 1
                        and rng.random() < self.crop_probability):
                    crop = 1 + int(rng.integers(len(ids) - 1 - CROP_MINIMUM))
                group.append(shifted(ids, first, crop))
            out[kind] = self.collate(group)
        return out

    def passes(self, step):
        return {kind: step * self.mix[kind] / len(store) for kind, store in self.stores.items()}


def build_plan(config, root=ROOT, verify_files=True, world=1):
    root = Path(root)
    if config.get("stage") != STAGE or config.get("architecture") not in FAMILIES:
        raise ValueError("Not a ja-kana configuration")
    configuration = FAMILIES[config["architecture"]][0](**config["model"])
    if configuration.vocab_size != 12288 or configuration.context_length != 128:
        raise ValueError("ja-kana models are 12K / context 128")
    s = dict(config["training"])
    for key in ("max_steps", "warmup_steps", "num_workers", "cpu_threads", "log_every", "eval_every",
                "checkpoint_every", "keep_every", "validation_batch", "validation_examples"):
        if type(s[key]) is not int or s[key] < (0 if key in ("warmup_steps", "num_workers") else 1):
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
    mix = {k: int(v) for k, v in config["data"]["mix"].items() if int(v)}
    if set(mix) - set(KINDS) or any(v < world for v in mix.values()):
        raise ValueError(f"data.mix names unknown kinds or a count below the {world} ranks")
    init = config["initialization"]
    if init.get("mode") not in ("random", "checkpoint"):
        raise ValueError("initialization.mode is random or checkpoint")
    data_dir = root / config["data"]["directory"]
    if file_sha(data_dir / "manifest.json") != config["data"]["manifest_sha256"]:
        raise ValueError("ja-kana store manifest changed or is not pinned")
    manifest = read(data_dir / "manifest.json")
    if manifest.get("format") != DATA_FORMAT or manifest.get("status") != "complete" or manifest.get("limit_jobs"):
        raise ValueError("Incomplete, smoke or unsupported ja-kana store")
    required = {data_dir / "manifest.json": config["data"]["manifest_sha256"]}
    if init["mode"] == "checkpoint":
        required[root / init["checkpoint"]] = init["checkpoint_sha256"]
    for split, stores in manifest["stores"].items():
        for kind, record in stores.items():
            if split == "validation" or kind in mix:
                for name, digest in record["files"].items():
                    required[data_dir / kind / name] = digest
    if verify_files:
        for path, digest in required.items():
            if file_sha(path) != digest:
                raise ValueError("Changed training dependency: " + str(path))
    train = {k: manifest["stores"]["train"][k]["examples"] for k in mix}
    if not all(train.values()):
        raise ValueError("A mixed kind has no training samples")
    code = {name: file_sha(ROOT / name) for name in SOURCE_FILES}
    signatures = {"data": config["data"]["manifest_sha256"], "source": init.get("checkpoint_sha256"),
                  "code": hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()}
    validation = [k for k in KINDS if manifest["stores"]["validation"].get(k, {}).get("examples")]
    model = FAMILIES[config["architecture"]][1](configuration)
    return {"stage": STAGE, "architecture": config["architecture"], "parameters": model.parameter_count(),
            "settings": s, "mix": mix, "batch": sum(mix.values()), "train_examples": train,
            "passes": {k: s["max_steps"] * v / train[k] for k, v in mix.items()},
            "validation_kinds": validation, "signatures": signatures, "code_sha256": code,
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "required_files": {Path(p).resolve().as_posix(): h for p, h in required.items()}}


def update(model, net, optimizer, batches, plan, device, precision, ctx):
    s = plan["settings"]
    local = {k: int((b["labels"] != -100).sum()) for k, b in batches.items()}
    tokens = dict(zip(local, (int(v) for v in ctx.sum([float(v) for v in local.values()], device))))
    total = sum(tokens.values())
    if total <= 0 or not all(tokens.values()):
        raise ValueError("A kind has no supervised targets")
    optimizer.zero_grad(set_to_none=True)
    sums = {}
    work = [k for k in batches if local[k]]
    for i, kind in enumerate(work):
        batch = batches[kind]
        ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with ctx.no_sync(net, i == len(work) - 1):
            with amp_context(device, precision):
                loss_sum = net(ids, labels)["loss_sum"]
                weight = 1 / total if s["normalization"] == "tokens" else plan["mix"][kind] / plan["batch"] / tokens[kind]
                loss = loss_sum * (weight * ctx.world)
            loss.backward()
        sums[kind] = float(loss_sum.detach())
    sums = dict(zip(sums, ctx.sum([sums.get(k, 0.0) for k in sums], device)))
    if not all(math.isfinite(v) for v in sums.values()):
        raise FloatingPointError("Non-finite training loss")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), s["grad_clip"], error_if_nonfinite=True))
    optimizer.step()
    return {"loss": sum(sums.values()) / total, "grad_norm": grad_norm, "supervised_tokens": total,
            "kinds": {k: {"loss": sums[k] / tokens[k], "supervised_tokens": tokens[k]} for k in sums}}


def validation_sets(directory, plan, seed):
    out = {}
    for kind in plan["validation_kinds"]:
        store = Store(Path(directory) / kind, "validation")
        indices = range(len(store))
        if len(store) > plan["settings"]["validation_examples"]:
            rng = np.random.default_rng([seed, 7, KINDS.index(kind)])
            indices = sorted(rng.choice(len(store), plan["settings"]["validation_examples"], replace=False))
        out[kind] = (store, [int(i) for i in indices])
    return out


@torch.no_grad()
def evaluate(model, sets, batch_size, device, limit=0):
    collate, report = TaskCollator(128), {}
    for kind, (store, indices) in sets.items():
        indices = indices[:limit] if limit else indices
        nll, count, exact = 0.0, 0, 0
        for start in range(0, len(indices), batch_size):
            batch = collate([shifted(*store.get(i)) for i in indices[start:start + batch_size]])
            ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            logits = model(ids).float()
            valid = labels != -100
            nll += float(torch.nn.functional.cross_entropy(logits[valid], labels[valid], reduction="sum"))
            count += int(valid.sum())
            exact += int(((logits.argmax(-1) == labels) | ~valid).all(dim=1).sum())
        report[kind] = {"loss": nll / count, "supervised_tokens": count, "examples": len(indices)}
        if kind not in PLAIN_KINDS:
            report[kind]["tf_exact"] = exact / len(indices)
    return report


def load_source(model, path, architecture):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("architecture") != architecture or saved.get("model_config") != model.configuration():
        raise ValueError("Source checkpoint has another architecture or shape")
    model.load_state_dict(saved["model"], strict=True)
    return {"step": saved.get("step"), "format": saved.get("format"), "stage": saved.get("stage"),
            "arm": (saved.get("config") or {}).get("arm")}


def run(config, resume=False, root=ROOT, log=print):
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from vimeml.posttraining.infer import fp32_evaluation
    root = Path(root)
    ctx = setup_parallel(config["training"]["device"])
    output, log_dir = root / config["output_dir"], root / config["log_dir"]
    if not resume and ((output.exists() and any(output.iterdir())) or (log_dir.exists() and any(log_dir.iterdir()))):
        raise ValueError("A fresh run requires empty output and log directories")
    ctx.barrier()
    if ctx.main:
        log("Verifying the ja-kana store and source...")
    plan = build_plan(config, root, verify_files=ctx.main, world=ctx.world)
    ctx.barrier()

    def put(path, value):
        if ctx.main:
            write_json(path, value)
    s = plan["settings"]
    torch.set_num_threads(s["cpu_threads"])
    torch.manual_seed(s["seed"])
    device = ctx.device(s["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("A GPU is required for formal training")
    precision = resolve_precision(device, s["precision"])
    torch.set_float32_matmul_precision("high")
    configuration, model_class, checkpoint_format = FAMILIES[config["architecture"]]
    model = model_class(configuration(**config["model"]))
    init = config["initialization"]
    source = {"mode": "random", "seed": s["seed"]}
    if init["mode"] == "checkpoint":
        source = {"mode": "checkpoint", **load_source(model, root / init["checkpoint"], config["architecture"])}
    model.to(device)
    optimizer = optimizer_for(model, s, device)
    state = {"step": 0, "training_seconds": 0.0, "evaluations": [],
             "totals": {k: {"supervised_tokens": 0, "loss_sum": 0.0} for k in plan["mix"]}}
    if resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if (saved.get("format") != checkpoint_format or saved["config"] != config
                or saved["signatures"] != plan["signatures"] or saved["precision"] != precision
                or saved.get("data_parallel_ranks", 1) != ctx.world):
            raise ValueError("Resume requires identical config, code, data, precision and rank count")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        state = copy.deepcopy(saved["training_state"])
        del saved
    smoke = s.get("smoke_updates", 0)
    end = smoke or s["max_steps"]
    if ctx.main:
        output.mkdir(parents=True, exist_ok=True)
    put(output / "config.json", config)
    put(output / "plan.json", plan)
    compilation = None
    if config["runtime"]["compile_backbone"]:
        from vimeml.training.runtime_v2 import compile_backbone
        compilation = compile_backbone(model)
    net = ctx.wrap(model, device)
    environment = {"data_parallel_ranks": ctx.world, "torch": str(torch.__version__), "cuda": torch.version.cuda,
                   "device": str(device), "precision": precision, "parameters": model.parameter_count(),
                   "source": source, "compilation": compilation, "smoke_updates": smoke}
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
    put(output / "environment.json", environment)
    data_dir = root / config["data"]["directory"]
    schedule = Schedule(data_dir, plan["mix"], s["seed"], s["prefix_crop_probability"], ctx.rank, ctx.world)
    sets = validation_sets(data_dir, plan, s["seed"]) if ctx.main else {}
    limit = s.get("smoke_validation_examples", 0)
    writer = SummaryWriter(str(log_dir), purge_step=state["step"] + 1 if resume else None,
                           flush_secs=30) if ctx.main else NullWriter()
    metrics = (output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) if ctx.main else None
    identity = {"tokenizer_model_sha256": plan["tokenizer_sha256"], "vocab_size": 12288, "special_ids": SPECIAL}
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

    def weights(name):
        return {"format": checkpoint_format, "architecture": config["architecture"], "stage": STAGE,
                "model": model.state_dict(), "model_config": model.configuration(), "step": state["step"],
                "config": config, "signatures": plan["signatures"], "tokenizer_identity": identity,
                "training": {"stage": STAGE, "arm": config.get("arm"), "mix": plan["mix"], "source": source}}

    def save(name="last"):
        if ctx.main:
            payload = weights(name)
            payload.update({"optimizer": optimizer.state_dict(), "precision": precision,
                            "training_state": copy.deepcopy(state), "data_parallel_ranks": ctx.world,
                            "scheduler": {"step": state["step"], "max_steps": s["max_steps"], "name": "cosine"}})
            atomic_checkpoint(output / f"{name}.pt", payload)
        ctx.barrier()

    def keep():
        if ctx.main:
            atomic_checkpoint(output / f"step-{state['step']:07d}.pt", weights("keep"))

    def evaluation():
        if any(r["step"] == state["step"] for r in state["evaluations"]):
            return
        report = None
        if ctx.main:
            with fp32_evaluation(model):
                report = {"step": state["step"], **evaluate(model, sets, s["validation_batch"], device, limit)}
            for kind in plan["validation_kinds"]:
                writer.add_scalar(f"{kind}/validation_loss", report[kind]["loss"], state["step"])
                if "tf_exact" in report[kind]:
                    writer.add_scalar(f"{kind}/validation_tf_exact", report[kind]["tf_exact"], state["step"])
            log(f"[eval step {state['step']}] " + " ".join(
                f"{k}={report[k]['loss']:.4f}" + (f"/{report[k]['tf_exact']:.3f}" if "tf_exact" in report[k] else "")
                for k in plan["validation_kinds"]))
        report = ctx.broadcast(report)
        state["evaluations"].append(report)
        put(output / "evaluations.json", state["evaluations"])
        net.train()

    options = {"num_workers": s["num_workers"], "pin_memory": device.type == "cuda"}
    if s["num_workers"]:
        options.update(multiprocessing_context="spawn", prefetch_factor=4, persistent_workers=True)
    try:
        evaluation()
        if not resume:
            save()
        loader = DataLoader(schedule, batch_size=None, sampler=range(state["step"], end), **options)
        net.train()
        for batches in loader:
            began = time.perf_counter()
            rate = learning_rate(state["step"] + 1, s)
            for group in optimizer.param_groups:
                group["lr"] = rate
            result = update(model, net, optimizer, batches, plan, device, precision, ctx)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            state["step"] += 1
            seconds = time.perf_counter() - began
            state["training_seconds"] += seconds
            for kind, r in result["kinds"].items():
                state["totals"][kind]["supervised_tokens"] += r["supervised_tokens"]
                state["totals"][kind]["loss_sum"] += r["loss"] * r["supervised_tokens"]
            if state["step"] % s["log_every"] == 0 and ctx.main:
                latest = {"step": state["step"], "learning_rate": rate, "seconds": seconds, **result}
                metrics.write(json.dumps(latest) + "\n")
                writer.add_scalar("train/loss", result["loss"], state["step"])
                for kind, r in result["kinds"].items():
                    writer.add_scalar(f"{kind}/train_loss", r["loss"], state["step"])
                writer.add_scalar("optimizer/learning_rate", rate, state["step"])
                writer.add_scalar("optimizer/grad_norm", result["grad_norm"], state["step"])
                speed = state["step"] / state["training_seconds"]
                put(output / "progress.json", {
                    "status": "running", "step": state["step"], "max_steps": end, "latest": latest,
                    "passes": schedule.passes(state["step"]),
                    "training_only_eta_seconds": (end - state["step"]) / speed if speed else None})
                log(f"step={state['step']}/{end} loss={result['loss']:.4f} " + " ".join(
                    f"{k}={r['loss']:.3f}" for k, r in result["kinds"].items()) + f" seconds={seconds:.3f}")
            if state["step"] % s["eval_every"] == 0 or state["step"] == end:
                evaluation()
            if state["step"] % s["keep_every"] == 0 or state["step"] == end:
                keep()
            if state["step"] % s["checkpoint_every"] == 0:
                save()
            if ctx.any(requested, device):
                break
        terminal = ("smoke_complete" if smoke else "complete") if state["step"] == end else "interrupted"
        save()
        if terminal != "interrupted":
            save("final")
        result = {"status": terminal, "step": state["step"], "max_steps": end, "totals": state["totals"],
                  "training_seconds": state["training_seconds"], "evaluations": state["evaluations"],
                  "checkpoints": sorted(p.name for p in output.glob("*.pt"))}
        if device.type == "cuda":
            result["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / 2**20
        put(output / "summary.json", result)
        put(output / "progress.json", result)
        return result
    except BaseException as error:
        put(output / "failure.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                                      "last_committed_step": state["step"],
                                      "safe_resume_checkpoint": str(output / "last.pt")})
        raise
    finally:
        writer.close()
        if metrics:
            metrics.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        ctx.close()
