"""Step 0 of the zh-800m arch2 plan (docs/training-zh-800m-arch2-plan.md): where an update's time goes.

Three measurements, none of which trains a model to keep:

1. updates: the trainer's own update loop (joint_zh200m.update_joint on the config's frozen data, model,
   optimizer, compilation and data parallelism) for a few updates from random weights, timed per update.
   With --emulate-world N in one process, every update runs the rows rank 0 of an N-rank run would get
   (same share, same packing), with no communication: the single-process control. The difference from
   the real N-rank run is the cost of communication and synchronization.
2. allreduce: the effective bandwidth of an FP32 all-reduce of the model's gradient size, as one tensor
   and as DDP-sized buckets (run with NCCL_DEBUG=INFO to see the transport).
3. --profile K on `updates`: torch.profiler over K updates on rank 0. The trace's GPU kernels are split per
   update into compute, exposed communication (NCCL kernels not overlapped by compute) and GPU idle; host
   work is the measured data wait, packing time and the CUDA synchronization calls.

It also runs arch2 configs (stage zh-800m-arch2), so the arch2 20-update smoke can use the same timing.
Output: summary.json (and trace-rank0.json with --profile) in the output directory, written by rank 0.
"""
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch

from vimeml.training import joint_zh200m as J
from vimeml.training.distributed import Context, setup as setup_parallel
from vimeml.training.train import optimizer_for, learning_rate, resolve_precision
from vimeml.posttraining.streams import CommittedStream, loader_for

# Synchronizations inside an update (.item(), .tolist(), copies to the host). cudaDeviceSynchronize is left out:
# it is the timing barrier at the end of every measured update, not the trainer's own work.
SYNC_CALLS = ("cudaStreamSynchronize", "cudaEventSynchronize")


class EmulatedRank(Context):
    """Rank 0 of a `world`-rank run inside one process: the same data share and loss scaling, no process
    group, no DDP and no collective (sums and flags are this rank's own)."""

    def __init__(self, world):
        super().__init__(world, 0, 0)

    def device(self, kind):
        return torch.device("cuda", torch.cuda.current_device()) if kind == "cuda" else torch.device(kind)

    def wrap(self, model, device):
        return model

    def no_sync(self, module, last):
        return nullcontext()

    def sum(self, values, device):
        return list(values)

    def any(self, flag, device):
        return bool(flag)

    def broadcast(self, value):
        return value

    def barrier(self):
        pass

    def close(self):
        pass


def union(intervals):
    """Total length of the union of [start, end) intervals."""
    total, end = 0.0, -math.inf
    for a, b in sorted(intervals):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def clip(intervals, start, end):
    return [(max(a, start), min(b, end)) for a, b in intervals if b > start and a < end]


def breakdown(trace):
    """Per-update time split from a chrome trace (torch.profiler export) with one "update" annotation per
    update. Times in seconds; compute + exposed_communication + gpu_idle = wall."""
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    updates, compute, comm, syncs, memcpy = [], [], [], [], []
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        span = (float(e["ts"]), float(e["ts"]) + float(e["dur"]))
        cat, name = e.get("cat", ""), e.get("name", "")
        if cat in ("user_annotation", "gpu_user_annotation") and name == "update":
            if cat == "user_annotation":
                updates.append(span)
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            (comm if "nccl" in name.lower() else compute).append(span)
        elif cat == "cuda_runtime" and name in SYNC_CALLS:
            syncs.append(span)
        elif cat == "cuda_runtime" and name == "cudaMemcpyAsync":
            memcpy.append(span)
    rows = []
    for start, end in sorted(updates):
        wall = end - start
        c, n = clip(compute, start, end), clip(comm, start, end)
        busy_compute, busy_comm, busy = union(c), union(n), union(c + n)
        rows.append({"wall": wall / 1e6, "compute": busy_compute / 1e6,
                     "communication": busy_comm / 1e6, "exposed_communication": (busy - busy_compute) / 1e6,
                     "gpu_idle": (wall - busy) / 1e6, "host_sync_calls": union(clip(syncs, start, end)) / 1e6,
                     "sync_call_count": len(clip(syncs, start, end)),
                     "host_memcpy_calls": union(clip(memcpy, start, end)) / 1e6})
    if not rows:
        return {"updates": 0}
    mean = {k: statistics.fmean(r[k] for r in rows) for k in rows[0]}
    share = {k: mean[k] / mean["wall"] for k in ("compute", "exposed_communication", "gpu_idle",
                                                  "host_sync_calls")}
    return {"updates": len(rows), "per_update": rows, "mean_seconds": mean, "share_of_wall": share}


def timed(function, bucket, shapes=None):
    """Wrap a packing function: its seconds go to `bucket`, the [rows, width] of each pass it returns to `shapes`."""
    def wrapper(*args, **kwargs):
        began = time.perf_counter()
        try:
            out = function(*args, **kwargs)
        finally:
            bucket.append(time.perf_counter() - began)
        if shapes is not None:
            shapes.extend(list(work[0].shape) for work in out)
        return out
    return wrapper


def graph_count():
    from torch._dynamo.utils import counters
    return int(counters["stats"]["unique_graphs"])


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))] if values else None


def run_updates(config, out, updates, warmup, profile=0, emulate_world=0, root=J.ROOT):
    """Time `updates` updates after `warmup` untimed ones (compilation); profile the last `profile` on rank 0."""
    root, out = Path(root), Path(out)
    if emulate_world:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        ctx = EmulatedRank(emulate_world)
    else:
        ctx = setup_parallel(config["training"]["device"])
    plan = J.build_plan(config, root, verify_files=False)
    s = plan["settings"]
    total = warmup + updates
    if total > s["updates_per_epoch"][0]:
        raise ValueError("Step 0 stays inside the first epoch")
    if profile > updates:
        raise ValueError("--profile counts updates inside the timed ones")
    torch.set_num_threads(s["cpu_threads"])
    random.seed(s["seed"])
    torch.manual_seed(s["seed"])
    device = ctx.device(s["device"])
    precision = resolve_precision(device, s["precision"])
    torch.set_float32_matmul_precision("high")
    family = J.family_of(config)
    model = family.model_class(family.config_class(**config["model"])).to(device)
    optimizer = optimizer_for(model, s, device)
    compilation = None
    if config["runtime"]["compile_backbone"]:
        if getattr(model, "packed_training", False):
            from vimeml.training.model_zh800m_arch2 import compile_backbone
        else:
            from vimeml.training.runtime_v2 import compile_backbone
        compilation = compile_backbone(model)
    net = ctx.wrap(model, device)
    lm_train = family.data.training_dataset(root / config["data_mixture"]["directory"], root)
    lm_loader = loader_for(lm_train, s["lm_batch"], seed=s["seed"], max_width=model.config.context_length,
                           workers=s["num_workers"], multiplier=s["bucket_multiplier"],
                           pin_memory=device.type == "cuda")
    lm_stream = CommittedStream(lm_loader, {"epoch": 0, "batch": 0, "offset": 0})
    joint = plan.get("pinyin")
    pinyin_stream = None
    if joint:
        from torch.utils.data import DataLoader
        from vimeml.training import joint_zh_pinyin as JP
        schedule = JP.Schedule(root / joint["directory"], {"plain": 0, **joint["mix"]}, s["seed"],
                               s["prefix_crop_probability"])
        extra = {"num_workers": s["num_workers"], "pin_memory": device.type == "cuda"}
        if s["num_workers"]:
            extra.update(multiprocessing_context="spawn", prefetch_factor=4, persistent_workers=True)
        pinyin_stream = iter(DataLoader(schedule, batch_size=None, sampler=range(0, total), **extra))
    packing, shapes = [], []
    saved = J.packed_rows, J.segment_microbatches
    J.packed_rows, J.segment_microbatches = timed(J.packed_rows, packing, shapes), timed(J.segment_microbatches, packing, shapes)
    records = []
    profiler = None
    model.train()
    try:
        for step in range(1, total + 1):
            if profile and ctx.main and step == total - profile + 1:
                from torch.profiler import profile as torch_profile, ProfilerActivity
                activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
                profiler = torch_profile(activities=activities)
                profiler.__enter__()
            ctx.barrier()
            began = time.perf_counter()
            batches = {"lm": lm_stream.next_batch()}
            if joint:
                batches.update(next(pinyin_stream))
            fetched = time.perf_counter()
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, s)
            del packing[:], shapes[:]
            annotation = torch.profiler.record_function("update") if profiler else nullcontext()
            with annotation:
                if joint:
                    result = J.update_joint(model, optimizer, batches, joint["weights"], s, device, precision, ctx,
                                            net)
                    result.pop("pinyin")
                else:
                    result = J.update_lm(model, optimizer, [batches["lm"]], s, device, precision, ctx, net)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            done = time.perf_counter()
            if emulate_world:   # padded_positions is rank 0's alone, so count rank 0's share of the inputs
                inputs = sum(int(p["lengths"].sum()) for p in map(ctx.share, batches.values()) if p is not None)
            else:
                inputs = sum(int(b["lengths"].sum()) for b in batches.values())
            records.append({"step": step, "seconds": done - began, "update_seconds": done - fetched,
                            "data_wait_seconds": fetched - began, "packing_seconds": sum(packing),
                            "padded_positions": result["padded_positions"], "input_tokens_all_tasks": inputs,
                            "lm_loss": result["loss"], "pass_shapes": [list(x) for x in shapes],
                            "compiled_graphs": graph_count() if compilation is not None else None,
                            "alloc_retries": (torch.cuda.memory_stats(device).get("num_alloc_retries", 0)
                                              if device.type == "cuda" else None)})
            if ctx.main:
                print(f"step {step}/{total} {'warmup' if step <= warmup else 'timed'} "
                      f"{done - began:.3f}s (update {done - fetched:.3f}s)", flush=True)
        if profiler:
            profiler.__exit__(None, None, None)
    finally:
        J.packed_rows, J.segment_microbatches = saved
        lm_train.close()
    timed_rows = records[warmup:]
    summary = None
    if ctx.main:
        out.mkdir(parents=True, exist_ok=True)
        seconds = [r["seconds"] for r in timed_rows]
        summary = {
            "stage": plan["stage"], "arm": plan["arm"], "parameters": model.parameter_count(),
            "lm_batch": s["lm_batch"], "lm_microbatch": s["lm_microbatch"],
            "mode": f"emulated rank 0 of {emulate_world}" if emulate_world else f"{ctx.world} ranks",
            "world": ctx.world, "warmup_updates": warmup, "timed_updates": len(timed_rows),
            "seconds_median": statistics.median(seconds), "seconds_p90": percentile(seconds, 0.9),
            "seconds_max": max(seconds),
            "update_seconds_median": statistics.median(r["update_seconds"] for r in timed_rows),
            "data_wait_seconds_median": statistics.median(r["data_wait_seconds"] for r in timed_rows),
            "packing_seconds_median": statistics.median(r["packing_seconds"] for r in timed_rows),
            "padding_ratio": sum(r["padded_positions"] for r in timed_rows)
                             / sum(r["input_tokens_all_tasks"] for r in timed_rows),
            "warmup_seconds": [round(r["seconds"], 3) for r in records[:warmup]],
            "updates_after_first_over_2s": sum(1 for r in records[1:] if r["seconds"] > 2),
            "compilation": compilation, "torch": torch.__version__, "records": records,
        }
        if emulate_world:
            summary["padding_ratio_note"] = "padded positions of rank 0's share over that share's input tokens"
        else:
            summary["padding_ratio_note"] = "padded positions summed over ranks over all input tokens"
        if compilation is not None:
            summary["compiled_graphs"] = graph_count()
        slow = [r for r in records[1:] if r["seconds"] > 2]
        summary["slow_updates"] = [{k: r[k] for k in ("step", "seconds", "pass_shapes", "compiled_graphs",
                                                      "alloc_retries")} for r in slow]
        timed_seconds = [r["seconds"] for r in timed_rows]
        summary["seconds_mean"] = statistics.fmean(timed_seconds)
        if device.type == "cuda":
            summary["gpu"] = torch.cuda.get_device_name(device)
            summary["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / 2**20
            summary["peak_reserved_MiB"] = torch.cuda.max_memory_reserved(device) / 2**20
        if profiler:
            path = out / "trace-rank0.json"
            profiler.export_chrome_trace(str(path))
            split = breakdown(json.loads(path.read_text("utf-8")))
            host = {"data_wait": statistics.fmean(r["data_wait_seconds"] for r in records[-profile:]),
                    "packing": statistics.fmean(r["packing_seconds"] for r in records[-profile:])}
            split["host_seconds"] = host
            mean_wall = statistics.fmean(r["seconds"] for r in records[-profile:])
            # host work proper: waiting for data and packing. Sync calls are the CPU waiting for the GPU,
            # reported apart (host_sync_calls); they overlap GPU work and do not add to it.
            split["host_share_of_update"] = (host["data_wait"] + host["packing"]) / mean_wall
            summary["profile"] = split
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    ctx.close()
    return summary


def run_allreduce(out, gigabytes, iterations, bucket_mb):
    """FP32 all-reduce bandwidth: one tensor of `gigabytes`, then the same bytes as bucket_mb buckets (DDP's
    default is 25 MB). busbw = algbw * 2 (n - 1) / n, the NCCL-tests convention."""
    import torch.distributed as dist
    ctx = setup_parallel("cuda")
    if not ctx.active:
        raise ValueError("allreduce runs under torchrun with at least 2 processes")
    device = ctx.device("cuda")
    count = int(gigabytes * 1e9 // 4)
    tensor = torch.ones(count, dtype=torch.float32, device=device)
    factor = 2 * (ctx.world - 1) / ctx.world
    results = {}
    for label, chunk in (("single_tensor", count), (f"buckets_{bucket_mb}MB", int(bucket_mb * 2**20 // 4))):
        pieces = list(tensor.split(chunk))
        times = []
        for i in range(iterations + 2):
            torch.cuda.synchronize(device)
            dist.barrier()
            began = time.perf_counter()
            for piece in pieces:
                dist.all_reduce(piece)
            torch.cuda.synchronize(device)
            if i >= 2:
                times.append(time.perf_counter() - began)
        median = statistics.median(times)
        results[label] = {"pieces": len(pieces), "seconds_median": median, "seconds_min": min(times),
                          "algbw_GBps": count * 4 / median / 1e9, "busbw_GBps": count * 4 / median / 1e9 * factor}
    summary = None
    if ctx.main:
        summary = {"world": ctx.world, "bytes": count * 4, "iterations": iterations,
                   "gpu": torch.cuda.get_device_name(device), "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                   "results": results}
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "allreduce.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=1), flush=True)
    ctx.close()
    return summary
