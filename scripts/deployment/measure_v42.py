"""Same-harness Mac measurements for V4.2 and V2.1 FP32/INT8 Core ML packages.

Stages:
  prepare-v21-kv  Measurement-only V2.1 KV packages from the deployed V2.1 bundle,
                  using the unchanged `kv_cache` converter and the INT8 block32 recipe.
  run             Compile every package with `coremlcompiler`, then measure each model
                  in fresh worker processes, interleaved across models.
Mac CPU_ONLY Python host-process numbers; they do not substitute for iPhone measurements.
"""
import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

DEPLOY = ROOT / "artifacts/deployment"
V21_KV = {"fp32": DEPLOY / "tiny-ja-v2.1-extend5-kv-fp32-mac-v42-measure-v1",
          "int8": DEPLOY / "tiny-ja-v2.1-extend5-kv-int8-b32-mac-v42-measure-v1"}
MODELS = [
    # name, family, graph, package directory, vocab, cache shape
    ("v42-fp32", "V4.2", "stateless", DEPLOY / "tiny-ja-v4.2-epoch20-fp32-v1", 8192, None),
    ("v42-int8", "V4.2", "stateless", DEPLOY / "tiny-ja-v4.2-epoch20-int8-b32-v1", 8192, None),
    ("v21-fp32", "V2.1", "stateless", DEPLOY / "tiny-ja-v2.1-extend5-fp32-v1", 16384, None),
    ("v21-int8", "V2.1", "stateless", DEPLOY / "tiny-ja-v2.1-extend5-int8-b32-v1", 16384, None),
    ("v42-kv-fp32", "V4.2", "kv_single_token", DEPLOY / "tiny-ja-v4.2-epoch20-kv-fp32-v1", 8192, (7, 1, 6, 128, 64)),
    ("v42-kv-int8", "V4.2", "kv_single_token", DEPLOY / "tiny-ja-v4.2-epoch20-kv-int8-b32-v1", 8192, (7, 1, 6, 128, 64)),
    ("v21-kv-fp32", "V2.1", "kv_chunk", V21_KV["fp32"], 16384, (6, 1, 5, 128, 64)),
    ("v21-kv-int8", "V2.1", "kv_chunk", V21_KV["int8"], 16384, (6, 1, 5, 128, 64)),
]
LENGTHS = (1, 8, 32, 64, 128)
PREFIXES = (7, 31, 63, 119)


class RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
        "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
        "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups",
        "ri_child_interrupt_wkups", "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
        "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
        "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
        "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
        "ri_serviced_system_time", "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions",
        "ri_cycles", "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
        "ri_runnable_time")]


def memory():
    info = RusageInfoV4()
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    mib = 1024 * 1024
    return {"phys_footprint_mib": info.ri_phys_footprint / mib, "resident_mib": info.ri_resident_size / mib,
            "lifetime_peak_footprint_mib": info.ri_lifetime_max_phys_footprint / mib}


def stats(values):
    return {"n": len(values), "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(min(values)), "max_ms": float(max(values))}


def timed(work):
    started = time.perf_counter()
    work()
    return (time.perf_counter() - started) * 1000


def worker(name, graph, compiled, vocab, cache_shape, repeats, warmup, seed):
    before_import = memory()
    import coremltools as ct
    after_import = memory()
    started = time.perf_counter()
    model = ct.models.CompiledMLModel(str(compiled), compute_units=ct.ComputeUnit.CPU_ONLY)
    load_ms = (time.perf_counter() - started) * 1000
    after_load = memory()
    rng = np.random.default_rng(seed)
    result = {"name": name, "graph": graph, "pid": os.getpid(), "load_ms": load_ms}
    if graph == "stateless":
        def call(length):
            ids = rng.integers(7, vocab, (1, length)).astype(np.int32)
            ids[0, 0] = 2
            return lambda: model.predict({"input_ids": ids})
        # Cold: first call after load at T=32, then the first call at each other length.
        cold = {"first_call_T32_ms": timed(call(32))}
        cold.update({f"first_call_T{t}_ms": timed(call(t)) for t in LENGTHS if t != 32})
        after_cold = memory()
        warm = {}
        for length in LENGTHS:
            work = call(length)
            for _ in range(warmup):
                work()
            warm[f"T{length}"] = stats([timed(work) for _ in range(repeats)])
    else:
        zeros = np.zeros(cache_shape, dtype=np.float32)

        def step(token, position, key, value):
            if graph == "kv_single_token":
                out = model.predict({"input_ids": np.array([[token]], np.int32), "position": np.array([position], np.int32),
                                     "key_cache": key, "value_cache": value})
            else:
                out = model.predict({"input_ids": np.array([[token]], np.int32), "cache_length": np.array([position], np.int32),
                                     "key_cache": key, "value_cache": value})
            return out["new_key_cache"], out["new_value_cache"]

        def prefill(ids):
            if graph == "kv_single_token":
                key, value = zeros, zeros
                for position, token in enumerate(ids):
                    key, value = step(int(token), position, key, value)
                return key, value
            out = model.predict({"input_ids": np.asarray([ids], np.int32), "cache_length": np.array([0], np.int32),
                                 "key_cache": zeros, "value_cache": zeros})
            return out["new_key_cache"], out["new_value_cache"]

        cold = {"first_call_decode_position0_ms": timed(lambda: step(2, 0, zeros, zeros))}
        after_cold = memory()
        warm = {}
        for prefix in PREFIXES:
            ids = rng.integers(7, vocab, prefix).astype(np.int32)
            ids[0] = 2
            key, value = prefill(ids)
            token = int(rng.integers(7, vocab))
            decode = lambda: step(token, prefix, key, value)
            for _ in range(warmup):
                decode()
            warm[f"decode_after_{prefix}"] = stats([timed(decode) for _ in range(repeats)])
            fill = lambda: prefill(ids)
            fill()
            warm[f"prefill_{prefix}"] = stats([timed(fill) for _ in range(max(5, repeats // 5))])
    after_workload = memory()
    result.update(cold=cold, warm=warm, memory={"before_import": before_import, "after_import": after_import,
        "after_load": after_load, "after_cold_calls": after_cold, "after_workload": after_workload,
        "load_delta_footprint_mib": after_load["phys_footprint_mib"] - after_import["phys_footprint_mib"],
        "peak_delta_footprint_mib": after_workload["lifetime_peak_footprint_mib"] - after_import["phys_footprint_mib"]})
    print(json.dumps(result))


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def conditions():
    def run(*command):
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            return f"unavailable: {error}"
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "loadavg": run("sysctl", "-n", "vm.loadavg"),
            "pmset_batt": run("pmset", "-g", "batt"), "pmset_therm": run("pmset", "-g", "therm"),
            "lowpowermode": [line.strip() for line in run("pmset", "-g").splitlines() if "lowpowermode" in line.lower()]}


def prepare_v21_kv():
    """Rebuild V2.1 KV (FP32 + INT8 block32) for timing only; gate on same-weight stateless packages."""
    import torch
    from vimeml.deployment import kv_cache as kv, v2
    from vimeml.deployment.coreml import coremltools, inspect_spec
    from vimeml.deployment.validation import numeric_error
    from vimeml.training.data import file_sha
    bundle = DEPLOY / "tiny-ja-v2.1-extend5-bundle-v1"
    v2.verify_bundle(bundle)
    if any(path.exists() for path in V21_KV.values()):
        raise ValueError("Measurement KV output exists; choose a new version.")
    kv.convert(bundle, V21_KV["fp32"])
    ct = coremltools()
    from coremltools.optimize import coreml as opt
    source = kv.verify_package(V21_KV["fp32"])
    model = ct.models.MLModel(str(V21_KV["fp32"] / "model.mlpackage"), skip_model_load=True)
    config = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_block",
                                        block_size=32, weight_threshold=2048)
    optimization, selection = v2.parameter_selection(opt, model, config)
    result = opt.linear_quantize_weights(model, config=optimization)
    if not any(name.startswith("constexpr_") for name in inspect_spec(result)["operations"]):
        raise ValueError("No compressed weights found.")
    kv.finish(V21_KV["int8"], result, {key: source[key] for key in (
        "minimum_ios", "bundle", "cache_shape", "cache_dtype", "compute_units", "cache_bytes_per_snapshot",
        "interface", "policy")} | {"kind": "linear8_fp32_compute", "measurement_only": True,
        "source_manifest_sha256": file_sha(V21_KV["fp32"] / "manifest.json"),
        "compression": {"method": "linear_symmetric", "bits": 8, "block_size": 32, "selection": selection},
        "conversion_source_sha256": file_sha(Path(__file__))})
    # Same-weight gate: KV prefill/decode equals the deployed stateless package within 3e-4.
    rng = np.random.default_rng(716)
    ids = [2, *rng.integers(4, 16384, 127).tolist()]
    report = {"format": "vimeml_v21_kv_measurement_gate_v1", "checks": [],
              "note": "Measurement-only rebuild; deployed V2.1 KV release assets are not part of this payload."}
    for kind, stateless in (("fp32", "tiny-ja-v2.1-extend5-fp32-v1"), ("int8", "tiny-ja-v2.1-extend5-int8-b32-v1")):
        runtime, old = kv.Runtime(bundle, V21_KV[kind]), v2.CoreMLLM(bundle, DEPLOY / stateless)
        expected = old.model(torch.tensor([ids])).numpy()
        one_shot, _ = runtime.step(ids)
        first, state = runtime.step(ids[:31])
        rest = [first]
        for position in range(31, 128):
            logits, state = runtime.step(ids[position:position + 1], state, position)
            rest.append(logits)
        decoded = np.concatenate(rest, axis=1)
        for check, actual in (("one_shot", one_shot), ("prefill31_then_single_token", decoded)):
            report["checks"].append({"kind": kind, "check": check, "stateless": stateless,
                                     **numeric_error(expected, actual, 3e-4, 3e-4)})
    report["passed"] = all(item["outside_tolerance"] == 0 for item in report["checks"])
    out = ROOT / "outputs/deployment/v21-kv-mac-v42-measure-gate-v1"
    out.mkdir(parents=True, exist_ok=False)
    (out / "gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "max_abs": max(c["max_abs"] for c in report["checks"])}))
    if not report["passed"]:
        raise SystemExit("V2.1 KV measurement gate failed; evidence preserved.")


def run(output, rounds, repeats, warmup):
    from vimeml.deployment.bundle import environment
    output.mkdir(parents=True, exist_ok=False)
    compiled_root = output / "compiled"
    packages = {}
    for name, family, graph, directory, vocab, cache in MODELS:
        package = directory / "model.mlpackage"
        destination = compiled_root / name
        destination.mkdir(parents=True)
        started = time.perf_counter()
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(destination)], check=True,
                       capture_output=True)
        compiled = destination / "model.mlmodelc"
        packages[name] = {"family": family, "graph": graph, "package": str(directory.relative_to(ROOT)),
                          "package_bytes": tree_bytes(package), "compiled_bytes": tree_bytes(compiled),
                          "compile_seconds": time.perf_counter() - started, "compiled": str(compiled)}
    records = {name: [] for name, *_ in MODELS}
    timeline = [{"phase": "start", **conditions()}]
    for index in range(rounds):
        for name, family, graph, directory, vocab, cache in MODELS:
            command = [sys.executable, __file__, "worker", "--name", name, "--graph", graph,
                       "--compiled", packages[name]["compiled"], "--vocab", str(vocab),
                       "--repeats", str(repeats), "--warmup", str(warmup), "--seed", str(716 + index)]
            if cache:
                command += ["--cache-shape", ",".join(map(str, cache))]
            line = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]
            record = json.loads(line)
            record["round"] = index
            records[name].append(record)
            print(f"round {index} {name}: load {record['load_ms']:.1f} ms", flush=True)
        timeline.append({"phase": f"after_round_{index}", **conditions()})

    def summary(name):
        items = records[name]
        first = items[0]
        warm_keys = first["warm"].keys()
        return {**{k: v for k, v in packages[name].items() if k != "compiled"},
                "load_ms_first_process": first["load_ms"], "load_ms_all": [r["load_ms"] for r in items],
                "load_ms_p50": float(np.median([r["load_ms"] for r in items])),
                "cold_first_process": first["cold"],
                "cold_p50_over_processes": {k: float(np.median([r["cold"][k] for r in items])) for k in first["cold"]},
                "warm_p50_of_process_p50_ms": {k: float(np.median([r["warm"][k]["p50_ms"] for r in items])) for k in warm_keys},
                "warm_p50_of_process_p95_ms": {k: float(np.median([r["warm"][k]["p95_ms"] for r in items])) for k in warm_keys},
                "memory_mib_p50": {key: float(np.median([r["memory"][key] for r in items]))
                                   for key in ("load_delta_footprint_mib", "peak_delta_footprint_mib")},
                "footprint_after_load_mib_p50": float(np.median([r["memory"]["after_load"]["phys_footprint_mib"] for r in items])),
                "lifetime_peak_footprint_mib_p50": float(np.median([r["memory"]["after_workload"]["lifetime_peak_footprint_mib"] for r in items])),
                "footprint_after_import_mib_p50": float(np.median([r["memory"]["after_import"]["phys_footprint_mib"] for r in items]))}

    hardware = {key: subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
                for key in ("machdep.cpu.brand_string", "hw.memsize", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu")}
    report = {"format": "vimeml_v42_mac_measurements_v1", "environment": environment(),
              "macos": platform.mac_ver()[0], "macos_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
              "xcode": subprocess.run(["xcodebuild", "-version"], capture_output=True, text=True).stdout.strip(),
              "hardware": hardware, "compute_units": "CPU_ONLY",
              "workload": {"rounds": rounds, "repeats": repeats, "warmup": warmup, "stateless_lengths": LENGTHS,
                           "kv_prefixes": PREFIXES, "process_order": "fresh process per model per round; models interleaved"},
              "conditions": timeline, "models": {name: summary(name) for name, *_ in MODELS}, "raw": records,
              "scope": "Mac CPU_ONLY Core ML through coremltools CompiledMLModel in a Python host process. Footprint includes the Python/coremltools runtime; deltas isolate the model. Not iPhone, simulator, or keyboard-extension measurements."}
    (output / "measurements.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    stages.add_parser("prepare-v21-kv")
    measure = stages.add_parser("run")
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--rounds", type=int, default=5)
    measure.add_argument("--repeats", type=int, default=50)
    measure.add_argument("--warmup", type=int, default=5)
    work = stages.add_parser("worker")
    for name in ("name", "graph", "compiled"):
        work.add_argument("--" + name, required=True)
    for name in ("vocab", "repeats", "warmup", "seed"):
        work.add_argument("--" + name, type=int, required=True)
    work.add_argument("--cache-shape")
    args = parser.parse_args()
    if args.stage == "prepare-v21-kv":
        prepare_v21_kv()
    elif args.stage == "run":
        if args.output.exists():
            parser.error("Fresh output path required.")
        run(args.output, args.rounds, args.repeats, args.warmup)
    else:
        shape = tuple(map(int, args.cache_shape.split(","))) if args.cache_shape else None
        worker(args.name, args.graph, args.compiled, args.vocab, shape, args.repeats, args.warmup, args.seed)


if __name__ == "__main__":
    main()
