"""Isolated Chinese IME research: real Rime pools and upstream Rust rerankers.

Weights, upstream clones, raw cases and outputs stay in ignored directories.
Run with the repository venv; PyYAML is loaded from the optional eval packages.
See the dated research report for setup commands and measurement boundaries.
"""
import argparse
import csv
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "artifacts/chinese-ime-research-20261017"
OUT = ROOT / "outputs/chinese-ime-research-20261017"
sys.path.insert(0, str(ROOT / "artifacts/external-eval/site-packages"))
sys.stdout.reconfigure(encoding="utf-8")


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def prepare(probes):
    import yaml
    upstream = EXP / "upstream"
    tsv = upstream / "chinese-ime-lm/eval/sentences-v1.tsv"
    lines = [line for line in tsv.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    cases = [{"id": row["id"], "input": row["input"], "gold": row["gold"], "context": "",
              "suite": "upstream_sentences", "tags": row["tags"]}
             for row in csv.DictReader(lines, delimiter="\t")]
    cases.extend(read_rows(probes))
    assert len({c["id"] for c in cases}) == len(cases)
    jsonl(EXP / "cases.jsonl", cases)
    (EXP / "inputs.tsv").write_text("".join(c["id"] + "\t" + c["input"] + "\n" for c in cases), encoding="utf-8")
    shared = EXP / "rime/shared"
    shared.mkdir(parents=True, exist_ok=True)
    for path in (upstream / "rime-prelude").glob("*.yaml"):
        shutil.copyfile(path, shared / path.name)
    shutil.copyfile(upstream / "rime-pinyin-simp/pinyin_simp.dict.yaml", shared / "pinyin_simp.dict.yaml")
    for path in (upstream / "rime-octagram-hans").glob("*.gram"):
        shutil.copyfile(path, shared / path.name)
    source = yaml.safe_load((upstream / "rime-pinyin-simp/pinyin_simp.schema.yaml").read_text(encoding="utf-8"))
    default = yaml.safe_load((shared / "default.yaml").read_text(encoding="utf-8"))
    default["schema_list"] = [{"schema": "research_plain"}, {"schema": "research_grammar"}]
    (shared / "default.yaml").write_text(yaml.safe_dump(default, allow_unicode=True), encoding="utf-8")
    for mode in ("plain", "grammar"):
        cfg = json.loads(json.dumps(source))
        cfg["schema"].update(schema_id="research_" + mode, name="research_" + mode, dependencies=[])
        cfg["engine"]["translators"] = ["punct_translator", "script_translator"]
        cfg["translator"].update(enable_user_dict=False, enable_completion=False,
                                 contextual_suggestions=(mode == "grammar"))
        if mode == "grammar":
            cfg["grammar"] = {"language": "zh-hans-t-essay-bgw"}
            cfg["translator"].update(max_homophones=7, max_homographs=7)
        (shared / f"research_{mode}.schema.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        (EXP / f"rime/user-{mode}").mkdir(parents=True, exist_ok=True)
    crate = EXP / "probe"
    (crate / "src").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "scripts/benchmarks/chinese_ime_probe.rs", crate / "src/main.rs")
    (crate / "Cargo.toml").write_text('''[package]
name = "vimeml-chinese-probe"
version = "0.1.0"
edition = "2021"
[dependencies]
chinese-ime-lm = { path = "../upstream/chinese-ime-lm/reference" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
''', encoding="utf-8")
    write(EXP / "frozen-inputs.json", {"cases_sha256": sha(EXP / "cases.jsonl"),
          "probes_sha256": sha(probes), "upstream_sentences_sha256": sha(tsv),
          "n": len(cases), "selection": "Both 4M and 25M selected before scoring; no threshold tuning."})
    print(f"Frozen {len(cases)} cases; prepared independent Rime workspaces and Rust adapter.")


def prepare_multi():
    import yaml
    shared = EXP / "rime-nbest/shared"
    shared.mkdir(parents=True, exist_ok=True)
    for path in (EXP / "rime/shared").iterdir():
        if path.is_file():
            shutil.copyfile(path, shared / path.name)
    for mode in ("plain", "grammar"):
        cfg = yaml.safe_load((shared / f"research_{mode}.schema.yaml").read_text(encoding="utf-8"))
        cfg["schema"].update(schema_id=f"research_multi_{mode}", name=f"research_multi_{mode}")
        cfg["translator"].update(max_sentences=9, sentence_cutoff_threshold=1.0)
        (shared / f"research_multi_{mode}.schema.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        (EXP / f"rime-nbest/user-{mode}").mkdir(parents=True, exist_ok=True)
    default = yaml.safe_load((shared / "default.yaml").read_text(encoding="utf-8"))
    default["schema_list"] = [{"schema": f"research_multi_{mode}"} for mode in ("plain", "grammar")]
    (shared / "default.yaml").write_text(yaml.safe_dump(default, allow_unicode=True), encoding="utf-8")
    print("Prepared separate multiple-sentence ablation; original candidate pools retained.")


def capture(multi=False):
    cases = read_rows(EXP / "cases.jsonl")
    # This is a dictionary-membership proxy, not Rime's internal candidate-source tag.
    dictionary = set()
    for line in (EXP / "rime/shared/pinyin_simp.dict.yaml").read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            dictionary.add((fields[0], fields[1].replace(" ", "").replace("'", "")))
    for mode in ("plain", "grammar"):
        rime_root = "rime-nbest" if multi else "rime"
        label = "nbest-" + mode if multi else mode
        schema = f"research_multi_{mode}" if multi else f"research_{mode}"
        command = [str(EXP / "capture.exe"), str(EXP / f"{rime_root}/shared"),
                   str(EXP / f"{rime_root}/user-{mode}"), schema, str(EXP / "inputs.tsv")]
        env = dict(os.environ)
        env["PATH"] = str(EXP / "librime/dist/lib") + os.pathsep + env["PATH"]
        proc = subprocess.run(command, capture_output=True, env=env, check=True)
        (OUT / f"rime-{label}-capture.tsv").write_bytes(proc.stdout)
        (OUT / f"rime-{label}-capture-stderr.txt").write_bytes(proc.stderr)
        pools = {c["id"]: [] for c in cases}
        for line in proc.stdout.decode("utf-8").splitlines():
            ident, index, text, complete = line.split("\t")
            assert int(index) == len(pools[ident])
            pools[ident].append({"text": text, "answers_key": complete == "1"})
        rows = []
        for case in cases:
            row = dict(case, candidates=pools[case["id"]])
            key = case["input"].replace("'", "")
            for candidate in row["candidates"]:
                candidate["trusted_dictionary_hit"] = candidate["answers_key"] and (candidate["text"], key) in dictionary
                candidate["source"] = 0 if candidate["trusted_dictionary_hit"] else 8
            rows.append(row)
        jsonl(OUT / f"rime-{label}-cases.jsonl", rows)
        print(label, "nonempty pools", sum(bool(r["candidates"]) for r in rows), "/", len(rows))


class ProcessMemory(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]


def working_set(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    handle = kernel.OpenProcess(0x410, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        memory = ProcessMemory()
        memory.cb = ctypes.sizeof(memory)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"working_set_bytes": memory.WorkingSetSize, "peak_working_set_bytes": memory.PeakWorkingSetSize}
    finally:
        kernel.CloseHandle(handle)


def percentile(values, p):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)]


def summarize(cases, scores):
    by_id = {r["id"]: r for r in scores}
    groups = sorted({c["suite"] for c in cases}) + ["all"]
    result = {}
    for group in groups:
        selected = [c for c in cases if group == "all" or c["suite"] == group]
        summary = {"n": len(selected), "engine": 0, "reachable_full_top9": 0, "gated": 0,
                   "ungated": 0, "no_context": 0, "rescued": 0, "broken": 0, "dictionary_leader": 0}
        latency = []
        for case in selected:
            cs = case["candidates"]
            if not cs:
                continue
            row = by_id[case["id"]]
            before = cs[0]["text"] == case["gold"] and cs[0]["answers_key"]
            after = cs[row["chosen"]]["text"] == case["gold"] and cs[row["chosen"]]["answers_key"]
            summary["engine"] += before
            summary["gated"] += after
            summary["reachable_full_top9"] += any(c["text"] == case["gold"] and c["answers_key"] for c in cs)
            for name in ("ungated", "no_context"):
                index = row[name]
                summary[name] += index is not None and cs[index]["text"] == case["gold"]
            summary["rescued"] += after and not before
            summary["broken"] += before and not after
            summary["dictionary_leader"] += cs[0]["trusted_dictionary_hit"]
            if cs[0]["answers_key"] and not cs[0]["trusted_dictionary_hit"] and sum(c["answers_key"] for c in cs) >= 2:
                latency.append(row["gated_ms"])
        summary["active_rerank_n"] = len(latency)
        summary["active_rerank_p50_ms"] = percentile(latency, .5)
        summary["active_rerank_p95_ms"] = percentile(latency, .95)
        result[group] = summary
    return result


def score(multi=False):
    manifest = json.loads((EXP / "models/manifest.json").read_text(encoding="utf-8-sig"))
    result = {}
    for item in manifest:
        model_name = item["id"].split("/")[-1]
        model = EXP / f"models/{model_name}/sentence-model.safetensors"
        assert sha(model) == item["sha256"]
        for mode in ("plain", "grammar"):
            label = "nbest-" + mode if multi else mode
            cases_path = OUT / f"rime-{label}-cases.jsonl"
            cases = read_rows(cases_path)
            active_path = OUT / f"rime-{label}-nonempty.jsonl"
            jsonl(active_path, [c for c in cases if c["candidates"]])
            proc = subprocess.Popen([str(EXP / "probe/target/release/vimeml-chinese-probe.exe"),
                str(model), str(active_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8")
            loaded = json.loads(proc.stdout.readline())
            loaded.update(working_set(proc.pid))
            stdout, stderr = proc.communicate("continue\n")
            if proc.returncode:
                raise RuntimeError(stderr)
            rows = [json.loads(line) for line in stdout.splitlines()]
            assert len(rows) == sum(bool(c["candidates"]) for c in cases)
            jsonl(OUT / f"{model_name}-{label}-scores.jsonl", rows)
            result[model_name + "/" + label] = {"model": item, "file_bytes": model.stat().st_size,
                "loaded": loaded, "cases_sha256": sha(cases_path), "metrics": summarize(cases, rows)}
            write(OUT / ("summary-multi.json" if multi else "summary.json"), result)
            print(model_name, label, json.dumps(result[model_name + "/" + label]["metrics"], ensure_ascii=False), flush=True)
    environment = {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                   "python": sys.version, "upstream_commits": {}}
    for path in (EXP / "upstream").iterdir():
        if (path / ".git").exists():
            environment["upstream_commits"][path.name] = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    write(OUT / "environment.json", environment)


def latency():
    # Shape-only growing-prefix trace: 9 distinct initial characters prevent duplicate
    # candidates, every step appends a new character, no complete list is revisited.
    seeds = []
    initials = set()
    for case in read_rows(EXP / "cases.jsonl"):
        text = case["gold"]
        if case["suite"] == "upstream_sentences" and text[0] not in initials:
            seeds.append((text * 15)[:15])
            initials.add(text[0])
        if len(seeds) == 9:
            break
    assert len(seeds) == 9
    trace = [{"id": f"step-{n}", "context": "", "candidates": [
        {"text": text[:n], "answers_key": True, "trusted_dictionary_hit": False} for text in seeds]} for n in range(1, 16)]
    trace_path = EXP / "growing-prefix-trace.jsonl"
    jsonl(trace_path, trace)
    summaries = {}
    for name in ("pinyin-ime-reranker-4M", "pinyin-ime-reranker-25M"):
        proc = subprocess.Popen([str(EXP / "probe/target/release/vimeml-chinese-probe.exe"),
            str(EXP / f"models/{name}/sentence-model.safetensors"), str(trace_path), "trace"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        loaded = json.loads(proc.stdout.readline())
        loaded.update(working_set(proc.pid))
        stdout, stderr = proc.communicate("continue\n")
        if proc.returncode:
            raise RuntimeError(stderr)
        rows = [json.loads(line) for line in stdout.splitlines()]
        assert len(rows) == 15 and max(r["cached_vs_fresh_max_abs_error"] for r in rows) < 1e-4
        jsonl(OUT / f"{name}-growing-trace.jsonl", rows)
        summaries[name] = {"loaded": loaded, "trace_sha256": sha(trace_path), "steps": rows}
        write(OUT / "growing-trace-summary.json", summaries)
        print(name, [(r["step"], round(r["p50_ms"], 2), round(r["p95_ms"], 2)) for r in rows], flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "capture", "score", "prepare-multi", "capture-multi", "score-multi", "latency"])
    parser.add_argument("--probes", type=Path, default=EXP / "probes.jsonl")
    args = parser.parse_args()
    {"prepare": lambda: prepare(args.probes), "capture": capture, "score": score,
     "prepare-multi": prepare_multi, "capture-multi": lambda: capture(True), "score-multi": lambda: score(True), "latency": latency}[args.action]()
