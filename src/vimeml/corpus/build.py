"""Run local cleaning workers concurrently, then merge into one corpus.

Each cleaning worker owns its SQLite database. Completed parts are then merged
through the parallel hash-bucket exporter.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.data.build import dump_json, file_sha
from vimeml.data.corpus_parts import AUDITS, load_plan, PART_SCHEMA_VERSION


def tail(path, maximum=8192):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - maximum))
        return stream.read().decode("utf-8", errors="replace")


def elapsed_text(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def preserve_interrupted(path, owner):
    path, owner = path.resolve(), owner.resolve()
    if not owner.is_relative_to(ROOT) or not path.is_relative_to(owner) or path == owner:
        raise ValueError("Interrupted output must stay inside its owned workspace.")
    destination = path.with_name(f"{path.name}-interrupted-{time.time_ns()}")
    if not destination.resolve().is_relative_to(owner):
        raise ValueError("Invalid interrupted-output destination.")
    path.rename(destination)
    print(f"Preserved interrupted output: {destination}", flush=True)


def completed_part(directory, index, workers, signature):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("stage") != "preprocessed_part"
            or manifest.get("part_schema_version") != PART_SCHEMA_VERSION
            or manifest.get("partition") != {"index": index, "count": workers}
            or manifest.get("pipeline_signature") != signature):
        raise ValueError(f"Completed worker has a different code/config version: {directory}")
    for name in ("index.sqlite", "part-stats.json", *(f"{name}.jsonl" for name in AUDITS)):
        if file_sha(directory / name) != manifest["artifact_sha256"].get(name):
            raise ValueError(f"Worker result is damaged: {directory / name}")
    return sum(manifest["selected_documents"].values())


def expected_documents(inventory):
    import pyarrow.parquet as pq

    total = 0
    for relative, source, limit in inventory:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if source == "fineweb":
            with pq.ParquetFile(path) as parquet:
                count = parquet.metadata.num_rows
        else:
            with path.open(encoding="utf-8-sig") as stream:
                count = sum(1 for _ in stream)
        total += min(count, limit) if limit else count
    return total


def run(config_path, output=None, workers=None, resume=False, progress_seconds=10):
    start = time.monotonic()
    config_path = config_path.resolve()
    config, inventory, _, _, _, _, signature = load_plan(config_path)
    cpu_count = os.cpu_count() or 1
    if workers is None:
        workers = min(10, max(1, cpu_count - 1), len(inventory))
    if type(workers) is not int or not 1 <= workers <= min(128, len(inventory)):
        raise ValueError(f"workers must be between 1 and {min(128, len(inventory))}.")
    if progress_seconds <= 0:
        raise ValueError("progress_seconds must be positive.")
    output = (ROOT / (output or config["build"]["output_dir"])).resolve()
    work = output.with_name(output.name + "-work")
    if not output.is_relative_to(ROOT) or output == ROOT or work == ROOT:
        raise ValueError("Parallel output must be a directory inside the repository.")
    planned = expected_documents(inventory)
    run_plan = {"pipeline_signature": signature, "workers": workers, "output": str(output),
                "expected_documents": planned, "config": str(config_path),
                "input_snapshots": {relative: {"bytes": (ROOT / relative).stat().st_size,
                                               "mtime_ns": (ROOT / relative).stat().st_mtime_ns}
                                    for relative, _, _ in inventory}}
    marker = work / "run-plan.json"
    if work.exists() and any(work.iterdir()):
        if not resume:
            raise FileExistsError(f"Work directory exists: {work}; use --resume or a new --output.")
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != run_plan:
            raise ValueError("Resume requires the same output, workers, inputs and processing policy.")
    else:
        if resume:
            raise FileNotFoundError(f"No saved parallel run to resume: {work}")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"Output must be absent or empty: {output}")
        work.mkdir(parents=True, exist_ok=True)
        dump_json(marker, run_plan)
    if output.exists() and any(output.iterdir()):
        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "complete" and manifest.get("pipeline_signature") == signature:
                print(f"Corpus already complete: {output}", flush=True)
                return json.loads((output / "stats.json").read_text(encoding="utf-8"))
            raise ValueError("Existing final output does not match this parallel run.")
        preserve_interrupted(output, ROOT)
    parts = [work / "parts" / f"worker-{index:03d}" for index in range(workers)]
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processed = [0] * workers
    reused = []
    for index, directory in enumerate(parts):
        if (directory / "manifest.json").is_file():
            processed[index] = completed_part(directory, index, workers, signature)
            reused.append(index)
        elif directory.exists():
            preserve_interrupted(directory, work)
    print(f"Parallel cleaning: {workers} processes / {cpu_count} logical CPUs; {planned:,} documents", flush=True)
    print(f"Worker logs: {logs}", flush=True)
    if reused:
        print(f"Reusing {len(reused)} completed workers; {sum(processed):,} documents already processed", flush=True)
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "ARROW_IO_THREADS": "1"})
    children = {}
    pids = {}
    merge_process = None
    try:
        with ExitStack() as stack:
            for index, directory in enumerate(parts):
                if index in reused:
                    continue
                log = stack.enter_context((logs / f"worker-{index:03d}.log").open("wb"))
                child = subprocess.Popen([sys.executable, "-X", "utf8", str(ROOT / "src/vimeml/data/preprocess_part.py"),
                                          "--config", str(config_path), "--part", f"{index}/{workers}", "--output", str(directory)],
                                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                children[index] = child
                pids[str(index)] = child.pid
            pending = set(children)
            next_progress = 0.0
            while pending:
                for index in list(pending):
                    child = children[index]
                    code = child.poll()
                    for match in re.finditer(r"read (\d+) document origins", tail(logs / f"worker-{index:03d}.log")):
                        processed[index] = max(processed[index], int(match[1]))
                    if code is None:
                        continue
                    if code != 0:
                        raise RuntimeError(f"Worker {index} failed (exit {code}). See {logs / f'worker-{index:03d}.log'}\n{tail(logs / f'worker-{index:03d}.log', 2500)}")
                    # Full checksums are verified by merge; read the success count now.
                    manifest = json.loads((parts[index] / "manifest.json").read_text(encoding="utf-8"))
                    processed[index] = sum(manifest["selected_documents"].values())
                    pending.remove(index)
                now = time.monotonic()
                if now >= next_progress or not pending:
                    total = sum(processed)
                    print(f"[cleaning] {total:,}/{planned:,} ({total/max(1,planned):.1%}); "
                          f"running={len(pending)}; completed={workers-len(pending)}/{workers}; elapsed={elapsed_text(now-start)}", flush=True)
                    next_progress = now + progress_seconds
                if pending:
                    time.sleep(0.25)
            preprocess_seconds = time.monotonic() - start
            print("[merge] All cleaning workers completed. Global deduplication and export now run in one process.", flush=True)
            log = stack.enter_context((logs / "merge.log").open("wb"))
            merge_work = work / "merge"
            if merge_work.exists() and any(merge_work.iterdir()):
                preserve_interrupted(merge_work, work)
            merge_process = subprocess.Popen([sys.executable, "-X", "utf8", str(ROOT / "scripts/corpus/merge.py"),
                                              "--config", str(config_path), "--parts", *(str(p) for p in parts),
                                              "--output", str(output), "--work", str(merge_work),
                                              "--workers", str(min(workers, 8))],
                                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            next_progress = 0.0
            last_message = ""
            while merge_process.poll() is None:
                now = time.monotonic()
                if now >= next_progress:
                    messages = [line for line in tail(logs / "merge.log").splitlines() if line.startswith(("Validating", "[fast-merge "))]
                    if messages:
                        last_message = messages[-1]
                    print(f"[merge] elapsed={elapsed_text(now-start)}; {last_message or 'Starting merge'}", flush=True)
                    next_progress = now + progress_seconds
                time.sleep(0.25)
            if merge_process.returncode:
                raise RuntimeError(f"Merge failed (exit {merge_process.returncode}). See {logs / 'merge.log'}\n{tail(logs / 'merge.log', 2500)}")
        stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
        dump_json(work / "run-summary.json", {
            "status": "complete", "workers": workers, "worker_pids": pids, "reused_workers": reused,
            "document_origins": stats["document_origins"], "preprocess_seconds": round(preprocess_seconds, 3),
            "total_seconds": round(time.monotonic()-start, 3), "output": str(output),
        })
        print(f"Complete: {stats['document_origins']:,} documents; {stats['unique_sentences']:,} unique sentences; "
              f"elapsed={elapsed_text(time.monotonic()-start)}\nCorpus: {output}", flush=True)
        return stats
    except BaseException:
        for child in [*children.values(), merge_process]:
            if child is not None and child.poll() is None:
                child.terminate()
        for child in [*children.values(), merge_process]:
            if child is not None and child.poll() is None:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        print(f"Run stopped. Completed workers are retained; retry with --resume. Logs: {logs}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/corpus-parallel.toml")
    parser.add_argument("--output", help="Final corpus directory; intermediate files use OUTPUT-work.")
    parser.add_argument("--workers", type=int, help="Default: up to 10 processes, limited by CPUs and input files.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed workers from this parallel run.")
    parser.add_argument("--progress-seconds", type=float, default=10)
    args = parser.parse_args()
    try:
        run(args.config, args.output, args.workers, args.resume, args.progress_seconds)
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
