"""Preregistered blind confirmation of vimeml-tiny-zh-v1 (docs/zh-CN/benchmarks/confirmation-zh-v1-plan.md).

Everything the plan fixes is a constant here. run() checks that the plan and code are committed and
unmodified and that the checkpoint and frozen files have the registered SHA256, rescores development
and requires the ranks recorded by the judgement, writes a consumption record, and only then opens
blind. Once an attempt has finished, blind is never scored again by this script.
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from vimeml.benchmarks.zh13_lm_scores import VERSION, read_jsonl, run as score_lm
from vimeml.training.data import file_sha

PLAN = "docs/zh-CN/benchmarks/confirmation-zh-v1-plan.md"
CHECKPOINT = "artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/final.pt"
TOKENIZER = "artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model"
FROZEN = "outputs/benchmarks/ime-standard-zh-v1/frozen-r1"
JUDGED = "outputs/benchmarks/zh13-20m/judgement/scores"
ARM = "12k-fresh-s1"
BASE = "outputs/benchmarks/zh-v1-confirmation"
SCORER = "scripts/benchmarks/score_standard_ime_zh.py"
SHA256 = {
    CHECKPOINT: "44c423a1e7cb0e32161afc94d6167cabb439a3d25fd9de6d1fbb150a927542d4",
    TOKENIZER: "68bd37c45bcb018b32b2eef5b241561ab5b6dc1b77c1c3aadcc1b41c4bf47e5a",
    f"{FROZEN}/items.jsonl": "43d6fe0a4d88d856a9d9d18fe7bd1bdaa0da6d229ca6154dcdfc6575f8d0adaa",
    f"{FROZEN}/pool-full.jsonl": "82bcd51f97886742659855e1f27e20f7bddccca231db81a4ce284c95f6467fd8",
    f"{FROZEN}/pool-abbrev.jsonl": "9773050b66851582424e595a14c9ff2041a51d94b9cf0b703e5e7a7f212b0e2e",
    f"{FROZEN}/manifest.json": "6b9d0a66aa46446a330301b00462c8e63b2f4ea63aa6d0a75ede47ef6a7bb701",
}
COMMITTED = (PLAN, SCORER, "scripts/benchmarks/confirm_zh_v1.py", "src/vimeml/benchmarks/zh_v1_confirm.py",
             "src/vimeml/benchmarks/zh13_lm_scores.py", "src/vimeml/benchmarks/statistics.py",
             "src/vimeml/training/model_zh13.py", "src/vimeml/training/joint_zh13.py")
RIME_BLIND_TOP1 = {"full": 1045, "abbrev": 147}  # recorded in docs/zh-CN/benchmarks/standard-ime-zh.md
SOURCES = ("crosswoz", "risawoz", "kdconv")
TRACKS = ("full", "abbrev")
ALPHA = 0.05


def decide(compare):
    """The plan's rule, from the full-track `compare` output of Rime -> the candidate."""
    top1, top3 = compare["top1"], compare["top3"]
    h1 = top1["net"] > 0 and top1["exact_mcnemar_p"] < ALPHA
    sources = {s: compare["top1_by_source"][s]["net"] > 0 for s in SOURCES}
    top3_guard = not (top3["net"] < 0 and top3["exact_mcnemar_p"] < ALPHA)
    return {"h1_confirmed": h1, "every_source_positive": sources, "top3_no_significant_drop": top3_guard,
            "passed": h1 and all(sources.values()) and top3_guard}


def wilson(hits, n, z=1.959963984540054):
    centre, margin = hits + z * z / 2, z * math.sqrt(hits * (n - hits) / n + z * z / 4)
    denominator = n + z * z
    return [(centre - margin) / denominator, (centre + margin) / denominator]


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def check_committed(root):
    for path in COMMITTED:
        if git(root, "ls-files", "--error-unmatch", path).returncode:
            raise SystemExit(f"not committed: {path}")
        if git(root, "diff", "--quiet", "HEAD", "--", path).returncode:
            raise SystemExit(f"differs from the committed version: {path}")
    if VERSION not in (root / PLAN).read_text(encoding="utf-8"):
        raise SystemExit(f"{PLAN} does not name {VERSION}")
    return {"path": PLAN, "commit": git(root, "log", "-1", "--format=%H", "--", PLAN).stdout.strip(),
            "head": git(root, "rev-parse", "HEAD").stdout.strip()}


def check_identities(root):
    for path, expected in SHA256.items():
        if file_sha(root / path) != expected:
            raise SystemExit(f"SHA256 differs from the plan: {path}")


def run_scorer(root, out, track, split="development", scores=None, name=None, plan=None):
    command = [sys.executable, "-I", "-X", "utf8", str(root / SCORER), "score", str(root / FROZEN), str(out), "--track", track]
    command += ["--split", split]
    if scores:
        command += ["--scores", str(scores), "--name", name]
    if plan:
        command += ["--blind-plan", str(root / plan)]
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if done.returncode:
        raise SystemExit(f"scorer failed for {out.name}:\n{done.stderr}")
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


def preflight(root, out, device):
    """Rescore development; every rank must equal the judged run, otherwise blind stays closed."""
    for track in TRACKS:
        scores = out / "development" / f"{track}.jsonl"
        score_lm(root / CHECKPOINT, root / TOKENIZER, root / FROZEN, scores, track, device)
        scored = out / "development" / f"score-{track}"
        run_scorer(root, scored, track, scores=scores, name=ARM)
        now = {r["id"]: r["rank"] for r in read_jsonl(scored / "items.jsonl")}
        judged = {r["id"]: r["rank"] for r in read_jsonl(root / JUDGED / f"{ARM}-{track}" / "items.jsonl")}
        if now != judged:
            changed = sum(now.get(i) != judged[i] for i in judged)
            raise SystemExit(f"preflight failed: {changed} {track} ranks differ from the judgement; blind not opened")
        print(f"preflight {track}: {len(now)} ranks equal the judgement", flush=True)


def next_directory(base, prefix):
    used = [int(p.name.rsplit("-", 1)[1]) for p in base.glob(f"{prefix}-*") if p.is_dir()]
    directory = base / f"{prefix}-{max(used, default=0) + 1:02d}"
    directory.mkdir(parents=True)
    return directory


def read_consumption(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"benchmark": VERSION, "split": "blind", "attempts": []}


def write_consumption(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def confirm(root, out, plan, device):
    tracks = {}
    for track in TRACKS:
        scores = out / "blind" / f"{track}.jsonl"
        meta = score_lm(root / CHECKPOINT, root / TOKENIZER, root / FROZEN, scores, track, device, split="blind", blind_plan=plan)
        rime = run_scorer(root, out / f"rime-{track}", track, "blind", plan=PLAN)
        if rime["all"]["top1"] != RIME_BLIND_TOP1[track]:
            raise SystemExit(f"Rime blind {track} Top-1 is {rime['all']['top1']}, recorded {RIME_BLIND_TOP1[track]}")
        model = run_scorer(root, out / f"v1-{track}", track, "blind", scores=scores, name="vimeml-tiny-zh-v1", plan=PLAN)
        compared = out / f"compare-{track}.json"
        done = subprocess.run([sys.executable, "-I", "-X", "utf8", str(root / SCORER), "compare",
                               str(out / f"rime-{track}"), str(out / f"v1-{track}"), str(compared)], capture_output=True, text=True, encoding="utf-8")
        if done.returncode:
            raise SystemExit(f"compare failed for {track}:\n{done.stderr}")
        tracks[track] = {"rime": rime, "model": model, "compare": json.loads(compared.read_text(encoding="utf-8")), "meta": meta}
    full = tracks["full"]["model"]["all"]
    result = {"version": VERSION, "split": "blind", "plan": plan, "arm": ARM, "sha256": SHA256,
              "decision": decide(tracks["full"]["compare"]), "top1_wilson_95": wilson(full["top1"], full["n"]),
              "fallbacks": {t: tracks[t]["model"]["fallbacks"] for t in TRACKS},
              "tracks": {t: {k: tracks[t][k] for k in ("rime", "model", "compare")} for t in TRACKS}}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return result


def run(root, device="cuda", preflight_only=False):
    root, base = Path(root), Path(root) / BASE
    ledger = base / "consumption.json"
    check_identities(root)
    if preflight_only:
        out = next_directory(base, "preflight")
        preflight(root, out, device)
        print(f"preflight passed, blind not opened: {out}")
        return None
    plan = check_committed(root)
    record = read_consumption(ledger)
    if any(a["status"] == "finished" for a in record["attempts"]):
        raise SystemExit(f"blind confirmation already finished, see {ledger}; it is not rerun")
    out = next_directory(base, "run")
    preflight(root, out, device)
    attempt = {"directory": out.name, "plan": plan, "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "opened"}
    record["attempts"].append(attempt)
    write_consumption(ledger, record)
    result = confirm(root, out, plan, device)
    attempt.update(status="finished", summary_sha256=file_sha(out / "summary.json"), passed=result["decision"]["passed"])
    write_consumption(ledger, record)
    return result
