"""Preregistered confirmation runs of vimeml-zh-v2-20m, -200m and -800m
(docs/zh-CN/benchmarks/confirmation-zh-v2-plan.md; v2-800m added by its 2026-11-23 amendment).

Everything the plan fixes is a constant here; MODELS is filled in when the plan is registered, and run()
refuses while any entry is missing. run(model):

1. checks that the plan and code are committed and unmodified and that every registered file has its SHA256;
2. preflight: regenerates ime-pinyin-zh-v1 development with the registered settings and compares each input's
   rescored Top-1 hit with the judged development rows; more than PREFLIGHT_FLIPS of inputs flipping, or the
   Top-1 moving by more than PREFLIGHT_TOP1 points, keeps test closed;
3. writes a consumption record, builds the test split's contexts and generates the test split;
4. compares with the registered baseline and decides by the plan's rule; both models also score the
   ime-standard-zh-v1-r1 blind full-pinyin rerank guard against v1's confirmation scores.

The production baseline of v2-20m (Rime 16 sentences + v1 rerank with the in-field context, Vime #70 logic)
comes from the local Rime harness, which is not in this repository: after step 3 it reads the test records
and contexts written into the run directory and writes rows {"id", "input", "v1-prod": [top texts]}, given
to finish(). v2-200m's baseline is v2-20m's finished test rows, v2-800m's is v2-200m's. A model whose attempt has finished is never
run again.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

from vimeml.benchmarks import pinyin_gen_zh as pg
from vimeml.training.data import file_sha

PLAN = "docs/zh-CN/benchmarks/confirmation-zh-v2-plan.md"
VERSION = "ime-pinyin-zh-v1-r1"
STANDARD = "ime-standard-zh-v1-r1"
FROZEN = "outputs/benchmarks/ime-pinyin-zh-v1/frozen-r1"
READINGS = "outputs/zh-readings/dict/vime_pinyin.dict.yaml"
TOKENIZER = "artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model"
STANDARD_FROZEN = "outputs/benchmarks/ime-standard-zh-v1/frozen-r1"
V1_STANDARD_BLIND = "outputs/benchmarks/zh-v1-confirmation/run-01/v1-full"
STANDARD_SCORER = "scripts/benchmarks/score_standard_ime_zh.py"
BASE = "outputs/benchmarks/zh-v2-confirmation"
MODE, BEAM, TOPK, LAMBDA = "task", 16, 10, 1.6
ALPHA = 0.05
PREFLIGHT_FLIPS = 0.005
PREFLIGHT_TOP1 = 0.2
# Filled in at registration: checkpoint, its SHA256, the judged development rows (a pinyin_gen_zh run with
# the checkpoint as its own prior) and their SHA256. "against": the H1 baseline.
MODELS = {
    "v2-20m": {"checkpoint": "artifacts/models/tiny-zh13-12k-pinyin-kd-cont/final.pt",
               "sha256": "412aee1d1c2d73c9f2ff52533188055e4655575472fe3be136e048fd240268a4",
               "judged": "outputs/eval-zh-20261117/eval-zh/self-prior/kd-cont",
               "judged_sha256": "0003a012b4b750bccc26c8013ada2aec089b15b8a6dacf0d0f92256b2f9b6a46",
               "against": "production"},
    "v2-200m": {"checkpoint": "artifacts/models/tiny-zh200m-12k-pinyin-pt-d1024/final.pt",
                "sha256": "90e48fe4cbaeacf5c58774ded243e57d94c86923a51d191555fbbfc7954150f8",
                "judged": "outputs/eval-zh-20261116/eval-zh/self-prior/pinyin-pt-200m",
                "judged_sha256": "9e1be73704483e0b298c2b1a18c03de24ea88de1bdb7b32e48b3e0e32b7a397c",
                "against": "v2-20m"},
    # amendment 2026-11-23: zh-800m arch2 (k = 4), chosen over arch1 on development
    # (docs/zh-CN/reports/zh-800m-arch2-eval-20261123/report.md)
    "v2-800m": {"checkpoint": "artifacts/models/tiny-zh800m-arch2-k4/final.pt",
                "sha256": "8af7dab5e134964678de99ac44aa671282683aaac82c0f83100dd10b44f2d58b",
                "judged": "outputs/zh800m-results/tiny-zh800m-arch2-k4/eval-zh800m/pinyin",
                "judged_sha256": "a79cb5157e55eca2bc029dbf7a5b365ece8e63ba3c26fe9df3d25fae0f4586ee",
                "against": "v2-200m"},
}
SHA256 = {
    TOKENIZER: "68bd37c45bcb018b32b2eef5b241561ab5b6dc1b77c1c3aadcc1b41c4bf47e5a",
    f"{FROZEN}/manifest.json": "19133b0afaed162034277fe3bbe72b195b46a101462c65697a1a1ec64589248f",
    READINGS: "3a1654cd5ce20ecacd88046186c34ad87e748a494f7078f23c1ad15492eb0c8a",
}
COMMITTED = (PLAN, "scripts/benchmarks/confirm_zh_v2.py", "src/vimeml/benchmarks/zh_v2_confirm.py",
             "src/vimeml/benchmarks/pinyin_gen_zh.py", "scripts/benchmarks/contexts_pinyin_ime_zh.py",
             "src/vimeml/deployment/zh_search.py", "src/vimeml/deployment/zh_torch_backend.py",
             "src/vimeml/benchmarks/zh13_lm_scores.py", "src/vimeml/training/zh_desktop.py", STANDARD_SCORER)


# ---- decision rules -----------------------------------------------------------------------------------

def significant_gain(test):
    return test["diff"] > 0 and test["p"] < ALPHA


def significant_loss(test):
    return test["diff"] < 0 and test["p"] < ALPHA


def decide(model, h1, full, standard):
    """h1, full: pinyin_gen_zh.sign_flip of baseline -> model on all inputs / full pinyin only;
    standard: the Standard compare output of v1 -> model. The phone loads one model, so for v2-20m a
    significant Standard drop fails the run; a desktop model can keep another reranker instead
    (200M: v1, rerank_stays_v1; 800M: the 200M, rerank_uses_200m), which the release notes then state."""
    top1 = standard["top1"]
    standard_drop = top1["net"] < 0 and top1["exact_mcnemar_p"] < ALPHA
    result = {"h1": significant_gain(h1)}
    if model == "v2-20m":
        result["full_pinyin_no_significant_drop"] = not significant_loss(full)
        result["standard_no_significant_drop"] = not standard_drop
    result["passed"] = all(result.values())
    if model == "v2-200m":
        result["rerank_stays_v1"] = standard_drop
    elif model == "v2-800m":  # the desktop can rerank with the confirmed 200M instead
        result["rerank_uses_200m"] = standard_drop
    return result


def production_rows(path, records):
    """The harness's v1-prod rows as pinyin_gen_zh rows (generation order = the production order)."""
    by_key = {(r["id"], r["input"]): r["v1-prod"] for r in pg.read_jsonl(path)}
    rows = []
    for record in records:
        for kind, keys in pg.inputs(record):
            if (record["id"], keys) not in by_key:
                raise ValueError(f"production rows miss {record['id']} {keys}")
            rows.append({"id": record["id"], "kind": kind, "input": keys, "text": record["text"],
                         "gen": by_key[(record["id"], keys)], "genScores": None, "prior": None, "calls": 0, "rows": 0})
    return rows


def preflight_check(judged, now):
    """Rescored Top-1 hits of a regenerated development run against the judged rows."""
    key = lambda r: (r["id"], r["kind"], r["input"])
    old = {key(r): pg.hit(r, LAMBDA) for r in judged}
    new = {key(r): pg.hit(r, LAMBDA) for r in now}
    if old.keys() != new.keys():
        raise ValueError("Regenerated development inputs differ from the judged rows")
    flips = sum(old[k] != new[k] for k in old)
    moved = 100 * (sum(new.values()) - sum(old.values())) / len(old)
    return {"inputs": len(old), "flips": flips, "top1_moved": moved,
            "passed": flips <= PREFLIGHT_FLIPS * len(old) and abs(moved) <= PREFLIGHT_TOP1}


# ---- checks and ledger ----------------------------------------------------------------------------------

def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def check_committed(root):
    for path in COMMITTED:
        if git(root, "ls-files", "--error-unmatch", path).returncode:
            raise SystemExit(f"not committed: {path}")
        if git(root, "diff", "--quiet", "HEAD", "--", path).returncode:
            raise SystemExit(f"differs from the committed version: {path}")
    text = (root / PLAN).read_text(encoding="utf-8")
    if VERSION not in text or STANDARD not in text:
        raise SystemExit(f"{PLAN} does not name {VERSION} and {STANDARD}")
    return {"path": PLAN, "commit": git(root, "log", "-1", "--format=%H", "--", PLAN).stdout.strip(),
            "head": git(root, "rev-parse", "HEAD").stdout.strip()}


def check_identities(root, model):
    spec = MODELS[model]
    if any(spec[k] is None for k in ("checkpoint", "sha256", "judged", "judged_sha256")):
        raise SystemExit(f"{model} is not registered yet (MODELS)")
    files = {**SHA256, spec["checkpoint"]: spec["sha256"], f"{spec['judged']}/rows.jsonl": spec["judged_sha256"]}
    for path, expected in files.items():
        if file_sha(root / path) != expected:
            raise SystemExit(f"SHA256 differs from the plan: {path}")


def ledger_path(root):
    return root / BASE / "consumption.json"


def read_ledger(root):
    path = ledger_path(root)
    return json.loads(path.read_text("utf-8")) if path.exists() else {"benchmark": VERSION, "split": "test", "attempts": []}


def write_ledger(root, record):
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", "utf-8")


def finished(record, model):
    return [a for a in record["attempts"] if a["model"] == model and a["status"] == "finished"]


# ---- the run --------------------------------------------------------------------------------------------

def generate(root, model, out, split, device, plan=None, contexts=None):
    spec = MODELS[model]
    checkpoint = root / spec["checkpoint"]
    return pg.run(checkpoint, root / TOKENIZER, root / READINGS, root / FROZEN, out, MODE, BEAM, TOPK, checkpoint,
                  device, split=split, contexts=contexts, confirmation=plan)


def open_test(root, model, device="cuda", preflight_only=False):
    """Steps 1-3 (only 1-2 with preflight_only). Returns the run directory; v2-20m then waits for the
    production rows."""
    root = Path(root)
    check_identities(root, model)
    plan = check_committed(root)
    record = read_ledger(root)
    if finished(record, model):
        raise SystemExit(f"{model} confirmation already finished; it is not rerun")
    if MODELS[model]["against"] != "production" and not finished(record, MODELS[model]["against"]):
        raise SystemExit(f"{model} is compared with {MODELS[model]['against']}, which has not finished")
    (root / BASE).mkdir(parents=True, exist_ok=True)
    out = root / BASE / f"{model}-run-{len(list((root / BASE).glob(f'{model}-run-*'))) + 1:02d}"
    out.mkdir()
    generate(root, model, out / "development", "development", device)
    check = preflight_check(pg.read_jsonl(root / MODELS[model]["judged"] / "rows.jsonl"),
                            pg.read_jsonl(out / "development" / "rows.jsonl"))
    (out / "preflight.json").write_text(json.dumps(check, indent=1) + "\n", "utf-8")
    if not check["passed"]:
        raise SystemExit(f"preflight failed ({check}); test not opened")
    if preflight_only:
        return out
    attempt = {"model": model, "directory": out.name, "plan": plan, "status": "opened",
               "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    record["attempts"].append(attempt)
    write_ledger(root, record)
    sys.path.insert(0, str(root / "scripts/benchmarks"))
    import contexts_pinyin_ime_zh as C
    (out / "test-contexts.jsonl").write_text(C.build(root / FROZEN, "test"), "utf-8")
    generate(root, model, out / "test", "test", device, plan, out / "test-contexts.jsonl")
    return out


def standard_guard(root, model, out, plan, device):
    from vimeml.benchmarks.zh13_lm_scores import run as score_lm
    scores = out / "standard-blind.jsonl"
    score_lm(root / MODELS[model]["checkpoint"], root / TOKENIZER, root / STANDARD_FROZEN, scores, "full", device,
             split="blind", blind_plan=plan)
    python = [sys.executable, "-I", "-X", "utf8", str(root / STANDARD_SCORER)]
    for command in (["score", str(root / STANDARD_FROZEN), str(out / "standard-full"), "--track", "full",
                     "--split", "blind", "--scores", str(scores), "--name", f"vimeml-zh-{model}",
                     "--blind-plan", str(root / PLAN)],
                    ["compare", str(root / V1_STANDARD_BLIND), str(out / "standard-full"), str(out / "standard-vs-v1.json")]):
        done = subprocess.run(python + command, capture_output=True, text=True, encoding="utf-8")
        if done.returncode:
            raise SystemExit(f"Standard {command[0]} failed:\n{done.stderr}")
    return json.loads((out / "standard-vs-v1.json").read_text("utf-8"))


def finish(root, model, production=None, device="cuda"):
    """Step 4 on the opened attempt: compare, decide, write summary.json, close the attempt."""
    root = Path(root)
    record = read_ledger(root)
    attempt = next((a for a in record["attempts"] if a["model"] == model and a["status"] == "opened"), None)
    if attempt is None:
        raise SystemExit(f"{model} has no opened attempt")
    out = root / BASE / attempt["directory"]
    records, _ = pg.load_set(root / FROZEN, 0, "test", out / "test-contexts.jsonl", attempt["plan"])
    rows = pg.read_jsonl(out / "test" / "rows.jsonl")
    against = MODELS[model]["against"]
    if against == "production":
        if production is None:
            raise SystemExit("v2-20m needs the production rows of the test split")
        base, base_lambda = production_rows(production, records), 0
        baseline = {"name": "production (Rime 16 + v1 rerank, in-field context)", "rows_sha256": file_sha(production)}
    else:
        done = finished(record, against)[-1]
        base, base_lambda = pg.read_jsonl(root / BASE / done["directory"] / "test" / "rows.jsonl"), LAMBDA
        baseline = {"name": against, "attempt": done["directory"]}
    h1 = pg.sign_flip(base, rows, base_lambda, LAMBDA)
    full = pg.sign_flip([r for r in base if r["kind"] == "full"], [r for r in rows if r["kind"] == "full"],
                        base_lambda, LAMBDA)
    standard = standard_guard(root, model, out, attempt["plan"], device)
    result = {"model": model, "version": VERSION, "split": "test", "plan": attempt["plan"], "baseline": baseline,
              "settings": {"mode": MODE, "beam": BEAM, "topk": TOPK, "lambda": LAMBDA, "prior": "self"},
              "decision": decide(model, h1, full, standard), "h1": h1, "full_pinyin": full,
              "standard_vs_v1": {"top1": standard["top1"]},
              "table": {"model": pg.table(rows, LAMBDA), "baseline": pg.table(base, base_lambda)}}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", "utf-8")
    attempt.update(status="finished", summary_sha256=file_sha(out / "summary.json"), passed=result["decision"]["passed"])
    write_ledger(root, record)
    return result
