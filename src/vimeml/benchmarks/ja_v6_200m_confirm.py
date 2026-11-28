"""Preregistered test confirmation of vimeml-ja-v6-200m (ja-gen 2.2 joint 200M) against the V4.2 deployment,
with vimeml-ja-v6-20m (colloq-c40) as the guard, docs/zh-CN/ja/benchmarks/confirmation-ja-v6-200m-plan.md.
One run; everything the plan fixes is a constant here. run():

1. checks that the plan and code are committed and unmodified and that every registered file has its SHA256;
2. checks the finished 20M confirmation (run-01): its consumption record, its summary.json SHA256, and that the
   V4.2 and colloq-c40 test rows it wrote reproduce its summary tables; those rows are the baselines here,
   neither V4.2 nor c40 is run on test again;
3. preflight on development only: the 200M regenerates ime-kana-ja-v1 and ime-kana-ja-colloq-v1 development and
   must reproduce the judged 2.2 Top-1 counts;
4. writes a consumption record, runs the 200M on both test pools, then H1 (kana test, V4.2 -> 200M, lenient
   Top-1), the two c40 guards and H2 in fixed sequence (kana test, c40 -> 200M, lenient Top-1); writes
   summary.json. An attempt that has finished is never run again.
"""
import json
from pathlib import Path
import time

from vimeml.benchmarks import ja_gen_confirm as jc
from vimeml.benchmarks import kana_gen_ja as kg
from vimeml.training.data import file_sha, write_json

PLAN = "docs/zh-CN/ja/benchmarks/confirmation-ja-v6-200m-plan.md"
BASE = "outputs/benchmarks/ja-v6-200m-confirmation"
MODEL = {"checkpoint": "artifacts/models/ja-kana-200m-joint/final.pt",
         "sha256": "47721ddd357cb3a720d7108e4aeaa60f671d83167504fd9f649365b37d70a843"}
# Judged 2.2 development counts (docs/*/ja/reports/ja-gen-stage2-200m-20261128): Top-1 hits out of 600.
JUDGED = {"kana-development": {"notation_lenient": 536, "exact": 432},
          "colloq-development": {"notation_lenient": 538, "exact": 451}}
PREFLIGHT_NET = 3        # largest allowed change of either Top-1 count (0.5% of 600)
BASELINE_RUN = "run-01"  # the finished 20M confirmation
BASELINE_ROWS = {"v42-kana": "v42-kana-test-rows.jsonl", "v42-colloq": "v42-colloq-test-rows.jsonl",
                 "c40-kana": "c40-kana-test/kana-test-rows.jsonl", "c40-colloq": "c40-colloq-test/kana-test-rows.jsonl"}
# The pools, tokenizer and index registered for the 20M confirmation, and the 200M checkpoint. V4.2 and c40
# are not run here, so their checkpoints are not needed.
SHA256 = {**{path: sha for path, sha in jc.SHA256.items()
             if path.startswith("artifacts/benchmarks/") or path in (jc.TOKENIZER, jc.INDEX)},
          MODEL["checkpoint"]: MODEL["sha256"]}
COMMITTED = (PLAN, "scripts/ja/benchmarks/confirm_ja_v6_200m.py", "src/vimeml/benchmarks/ja_v6_200m_confirm.py",
             *jc.COMMITTED[1:])


def check_committed(root):
    for path in COMMITTED:
        if jc.git(root, "ls-files", "--error-unmatch", path).returncode:
            raise SystemExit(f"not committed: {path}")
        if jc.git(root, "diff", "--quiet", "HEAD", "--", path).returncode:
            raise SystemExit(f"differs from the committed version: {path}")
    text = (root / PLAN).read_text(encoding="utf-8")
    if not all(name in text for name in (jc.KANA, jc.COLLOQ, MODEL["sha256"], jc.MODELS["c40"]["sha256"])):
        raise SystemExit(f"{PLAN} does not name both test sets, the 200M and the c40 checkpoint")
    return {"path": PLAN, "commit": jc.git(root, "log", "-1", "--format=%H", "--", PLAN).stdout.strip(),
            "head": jc.git(root, "rev-parse", "HEAD").stdout.strip()}


def check_identities(root):
    for path, expected in SHA256.items():
        if not (root / path).exists() or file_sha(root / path) != expected:
            raise SystemExit(f"missing or SHA256 differs from the plan: {path}")


def as_json(value):
    return json.loads(json.dumps(value))


def load_baselines(root):
    """The 20M confirmation's test rows, checked against its finished record and its own summary tables."""
    base = root / jc.BASE
    record = json.loads((base / "consumption.json").read_text("utf-8"))
    finished = [a for a in record["attempts"] if a["status"] == "finished"]
    if [a["directory"] for a in finished] != [BASELINE_RUN] or finished[0]["ships"] != "c40":
        raise SystemExit(f"the 20M confirmation record is not a single finished {BASELINE_RUN} shipping c40")
    run_dir = base / BASELINE_RUN
    summary_sha = file_sha(run_dir / "summary.json")
    if summary_sha != finished[0]["summary_sha256"]:
        raise SystemExit("the 20M confirmation summary.json differs from its consumption record")
    summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
    rows = {key: kg.read_jsonl(run_dir / path) for key, path in BASELINE_ROWS.items()}
    for s in ("kana", "colloq"):
        if as_json(jc.table(rows[f"v42-{s}"], jc.v42_order)) != summary["table"][s]["v42"] or \
                as_json(jc.table(rows[f"c40-{s}"], jc.gen_order, True)) != summary["table"][s]["c40"]:
            raise SystemExit(f"the 20M confirmation {s} rows do not reproduce its summary tables")
    return rows, {"directory": f"{jc.BASE}/{BASELINE_RUN}", "summary_sha256": summary_sha,
                  "rows_sha256": {key: file_sha(run_dir / path) for key, path in BASELINE_ROWS.items()}}


def run_200m(root, pool, out, device, confirmation=None):
    kg.evaluate_arm(root / MODEL["checkpoint"], root / jc.TOKENIZER, root / jc.POOLS[pool], out_dir=out,
                    mode="read", index_path=root / jc.INDEX, device=device, beam=jc.BEAM, topk=jc.TOPK,
                    lam=jc.LAMBDA, lambdas=(jc.LAMBDA,), search_mode="constrained", fallback_free=True,
                    confirmation=confirmation)
    split = "test" if confirmation else "development"
    return kg.read_jsonl(out / f"kana-{split}-rows.jsonl")


def preflight(root, out, device):
    check = {"passed": True}
    for pool, judged in JUDGED.items():
        rows = run_200m(root, pool, out / f"200m-{pool}", device)
        item = {"items": len(rows)}
        for mode, hits_judged in judged.items():
            hits = sum(jc.top1(rows, jc.gen_order, mode).values())
            item[mode] = {"judged_hits": hits_judged, "hits": hits, "ok": abs(hits - hits_judged) <= PREFLIGHT_NET}
        item["fallbacks"] = sum(bool(r.get("search_fallback")) for r in rows)
        item["passed"] = len(rows) == 600 and item["fallbacks"] == 0 and all(item[m]["ok"] for m in judged)
        check[pool.replace("-", "_")] = item
        check["passed"] &= item["passed"]
    return check


def decide(h1, guard_kana, guard_colloq, h2):
    """Ship the 200M as vimeml-ja-v6-200m when H1 passes and neither guard shows a significant drop against
    c40; otherwise the desktop uses vimeml-ja-v6-20m. H2 is tested only after H1 and changes only the claim."""
    result = {"h1": jc.significant_gain(h1),
              "kana_lenient_not_below_c40": not jc.significant_loss(guard_kana),
              "colloq_lenient_not_below_c40": not jc.significant_loss(guard_colloq)}
    result["h2"] = jc.significant_gain(h2) if result["h1"] else None
    result["passed"] = result["h1"] and result["kana_lenient_not_below_c40"] and result["colloq_lenient_not_below_c40"]
    result["desktop"] = "vimeml-ja-v6-200m" if result["passed"] else "vimeml-ja-v6-20m"
    return result


def run(root, device="cuda", preflight_only=False):
    root = Path(root)
    check_identities(root)
    plan = check_committed(root)
    baselines, baseline_record = load_baselines(root)
    ledger = root / BASE / "consumption.json"
    record = json.loads(ledger.read_text("utf-8")) if ledger.exists() else \
        {"benchmarks": [f"{jc.KANA} test", f"{jc.COLLOQ} test"], "model": "vimeml-ja-v6-200m", "attempts": []}
    if any(a["status"] == "finished" for a in record["attempts"]):
        raise SystemExit("the vimeml-ja-v6-200m confirmation already finished; it is not rerun")
    (root / BASE).mkdir(parents=True, exist_ok=True)
    out = root / BASE / f"run-{len(list((root / BASE).glob('run-*'))) + 1:02d}"
    out.mkdir()
    check = preflight(root, out, device)
    write_json(out / "preflight.json", {**check, "baselines": baseline_record})
    if not check["passed"]:
        raise SystemExit(f"preflight failed ({json.dumps(check)}); test not opened")
    if preflight_only:
        return {"preflight": check, "baselines": baseline_record, "directory": str(out)}
    attempt = {"directory": out.name, "plan": plan, "status": "opened",
               "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    record["attempts"].append(attempt)
    write_json(ledger, record)

    big = {s: run_200m(root, f"{s}-test", out / f"200m-{s}-test", device, plan) for s in ("kana", "colloq")}
    v42 = {s: baselines[f"v42-{s}"] for s in ("kana", "colloq")}
    c40 = {s: baselines[f"c40-{s}"] for s in ("kana", "colloq")}
    lenient, exact = "notation_lenient", "exact"
    h1 = jc.paired(v42["kana"], jc.v42_order, big["kana"], jc.gen_order, lenient, False)
    guard_kana = jc.paired(c40["kana"], jc.gen_order, big["kana"], jc.gen_order, lenient, False)
    guard_colloq = jc.paired(c40["colloq"], jc.gen_order, big["colloq"], jc.gen_order, lenient, True)
    h2 = guard_kana
    descriptive = {}
    for s, item_level in (("kana", False), ("colloq", True)):
        for mode in (lenient, exact):
            if (s, mode) != ("kana", lenient):
                descriptive[f"{s}_200m_vs_v42_{mode}"] = jc.paired(v42[s], jc.v42_order, big[s], jc.gen_order,
                                                                   mode, item_level)
        descriptive[f"{s}_200m_vs_c40_exact"] = jc.paired(c40[s], jc.gen_order, big[s], jc.gen_order, exact, item_level)
    result = {"candidate": "vimeml-ja-v6-200m (ja-gen 2.2 joint 200M)",
              "baseline": "V4.2 rerank (AzooKey N-best 20 + V4.2), from the 20M confirmation run-01",
              "guard": "vimeml-ja-v6-20m (colloq-c40), from the 20M confirmation run-01",
              "plan": plan, "baselines": baseline_record,
              "settings": {"beam": jc.BEAM, "topk": jc.TOPK, "lambda": jc.LAMBDA, "search": "constrained",
                           "alpha": jc.ALPHA},
              "decision": decide(h1, guard_kana, guard_colloq, h2),
              "h1": h1, "guard_kana_lenient_vs_c40": guard_kana, "guard_colloq_lenient_vs_c40": guard_colloq,
              "h2": h2, "descriptive": descriptive,
              "search_fallbacks": {s: sum(bool(r.get("search_fallback")) for r in rows) for s, rows in big.items()},
              "table": {s: {"v42": jc.table(v42[s], jc.v42_order), "c40": jc.table(c40[s], jc.gen_order, True),
                            "200m": jc.table(big[s], jc.gen_order, True)} for s in ("kana", "colloq")}}
    write_json(out / "summary.json", result)
    attempt.update(status="finished", summary_sha256=file_sha(out / "summary.json"),
                   passed=result["decision"]["passed"], desktop=result["decision"]["desktop"])
    write_json(ledger, record)
    return result
