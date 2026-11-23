"""Preregistered test confirmation of the colloquial post-training candidate colloq-c40 against vimeml-zh-v2-20m
(kd-cont), docs/zh-CN/benchmarks/confirmation-zh-colloq-plan.md. One run; everything the plan fixes is a
constant here. run():

1. checks that the plan and code are committed and unmodified and that every registered file has its SHA256;
2. preflight: regenerates ime-pinyin-zh-colloq-v1 development for c40 with its own prior and compares each
   input's rescored Top-1 hit with the judged development rows (zh_v2_confirm.preflight_check thresholds);
3. writes a consumption record, then generates the colloquial test split for kd-cont, c40 and c20 (c20 is
   descriptive), and the ime-pinyin-zh-v1 test split for c40 with the contexts of the v2-20m confirmation run;
4. H1 (colloquial test, kd-cont -> c40), the public test guards against the v2-20m confirmation's kd-cont rows
   (all inputs and full pinyin) and the Standard blind guard against v1; writes summary.json.
An attempt that has finished is never run again.
"""
import json
from pathlib import Path
import time

from vimeml.benchmarks import pinyin_gen_zh as pg
from vimeml.benchmarks import zh_v2_confirm as v2
from vimeml.training.data import file_sha

PLAN = "docs/zh-CN/benchmarks/confirmation-zh-colloq-plan.md"
COLLOQ = "ime-pinyin-zh-colloq-v1-r1"
COLLOQ_FROZEN = "outputs/benchmarks/ime-pinyin-zh-colloq-v1/frozen-r1"
BASE = "outputs/benchmarks/zh-colloq-confirmation"
V2_RUN = f"{v2.BASE}/v2-20m-run-02"   # the finished v2-20m confirmation: kd-cont's public test rows and contexts
MODELS = {
    "c40": {"checkpoint": "artifacts/models/tiny-zh13-12k-pinyin-colloq-c40/final.pt",
            "sha256": "0ec5ffb110fc3959534858fe1a931f4c177878daa8db33bea861c843ca1c1027"},
    "kd-cont": {"checkpoint": v2.MODELS["v2-20m"]["checkpoint"], "sha256": v2.MODELS["v2-20m"]["sha256"]},
    "c20": {"checkpoint": "artifacts/models/tiny-zh13-12k-pinyin-colloq-c20/final.pt",
            "sha256": "658e6c0e600eec400657a9f61546f48994596a42ddc585c33509d1672d6f0e29"},
}
# c40's judged colloquial development run (cluster_eval_zh.sh colloq/self: generation, then its own prior)
JUDGED = "outputs/zh-colloq/results-20261123/colloq-20261123-results/outputs/eval-zh/colloq-c40/colloq/self"
SHA256 = {
    **v2.SHA256,
    f"{COLLOQ_FROZEN}/manifest.json": "f94a76df52a31805bccd028e41518ffab2673c8c5367707d7ed758d0c3e88c21",
    f"{COLLOQ_FROZEN}/test-contexts.jsonl": "8275d9617c7d698b592f2efb2abba264cc7923fc230fd7a31afafcbef997258c",
    f"{JUDGED}/rows.jsonl": "70a1f5be08aa8fc38e52131ed6f0cb4b8d3f4aca51b14ade473560faf0d2036d",
    f"{V2_RUN}/test/rows.jsonl": "6f6a31d2e74b1908f2cd0a370dc0fb715af2adde3ec0a8767037800e94741628",
    f"{V2_RUN}/test-contexts.jsonl": "f45c5d0ab4bc9c79262a9cf26c7c871cd031d038464d1dc5aa86aaed5af35cfc",
    **{m["checkpoint"]: m["sha256"] for m in MODELS.values()},
}
COMMITTED = (PLAN, "scripts/benchmarks/confirm_zh_colloq.py", "src/vimeml/benchmarks/zh_colloq_confirm.py",
             *v2.COMMITTED[1:])


def decide(h1, public, public_full, standard):
    top1 = standard["top1"]
    result = {"h1": v2.significant_gain(h1),
              "public_no_significant_drop": not v2.significant_loss(public),
              "public_full_pinyin_no_significant_drop": not v2.significant_loss(public_full),
              "standard_no_significant_drop": not (top1["net"] < 0 and top1["exact_mcnemar_p"] < v2.ALPHA)}
    result["passed"] = all(result.values())
    return result


def check_committed(root):
    for path in COMMITTED:
        if v2.git(root, "ls-files", "--error-unmatch", path).returncode:
            raise SystemExit(f"not committed: {path}")
        if v2.git(root, "diff", "--quiet", "HEAD", "--", path).returncode:
            raise SystemExit(f"differs from the committed version: {path}")
    text = (root / PLAN).read_text(encoding="utf-8")
    if not all(name in text for name in (COLLOQ, v2.VERSION, v2.STANDARD)):
        raise SystemExit(f"{PLAN} does not name {COLLOQ}, {v2.VERSION} and {v2.STANDARD}")
    return {"path": PLAN, "commit": v2.git(root, "log", "-1", "--format=%H", "--", PLAN).stdout.strip(),
            "head": v2.git(root, "rev-parse", "HEAD").stdout.strip()}


def check_identities(root):
    for path, expected in SHA256.items():
        if file_sha(root / path) != expected:
            raise SystemExit(f"SHA256 differs from the plan: {path}")


def generate(root, model, frozen, out, split, device, plan=None, contexts=None):
    checkpoint = root / MODELS[model]["checkpoint"]
    return pg.run(checkpoint, root / v2.TOKENIZER, root / v2.READINGS, root / frozen, out, v2.MODE, v2.BEAM, v2.TOPK,
                  checkpoint, device, split=split, contexts=contexts, confirmation=plan)


def standard_guard(root, out, plan, device):
    """zh_v2_confirm.standard_guard for c40, recorded under this plan."""
    saved = (v2.MODELS.get("c40"), v2.PLAN)
    v2.MODELS["c40"], v2.PLAN = MODELS["c40"], PLAN
    try:
        return v2.standard_guard(root, "c40", out, plan, device)
    finally:
        v2.PLAN = saved[1]
        if saved[0] is None:
            del v2.MODELS["c40"]


def run(root, device="cuda", preflight_only=False):
    root = Path(root)
    check_identities(root)
    plan = check_committed(root)
    ledger = root / BASE / "consumption.json"
    record = json.loads(ledger.read_text("utf-8")) if ledger.exists() else \
        {"benchmarks": [COLLOQ, f"{v2.VERSION} (second use)", f"{v2.STANDARD} blind"], "attempts": []}
    if any(a["status"] == "finished" for a in record["attempts"]):
        raise SystemExit("the colloquial confirmation already finished; it is not rerun")
    (root / BASE).mkdir(parents=True, exist_ok=True)
    out = root / BASE / f"run-{len(list((root / BASE).glob('run-*'))) + 1:02d}"
    out.mkdir()
    generate(root, "c40", COLLOQ_FROZEN, out / "c40-colloq-development", "development", device)
    check = v2.preflight_check(pg.read_jsonl(root / JUDGED / "rows.jsonl"),
                               pg.read_jsonl(out / "c40-colloq-development" / "rows.jsonl"))
    (out / "preflight.json").write_text(json.dumps(check, indent=1) + "\n", "utf-8")
    if not check["passed"]:
        raise SystemExit(f"preflight failed ({check}); test not opened")
    if preflight_only:
        return {"preflight": check, "directory": str(out)}
    attempt = {"directory": out.name, "plan": plan, "status": "opened", "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    record["attempts"].append(attempt)
    ledger.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", "utf-8")

    contexts = root / COLLOQ_FROZEN / "test-contexts.jsonl"
    for model in ("kd-cont", "c40", "c20"):
        generate(root, model, COLLOQ_FROZEN, out / f"{model}-colloq-test", "test", device, plan, contexts)
    generate(root, "c40", v2.FROZEN, out / "c40-public-test", "test", device, plan, root / V2_RUN / "test-contexts.jsonl")
    rows = {m: pg.read_jsonl(out / f"{m}-colloq-test" / "rows.jsonl") for m in ("kd-cont", "c40", "c20")}
    public_base = pg.read_jsonl(root / V2_RUN / "test" / "rows.jsonl")
    public = pg.read_jsonl(out / "c40-public-test" / "rows.jsonl")
    lam = v2.LAMBDA
    full = lambda rs: [r for r in rs if r["kind"] == "full"]
    h1 = pg.sign_flip(rows["kd-cont"], rows["c40"], lam, lam)
    public_test = pg.sign_flip(public_base, public, lam, lam)
    public_full = pg.sign_flip(full(public_base), full(public), lam, lam)
    standard = standard_guard(root, out, plan, device)
    result = {"candidate": "c40", "baseline": "kd-cont (vimeml-zh-v2-20m)", "plan": plan,
              "settings": {"mode": v2.MODE, "beam": v2.BEAM, "topk": v2.TOPK, "lambda": lam, "prior": "self"},
              "decision": decide(h1, public_test, public_full, standard),
              "h1": h1, "public": public_test, "public_full_pinyin": public_full,
              "standard_vs_v1": {"top1": standard["top1"]},
              "descriptive": {"colloq_c20_vs_kd_cont": pg.sign_flip(rows["kd-cont"], rows["c20"], lam, lam),
                              "colloq_c40_vs_c20": pg.sign_flip(rows["c20"], rows["c40"], lam, lam)},
              "table": {"colloq": {m: pg.table(r, lam) for m, r in rows.items()},
                        "public": {"c40": pg.table(public, lam), "kd-cont": pg.table(public_base, lam)}}}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", "utf-8")
    attempt.update(status="finished", summary_sha256=file_sha(out / "summary.json"), passed=result["decision"]["passed"])
    ledger.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", "utf-8")
    return result
