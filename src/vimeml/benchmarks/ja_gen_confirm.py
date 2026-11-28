"""Preregistered test confirmation of the ja-gen 20M candidate colloq-c40 against the V4.2 deployment
(AzooKey N-best 20 + V4.2 reranking), docs/zh-CN/ja/benchmarks/confirmation-ja-gen-plan.md. One run;
everything the plan fixes is a constant here. run():

1. checks that the plan and code are committed and unmodified and that every registered file has its SHA256;
2. preflight on development only: V4.2 reranking of ime-kana-ja-v1 development against the stage 0 orders,
   and colloq-c40 on ime-kana-ja-v1 and ime-kana-ja-colloq-v1 development against its judged 3.3 rows;
3. writes a consumption record, then V4.2-reranks both test pools and runs colloq-c40 and kd-cont on them;
4. H1 (kana test, V4.2 -> c40, lenient Top-1), H2 in fixed sequence (kana test, kd-cont -> c40, exact Top-1),
   the two kd-cont guards and the fallback test; writes summary.json.
An attempt that has finished is never run again.
"""
import copy
import json
from pathlib import Path
import subprocess
import time

from vimeml.benchmarks import kana_gen_ja as kg
from vimeml.benchmarks import statistics
from vimeml.training.data import file_sha, write_json

PLAN = "docs/zh-CN/ja/benchmarks/confirmation-ja-gen-plan.md"
KANA = "ime-kana-ja-v1"
COLLOQ = "ime-kana-ja-colloq-v1"
BASE = "outputs/benchmarks/ja-gen-confirmation"
ALPHA = 0.05
LAMBDA, BEAM, TOPK = 1.6, 16, 10
PREFLIGHT_FLIPS = 0.01   # share of items whose Top-1 hit (lenient or exact) may differ from the judged rows
PREFLIGHT_NET = 0.005    # largest allowed change of either Top-1 rate
V42_ORDER = "lm_context_sum"

POOLS = {
    "kana-development": "artifacts/benchmarks/ime-candidates-v3-imported/kana-ja-v1-development-n20",
    "kana-test": "artifacts/benchmarks/ime-candidates-v3-imported/kana-ja-v1-test-n20",
    "colloq-development": "artifacts/benchmarks/ime-candidates-colloq-dev-v1-imported/kana-ja-colloq-v1-development-n20",
    "colloq-test": "artifacts/benchmarks/ime-candidates-colloq-test-v1-imported/kana-ja-colloq-v1-test-n20",
}
TOKENIZER = "artifacts/tokenizers/ja-kana-v1/ja-unigram-12288/tokenizer.model"
INDEX = "outputs/ja-gen-stage2/kana-index.json"
V42 = {"checkpoint": "outputs/training-v42-20261020/remote/checkpoints/best-lm.pt",
       "tokenizer": "artifacts/tokenizers/v4-a/ja-unigram-8192"}
MODELS = {
    "c40": {"checkpoint": "artifacts/models/ja-kana-20m-colloq-c40/final.pt",
            "sha256": "19b04ddfe0f98041ffe50511cab9d64d2f2a20b2918f5e78cc78c0768981d15f"},
    "kd-cont": {"checkpoint": "artifacts/models/ja-kana-20m-kd-cont/final.pt",
                "sha256": "5a44839657569b34dea57fae4a45a2050506f892a441e38715756727fe46a92f"},
}
# Judged development rows: c40 in 3.3 (evaluate_ja_gen_stage2.py, lambda 1.6) and V4.2 in stage 0.
JUDGED = {
    "c40-kana-development": "outputs/ja-gen-confirmation-inputs/colloq-c40-main/kana-development-rows.jsonl",
    "c40-colloq-development": "outputs/ja-gen-confirmation-inputs/colloq-c40-colloq/kana-development-rows.jsonl",
    "v42-kana-development": "outputs/ja-gen-stage0-20261125/kana_ja_v1_development-v42-scores.jsonl",
}
SHA256 = {
    f"{POOLS['kana-development']}/manifest.json": "9f14c1a49159ddc5dc919104d421b8e66d67c9d126b3d66e0de561c6f0d5af77",
    f"{POOLS['kana-development']}/ajimee-results/azookey-candidates.json":
        "3490005984c108a3c1d6557c1bba943da2adc20736ef98a47073b895fcb884a2",
    f"{POOLS['kana-test']}/manifest.json": "b11df32194e36ba5af37566049c3953ac7cd1049b57c7afbce3b75b268aedc08",
    f"{POOLS['kana-test']}/evaluation_items.json": "6c6598dee0a5670401f02625e7d1a74dc83a51a168c8d889e6630d6310e77ff0",
    f"{POOLS['kana-test']}/ajimee-results/azookey-candidates.json":
        "21acc93eeabbc256bfef723ca5910d62fe45a89d7e23c9f678949d3d5364c2f8",
    f"{POOLS['colloq-development']}/manifest.json": "PIN_AFTER_DOWNLOAD",
    f"{POOLS['colloq-test']}/manifest.json": "PIN_AFTER_IMPORT",
    f"{POOLS['colloq-test']}/evaluation_items.json": "PIN_AFTER_IMPORT",
    f"{POOLS['colloq-test']}/ajimee-results/azookey-candidates.json": "PIN_AFTER_IMPORT",
    TOKENIZER: "61d71460606eed32f0a976611e5fde0dca12efeecb413c516838911a49d3aa53",
    INDEX: "PIN_AFTER_DOWNLOAD",
    V42["checkpoint"]: "0e83ea7417b9816b37183044468343b151f4e610da0764c31f1f780c5fa5796d",
    f"{V42['tokenizer']}/tokenizer.model": "296a2bdf4f4862be8bda848ed1cd261f0b0fd43064181fad5a513c6ce73081b5",
    JUDGED["c40-kana-development"]: "adef0a8764ee65de1b75ac29d42dcab446cc4d34ebda160c1e7b492ac7ae007f",
    JUDGED["c40-colloq-development"]: "9ecfc1b179ce46f00d23cbfbf1b2607f8db4e9a4ca07c3a2d59f2b3603e53234",
    JUDGED["v42-kana-development"]: "69759d1e8d5874de4c8a2cb9d0caff7cd3d0a5077e0e39380f7bc791b64dd7ac",
    **{m["checkpoint"]: m["sha256"] for m in MODELS.values()},
}
COMMITTED = (PLAN, "scripts/ja/benchmarks/confirm_ja_gen.py", "src/vimeml/benchmarks/ja_gen_confirm.py",
             "src/vimeml/benchmarks/kana_gen_ja.py", "src/vimeml/deployment/ja_search.py",
             "src/vimeml/benchmarks/evaluate_ajimee.py", "src/vimeml/benchmarks/statistics.py",
             "src/vimeml/benchmarks/notation.py")


def git(root, *args):
    return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(root), *args], capture_output=True, text=True)


def check_committed(root):
    for path in COMMITTED:
        if git(root, "ls-files", "--error-unmatch", path).returncode:
            raise SystemExit(f"not committed: {path}")
        if git(root, "diff", "--quiet", "HEAD", "--", path).returncode:
            raise SystemExit(f"differs from the committed version: {path}")
    text = (root / PLAN).read_text(encoding="utf-8")
    if not all(name in text for name in (KANA, COLLOQ, MODELS["c40"]["sha256"])):
        raise SystemExit(f"{PLAN} does not name {KANA}, {COLLOQ} and the c40 checkpoint")
    return {"path": PLAN, "commit": git(root, "log", "-1", "--format=%H", "--", PLAN).stdout.strip(),
            "head": git(root, "rev-parse", "HEAD").stdout.strip()}


def check_identities(root):
    for path, expected in SHA256.items():
        if not (root / path).exists() or file_sha(root / path) != expected:
            raise SystemExit(f"missing or SHA256 differs from the plan: {path}")


# ---- display orders and paired tests ----

def gen_order(row):
    return kg.merge_and_rank(row["azookey"], row["gen"], row["union_gen_scores"], row["union_lm_scores"], lam=LAMBDA)


def v42_order(row):
    return row["orders"][V42_ORDER]


def azookey_order(row):
    return row["orders"]["azookey"]


def top1(rows, order, mode):
    return {r["id"]: kg.hit_at_k(order(r), r["answers"], 1, mode) for r in rows}


def paired(base_rows, base_order, arm_rows, arm_order, mode, item_level):
    """base -> arm on Top-1, sign-flip clustered by author/dialogue (kana) or by item (colloquial)."""
    before, after = top1(base_rows, base_order, mode), top1(arm_rows, arm_order, mode)
    if set(before) != set(after):
        raise ValueError("paired rows differ in their ids")
    rows = sorted(arm_rows, key=lambda r: r["id"])
    labels = [r["id"] for r in rows] if item_level else statistics.clusters(rows, template=None)
    test = statistics.paired([before[r["id"]] for r in rows], [after[r["id"]] for r in rows], labels)
    return {**test, "diff": test["net"], "p": test["cluster_sign_flip_p"]}


def significant_gain(test):
    return test["diff"] > 0 and test["p"] < ALPHA


def significant_loss(test):
    return test["diff"] < 0 and test["p"] < ALPHA


def decide(h1, h2, guard_kana, guard_colloq, kd_vs_v42):
    """Ship c40 when H1 passes and neither guard shows a significant drop against kd-cont; H2 is tested only
    after H1 and does not change what ships. When H1 passes but a guard fails, kd-cont ships if it is
    itself significantly above V4.2 on the H1 metric; otherwise, and whenever H1 fails, V4.2 stays."""
    result = {"h1": significant_gain(h1),
              "kana_lenient_not_below_kd_cont": not significant_loss(guard_kana),
              "colloq_lenient_not_below_kd_cont": not significant_loss(guard_colloq)}
    result["h2"] = significant_gain(h2) if result["h1"] else None
    result["passed"] = result["h1"] and result["kana_lenient_not_below_kd_cont"] and \
        result["colloq_lenient_not_below_kd_cont"]
    if result["passed"]:
        result["ships"] = "c40"
    elif result["h1"] and significant_gain(kd_vs_v42):
        result["ships"] = "kd-cont"
    else:
        result["ships"] = "V4.2"
    return result


def table(rows, order, generated=False):
    out = kg.summarize_rows(rows, order)
    if generated:
        out["generation_only_notation_lenient_top1"] = kg.share(
            [kg.hit_at_k(r["gen"], r["answers"], 1, "notation_lenient") for r in rows])
        out["generation_only_exact_top1"] = kg.share([kg.hit_at_k(r["gen"], r["answers"], 1, "exact") for r in rows])
    return out


def preflight_compare(judged, now, order, modes=("notation_lenient", "exact")):
    if sorted(r["id"] for r in judged) != sorted(r["id"] for r in now):
        return {"passed": False, "reason": "ids differ"}
    n = len(now)
    result = {"items": n, "passed": True}
    for mode in modes:
        before, after = top1(judged, order, mode), top1(now, order, mode)
        flips = sum(before[i] != after[i] for i in before)
        net = sum(after.values()) - sum(before.values())
        ok = flips <= PREFLIGHT_FLIPS * n and abs(net) <= PREFLIGHT_NET * n
        result[mode] = {"judged_hits": sum(before.values()), "hits": sum(after.values()), "flips": flips, "ok": ok}
        result["passed"] &= ok
    return result


# ---- runs ----

def run_arm(root, model, pool, out, device, confirmation=None):
    kg.evaluate_arm(root / MODELS[model]["checkpoint"], root / TOKENIZER, root / POOLS[pool], out_dir=out,
                    mode="read", index_path=root / INDEX, device=device, beam=BEAM, topk=TOPK, lam=LAMBDA,
                    lambdas=(LAMBDA,), search_mode="constrained", fallback_free=True, confirmation=confirmation)
    split = "test" if confirmation else "development"
    return kg.read_jsonl(out / f"kana-{split}-rows.jsonl")


def v42_rerank(root, pool, out_file, confirmation=None, device="cuda"):
    import torch
    from vimeml.benchmarks.evaluate_ajimee import rerank
    from vimeml.training.infer_v4 import JapaneseLMV4

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    _, rows = kg.load_pool(root / POOLS[pool], confirmation)
    rows = copy.deepcopy(rows)
    lm = JapaneseLMV4(root / V42["checkpoint"], root / V42["tokenizer"], device)
    with torch.inference_mode():
        rerank(lm, rows)
    for row in rows:
        row["d5_item"] = kg.is_d5_item(row)
    kg.write_jsonl(out_file, rows)
    return rows


def preflight(root, out, device):
    v42_now = v42_rerank(root, "kana-development", out / "v42-kana-development-rows.jsonl", device=device)
    v42_judged = {r["id"]: r["orders"][V42_ORDER] for r in kg.read_jsonl(root / JUDGED["v42-kana-development"])}
    same = sum(r["orders"][V42_ORDER] == v42_judged.get(r["id"]) for r in v42_now)
    check = {"v42_kana_development": {"items": len(v42_now), "same_order": same,
                                      "passed": same >= (1 - PREFLIGHT_FLIPS) * len(v42_now)}}
    for pool in ("kana-development", "colloq-development"):
        now = run_arm(root, "c40", pool, out / f"c40-{pool}", device)
        judged = kg.read_jsonl(root / JUDGED[f"c40-{pool}"])
        check[f"c40_{pool.replace('-', '_')}"] = preflight_compare(judged, now, gen_order)
    check["passed"] = all(v["passed"] for v in check.values())
    return check


def run(root, device="cuda", preflight_only=False):
    root = Path(root)
    check_identities(root)
    plan = check_committed(root)
    ledger = root / BASE / "consumption.json"
    record = json.loads(ledger.read_text("utf-8")) if ledger.exists() else \
        {"benchmarks": [f"{KANA} test", f"{COLLOQ} test"], "attempts": []}
    if any(a["status"] == "finished" for a in record["attempts"]):
        raise SystemExit("the ja-gen confirmation already finished; it is not rerun")
    (root / BASE).mkdir(parents=True, exist_ok=True)
    out = root / BASE / f"run-{len(list((root / BASE).glob('run-*'))) + 1:02d}"
    out.mkdir()
    check = preflight(root, out, device)
    write_json(out / "preflight.json", check)
    if not check["passed"]:
        raise SystemExit(f"preflight failed ({json.dumps(check)}); test not opened")
    if preflight_only:
        return {"preflight": check, "directory": str(out)}
    attempt = {"directory": out.name, "plan": plan, "status": "opened",
               "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    record["attempts"].append(attempt)
    write_json(ledger, record)

    v42 = {s: v42_rerank(root, f"{s}-test", out / f"v42-{s}-test-rows.jsonl", plan, device) for s in ("kana", "colloq")}
    rows = {(m, s): run_arm(root, m, f"{s}-test", out / f"{m}-{s}-test", device, plan)
            for m in ("c40", "kd-cont") for s in ("kana", "colloq")}
    lenient, exact = "notation_lenient", "exact"
    h1 = paired(v42["kana"], v42_order, rows["c40", "kana"], gen_order, lenient, False)
    h2 = paired(rows["kd-cont", "kana"], gen_order, rows["c40", "kana"], gen_order, exact, False)
    guard_kana = paired(rows["kd-cont", "kana"], gen_order, rows["c40", "kana"], gen_order, lenient, False)
    guard_colloq = paired(rows["kd-cont", "colloq"], gen_order, rows["c40", "colloq"], gen_order, lenient, True)
    kd_vs_v42 = paired(v42["kana"], v42_order, rows["kd-cont", "kana"], gen_order, lenient, False)
    descriptive = {}
    for s, item_level in (("kana", False), ("colloq", True)):
        descriptive[f"{s}_v42_vs_azookey_lenient"] = paired(v42[s], azookey_order, v42[s], v42_order, lenient, item_level)
        for mode in (lenient, exact):
            for m in ("c40", "kd-cont"):
                if (s, mode, m) in (("kana", lenient, "c40"), ("kana", lenient, "kd-cont")):
                    continue
                descriptive[f"{s}_{m}_vs_v42_{mode}"] = paired(v42[s], v42_order, rows[m, s], gen_order, mode, item_level)
        if s == "colloq":
            descriptive["colloq_c40_vs_kd_cont_exact"] = paired(rows["kd-cont", s], gen_order, rows["c40", s], gen_order,
                                                                exact, True)
    fallbacks = {f"{m}-{s}": sum(bool(r.get("search_fallback")) for r in rs) for (m, s), rs in rows.items()}
    result = {"candidate": "colloq-c40 (s1)", "baseline": "V4.2 rerank (AzooKey N-best 20 + V4.2)", "plan": plan,
              "settings": {"beam": BEAM, "topk": TOPK, "lambda": LAMBDA, "search": "constrained", "alpha": ALPHA},
              "decision": decide(h1, h2, guard_kana, guard_colloq, kd_vs_v42),
              "h1": h1, "h2": h2, "guard_kana_lenient_vs_kd_cont": guard_kana,
              "guard_colloq_lenient_vs_kd_cont": guard_colloq, "fallback_kd_cont_vs_v42": kd_vs_v42,
              "descriptive": descriptive, "search_fallbacks": fallbacks,
              "table": {s: {"azookey": table(v42[s], azookey_order), "v42": table(v42[s], v42_order),
                            "c40": table(rows["c40", s], gen_order, True),
                            "kd-cont": table(rows["kd-cont", s], gen_order, True)} for s in ("kana", "colloq")}}
    write_json(out / "summary.json", result)
    attempt.update(status="finished", summary_sha256=file_sha(out / "summary.json"),
                   passed=result["decision"]["passed"], ships=result["decision"]["ships"])
    write_json(ledger, record)
    return result
