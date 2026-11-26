"""ja-gen stage 2 evaluation CLI: build reading index, evaluate one arm, or aggregate a 4-seed report.

Development pools only (test and blind splits are refused).

    python -X utf8 scripts/ja/benchmarks/evaluate_ja_gen_stage2.py build-index \
        --tokenizer artifacts/tokenizers/ja-kana-v1/tokenizer.model \
        --table configs/ja/ja-reading-table.tsv \
        --out outputs/ja-gen-stage2/kana-index.json

    python -X utf8 scripts/ja/benchmarks/evaluate_ja_gen_stage2.py run-arm \
        --mode read --checkpoint artifacts/models/ja-kana-20m-read-s1/final.pt \
        --tokenizer artifacts/tokenizers/ja-kana-v1/tokenizer.model \
        --index outputs/ja-gen-stage2/kana-index.json \
        --kana-pool artifacts/benchmarks/ime-candidates-v3-imported/kana-ja-v1-development-n20 \
        --standard-pool artifacts/benchmarks/ime-candidates-v2-imported/standard-v1-development-n20 \
        --out outputs/ja-gen-stage2/read-s1 --device cuda

    python -X utf8 scripts/ja/benchmarks/evaluate_ja_gen_stage2.py report \
        --read outputs/ja-gen-stage2/read-s1 ... --lm outputs/ja-gen-stage2/lm-s1 ... \
        --v42-kana outputs/ja-gen-stage0-20261125/kana_ja_v1_development-v42-scores.jsonl \
        --out outputs/ja-gen-stage2/stage21-report
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.benchmarks.kana_gen_ja import (  # noqa: E402
    DEFAULT_LAMBDAS,
    best_lambda,
    evaluate_arm,
    evaluate_stage21_decision,
    merge_arm_shards,
    paired_against_baseline,
    read_jsonl,
)
from vimeml.deployment.ja_search import build_index  # noqa: E402
from vimeml.training.data import file_sha, write_json  # noqa: E402


def cmd_build_index(args):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    extra = json.loads(Path(args.extra_readings).read_text("utf-8")) if args.extra_readings else None
    index, stats = build_index(sp, reading_table_path=args.table, extra_piece_readings=extra)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "tokenizer_sha256": file_sha(args.tokenizer),
        "table_sha256": file_sha(args.table) if args.table and Path(args.table).exists() else None,
        **stats,
    }
    index.save(out, metadata=meta)
    print(json.dumps({"out": str(out), "readings": len(index.by_reading), "max_len": index.max_len, **meta},
                     ensure_ascii=False))


def cmd_run_arm(args):
    lambdas = tuple(float(x) for x in args.lambdas.split(",")) if args.lambdas else DEFAULT_LAMBDAS
    summary = evaluate_arm(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        kana_pool_dir=args.kana_pool,
        standard_pool_dir=args.standard_pool,
        out_dir=args.out,
        mode=args.mode,
        index_path=args.index,
        device=args.device,
        beam=args.beam,
        topk=args.topk,
        lam=args.lam,
        lambdas=lambdas,
        search_mode=args.search_mode,
        fallback_free=not args.no_fallback_free,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    std = summary.get("standard_development")
    print(json.dumps({
        "arm": summary.get("arm"),
        "mode": summary["mode"],
        "shard": summary.get("shard"),
        "kana_lenient_top1": summary["kana_development"]["notation_lenient_top1"],
        "kana_exact_top1": summary["kana_development"]["exact_top1"],
        "standard_exact_top1": std["exact_top1"] if std else None,
        "seconds": summary["seconds"],
    }, ensure_ascii=False))


def cmd_merge_shards(args):
    lambdas = tuple(float(x) for x in args.lambdas.split(",")) if args.lambdas else DEFAULT_LAMBDAS
    summary = merge_arm_shards(args.shards, args.out, lam=args.lam, lambdas=lambdas)
    std = summary.get("standard_development")
    best_l, best_r = best_lambda(summary)
    print(json.dumps({
        "arm": summary.get("arm"),
        "merged_shards": summary["merged_shards"],
        "kana_lenient_top1": summary["kana_development"]["notation_lenient_top1"],
        "kana_exact_top1": summary["kana_development"]["exact_top1"],
        "best_lambda": best_l,
        "best_lambda_rate": best_r,
        "standard_exact_top1": std["exact_top1"] if std else None,
    }, ensure_ascii=False))


def cmd_compare(args):
    arm_dir = Path(args.arm)
    arm_sum = json.loads((arm_dir / "summary.json").read_text("utf-8"))
    arm_rows = read_jsonl(arm_dir / "kana-development-rows.jsonl")
    best_l, best_r = best_lambda(arm_sum)
    eval_lam = args.lam if args.lam is not None else best_l

    paired = {}
    if args.baseline:
        base_dir = Path(args.baseline)
        base_rows = read_jsonl(base_dir / "kana-development-rows.jsonl")
        paired["vs_baseline"] = {
            "baseline": base_dir.name,
            "default_lambda": paired_against_baseline(arm_rows, base_rows),
            **({"eval_lambda": eval_lam,
                "at_eval_lambda": paired_against_baseline(arm_rows, base_rows, lam=eval_lam)}
               if eval_lam is not None else {}),
        }
        arm_std_path = arm_dir / "standard-development-rows.jsonl"
        base_std_path = base_dir / "standard-development-rows.jsonl"
        if arm_std_path.is_file() and base_std_path.is_file():
            paired["standard_vs_baseline"] = paired_against_baseline(
                read_jsonl(arm_std_path), read_jsonl(base_std_path), mode="exact"
            )
    if args.v42_kana:
        v42_rows = read_jsonl(args.v42_kana)
        paired["vs_v42"] = {
            "default_lambda": paired_against_baseline(arm_rows, v42_rows),
            **({"eval_lambda": eval_lam,
                "at_eval_lambda": paired_against_baseline(arm_rows, v42_rows, lam=eval_lam)}
               if eval_lam is not None else {}),
        }

    k = arm_sum["kana_development"]
    d5 = k["notation_lenient_top1_by_stratum"].get("d5", {})
    std = (arm_sum.get("standard_development") or {}).get("exact_top1")
    lines = [
        f"# ja-gen development comparison: {arm_dir.name}",
        "",
        f"- Checkpoint: `{arm_sum['checkpoint']}` (step {arm_sum['step']}, sha `{arm_sum['checkpoint_sha256'][:12]}`)",
        f"- kana development lenient Top-1: **{k['notation_lenient_top1']['rate']:.2f}%** "
        f"({k['notation_lenient_top1']['hits']}/{k['notation_lenient_top1']['cases']})",
        f"- kana development exact Top-1: **{k['exact_top1']['rate']:.2f}%** "
        f"({k['exact_top1']['hits']}/{k['exact_top1']['cases']})",
    ]
    if d5:
        lines.append(f"- D5 subset lenient Top-1: **{d5['rate']:.2f}%** ({d5['hits']}/{d5['cases']})")
    if best_l is not None:
        lines.append(f"- Best dev lambda: **{best_l:.2f}** -> **{best_r:.2f}%** lenient Top-1")
    if std:
        lines.append(f"- Standard development exact Top-1: **{std['hits']}/{std['cases']} ({std['rate']:.2f}%)**")
    lines.append("")

    if arm_sum.get("kana_lambda_sweep"):
        lines += ["## Lambda sweep", "", "| lambda | lenient Top-1 | exact Top-1 | lenient Top-3 |", "| ---: | ---: | ---: | ---: |"]
        for lv, m in sorted(arm_sum["kana_lambda_sweep"].items(), key=lambda kv: float(kv[0])):
            lines.append(
                f"| {lv} | {m['notation_lenient_top1']['rate']:.2f}% | "
                f"{m['exact_top1']['rate']:.2f}% | {m['notation_lenient_top3']['rate']:.2f}% |"
            )
        lines.append("")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_data = {
        "arm": arm_dir.name,
        "checkpoint_sha256": arm_sum["checkpoint_sha256"],
        "step": arm_sum["step"],
        "best_lambda": best_l,
        "best_lambda_rate": best_r,
        "kana_development": k,
        "standard_development": arm_sum.get("standard_development"),
        "paired": paired,
    }
    write_json(out / "summary.json", report_data)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "arm": arm_dir.name,
        "kana_lenient_top1": k["notation_lenient_top1"],
        "best_lambda": best_l,
        "best_lambda_rate": best_r,
        "standard_exact_top1": std,
    }, ensure_ascii=False))


def cmd_report(args):
    read_dirs = [Path(d) for d in args.read]
    lm_dirs = [Path(d) for d in args.lm]
    read_sums = [json.loads((d / "summary.json").read_text("utf-8")) for d in read_dirs]
    lm_sums = [json.loads((d / "summary.json").read_text("utf-8")) for d in lm_dirs]
    decision = evaluate_stage21_decision(read_sums, lm_sums)

    v42_rows = read_jsonl(args.v42_kana) if args.v42_kana else None
    paired = {"vs_lm": [], "vs_v42": []}
    for rd, ld in zip(read_dirs, lm_dirs):
        r_rows = read_jsonl(rd / "kana-development-rows.jsonl")
        l_rows = read_jsonl(ld / "kana-development-rows.jsonl")
        paired["vs_lm"].append({"read": rd.name, "lm": ld.name, **paired_against_baseline(r_rows, l_rows)})
        if v42_rows is not None:
            paired["vs_v42"].append({"read": rd.name, **paired_against_baseline(r_rows, v42_rows)})

    lines = [
        "# ja-gen Stage 2.1 development summary",
        "",
        "| Arm | Mode | kana lenient Top-1 | kana exact Top-1 | kana D5 lenient | Standard exact |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for d, s in list(zip(read_dirs, read_sums)) + list(zip(lm_dirs, lm_sums)):
        k = s["kana_development"]
        d5 = k["notation_lenient_top1_by_stratum"].get("d5", {}).get("rate", float("nan"))
        st = s["standard_development"]["exact_top1"]
        lines.append(
            f"| {d.name} | {s['mode']} | {k['notation_lenient_top1']['rate']:.2f}% | "
            f"{k['exact_top1']['rate']:.2f}% | {d5:.2f}% | {st['hits']}/{st['cases']} ({st['rate']:.2f}%) |"
        )
    lines += [
        "",
        "## Decision check",
        "",
        f"- Read 4-seed kana lenient Top-1 mean: **{decision['read_kana_lenient_mean']:.2f}%** "
        f"(V4.2 baseline {decision['v42_kana_lenient']:.1f}%, LM control mean {decision['lm_kana_lenient_mean']:.2f}%)",
        f"- Read 4-seed Standard exact Top-1 mean: **{decision['read_standard_exact_mean']:.2f}** "
        f"(floor {decision['standard_floor']:.0f}, LM control mean {decision['lm_standard_exact_mean']:.2f})",
        f"- Keep reading recipe: **{decision['keep_reading_recipe']}**",
        "",
    ]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", {"decision": decision, "paired": paired,
                                      "read": [d.name for d in read_dirs], "lm": [d.name for d in lm_dirs]})
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    bi = sub.add_parser("build-index")
    bi.add_argument("--tokenizer", required=True)
    bi.add_argument("--table", default=str(ROOT / "configs/ja/ja-reading-table.tsv"))
    bi.add_argument("--extra-readings", default=None)
    bi.add_argument("--out", required=True)
    bi.set_defaults(fn=cmd_build_index)

    ra = sub.add_parser("run-arm")
    ra.add_argument("--mode", choices=("read", "lm"), required=True)
    ra.add_argument("--checkpoint", required=True)
    ra.add_argument("--tokenizer", required=True)
    ra.add_argument("--index", default=None)
    ra.add_argument("--kana-pool", required=True)
    ra.add_argument("--standard-pool", default=None)
    ra.add_argument("--out", required=True)
    ra.add_argument("--device", default="cuda")
    ra.add_argument("--beam", type=int, default=16)
    ra.add_argument("--topk", type=int, default=10)
    ra.add_argument("--lambda", dest="lam", type=float, default=1.6)
    ra.add_argument("--lambdas", default=None)
    ra.add_argument("--search-mode", choices=("constrained", "free"), default="constrained")
    ra.add_argument("--no-fallback-free", action="store_true")
    ra.add_argument("--limit", type=int, default=0)
    ra.add_argument("--shard-index", type=int, default=0)
    ra.add_argument("--num-shards", type=int, default=1)
    ra.set_defaults(fn=cmd_run_arm)

    ms = sub.add_parser("merge-shards")
    ms.add_argument("--shards", nargs="+", required=True)
    ms.add_argument("--out", required=True)
    ms.add_argument("--lambda", dest="lam", type=float, default=None)
    ms.add_argument("--lambdas", default=None)
    ms.set_defaults(fn=cmd_merge_shards)

    cp = sub.add_parser("compare")
    cp.add_argument("--arm", required=True)
    cp.add_argument("--baseline", default=None)
    cp.add_argument("--v42-kana", default=None)
    cp.add_argument("--lambda", dest="lam", type=float, default=None)
    cp.add_argument("--out", required=True)
    cp.set_defaults(fn=cmd_compare)

    rp = sub.add_parser("report")
    rp.add_argument("--read", nargs="+", required=True)
    rp.add_argument("--lm", nargs="+", required=True)
    rp.add_argument("--v42-kana", default=None)
    rp.add_argument("--out", required=True)
    rp.set_defaults(fn=cmd_report)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
