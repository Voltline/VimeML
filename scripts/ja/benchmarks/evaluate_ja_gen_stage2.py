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
    evaluate_arm,
    evaluate_stage21_decision,
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
    )
    print(json.dumps({
        "arm": summary.get("arm"),
        "mode": summary["mode"],
        "kana_lenient_top1": summary["kana_development"]["notation_lenient_top1"],
        "kana_exact_top1": summary["kana_development"]["exact_top1"],
        "standard_exact_top1": summary["standard_development"]["exact_top1"],
        "seconds": summary["seconds"],
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
    ra.add_argument("--standard-pool", required=True)
    ra.add_argument("--out", required=True)
    ra.add_argument("--device", default="cuda")
    ra.add_argument("--beam", type=int, default=16)
    ra.add_argument("--topk", type=int, default=10)
    ra.add_argument("--lambda", dest="lam", type=float, default=1.6)
    ra.add_argument("--lambdas", default=None)
    ra.add_argument("--search-mode", choices=("constrained", "free"), default="constrained")
    ra.add_argument("--no-fallback-free", action="store_true")
    ra.add_argument("--limit", type=int, default=0)
    ra.set_defaults(fn=cmd_run_arm)

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
