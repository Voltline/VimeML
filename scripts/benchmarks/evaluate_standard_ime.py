"""Evaluate standard IME tracks; never tune on a consumed blind release."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.evaluate_ajimee import metrics, rerank, summarize
from vimeml.benchmarks.standard_ime import read, load_standard_export, allow_evaluation, source_metrics
from vimeml.training.data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path, help="One imported development/blind/regression directory")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/tokenizers/ja-unigram-16k-v2")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-draft-development", action="store_true")
    parser.add_argument("--evaluation-plan", type=Path)
    parser.add_argument("--cached-scores", type=Path, help="Reuse matching FP32 legacy scores; regression track only")
    args = parser.parse_args()
    manifest = read(args.benchmark / "manifest.json")
    plan = read(args.evaluation_plan) if args.evaluation_plan else None
    allow_evaluation(manifest, allow_draft=args.allow_draft_development, plan=plan, checkpoint=args.checkpoint)
    if args.output.exists():
        raise ValueError("Preserve prior results; choose a new output.")
    receipt_path = args.benchmark.parent / "blind-consumption.json"
    if manifest["split"] == "blind":
        receipt = read(receipt_path) if receipt_path.exists() else {"plan_id": plan["plan_id"], "evaluations": []}
        if receipt["plan_id"] != plan["plan_id"] and plan.get("role") != "reference_test_after_consumption":
            raise ValueError("Blind release has been opened under a different plan; use a new held-out release.")
        model = str(args.checkpoint.resolve())
        if any(r["checkpoint"] == model and r["status"] == "complete" for r in receipt["evaluations"]):
            raise ValueError("This planned model already has completed blind results; reuse them.")
        receipt["evaluations"].append({"checkpoint": model, "output": str(args.output.resolve()), "status": "started",
                                       "plan_id": plan["plan_id"], "result_role": plan.get("role", "blind"),
                                       "utc": datetime.now(timezone.utc).isoformat()})
        write_json(receipt_path, receipt)
    started = time.perf_counter()
    manifest, provenance, rows = load_standard_export(args.benchmark)
    if args.cached_scores:
        if manifest["split"] != "regression":
            raise ValueError("Cache reuse is restricted to historical regression cases.")
        cached_report = read(args.cached_scores / "metrics.json")
        metadata = cached_report["model"]
        if metadata["precision"] != "fp32" or Path(metadata["checkpoint"]).resolve() != args.checkpoint.resolve():
            raise ValueError("Cache model/precision does not match.")
        cached_rows = {r["id"]: r for r in (json.loads(line) for line in (args.cached_scores / "scores.jsonl").read_text(encoding="utf-8").splitlines())}
        for row in rows:
            cached = cached_rows[row["id"]]
            if (any(row[k] != cached[k] for k in ("query", "left_context", "answers")) or
                    row["candidates"] != cached["candidates"] or row["orders"]["azookey"] != cached["orders"]["azookey"]):
                raise ValueError("Frozen legacy case/candidate pool differs from the cache.")
            row.update({k: cached[k] for k in ("orders", "fallbacks", "lm_scores")})
    else:
        import torch
        from vimeml.training.infer import JapaneseLM
        torch.set_num_threads(4)
        lm = JapaneseLM(args.checkpoint, args.tokenizer, args.device, hash_checkpoint=False)
        metadata = lm.metadata
        if plan:
            expected = next(m for m in plan["models"] if m["checkpoint"] == str(args.checkpoint.resolve()))
            if (metadata["checkpoint_step"] != expected["step"] or
                    lm.checkpoint_identity != {"config": expected["training_config"], "signatures": expected["signatures"]}):
                raise ValueError("Checkpoint identity differs from the precommitted snapshot.")
        rerank(lm, rows)
    args.output.mkdir(parents=True)
    (args.output / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    report = {"format": "vimeml_standard_ime_metrics_v1", "status": "complete", "model": metadata,
              "benchmark_manifest": manifest, "export_provenance": provenance, "metrics": summarize(rows),
              "source_metrics": source_metrics(rows, metrics), "elapsed_seconds": time.perf_counter()-started,
              "labels_formal_gold": manifest["labels_formal_gold"], "split": manifest["split"],
              "label_quality": manifest.get("label_quality", "unreviewed_draft"),
              "evaluation_plan": plan, "policy": manifest["score_policy"],
              "result_role": plan.get("role", "blind") if plan else manifest["split"],
              "cached_scores_reused": str(args.cached_scores) if args.cached_scores else None}
    write_json(args.output / "metrics.json", report)
    if manifest["split"] == "blind":
        receipt["evaluations"][-1]["status"] = "complete"
        write_json(receipt_path, receipt)
    print(json.dumps(report["source_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
