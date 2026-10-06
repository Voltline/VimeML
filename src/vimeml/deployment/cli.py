"""Independent stage commands; user starts every real experiment manually."""
import argparse
import json
import math
from pathlib import Path

import torch

from vimeml.deployment.bundle import BundleLM, ROOT, export_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Windows: inference-only bundle; never alters frozen inputs.")
    export.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/models/tiny-ja-v1/best.pt")
    export.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/tokenizers/ja-unigram-16k-v1/tokenizer.model")
    export.add_argument("--token-manifest", type=Path, default=ROOT / "artifacts/token-data/corpus-v1-16k/manifest.json")
    export.add_argument("--output", type=Path, required=True)
    ref = sub.add_parser("reference", help="Windows CPU FP32 portable tensor/scoring fixtures.")
    convert = sub.add_parser("convert", help="Mac: uncompressed FP16 ML Program.")
    convert.add_argument("--target", type=int, choices=(16, 18), default=18)
    compress = sub.add_parser("compress", help="Mac: weight-only post-training compression.")
    compress.add_argument("--source", type=Path, required=True)
    compress.add_argument("--fp16-alignment", type=Path, required=True)
    compress.add_argument("--method", choices=("palette", "linear"), default="palette")
    compress.add_argument("--bits", type=int, choices=(4, 8), default=4)
    compress.add_argument("--group-size", type=int, choices=(0, 8, 16, 32), default=16,
                          help="0=per-tensor palette, positive=grouped channel (iOS18).")
    compress.add_argument("--block-size", type=int, choices=(16, 32, 64, 128), default=32)
    alignment = sub.add_parser("validate", help="Mac: compare exact Windows FP32 fixtures.")
    alignment.add_argument("--reference", type=Path, required=True)
    alignment.add_argument("--atol", type=float, default=.1)
    alignment.add_argument("--rtol", type=float, default=.01)
    alignment.add_argument("--score-atol", type=float, default=.2)
    evaluate = sub.add_parser("evaluate", help="Fresh dev/AJIMEE scores; LM-only, no lambda.")
    evaluate.add_argument("--benchmark", type=Path, required=True)
    evaluate.add_argument("--role", choices=("dev", "ajimee"), required=True)
    evaluate.add_argument("--baseline", type=Path, help="Existing frozen FP32 or new deployment scores directory.")
    phrases = sub.add_parser("phrases", help="Same fixed 20-prefix demo suite; preserve raw outputs.")
    timing = sub.add_parser("timing", help="Mac warm load/reranking/beam latency; no device claims.")
    timing.add_argument("--benchmark", type=Path, required=True)
    timing.add_argument("--repeats", type=int, default=5)
    timing.add_argument("--warmup", type=int, default=1)
    for command in (phrases, timing):
        command.add_argument("--prompts", type=Path, default=ROOT / "configs/phrase-demo-prompts.json")
    phrases.add_argument("--mode", choices=("beam", "sample"), default="beam")
    for command in (ref, convert, alignment, evaluate, phrases, timing):
        command.add_argument("--bundle", type=Path, required=True)
    for command in (alignment, evaluate, phrases, timing):
        command.add_argument("--model", type=Path, help="Core ML experiment directory; omit for bundle FP32.")
        command.add_argument("--compute-units", choices=("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"), default="CPU_ONLY")
    for command in (ref, convert, compress, alignment, evaluate, phrases, timing):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive.")
    if args.command == "validate" and any(not math.isfinite(value) or value < 0
                                          for value in (args.atol, args.rtol, args.score_atol)):
        parser.error("tolerances must be finite and nonnegative.")
    if args.command == "timing" and (args.repeats < 1 or args.warmup < 0):
        parser.error("repeats must be positive and warmup nonnegative.")
    if args.output.exists():
        parser.error("Output exists; use a new versioned directory (no overwrite option).")
    torch.set_num_threads(args.threads)
    if args.command == "export":
        export_bundle(args.checkpoint, args.tokenizer, args.token_manifest, args.output)
    elif args.command == "convert":
        from vimeml.deployment.coreml import convert as run
        run(args.bundle, args.output, args.target)
    elif args.command == "compress":
        from vimeml.deployment.coreml import compress as run
        run(args.source, args.output, args.method, args.bits, args.group_size, args.block_size, args.fp16_alignment)
    else:
        if getattr(args, "model", None):
            from vimeml.deployment.coreml import CoreMLLM
            lm = CoreMLLM(args.bundle, args.model, args.compute_units)
        else:
            lm = BundleLM(args.bundle)
        from vimeml.deployment import validation
        if args.command == "reference":
            validation.reference(lm, args.output)
        elif args.command == "validate":
            report = validation.validate(lm, args.reference, args.output, args.atol, args.rtol, args.score_atol)
            print(json.dumps({key: report[key] for key in ("passed", "logits", "invariants")}, indent=2))
            if not report["passed"]:
                raise SystemExit("Alignment failed; detailed report preserved. Inspect it before proceeding.")
        elif args.command == "evaluate":
            report = validation.evaluate(lm, args.benchmark, args.role, args.output, args.baseline)
            print(json.dumps(report["metrics"]["all"], ensure_ascii=False, indent=2))
        elif args.command == "phrases":
            from vimeml.tools.phrase_demo import PhraseDemo, run_suite
            from vimeml.deployment.bundle import read_json
            run_suite(PhraseDemo(lm, read_json(args.prompts)), args.prompts, args.output, max_tokens=8, mode=args.mode)
        elif args.command == "timing":
            validation.timing(lm, args.benchmark, args.prompts, args.output, args.repeats, args.warmup)
    print(f"Output: {args.output.resolve()}")
