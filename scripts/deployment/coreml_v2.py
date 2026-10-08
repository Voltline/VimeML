"""Independent V2 deployment stages; preserve all experiment directories."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from vimeml.deployment import v2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    stages = parser.add_subparsers(dest="stage", required=True)
    export = stages.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--tokenizer", type=Path, required=True)
    export.add_argument("--token-manifest", type=Path, required=True)
    convert = stages.add_parser("convert")
    convert.add_argument("--bundle", type=Path, required=True)
    compress = stages.add_parser("compress")
    compress.add_argument("--source", type=Path, required=True)
    compress.add_argument("--alignment", type=Path, required=True)
    align = stages.add_parser("align")
    align.add_argument("--reference", type=Path, required=True)
    evaluate = stages.add_parser("evaluate")
    evaluate.add_argument("--benchmark", type=Path, required=True)
    evaluate.add_argument("--baseline", type=Path, required=True)
    samples = stages.add_parser("samples")
    samples.add_argument("--baseline", type=Path, required=True)
    fixtures = stages.add_parser("device-fixtures")
    for command in (align, evaluate, samples, fixtures):
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--model", type=Path)
    for command in (export, convert, compress, align, evaluate, samples, fixtures):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        parser.error("Positive threads and a fresh output path are required.")
    torch.set_num_threads(args.threads)
    if args.stage == "export":
        v2.export_bundle(args.checkpoint, args.tokenizer, args.token_manifest, args.output)
    elif args.stage == "convert":
        v2.convert(args.bundle, args.output)
    elif args.stage == "compress":
        v2.compress(args.source, args.output, args.alignment)
    else:
        lm = v2.CoreMLLM(args.bundle, args.model) if args.model else v2.BundleLM(args.bundle)
        if args.stage == "align":
            report = v2.align(lm, args.reference, args.output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not report["passed"]:
                raise SystemExit("Strict alignment failed; report preserved. Review ranking quality separately.")
        elif args.stage == "evaluate":
            report = v2.evaluate(lm, args.benchmark, args.baseline, args.output)
            print(json.dumps({"metrics": report["metrics"]["all"],
                              "comparison": {k: v for k, v in report["comparison"].items() if k != "changes"}}, indent=2))
        elif args.stage == "samples":
            v2.samples(lm, args.baseline, args.output)
        else:
            v2.device_fixtures(lm, args.output)
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
