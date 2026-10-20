"""Independent V4.2 deployment stages; every stage writes a fresh output directory."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from vimeml.deployment import v4_chunk, v4_coreml as v4
from vimeml.deployment.bundle import environment, fresh_directory, write_json
from vimeml.training.data import file_sha

CHECKPOINT = ROOT / "artifacts/models/tiny-ja-v42-e8k-d384-l7/deployment.pt"
TOKENIZER = ROOT / "artifacts/tokenizers/v4-a/ja-unigram-8192"
REFERENCE = ROOT / "outputs/deployment/tiny-ja-v42-epoch20-fp32-input"
CUDA_SCORES = ROOT / "outputs/evaluation-v42-20261020"


@torch.inference_mode()
def baseline(output):
    """Task 1: Mac PyTorch CPU FP32 fixtures and four frozen tracks vs the CUDA run."""
    from vimeml.training.infer_v4 import JapaneseLMV4
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    lm = JapaneseLMV4(CHECKPOINT, TOKENIZER, "cpu")
    lm.metadata.update(deployment_checkpoint_sha256=lm.metadata["checkpoint_sha256"],
                       checkpoint_sha256=saved["source_checkpoint_sha256"], device="cpu",
                       environment=environment(), threads=torch.get_num_threads())
    info, arrays = v4.read_reference(REFERENCE, lm)
    errors = []
    for name, case in info["cases"].items():
        actual = lm.model(torch.from_numpy(arrays[name + "_input_ids"].copy()).long()).numpy()
        valid = case["valid_prefix"]
        errors.append({"case": name, "check": "stateless", "valid_prefix": valid,
                       **v4.error_distribution(arrays[name + "_logits"][:, :valid], actual[:, :valid])})
    cache, steps = lm.model.new_cache(), []
    ids = torch.from_numpy(arrays["long_input_ids"].copy())
    for position in range(ids.shape[1]):
        steps.append(lm.model.forward_cached(ids[:, position:position + 1], cache))
    errors.append({"case": "long", "check": "forward_cached_steps",
                   **v4.error_distribution(arrays["long_kv_step_logits"], torch.cat(steps, 1).numpy())})
    fresh_directory(output)
    passed = all(e["outside_tolerance"] == 0 for e in errors)
    write_json(output / "fixtures.json", {"format": "vimeml_v4_mac_pytorch_fixtures_v1", "passed": passed,
        "model": lm.metadata, "reference": info, "tolerances": {"atol": v4.TOLERANCE, "rtol": v4.TOLERANCE},
        "errors": errors})
    print(json.dumps({"fixtures_passed": passed, "max_abs": max(e["max_abs"] for e in errors)}), flush=True)
    summary = v4.evaluate_tracks(lm, output / "tracks", CUDA_SCORES, "V4.2 FP32 CUDA (2026-10-20)")
    return passed, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    stages = parser.add_subparsers(dest="stage", required=True)
    commands = {name: stages.add_parser(name) for name in (
        "baseline", "export", "convert", "convert-kv", "align", "align-kv", "compress", "compress-kv",
        "evaluate", "evaluate-kv", "evaluate-v21",
        "convert-kvchunk", "align-kvchunk", "compress-kvchunk", "evaluate-kvchunk")}
    for name, command in commands.items():
        command.add_argument("--output", type=Path, required=True)
        if name not in ("baseline", "export", "compress", "compress-kv", "evaluate-v21", "compress-kvchunk"):
            command.add_argument("--bundle", type=Path, required=True)
        if name in ("align", "align-kv", "evaluate", "evaluate-kv", "align-kvchunk", "evaluate-kvchunk"):
            command.add_argument("--model", type=Path, required=name.endswith(("-kv", "-kvchunk")))
        if name in ("compress", "compress-kv", "compress-kvchunk"):
            command.add_argument("--source", type=Path, required=True)
            command.add_argument("--alignment", type=Path, required=True)
        if name == "evaluate-v21":
            command.add_argument("--model", type=Path, required=True, help="Deployed V2.1 stateless package")
        if name in ("evaluate", "evaluate-kv", "evaluate-v21", "evaluate-kvchunk"):
            command.add_argument("--reference", type=Path)
            command.add_argument("--reference-label")
    commands["align-kv"].add_argument("--stateless", type=Path,
                                      help="Same-precision stateless package; default compares with PyTorch fixtures")
    commands["align-kvchunk"].add_argument("--stateless", type=Path,
                                           help="Same-precision stateless package; default compares with PyTorch fixtures")
    commands["align-kvchunk"].add_argument("--single", type=Path, help="Same-precision single-token KV package")
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        parser.error("Positive threads and a fresh output path are required.")
    torch.set_num_threads(args.threads)
    if args.stage == "baseline":
        passed, _ = baseline(args.output)
        if not passed:
            raise SystemExit("Mac PyTorch fixtures failed; evidence preserved.")
    elif args.stage == "export":
        v4.export_bundle(CHECKPOINT, TOKENIZER, args.output)
    elif args.stage == "convert":
        v4.convert(args.bundle, args.output)
    elif args.stage == "convert-kv":
        v4.convert_kv(args.bundle, args.output)
    elif args.stage in ("compress", "compress-kv"):
        v4.compress(args.source, args.output, args.alignment, kv=args.stage == "compress-kv")
    elif args.stage == "align":
        lm = v4.CoreMLLMV4(args.bundle, args.model) if args.model else v4.BundleLMV4(args.bundle)
        report = v4.align(lm, REFERENCE, args.output)
        print(json.dumps({"passed": report["passed"], **report["summary"]}, indent=2))
        if not report["passed"]:
            raise SystemExit("Strict alignment failed; report preserved. Review ranking quality separately.")
    elif args.stage == "align-kv":
        runtime = v4.KVRuntimeV4(args.bundle, args.model)
        report = v4.align_kv(runtime, args.bundle, REFERENCE, args.output, args.stateless)
        print(json.dumps({"passed": report["passed"], "comparison": report["comparison"], **report["summary"]}, indent=2))
        if not report["passed"]:
            raise SystemExit("KV alignment failed; evidence preserved.")
    elif args.stage == "convert-kvchunk":
        v4_chunk.convert(args.bundle, args.output)
    elif args.stage == "compress-kvchunk":
        v4_chunk.compress(args.source, args.output, args.alignment)
    elif args.stage == "align-kvchunk":
        runtime = v4_chunk.KVChunkRuntimeV4(args.bundle, args.model)
        report = v4_chunk.align(runtime, args.bundle, REFERENCE, args.output, args.stateless, args.single)
        print(json.dumps({"passed": report["passed"], "comparison": report["comparison"], **report["summary"]}, indent=2))
        if not report["passed"]:
            raise SystemExit("KV chunk alignment failed; evidence preserved.")
    elif args.stage == "evaluate-kvchunk":
        v4.evaluate_tracks(v4_chunk.CachedChunkLMV4(args.bundle, args.model), args.output, args.reference,
                           args.reference_label)
    elif args.stage == "evaluate-v21":
        # Same frozen pools and scorer for the deployed V2.1 packages (same Mac harness).
        from vimeml.deployment import v2
        lm = v2.CoreMLLM(ROOT / "artifacts/deployment/tiny-ja-v2.1-extend5-bundle-v1", args.model)
        v4.evaluate_tracks(lm, args.output, args.reference, args.reference_label)
    else:
        if args.stage == "evaluate-kv":
            lm = v4.CachedLMV4(args.bundle, args.model)
        else:
            lm = v4.CoreMLLMV4(args.bundle, args.model) if args.model else v4.BundleLMV4(args.bundle)
        v4.evaluate_tracks(lm, args.output, args.reference, args.reference_label)
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
