"""vimeml-tiny-zh-v1 deployment stages; every stage writes a fresh output directory.

Brief: docs/reference/mac-zh-v1-coreml.md. Packages are `artifacts/deployment/tiny-zh-v1-<kind>-v1`;
records go under `outputs/deployment/zh-v1-*`.

  baseline        Mac PyTorch CPU FP32: fixtures (stateless + forward_cached) and development
                  rescoring on both tracks, scored and compared with the reference results.
  export          Inference bundle from deployment.pt.
  convert         Stateless FP32 package.         convert-chunk   Chunk-KV FP32 package.
  align           Stateless fixture gate.         align-chunk     Chunk-KV gate (fixtures or --stateless).
  compress        --recipe int8-b32 (deployed) or a survey recipe; --chunk for the chunk-KV package.
  evaluate        Development, both tracks, through a package (or PyTorch from the bundle), scored
                  and compared with --reference (a previous evaluate/baseline output).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from vimeml.benchmarks.zh13_lm_scores import load  # noqa: E402
from vimeml.deployment import zh_v1 as zh  # noqa: E402
from vimeml.deployment.bundle import environment, fresh_directory, read_json, write_json  # noqa: E402

CHECKPOINT = ROOT / "artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/deployment.pt"
TOKENIZER = ROOT / "artifacts/tokenizers/zh-v13/zh-unigram-12288"
REFERENCE = ROOT / "outputs/deployment/vimeml-tiny-zh-v1-fp32-input"
FROZEN = ROOT / "outputs/benchmarks/ime-standard-zh-v1/frozen-r1"
REFERENCE_SCORES = ROOT / "outputs/benchmarks/zh13-20m/runs/12k-fresh-s1"
REFERENCE_RESULTS = ROOT / "outputs/benchmarks/zh13-20m/judgement/scores"
SCORER = ROOT / "scripts/benchmarks/score_standard_ime_zh.py"
TRACKS = ("full", "abbrev")
EXPECTED_TOP1 = {"full": 1363, "abbrev": 337}


def scorer(*args):
    result = subprocess.run([sys.executable, "-I", "-X", "utf8", str(SCORER), *map(str, args)],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def judge(output, label, reference_scores, reference_results, reference_label):
    """score_standard_ime_zh.py score + compare per track; rank differences by item id."""
    report = {"label": label, "reference": reference_label, "tracks": {}}
    for track in TRACKS:
        results = output / "results" / track
        line = scorer("score", FROZEN, results, "--track", track, "--scores", output / "scores" / f"{track}.jsonl",
                      "--name", f"{label}-{track}")
        compared = scorer("compare", reference_results(track), results, output / f"compare-{track}.json")
        summary = read_json(results / "summary.json")
        report["tracks"][track] = {
            "scorer": line, "compare": compared, "all": summary["all"],
            "rank_differences": zh.rank_differences(reference_scores(track), output / "scores" / f"{track}.jsonl"),
            **zh.changed_items(reference_results(track), results)}
        print(line, "\n", compared, flush=True)
    write_json(output / "judgement.json", report)
    return report


@torch.inference_mode()
def baseline(output):
    model, processor, meta = load(CHECKPOINT, TOKENIZER / "tokenizer.model")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    lm_meta = {"checkpoint_step": saved["step"], "checkpoint_sha256": saved["source_checkpoint_sha256"],
               "tokenizer_sha256": meta["tokenizer_model_sha256"]}
    info, arrays = zh.read_reference(REFERENCE, type("Meta", (), {"metadata": lm_meta}))
    errors = []
    for name, case in info["cases"].items():
        actual = model(torch.from_numpy(arrays[name + "_input_ids"].copy()).long()).numpy()
        valid = case["valid_prefix"]
        errors.append({"case": name, "check": "stateless", "valid_prefix": valid,
                       **zh.error_distribution(arrays[name + "_logits"][:, :valid], actual[:, :valid])})
    cache, steps = model.new_cache(), []
    ids = torch.from_numpy(arrays["long_input_ids"].copy())
    for position in range(ids.shape[1]):
        steps.append(model.forward_cached(ids[:, position:position + 1], cache))
    errors.append({"case": "long", "check": "forward_cached_steps",
                   **zh.error_distribution(arrays["long_kv_step_logits"], torch.cat(steps, 1).numpy())})
    fresh_directory(output)
    passed = all(e["outside_tolerance"] == 0 for e in errors)
    write_json(output / "fixtures.json", {"format": "vimeml_zh_v1_mac_pytorch_fixtures_v1", "passed": passed,
        "model": {**meta, **lm_meta, "device": "cpu", "threads": torch.get_num_threads()},
        "environment": environment(), "reference": info,
        "tolerances": {"atol": zh.TOLERANCE, "rtol": zh.TOLERANCE}, "errors": errors})
    print(json.dumps({"fixtures_passed": passed, "max_abs": max(e["max_abs"] for e in errors)}), flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=4)
    stages = parser.add_subparsers(dest="stage", required=True)
    commands = {name: stages.add_parser(name) for name in (
        "baseline", "export", "convert", "convert-chunk", "align", "align-chunk", "compress", "evaluate")}
    for name, command in commands.items():
        command.add_argument("--output", type=Path, required=True)
        if name not in ("baseline", "export", "compress"):
            command.add_argument("--bundle", type=Path, required=True)
        if name in ("align", "align-chunk", "evaluate"):
            command.add_argument("--model", type=Path, required=name == "align-chunk")
    commands["align-chunk"].add_argument("--stateless", type=Path,
                                         help="Same-precision stateless package; default compares with PyTorch fixtures")
    commands["compress"].add_argument("--source", type=Path, required=True)
    commands["compress"].add_argument("--alignment", type=Path, required=True)
    commands["compress"].add_argument("--recipe", choices=sorted(zh.RECIPES), default=zh.DEPLOYED_RECIPE)
    commands["compress"].add_argument("--chunk", action="store_true")
    commands["evaluate"].add_argument("--chunk", action="store_true", help="--model is a chunk-KV package")
    commands["evaluate"].add_argument("--reference", type=Path,
                                      help="Earlier evaluate/baseline output; default: the Windows reference run")
    commands["evaluate"].add_argument("--reference-label")
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        parser.error("Positive threads and a fresh output path are required.")
    torch.set_num_threads(args.threads)
    if args.stage == "baseline":
        passed = baseline(args.output / "fixtures")
        model, processor, meta = load(CHECKPOINT, TOKENIZER / "tokenizer.model")

        class Loaded:
            def loaded(self):
                return model, processor, meta
        zh.score_development(Loaded(), FROZEN, args.output / "scores")
        judge(args.output, "mac-pytorch", lambda t: REFERENCE_SCORES / f"{t}.jsonl",
              lambda t: REFERENCE_RESULTS / f"12k-fresh-s1-{t}", "Windows CUDA FP32 reference")
        if not passed:
            raise SystemExit("Mac PyTorch fixtures failed; evidence preserved.")
    elif args.stage == "export":
        zh.export_bundle(CHECKPOINT, TOKENIZER, args.output)
    elif args.stage == "convert":
        zh.convert(args.bundle, args.output)
    elif args.stage == "convert-chunk":
        zh.convert_chunk(args.bundle, args.output)
    elif args.stage == "compress":
        zh.compress(args.source, args.output, args.alignment, args.recipe, args.chunk)
    elif args.stage == "align":
        lm = zh.CoreMLLMZh(args.bundle, args.model) if args.model else zh.BundleLMZh(args.bundle)
        report = zh.align(lm, REFERENCE, args.output)
        print(json.dumps({"passed": report["passed"], **report["summary"]}, indent=2))
        if not report["passed"]:
            raise SystemExit("Strict alignment failed; report preserved. Review ranking quality separately.")
    elif args.stage == "align-chunk":
        runtime = zh.ChunkRuntimeZh(args.bundle, args.model)
        report = zh.align_chunk(runtime, args.bundle, REFERENCE, args.output, args.stateless)
        print(json.dumps({"passed": report["passed"], "comparison": report["comparison"], **report["summary"]}, indent=2))
        if not report["passed"]:
            raise SystemExit("KV chunk alignment failed; evidence preserved.")
    elif args.stage == "evaluate":
        if args.chunk:
            lm = zh.CachedChunkLMZh(args.bundle, args.model)
        else:
            lm = zh.CoreMLLMZh(args.bundle, args.model) if args.model else zh.BundleLMZh(args.bundle)
        zh.score_development(lm, FROZEN, args.output / "scores")
        if args.reference:
            judge(args.output, lm.metadata.get("package", "pytorch-bundle"),
                  lambda t: args.reference / "scores" / f"{t}.jsonl", lambda t: args.reference / "results" / t,
                  args.reference_label or str(args.reference))
        else:
            judge(args.output, lm.metadata.get("package", "pytorch-bundle"),
                  lambda t: REFERENCE_SCORES / f"{t}.jsonl", lambda t: REFERENCE_RESULTS / f"12k-fresh-s1-{t}",
                  "Windows CUDA FP32 reference")
        write_json(args.output / "model.json", lm.metadata)
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
