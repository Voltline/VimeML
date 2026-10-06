"""Package an evaluated INT8 candidate-reranking deployment without overwriting experiments."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.deployment.coreml import verify_package
from vimeml.training.data import file_sha


def package(source, compiled, tokenizer, review, output):
    if output.exists():
        raise ValueError("Output exists; choose a new deployment version.")
    model = verify_package(source)
    decision = json.loads(review.read_text())
    if (model["kind"] != "linear8_fp32_compute" or model["minimum_ios"] != 18 or
            model["compression"]["bits"] != 8 or
            decision["source_manifest_sha256"] != file_sha(source / "manifest.json") or
            decision["decision"] != "integrate_cpu_candidate_reranking_for_simulator_validation"):
        raise ValueError("Review/source identity or reviewed INT8 deployment policy mismatch.")
    if file_sha(tokenizer) != model["bundle"]["tokenizer_sha256"]:
        raise ValueError("Wrong tokenizer for this frozen model.")
    if not (compiled / "coremldata.bin").is_file():
        raise ValueError("Supply an iOS18 coremlcompiler output directory.")
    inventory = {p.relative_to(compiled).as_posix(): file_sha(p)
                 for p in sorted(compiled.rglob("*")) if p.is_file()}
    output.mkdir(parents=True)
    resources = output / "Resources"
    resources.mkdir()
    shutil.copytree(compiled, resources / "TinyJapaneseINT8.mlmodelc")
    shutil.copyfile(tokenizer, resources / "VimeJapaneseTokenizer.model")
    manifest = {"format": "vime_ios_lm_v1", "minimum_ios": 18, "vocab_size": 16384,
        "context_length": 128, "compute_units": "CPU_ONLY", "tokenizer_sha256": file_sha(tokenizer),
        "compiled_files_sha256": inventory, "source_manifest_sha256": file_sha(source / "manifest.json"),
        "bundle_manifest_sha256": model["bundle"]["bundle_manifest_sha256"], "kind": model["kind"],
        "quantization": decision["quantization"],
        "strict_fp32_logits_alignment_passed": decision["strict_fp32_alignment_passed"],
        "quality_review": "See the dated review.json; this packaging action does not validate the current client or installed device build.",
        "integration_snapshot": "examples/ios/integration: first candidate-reranking integration, not the latest Vime client"}
    (resources / "VimeLMManifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    shutil.copyfile(review, output / "review.json")
    shutil.copyfile(source / "manifest.json", output / "coreml-source-manifest.json")
    shutil.copytree(ROOT / "examples/ios/VimeSentencePiece", output / "Dependencies/VimeSentencePiece")
    shutil.copytree(ROOT / "examples/ios/integration", output / "Integration")
    (output / "README.md").write_text("""# Vime iOS18 INT8 resource/reference package

Weight-only symmetric INT8 block32, FP32 computation, CPU_ONLY, iOS18.
Resources and tokenizer hashes are in Resources/VimeLMManifest.json.
Official SentencePiece0.2.1 and third-party license notices are preserved.

review.json is the dated first-integration review, not a fresh device acceptance.
Strict FP32 logits alignment failed; fixture/PAD/causal and dev/AJIMEE evidence
supported a limited candidate-reranking integration. Packaging does not run tests.

Integration/ is the first examples/ios/integration snapshot. It predates the latest
Vime client's sentence context, next-word suggestions, preferences and cancellation.
Use the latest Vime repository (or the archived Vime-client.zip) for those behaviors.
This package does not attest installed build identity, enabled settings or device stability.
See docs/coreml.md and docs/reports/mac-20261006/ for the subsequent device evidence.
""", encoding="utf-8")
    print(output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "compiled", "tokenizer", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    package(args.source, args.compiled, args.tokenizer, args.review, args.output)


if __name__ == "__main__":
    main()
