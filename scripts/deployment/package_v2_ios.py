"""Prepare V2 resources and native fixtures, leaving existing V1 releases intact."""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from vimeml.deployment import v2
from vimeml.deployment.bundle import fresh_directory, read_json, write_json
from vimeml.training.data import file_sha
from vimeml.training.evaluate_ime import score_candidates
from vimeml.tools.phrase_demo import PhraseDemo


def package(bundle, source, compiled, template, output):
    if output.exists():
        raise ValueError("Resource output exists.")
    source_manifest = v2.verify_package(source)
    if source_manifest["kind"] != "linear8_fp32_compute" or source_manifest["minimum_ios"] != 18:
        raise ValueError("Expected V2 INT8 FP32-compute iOS18 experiment.")
    if not (compiled / "coremldata.bin").is_file():
        raise ValueError("Missing iOS compiled model.")
    lm = v2.CoreMLLM(bundle, source)
    old = read_json(template)
    fixtures = {"format": "vime_client_fixtures_v2", "model": lm.metadata,
                "tokenization": [], "scoring": [], "phrases": []}
    for item in old["tokenization"]:
        ids = lm.processor.encode(item["text"], out_type=int)
        fixtures["tokenization"].append({"text": item["text"], "ids": ids,
                                         "decoded": lm.processor.decode(ids)})
    for item in old["scoring"]:
        scored = score_candidates(lm, item["context"], item["candidates"])
        fixtures["scoring"].append({"context": item["context"], "candidates": item["candidates"],
                                  "sums": [candidate["log_probability_sum"] for candidate in scored["candidates"]]})
    demo = PhraseDemo(lm, [])
    for index, item in enumerate(old["phrases"]):
        result = demo.suggest({"prompt": item["prompt"], "mode": "beam", "count": 5, "max_tokens": 8})
        first = result["suggestions"][0]
        fixtures["phrases"].append({"prompt": item["prompt"], "firstText": first["text"],
                                   "firstIDs": first["new_token_ids"], "firstSum": first["log_probability_sum"]})
        print(f"Native phrase fixture {index+1}/{len(old['phrases'])}", flush=True)
    fresh_directory(output)
    resources = output / "Resources"
    resources.mkdir()
    shutil.copytree(compiled, resources / "TinyJapaneseV21INT8.mlmodelc")
    shutil.copyfile(bundle / "tokenizer.model", resources / "VimeJapaneseTokenizerV2.model")
    manifest = {"format": "vime_ios_lm_v2", "architecture": v2.ARCHITECTURE,
        "model_version": "2.1-extend5-step40000-int8-b32-v1", "minimum_ios": 18,
        "vocab_size": 16384, "context_length": 128, "compute_units": "CPU_ONLY",
        "special_ids": lm.special, "tokenizer_sha256": lm.metadata["tokenizer_sha256"],
        "compiled_files_sha256": {name: item["sha256"] for name, item in v2.inventory(compiled, hashes=True).items()},
        "source_manifest_sha256": lm.metadata["coreml_manifest_sha256"],
        "bundle_manifest_sha256": lm.metadata["bundle_manifest_sha256"],
        "checkpoint_sha256": lm.metadata["checkpoint_sha256"],
        "kind": source_manifest["kind"], "quantization": "weight-only symmetric INT8 block32; FP32 computation",
        "strict_fp32_logits_alignment_passed": False,
        "quality_review": "Candidate for V2 integration; see VimeML dated quality/device reports. Device acceptance is separate."}
    write_json(resources / "VimeLMManifestV21.json", manifest)
    write_json(output / "VimeLMFixturesV21.json", fixtures)
    write_json(output / "manifest.json", {"format": "vimeml_v2_ios_resources_v1", "model": lm.metadata,
        "files": v2.inventory(output), "note": "Resources/ is installed by the Vime client resource installer. No client source snapshot included."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "source", "compiled", "template", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    package(args.bundle, args.source, args.compiled, args.template, args.output)


if __name__ == "__main__":
    main()
