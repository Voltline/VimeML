"""Prepare V4.2 INT8 client resources and native fixtures; V1/V2 releases stay intact.

Device runs belong to the separate Vime repository; this only writes resources.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from vimeml.deployment import v2, v4_chunk, v4_coreml as v4
from vimeml.deployment.bundle import fresh_directory, read_json, write_json
from vimeml.training.data import file_sha
from vimeml.training.evaluate_ime import score_candidates
from vimeml.tools.phrase_demo import PhraseDemo

MODEL_VERSION = "4.2-epoch20-step1081942-int8-b32-v1"


def compile_package(package, destination):
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package / "model.mlpackage"), str(destination)],
                   check=True, capture_output=True)
    compiled = destination / "model.mlmodelc"
    if not (compiled / "coremldata.bin").is_file():
        raise ValueError("Missing compiled model.")
    return compiled


def package(bundle, stateless, kv, alignments, template, output, chunk=False):
    """`chunk=True`: `kv` is the chunked KV package (resources v2); otherwise the single-token one (v1)."""
    if output.exists():
        raise ValueError("Resource output exists.")
    stateless_manifest = v4.verify_package(stateless)
    kv_manifest = v4.verify_package(kv, v4_chunk.FORMAT if chunk else v4.KV_FORMAT)
    if any(m["kind"] != "linear8_fp32_compute" or m["minimum_ios"] != 18 for m in (stateless_manifest, kv_manifest)):
        raise ValueError("Expected V4 INT8 FP32-compute iOS18 packages.")
    gate = read_json(alignments["kv_vs_stateless"])
    if (gate.get("format") != (v4_chunk.ALIGNMENT_FORMAT if chunk else "vimeml_kv_alignment_v4") or not gate.get("passed")
            or gate.get("comparison") != "same-weight stateless Core ML"
            or gate["coreml_manifest_sha256"] != file_sha(kv / "manifest.json")):
        raise ValueError("Passed same-weight KV/stateless INT8 comparison required.")
    lm = v4.CoreMLLMV4(bundle, stateless)
    runtime = v4_chunk.KVChunkRuntimeV4(bundle, kv) if chunk else v4.KVRuntimeV4(bundle, kv)
    old = read_json(template)
    fixtures = {"format": "vime_client_fixtures_v4", "model": lm.metadata, "tokenization": [], "scoring": [],
                "phrases": [], "kv_chunk_steps" if chunk else "kv_steps": []}
    deltas = []
    for item in old["tokenization"]:
        ids = lm.processor.encode(item["text"], out_type=int)
        fixtures["tokenization"].append({"text": item["text"], "ids": ids, "decoded": lm.processor.decode(ids)})
    for item in old["scoring"]:
        scored = score_candidates(lm, item["context"], item["candidates"])
        fixtures["scoring"].append({"context": item["context"], "candidates": item["candidates"],
                                    "common_prefix_tokens_including_bos": scored["common_prefix_tokens_including_bos"],
                                    "sums": [c["log_probability_sum"] for c in scored["candidates"]]})
        if chunk:
            # One chunk call for the common prefix, then single-token decode of the suffix.
            common = scored["common_prefix_tokens_including_bos"]
            prefix_ids = [lm.special["bos"], *lm.processor.encode(item["context"] + scored["candidates"][0]["text"],
                                                                  out_type=int)][:common]
            prefix_logits, snapshot = runtime.run(prefix_ids)
            for candidate in scored["candidates"]:
                sequence = [lm.special["bos"], *lm.processor.encode(item["context"] + candidate["text"], out_type=int)]
                if sequence[:common] != prefix_ids:
                    raise ValueError("Common prefix differs from the scored boundary.")
                rows, state = [prefix_logits[0, -1]], snapshot
                for position in range(common, len(sequence) - 1):
                    logits, state = runtime.run([sequence[position]], state, position)
                    rows.append(logits[0, -1])
                logp = torch.from_numpy(np.stack(rows)).log_softmax(-1)
                targets = sequence[common:]
                values = [float(logp[i, t]) for i, t in enumerate(targets)]
                deltas.append(abs(sum(values) - candidate["log_probability_sum"]))
                fixtures["kv_chunk_steps"].append({"context": item["context"], "text": candidate["text"],
                    "prefix_ids": prefix_ids, "prefix_chunk_calls": 1, "decode_ids": sequence[common:-1],
                    "targets": targets, "target_log_probabilities": values, "suffix_sum": sum(values),
                    "stateless_sum": candidate["log_probability_sum"]})
            continue
        # KV contract fixture: single-token steps over each joint sequence.
        for candidate in scored["candidates"]:
            sequence = [lm.special["bos"], *lm.processor.encode(item["context"] + candidate["text"], out_type=int)]
            logits, _ = runtime.run(sequence[:-1])
            logp = torch.from_numpy(logits[0]).log_softmax(-1)
            targets = sequence[1:]
            fixtures["kv_steps"].append({"context": item["context"], "text": candidate["text"],
                "input_ids": sequence[:-1], "targets": targets,
                "score_start": scored["common_prefix_tokens_including_bos"] - 1,
                "step_target_log_probabilities": [float(logp[i, t]) for i, t in enumerate(targets)],
                "step_argmax_ids": [int(i) for i in logits[0].argmax(-1)]})
    demo = PhraseDemo(lm, [])
    for index, item in enumerate(old["phrases"]):
        result = demo.suggest({"prompt": item["prompt"], "mode": "beam", "count": 5, "max_tokens": 8})
        first = result["suggestions"][0]
        fixtures["phrases"].append({"prompt": item["prompt"], "firstText": first["text"],
                                    "firstIDs": first["new_token_ids"], "firstSum": first["log_probability_sum"]})
        print(f"Native phrase fixture {index + 1}/{len(old['phrases'])}", flush=True)
    fresh_directory(output)
    resources = output / "Resources"
    resources.mkdir()
    staging = output / "compile"
    compiled = {"stateless": compile_package(stateless, staging / "stateless"),
                "kv": compile_package(kv, staging / "kv")}
    names = {"stateless": "TinyJapaneseV42INT8.mlmodelc",
             "kv": "TinyJapaneseV42KVChunkINT8.mlmodelc" if chunk else "TinyJapaneseV42KVINT8.mlmodelc"}
    for key, name in names.items():
        shutil.copytree(compiled[key], resources / name)
    shutil.rmtree(staging)
    shutil.copyfile(Path(bundle) / "tokenizer.model", resources / "VimeJapaneseTokenizerV4.model")
    manifest = {"format": "vime_ios_lm_v4", "architecture": v4.ARCHITECTURE, "model_version": MODEL_VERSION,
        "minimum_ios": 18, "vocab_size": 8192, "context_length": 128, "compute_units": "CPU_ONLY",
        "special_ids": lm.special, "forbidden_generation_ids": list(lm.forbidden),
        "reserved_marker_ids": [4, 5, 6], "tokenizer_sha256": lm.metadata["tokenizer_sha256"],
        "checkpoint_sha256": lm.metadata["checkpoint_sha256"],
        "bundle_manifest_sha256": lm.metadata["bundle_manifest_sha256"],
        "models": {key: {"resource": names[key],
                         "source_manifest_sha256": file_sha((stateless if key == "stateless" else kv) / "manifest.json"),
                         "compiled_files_sha256": {n: e["sha256"] for n, e in v2.inventory(resources / names[key], hashes=True).items()}}
                   for key in names},
        "stateless_interface": stateless_manifest["interface"], "kv_interface": kv_manifest["interface"],
        "cache_shape": kv_manifest["cache_shape"], "cache_dtype": "float32",
        "kind": "linear8_fp32_compute", "quantization": "weight-only symmetric INT8 block32; FP32 computation",
        "strict_fp32_logits_alignment_passed": False,
        "kv_vs_stateless_alignment_sha256": file_sha(alignments["kv_vs_stateless"]),
        "kv_graph": "chunk" if chunk else "single_token",
        **({"kv_chunk_fixture_max_suffix_sum_delta": max(deltas),
            "kv_chunk_fixture_within_1e-5": max(deltas) <= 1e-5} if chunk else {}),
        "quality_review": "Candidate only; V2.1 remains the deployed default. See VimeML docs/reports/mac-20261020. Device acceptance is separate."}
    write_json(resources / "VimeLMManifestV42.json", manifest)
    write_json(output / "VimeLMFixturesV42.json", fixtures)
    write_json(output / "manifest.json", {"format": "vimeml_v4_ios_resources_v1", "model": lm.metadata,
        "files": v2.inventory(output), "template_sha256": file_sha(template),
        "note": "Resources/ is for a future Vime client installer. No client source snapshot or device run included."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "stateless", "kv", "kv-alignment", "template", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--chunk", action="store_true", help="--kv is the chunked KV package")
    args = parser.parse_args()
    torch.set_num_threads(2)
    package(args.bundle, args.stateless, args.kv, {"kv_vs_stateless": args.kv_alignment}, args.template, args.output,
            args.chunk)


if __name__ == "__main__":
    main()
