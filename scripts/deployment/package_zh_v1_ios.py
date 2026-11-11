"""Prepare vimeml-tiny-zh-v1 INT8 client resources and native fixtures for the Vime test host.

Resources are named so that Japanese resources are unaffected:
  TinyChineseV1INT8.mlmodelc, TinyChineseV1KVChunkINT8.mlmodelc,
  VimeChineseTokenizerV1.model, VimeLMManifestZhV1.json (+ VimeLMFixturesZhV1.json beside Resources/).

Fixture texts are written here; none comes from the benchmark. Device runs belong to the Vime repository.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vimeml.benchmarks.zh13_lm_scores import score_item  # noqa: E402
from vimeml.deployment import v2, zh_v1 as zh  # noqa: E402
from vimeml.deployment.bundle import fresh_directory, read_json, write_json  # noqa: E402
from vimeml.training.data import file_sha  # noqa: E402

TAG = "ZhV1"
TOKENIZATION = ["你好，今天天气不错。", "我们明天下午三点在地铁站见面吧。", "这家餐厅的火锅很好吃，推荐给你。",
                "Vime 输入法支持拼音。", "一二三四五六七八九十"]
SCORING = [("我们晚上去吃", ["火锅", "活过", "伙夫", "或"]),
           ("明天上午我要去", ["开会", "看会", "开灰", "凯"]),
           ("这本书写得", ["很好", "恨好", "很号"]),
           ("", ["你好", "拟好", "泥嚎"])]


def compile_package(package, destination):
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package / "model.mlpackage"), str(destination)],
                   check=True, capture_output=True)
    compiled = destination / "model.mlmodelc"
    if not (compiled / "coremldata.bin").is_file():
        raise ValueError("Missing compiled model.")
    return compiled


@torch.inference_mode()
def package(bundle, stateless, chunk, alignment, output):
    if output.exists():
        raise ValueError("Resource output exists.")
    stateless_manifest = zh.verify_package(stateless)
    chunk_manifest = zh.verify_package(chunk, zh.CHUNK_FORMAT)
    for manifest in (stateless_manifest, chunk_manifest):
        if manifest.get("recipe") != zh.DEPLOYED_RECIPE or manifest["minimum_ios"] != 18:
            raise ValueError("Expected the deployed INT8 block32 FP32-compute iOS18 recipe.")
    gate = read_json(alignment)
    if (gate.get("format") != zh.CHUNK_ALIGNMENT_FORMAT or not gate.get("passed")
            or gate.get("comparison") != "same-weight stateless Core ML"
            or gate["coreml_manifest_sha256"] != file_sha(chunk / "manifest.json")
            or gate["baseline_manifest_sha256"] != file_sha(stateless / "manifest.json")):
        raise ValueError("Passed same-weight chunk-KV/stateless INT8 comparison required.")
    lm = zh.CoreMLLMZh(bundle, stateless)
    runtime = zh.ChunkRuntimeZh(bundle, chunk)
    bos = lm.special["bos"]
    fixtures = {"format": "vime_client_fixtures_zh_v1", "model": lm.metadata, "tokenization": [], "scoring": [],
                "kv_chunk_steps": []}
    for text in TOKENIZATION:
        ids = lm.processor.encode(text)
        fixtures["tokenization"].append({"text": text, "ids": ids, "decoded": lm.processor.decode(ids)})
    deltas = []
    for context, candidates in SCORING:
        scored = score_item(lm.model, lm.processor, context, candidates)
        common = scored["common_prefix_tokens_including_bos"]
        fixtures["scoring"].append({"context": context, "candidates": candidates,
                                    "common_prefix_tokens_including_bos": common, "sums": scored["scores"],
                                    "scored_tokens": scored["scored_tokens"]})
        sequences = [[bos, *lm.processor.encode(context + c)] for c in candidates]
        prefix_ids = sequences[0][:common]
        # One chunk call for the common prefix, then single-token decode of each suffix.
        prefix_logits, snapshot = runtime.run(prefix_ids)
        for text, sequence, expected in zip(candidates, sequences, scored["scores"], strict=True):
            if sequence[:common] != prefix_ids:
                raise ValueError("Common prefix differs from the scored boundary.")
            rows, state = [prefix_logits[0, -1]], snapshot
            for position in range(common, len(sequence) - 1):
                logits, state = runtime.run([sequence[position]], state, position)
                rows.append(logits[0, -1])
            logp = torch.from_numpy(np.stack(rows)).log_softmax(-1)
            targets = sequence[common:]
            values = [float(logp[i, t]) for i, t in enumerate(targets)]
            deltas.append(abs(sum(values) - expected))
            fixtures["kv_chunk_steps"].append({"context": context, "text": text, "prefix_ids": prefix_ids,
                "prefix_chunk_calls": 1, "decode_ids": sequence[common:-1], "targets": targets,
                "target_log_probabilities": values, "suffix_sum": sum(values), "stateless_sum": expected})
    fresh_directory(output)
    resources = output / "Resources"
    resources.mkdir()
    staging = output / "compile"
    compiled = {"stateless": compile_package(stateless, staging / "stateless"),
                "kv": compile_package(chunk, staging / "kv")}
    names = {"stateless": "TinyChineseV1INT8.mlmodelc", "kv": "TinyChineseV1KVChunkINT8.mlmodelc"}
    for key, name in names.items():
        shutil.copytree(compiled[key], resources / name)
    shutil.rmtree(staging)
    shutil.copyfile(Path(bundle) / "tokenizer.model", resources / "VimeChineseTokenizerV1.model")
    bundle_manifest = read_json(Path(bundle) / "manifest.json")
    manifest = {"format": "vime_ios_lm_zh_v1", "architecture": zh.ARCHITECTURE, "release_name": zh.RELEASE,
        "model_version": f"zh-v1-step{lm.metadata['checkpoint_step']}-int8-b32-v1",
        "minimum_ios": 18, "vocab_size": zh.VOCAB, "n_layers": zh.LAYERS, "context_length": zh.CONTEXT,
        "compute_units": "CPU_ONLY", "special_ids": lm.special,
        "reserved_ids": bundle_manifest.get("reserved_ids", {}),
        "tokenizer_sha256": lm.metadata["tokenizer_sha256"], "checkpoint_sha256": lm.metadata["checkpoint_sha256"],
        "bundle_manifest_sha256": lm.metadata["bundle_manifest_sha256"],
        "models": {key: {"resource": names[key],
                         "source_manifest_sha256": file_sha((stateless if key == "stateless" else chunk) / "manifest.json"),
                         "compiled_files_sha256": {n: e["sha256"] for n, e in
                                                   v2.inventory(resources / names[key], hashes=True).items()}}
                   for key in names},
        "stateless_interface": stateless_manifest["interface"], "kv_interface": chunk_manifest["interface"],
        "cache_shape": chunk_manifest["cache_shape"], "cache_dtype": "float32", "kv_graph": "chunk",
        "kind": "linear8_fp32_compute", "quantization": "weight-only symmetric INT8 block32; FP32 computation",
        "strict_fp32_logits_alignment_passed": False,
        "kv_vs_stateless_alignment_sha256": file_sha(alignment),
        "kv_chunk_fixture_max_suffix_sum_delta": max(deltas),
        "kv_chunk_fixture_within_1e-4": max(deltas) <= 1e-4,
        "scoring": "BOS + joint context/candidate tokenization; logP sum from the first token differing from the context alone; no EOS",
        "quality_review": "Test-host measurement candidate only; not shipped in Vime. See VimeML docs/reports/mac-*/zh-v1-coreml.md."}
    write_json(resources / f"VimeLMManifest{TAG}.json", manifest)
    write_json(output / f"VimeLMFixtures{TAG}.json", fixtures)
    write_json(output / "manifest.json", {"format": "vimeml_zh_v1_ios_resources_v1", "model": lm.metadata,
        "files": v2.inventory(output),
        "note": "Resources/ is for the Vime test host. No client source snapshot or device run included."})
    print(f"KV chunk fixture max suffix-sum delta: {max(deltas):.3g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "stateless", "chunk", "chunk-alignment", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    package(args.bundle, args.stateless, args.chunk, args.chunk_alignment, args.output)


if __name__ == "__main__":
    main()
