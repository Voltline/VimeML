"""Prepare the vimeml-tiny-zh-v1 batched-KV client resource for Vime (Mac only: needs coremlcompiler).

Writes Resources/TinyChineseV1KVBatchINT8.mlmodelc and Resources/VimeLMManifestZhV1KVBatch.json, a
`kv_batch` section with the fields of the existing manifest's kv entry (resource, compiled_files_sha256,
interface, cache_shape, alignment_tolerance, alignment_checks, policy) plus the batch range. The existing
TinyChineseV1INT8 / TinyChineseV1KVChunkINT8 resources and VimeLMManifestZhV1.json are not touched.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/deployment"))

from package_zh_v1_ios import compile_package  # noqa: E402
from vimeml.deployment import v2, zh_v1 as zh, zh_v1_batch as zb  # noqa: E402
from vimeml.deployment.bundle import fresh_directory, read_json, write_json  # noqa: E402
from vimeml.training.data import file_sha  # noqa: E402

RESOURCE = "TinyChineseV1KVBatchINT8.mlmodelc"


def package(batch, alignment, search, fixture, output):
    manifest = zh.verify_package(batch, zb.FORMAT)
    if manifest.get("recipe") != zh.DEPLOYED_RECIPE or manifest["minimum_ios"] != 18:
        raise ValueError("Expected the deployed INT8 block32 FP32-compute iOS18 recipe.")
    gate = read_json(alignment)
    if (gate.get("format") != zb.ALIGNMENT_FORMAT or not gate.get("passed")
            or gate["coreml_manifest_sha256"] != file_sha(batch / "manifest.json")
            or not gate["reference"].startswith("stateless Core ML")):
        raise ValueError("Passed same-precision batch/stateless alignment of this package required.")
    checked = read_json(search / "summary.json")
    if checked["package_manifest_sha256"] != file_sha(batch / "manifest.json"):
        raise ValueError("Search check belongs to another package.")
    fresh_directory(output)
    resources = output / "Resources"
    resources.mkdir()
    compiled = compile_package(batch, output / "compile")
    shutil.copytree(compiled, resources / RESOURCE)
    shutil.rmtree(output / "compile")
    for name in ("batch16.jsonl", "batch16_logits.npz", "meta.json"):
        shutil.copyfile(fixture / name, output / f"KVBatchFixture-{name}")
    section = {"resource": RESOURCE, "source_manifest_sha256": file_sha(batch / "manifest.json"),
        "compiled_files_sha256": {n: e["sha256"] for n, e in v2.inventory(resources / RESOURCE, hashes=True).items()},
        "interface": manifest["interface"], "cache_shape": manifest["cache_shape"], "cache_slots": manifest["cache_slots"],
        "cache_dtype": manifest["cache_dtype"], "batch_range": manifest["batch_range"], "batch": manifest["batch"],
        "rows": manifest["rows"], "prompt": manifest["prompt"],
        "alignment_tolerance": gate["tolerance"], "alignment_reference": gate["reference"],
        "alignment_checks": gate["checks"], "alignment_sha256": file_sha(alignment),
        "search_check": checked["summary"],
        "policy": manifest["policy"] + " Write new_key/new_value into slot position[b] of row b yourself.",
        "kind": manifest["kind"], "compute_units": manifest["compute_units"]}
    write_json(resources / "VimeLMManifestZhV1KVBatch.json", {"format": "vime_ios_lm_zh_v1_kv_batch",
        "release_name": zh.RELEASE, "bundle_manifest_sha256": manifest["bundle"]["bundle_manifest_sha256"],
        "tokenizer_sha256": manifest["bundle"]["tokenizer_sha256"], "kv_batch": section})
    write_json(output / "manifest.json", {"format": "vimeml_zh_v1_ios_kv_batch_resources_v1",
        "files": v2.inventory(output), "note": "Add Resources/ beside the existing zh-v1 resources."})
    print(f"{RESOURCE} ready")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("batch", "alignment", "search", "fixture", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    package(a.batch, a.alignment, a.search, a.fixture, a.output)


if __name__ == "__main__":
    main()
