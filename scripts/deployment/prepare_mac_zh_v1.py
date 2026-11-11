"""Package the committed main branch, vimeml-tiny-zh-v1 inference weights, PyTorch FP32 fixtures and the
development split of ime-standard-zh-v1-r1 with v1's reference scores, for Core ML and quantization on a Mac.

Blind items never enter the package: every benchmark file is filtered to split == development.
Brief: docs/reference/mac-zh-v1-coreml.md.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CHECKPOINT = "artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/final.pt"
CHECKPOINT_SHA256 = "44c423a1e7cb0e32161afc94d6167cabb439a3d25fd9de6d1fbb150a927542d4"
DEPLOYMENT = "artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/deployment.pt"
TOKENIZER_DIR = "artifacts/tokenizers/zh-v13/zh-unigram-12288"
TOKENIZER_SHA256 = "68bd37c45bcb018b32b2eef5b241561ab5b6dc1b77c1c3aadcc1b41c4bf47e5a"
REFERENCE = "outputs/deployment/vimeml-tiny-zh-v1-fp32-input"
FROZEN = "outputs/benchmarks/ime-standard-zh-v1/frozen-r1"
FILTERED = ("items.jsonl", "pool-full.jsonl", "pool-abbrev.jsonl", "rime-full.jsonl", "rime-abbrev.jsonl",
            "cand-full.jsonl", "cand-abbrev.jsonl", "overlap/overlap.jsonl")
SCORES = "outputs/benchmarks/zh13-20m/runs/12k-fresh-s1"
RESULTS = ("outputs/benchmarks/zh13-20m/judgement/scores/12k-fresh-s1-full",
           "outputs/benchmarks/zh13-20m/judgement/scores/12k-fresh-s1-abbrev",
           "outputs/benchmarks/zh13-20m/judgement/scores/rime-full",
           "outputs/benchmarks/zh13-20m/judgement/scores/rime-abbrev")
EXPECTED_TOP1 = {"full": 1363, "abbrev": 337}
COPIED = (f"{TOKENIZER_DIR}/tokenizer.model", f"{TOKENIZER_DIR}/tokenizer.vocab",
          "artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/config.json",
          "artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/summary.json",
          "artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/environment.json")
TEXT = "你好，我想找一家离地铁站近一点的酒店，晚上想吃火锅，你有推荐的餐馆吗？明天上午我们去故宫参观，下午再去逛街。"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_weights(payload):
    import numpy as np
    import sentencepiece as spm
    import torch
    from vimeml.benchmarks.zh13_lm_scores import load
    from vimeml.training.data import file_sha

    torch.set_num_threads(4)
    if file_sha(ROOT / CHECKPOINT) != CHECKPOINT_SHA256 or file_sha(ROOT / TOKENIZER_DIR / "tokenizer.model") != TOKENIZER_SHA256:
        raise ValueError("Checkpoint or tokenizer differs from the confirmed vimeml-tiny-zh-v1")
    saved = torch.load(ROOT / CHECKPOINT, map_location="cpu", weights_only=True)
    if any(t.dtype != torch.float32 for t in saved["model"].values()):
        raise ValueError("Expected FP32 master weights")
    exported = {key: saved[key] for key in ("format", "architecture", "model", "model_config", "step", "epoch",
                                            "signatures", "tokenizer_identity")}
    exported["config"] = {key: saved["config"][key] for key in ("arm", "architecture", "stage")}
    exported.update(deployment_only=True, source_checkpoint_sha256=CHECKPOINT_SHA256, release_name="vimeml-tiny-zh-v1")
    destination = payload / DEPLOYMENT
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, destination)
    reloaded = torch.load(destination, map_location="cpu", weights_only=True)
    if reloaded["model"].keys() != saved["model"].keys() or any(
            not torch.equal(value, reloaded["model"][name]) for name, value in saved["model"].items()):
        raise ValueError("Inference export changed a model tensor")
    del saved

    model, _, meta = load(destination, ROOT / TOKENIZER_DIR / "tokenizer.model", "cpu")
    processor = spm.SentencePieceProcessor(model_file=str(ROOT / TOKENIZER_DIR / "tokenizer.model"))
    content = processor.encode(TEXT * 8, out_type=int)
    bos, pad = processor.bos_id(), processor.pad_id()
    if (bos, pad) != (2, 0) or len(content) < 127:
        raise ValueError("Unexpected tokenizer specials or fixture text too short")
    cases = {"bos": [bos], "short": [bos, *content[:15]], "long": [bos, *content[:127]]}
    cases["right_pad"] = cases["short"] + [pad] * (128 - len(cases["short"]))
    cases["changed_future"] = cases["short"] + content[:128 - len(cases["short"])]
    arrays = {}
    with torch.inference_mode():
        for name, ids in cases.items():
            inputs = torch.tensor([ids], dtype=torch.int32)
            arrays[f"{name}_input_ids"] = inputs.numpy()
            arrays[f"{name}_logits"] = model(inputs.long()).numpy()
        cache, steps = model.new_cache(), []
        long_ids = torch.tensor([cases["long"]], dtype=torch.int32)
        for position in range(long_ids.shape[1]):
            steps.append(model.forward_cached(long_ids[:, position:position + 1], cache))
        arrays["long_kv_step_logits"] = torch.cat(steps, dim=1).numpy()
    worst = float(np.abs(arrays["long_kv_step_logits"] - arrays["long_logits"]).max())
    if worst > 3e-4:
        raise ValueError(f"forward_cached differs from forward by {worst}")
    reference = payload / REFERENCE
    reference.mkdir(parents=True)
    np.savez_compressed(reference / "logits.npz", **arrays)
    write_json(reference / "reference.json", {
        "format": "vimeml_zh_v1_mac_preparation_reference_v1",
        "release_name": "vimeml-tiny-zh-v1", "checkpoint_step": meta["step"],
        "checkpoint_sha256": CHECKPOINT_SHA256, "deployment_checkpoint_sha256": file_sha(destination),
        "tokenizer_sha256": TOKENIZER_SHA256, "precision": "fp32", "device": "cpu", "torch_version": torch.__version__,
        "cases": {name: {"length": len(ids),
                         "valid_prefix": len(cases["short"]) if name in {"right_pad", "changed_future"} else len(ids)}
                  for name, ids in cases.items()},
        "kv_steps": {"case": "long", "array": "long_kv_step_logits", "max_abs_vs_stateless": worst,
                     "method": "TinyGPTZh13.forward_cached one token at a time from an empty cache"},
        "policy": "PyTorch preparation fixtures; not Core ML validation, production quality or device timing.",
    })
    return {"release_name": "vimeml-tiny-zh-v1", "arm": "12k-fresh-s1", "checkpoint_step": meta["step"],
            "source_checkpoint": CHECKPOINT, "source_checkpoint_sha256": CHECKPOINT_SHA256,
            "deployment_checkpoint": DEPLOYMENT, "deployment_checkpoint_sha256": file_sha(destination),
            "tokenizer": TOKENIZER_DIR, "tokenizer_sha256": TOKENIZER_SHA256,
            "model_config": exported["model_config"], "parameter_count": model.parameter_count(),
            "kv_cache_shape": [exported["model_config"]["n_layers"], 1, 6, 128, 64], "optimizer_included": False}


def prepare_benchmark(payload):
    items = [i for i in read_jsonl(ROOT / FROZEN / "items.jsonl") if i["split"] == "development"]
    ids = {i["id"] for i in items}
    if len(ids) != 1456:
        raise ValueError("Expected 1,456 development items")
    for name in FILTERED:
        rows = items if name == "items.jsonl" else [r for r in read_jsonl(ROOT / FROZEN / name) if r["id"] in ids]
        write_jsonl(payload / FROZEN / name, rows)
    source = json.loads((ROOT / FROZEN / "manifest.json").read_text(encoding="utf-8"))
    write_json(payload / FROZEN / "manifest.json", {
        "version": source["version"], "split_included": "development only", "labels": source["labels"],
        "reading_of_谁": source["reading_of_谁"], "items": len(items),
        "abbrev_items": sum(i["keys_abbrev"] is not None for i in items),
        "strata": {k: v for k, v in source["strata"].items() if "/development/" in k}})
    (payload / SCORES).mkdir(parents=True)
    for track in ("full", "abbrev"):
        if {r["id"] for r in read_jsonl(ROOT / SCORES / f"{track}.jsonl")} - ids:
            raise ValueError("Reference scores contain non-development items")
        for name in (f"{track}.jsonl", f"{track}.meta.json"):
            shutil.copy2(ROOT / SCORES / name, payload / SCORES / name)
    for directory in RESULTS:
        summary = json.loads((ROOT / directory / "summary.json").read_text(encoding="utf-8"))
        if summary["split"] != "development":
            raise ValueError(f"{directory} is not a development result")
        (payload / directory).mkdir(parents=True)
        for name in ("items.jsonl", "summary.json"):
            shutil.copy2(ROOT / directory / name, payload / directory / name)
    for track, expected in EXPECTED_TOP1.items():
        summary = json.loads((ROOT / RESULTS[0 if track == "full" else 1] / "summary.json").read_text(encoding="utf-8"))
        if summary["all"]["top1"] != expected:
            raise ValueError(f"v1 {track} development Top-1 is not {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output exists; choose a new handoff version.")
    if git("status", "--porcelain", "--untracked-files=no"):
        parser.error("Commit tracked source changes before creating the portable Git bundle.")
    if git("branch", "--show-current") != "main" or git("rev-parse", "HEAD") != git("rev-parse", "origin/main"):
        parser.error("Package the pushed main branch.")
    from vimeml.training.data import file_sha

    maintenance = ROOT / "outputs/maintenance"
    maintenance.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mac-zh-v1-package-", dir=maintenance) as temporary:
        stage = Path(temporary)
        payload = stage / "payload"
        model = prepare_weights(payload)
        prepare_benchmark(payload)
        for relative in COPIED:
            target = payload / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        for path in payload.rglob("*"):
            if path.is_file() and (path.is_symlink() or path.name.startswith(".env") or path.suffix == ".pyc"):
                raise ValueError(f"Unexpected input: {path}")
        subprocess.run(["git", "bundle", "create", str(stage / "repository.bundle"), "main"], cwd=ROOT, check=True)
        subprocess.run(["git", "bundle", "verify", str(stage / "repository.bundle")], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        manifest = {
            "format": "vimeml_windows_to_mac_zh_v1",
            "source_commit": git("rev-parse", "HEAD"),
            "source_branch": "main",
            "mac_branch": "main",
            "checkout_name": "VimeML-zh-v1-coreml",
            "handoff_record": "outputs/deployment/mac-zh-v1-handoff.json",
            "remote": git("remote", "get-url", "origin"),
            "model": model,
            "reference": REFERENCE,
            "benchmark": {"version": "ime-standard-zh-v1-r1", "split_included": "development",
                          "frozen": FROZEN, "reference_scores": SCORES,
                          "reference_top1": EXPECTED_TOP1, "rime_top1": {"full": 1001, "abbrev": 142}},
            "blind_included": False,
            "coreml_conversion_done": False,
            "payload_files": [
                {"path": p.relative_to(payload).as_posix(), "bytes": p.stat().st_size, "sha256": file_sha(p)}
                for p in sorted(payload.rglob("*")) if p.is_file()
            ],
            "verification": "Checkpoint and tokenizer SHA256; exact tensor equality; KV vs stateless logits; "
                            "development-only filter; reference Top-1; Git bundle verify; per-file SHA256; ZIP CRC.",
        }
        write_json(stage / "handoff-manifest.json", manifest)
        shutil.copy2(ROOT / "scripts/deployment/setup_mac.py", stage / "setup_mac.py")
        shutil.copy2(ROOT / "docs/reference/mac-zh-v1-coreml.md", stage / "README.md")
        output.parent.mkdir(parents=True, exist_ok=True)
        expected = {}
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    name = path.relative_to(stage).as_posix()
                    archive.write(path, name)
                    expected[name] = path.stat().st_size
        with zipfile.ZipFile(output) as archive:
            if {p.filename: p.file_size for p in archive.infolist()} != expected or archive.testzip() is not None:
                raise ValueError("ZIP directory or CRC differs from packaged files.")
        summary = {"zip": output.name, "bytes": output.stat().st_size, "sha256": file_sha(output),
                   "file_count": len(expected), "source_commit": manifest["source_commit"], "model": model}
        write_json(output.with_suffix(".json"), summary)
        print(json.dumps({k: summary[k] for k in ("zip", "bytes", "sha256", "file_count", "source_commit")},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
