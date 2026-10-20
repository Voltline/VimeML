"""Package committed Git history, V4.3 epoch 30/27 inference weights, PyTorch fixtures, V4.2/V2.1
baseline packages and frozen evidence for Mac Core ML work."""

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

from vimeml.deployment.v4_variants import VARIANTS

TOKENIZER_DIR = "artifacts/tokenizers/v4-a/ja-unigram-8192"
TRAINING = "outputs/training-v43-20261020"
EVALUATION = "outputs/evaluation-v43-20261020"
EPOCHS = (30, 27)
INPUTS = [
    f"{TOKENIZER_DIR}/tokenizer.model",
    f"{TOKENIZER_DIR}/tokenizer.vocab",
    f"{TOKENIZER_DIR}/manifest.json",
    "artifacts/token-data/corpus-v4-8k/manifest.json",
    "outputs/evaluation-v4-20261015/inputs",
    "outputs/evaluation-v4-20261015/v21",
    "outputs/evaluation-v42-20261020",
    EVALUATION,
    f"{TRAINING}/download-receipt.json",
    f"{TRAINING}/remote/README.md",
    f"{TRAINING}/remote/manifest.json",
    f"{TRAINING}/remote/records/config.json",
    f"{TRAINING}/remote/records/plan.json",
    f"{TRAINING}/remote/records/environment.json",
    f"{TRAINING}/remote/records/summary.json",
    f"{TRAINING}/remote/records/full-validation.json",
    f"{TRAINING}/remote/records/epoch-validation-history.json",
    f"{TRAINING}/remote/records/progress.json",
    # V4.2 epoch 20 and V2.1 baselines for same-session size/latency comparisons (no rebuild).
    "artifacts/deployment/tiny-ja-v4.2-epoch20-bundle-v1",
    "artifacts/deployment/tiny-ja-v4.2-epoch20-fp32-v1",
    "artifacts/deployment/tiny-ja-v4.2-epoch20-int8-b32-v1",
    "artifacts/deployment/tiny-ja-v4.2-epoch20-kvchunk-fp32-v1",
    "artifacts/deployment/tiny-ja-v4.2-epoch20-kvchunk-int8-b32-v1",
    "artifacts/deployment/tiny-ja-v4.2-epoch20-client-resources-v1",
    "artifacts/deployment/tiny-ja-v2.1-extend5-bundle-v1",
    "artifacts/deployment/tiny-ja-v2.1-extend5-fp32-v1",
    "artifacts/deployment/tiny-ja-v2.1-extend5-int8-b32-v1",
    "artifacts/deployment/tiny-ja-v2.1-extend5-kv-fp32-mac-v42-measure-v1",
    "artifacts/deployment/tiny-ja-v2.1-extend5-kv-int8-b32-mac-v42-measure-v1",
]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_weights(stage, epoch):
    import numpy as np
    import sentencepiece as spm
    import torch
    from vimeml.training.data import file_sha
    from vimeml.training.infer_v4 import JapaneseLMV4

    torch.set_num_threads(4)
    variant = VARIANTS[f"v43-epoch{epoch}"]
    source = f"{TRAINING}/remote/checkpoints/epoch-{epoch}.pt"
    checkpoint = ROOT / source
    receipt = json.loads((ROOT / TRAINING / "download-receipt.json").read_text(encoding="utf-8"))
    expected = next(r["sha256"] for r in receipt["files"] if r["path"] == f"checkpoints/epoch-{epoch}.pt")
    digest = file_sha(checkpoint)
    if receipt["status"] != "verified" or digest != expected:
        raise ValueError(f"epoch-{epoch}.pt differs from the verified download receipt.")
    plan = json.loads((ROOT / variant.cuda_scores / "plan.json").read_text(encoding="utf-8"))
    if plan["checkpoint_sha256"] != digest:
        raise ValueError("Checkpoint differs from the frozen local evaluation.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["format"] != "vimeml_tiny_gpt_v4" or saved["step"] != variant.step or saved["epoch"] != epoch - 1:
        raise ValueError(f"Expected frozen V4.3 epoch {epoch}.")
    exported = {key: saved[key] for key in ("format", "architecture", "model", "model_config",
                                            "step", "epoch", "signatures", "tokenizer_identity")}
    exported["config"] = {key: saved["config"][key] for key in ("architecture", "token_dir")}
    exported["deployment_only"] = True
    exported["source_checkpoint_sha256"] = digest
    destination = stage / "payload" / variant.checkpoint
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, destination)
    reloaded = torch.load(destination, map_location="cpu", weights_only=True)
    if reloaded["model"].keys() != saved["model"].keys() or any(
            not torch.equal(value, reloaded["model"][name]) for name, value in saved["model"].items()):
        raise ValueError("Inference export changed a model tensor.")
    del saved

    lm = JapaneseLMV4(destination, ROOT / TOKENIZER_DIR, "cpu")
    model = lm.model
    processor = spm.SentencePieceProcessor(model_file=str(ROOT / TOKENIZER_DIR / "tokenizer.model"))
    text = "今日はとてもいい天気です。駅まで歩いて、この本を読んで、日本語を勉強します。"
    content = processor.encode(text * 20, out_type=int)
    bos, pad = processor.bos_id(), processor.pad_id()
    cases = {"bos": [bos], "short": [bos, *content[:15]], "long": [bos, *content[:127]]}
    cases["right_pad"] = cases["short"] + [pad] * (128 - len(cases["short"]))
    cases["changed_future"] = cases["short"] + content[:128 - len(cases["short"])]
    arrays = {}
    with torch.inference_mode():
        for name, ids in cases.items():
            inputs = torch.tensor([ids], dtype=torch.int32)
            arrays[f"{name}_input_ids"] = inputs.numpy()
            arrays[f"{name}_logits"] = model(inputs.long()).numpy()
        # Single-token KV steps over the long case, the Core ML KV graph's contract.
        cache, steps = model.new_cache(), []
        long_ids = torch.tensor([cases["long"]], dtype=torch.int32)
        for position in range(long_ids.shape[1]):
            steps.append(model.forward_cached(long_ids[:, position:position + 1], cache))
        arrays["long_kv_step_logits"] = torch.cat(steps, dim=1).numpy()
    reference = stage / "payload" / variant.reference
    reference.mkdir(parents=True)
    np.savez_compressed(reference / "logits.npz", **arrays)
    write_json(reference / "reference.json", {
        "format": "vimeml_v4_mac_preparation_reference_v1",
        "checkpoint_step": variant.step,
        "checkpoint_epoch": epoch,
        "checkpoint_sha256": digest,
        "tokenizer_sha256": lm.metadata["tokenizer_sha256"],
        "precision": "fp32", "device": "cpu", "torch_version": torch.__version__,
        "cases": {name: {"length": len(ids),
                         "valid_prefix": len(cases["short"]) if name in {"right_pad", "changed_future"} else len(ids)}
                  for name, ids in cases.items()},
        "kv_steps": {"case": "long", "array": "long_kv_step_logits",
                     "method": "TinyGPTV4.forward_cached one token at a time from an empty cache"},
        "policy": "PyTorch preparation fixtures; not Core ML validation, production quality or device timing.",
    })
    return {
        "variant": variant.name,
        "checkpoint_step": variant.step,
        "checkpoint_epoch": epoch,
        "selection": ("best complete LM validation BPC" if epoch == 30 else "requested comparison epoch")
                     + "; named before local IME evaluation",
        "source_checkpoint": source,
        "source_checkpoint_sha256": digest,
        "deployment_checkpoint": variant.checkpoint,
        "deployment_checkpoint_sha256": file_sha(destination),
        "tokenizer": TOKENIZER_DIR,
        "tokenizer_sha256": lm.metadata["tokenizer_sha256"],
        "model_config": lm.metadata["model"],
        "parameter_count": model.parameter_count(),
        "optimizer_included": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output exists; choose a new handoff version.")
    if git("status", "--porcelain", "--untracked-files=no"):
        parser.error("Commit tracked source changes before creating the portable Git bundle.")
    branch = git("branch", "--show-current")
    if not branch:
        parser.error("A named source branch is required.")
    from vimeml.training.data import file_sha
    from vimeml.posttraining.infer import TRACKS, read_snapshot

    for track in TRACKS:
        read_snapshot(track)  # Verifies pinned candidate-pool hashes before packaging.
    maintenance = ROOT / "outputs/maintenance"
    maintenance.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mac-v43-package-", dir=maintenance) as temporary:
        stage = Path(temporary)
        metadata = {f"epoch{epoch}": prepare_weights(stage, epoch) for epoch in EPOCHS}
        payload = stage / "payload"
        for relative in INPUTS:
            source = ROOT / relative
            if not source.exists():
                raise FileNotFoundError(source)
            files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
            for path in files:
                if (path.is_symlink() or path.name.startswith(".env")
                        or path.name in {".netrc", "credentials"} or path.suffix == ".pyc"):
                    raise ValueError(f"Unexpected input: {path}")
                target = payload / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        subprocess.run(["git", "bundle", "create", str(stage / "repository.bundle"), branch], cwd=ROOT, check=True)
        subprocess.run(["git", "bundle", "verify", str(stage / "repository.bundle")], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        manifest = {
            "format": "vimeml_windows_to_mac_v43",
            "source_commit": git("rev-parse", "HEAD"),
            "source_branch": branch,
            "mac_branch": "claude/mac-v43-coreml",
            "checkout_name": "VimeML-v43-coreml",
            "handoff_record": "outputs/deployment/mac-v43-handoff.json",
            "remote": git("remote", "get-url", "origin"),
            "models": metadata,
            "references": {f"epoch{e}": VARIANTS[f"v43-epoch{e}"].reference for e in EPOCHS},
            "frozen_tracks": {k: {"cases": v[0], "sha256": v[1]} for k, v in TRACKS.items()},
            "blind_included": False,
            "test_included": False,
            "coreml_conversion_done": False,
            "payload_files": [
                {"path": p.relative_to(payload).as_posix(), "bytes": p.stat().st_size, "sha256": file_sha(p)}
                for p in sorted(payload.rglob("*")) if p.is_file()
            ],
            "verification": "Receipt/evaluation checkpoint hash; exact tensor equality; pinned track hashes; Git bundle verify; per-file SHA256; ZIP entry paths/sizes.",
        }
        write_json(stage / "handoff-manifest.json", manifest)
        shutil.copy2(ROOT / "scripts/deployment/setup_mac.py", stage / "setup_mac.py")
        shutil.copy2(ROOT / "docs/reference/mac-v43-coreml.md", stage / "README.md")
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
                   "file_count": len(expected), "source_commit": manifest["source_commit"], "models": metadata}
        write_json(output.with_suffix(".json"), summary)
        print(json.dumps({k: summary[k] for k in ("zip", "bytes", "sha256", "file_count", "source_commit")},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
