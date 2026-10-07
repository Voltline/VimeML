"""Package committed Git history, V2 inference weights and frozen development evidence."""

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


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def prepare_weights(stage):
    import numpy as np
    import sentencepiece as spm
    import torch
    from vimeml.training.data import file_sha
    from vimeml.training.model_factory import model_from_checkpoint

    torch.set_num_threads(4)
    model_dir = "artifacts/models/tiny-ja-v2.1-e16k-d320-l6-extend5"
    checkpoint = ROOT / model_dir / "best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    score = json.loads(
        (
            ROOT / "outputs/ime-eval/tiny-ja-v2.1-extend5-best-ajimee/metrics.json"
        ).read_text()
    )
    if saved["format"] != "vimeml_tiny_gpt_v2" or saved["step"] != 40000:
        raise ValueError("Expected frozen V2.1 extend5 best step40000.")
    if saved["model_config"] != score["model"]["model"]:
        raise ValueError("Model configuration differs from the frozen FP32 report.")
    token_manifest = ROOT / saved["config"]["token_dir"] / "manifest.json"
    tokenizer = ROOT / "artifacts/tokenizers/ja-unigram-16k-v2/tokenizer.model"
    if (
        file_sha(token_manifest) != saved["signatures"]["tokens"]
        or file_sha(tokenizer) != score["model"]["tokenizer_sha256"]
    ):
        raise ValueError(
            "Tokenizer or small token manifest differs from frozen provenance."
        )
    exported = {
        key: saved[key]
        for key in (
            "format",
            "architecture",
            "model",
            "model_config",
            "step",
            "signatures",
        )
    }
    exported["config"] = {
        key: saved["config"][key] for key in ("architecture", "token_dir")
    }
    exported["deployment_only"] = True
    exported["source_checkpoint_sha256"] = score["model"]["checkpoint_sha256"]
    destination = stage / "payload" / model_dir / "deployment.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, destination)
    reloaded = torch.load(destination, map_location="cpu", weights_only=True)
    if any(
        not torch.equal(value, reloaded["model"][name])
        for name, value in saved["model"].items()
    ):
        raise ValueError("Inference export changed a model tensor.")
    model = model_from_checkpoint(reloaded).eval()
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    text = (
        "今日はとてもいい天気です。駅まで歩いて、この本を読んで、日本語を勉強します。"
    )
    content = processor.encode(text * 20, out_type=int)
    cases = {
        "bos": [processor.bos_id()],
        "short": [processor.bos_id(), *content[:15]],
        "long": [processor.bos_id(), *content[:127]],
    }
    cases["right_pad"] = cases["short"] + [processor.pad_id()] * (
        128 - len(cases["short"])
    )
    cases["changed_future"] = cases["short"] + content[: 128 - len(cases["short"])]
    arrays = {}
    with torch.inference_mode():
        for name, ids in cases.items():
            inputs = torch.tensor([ids], dtype=torch.int32)
            arrays[f"{name}_input_ids"] = inputs.numpy()
            arrays[f"{name}_logits"] = model(inputs.long()).numpy()
    reference = stage / "payload/outputs/deployment/tiny-ja-v2.1-extend5-fp32-input"
    reference.mkdir(parents=True)
    np.savez_compressed(reference / "logits.npz", **arrays)
    write_json(
        reference / "reference.json",
        {
            "format": "vimeml_v2_mac_preparation_reference_v1",
            "checkpoint_step": saved["step"],
            "precision": "fp32",
            "device": "cpu",
            "torch_version": torch.__version__,
            "cases": {
                name: {
                    "length": len(ids),
                    "valid_prefix": len(cases["short"])
                    if name in {"right_pad", "changed_future"}
                    else len(ids),
                }
                for name, ids in cases.items()
            },
            "policy": "PyTorch preparation fixtures; not Core ML validation, production quality or device timing.",
        },
    )
    return {
        "checkpoint_step": saved["step"],
        "source_checkpoint": f"{model_dir}/best.pt",
        "source_checkpoint_sha256": score["model"]["checkpoint_sha256"],
        "deployment_checkpoint": f"{model_dir}/deployment.pt",
        "model_config": saved["model_config"],
        "parameter_count": model.parameter_count(),
        "optimizer_included": False,
        "source_fingerprint_policy": "Reused existing FP32 evaluation provenance; no repeated checkpoint SHA256.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output exists; choose a new handoff version.")
    if git("status", "--porcelain"):
        parser.error("Commit source changes before creating the portable Git bundle.")
    branch = git("branch", "--show-current")
    if not branch:
        parser.error("A named source branch is required.")
    maintenance = ROOT / "outputs/maintenance"
    maintenance.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="mac-v21-package-", dir=maintenance
    ) as temporary:
        stage = Path(temporary)
        metadata = prepare_weights(stage)
        payload = stage / "payload"
        inputs = [
            "artifacts/tokenizers/ja-unigram-16k-v2/tokenizer.model",
            "artifacts/tokenizers/ja-unigram-16k-v2/tokenizer.vocab",
            "artifacts/token-data/corpus-v2-16k/manifest.json",
            "artifacts/benchmarks/ajimee-jwtd-v2-v1",
            "artifacts/benchmarks/ime-dev-v2",
            "artifacts/benchmarks/ime-expanded-v21-candidates-v1/development",
            "artifacts/deployment/tiny-ja-v1-conservative-int8-b32-v1",
            "artifacts/tokenizers/ja-unigram-16k-v1/tokenizer.model",
            "outputs/ime-eval/tiny-ja-v2.1-extend5-best-ajimee",
            "outputs/ime-eval/tiny-ja-v2.1-extend5-best-development",
            "outputs/ime-eval/expanded-v21-dev-draft-extend5-best",
            "outputs/ime-eval/tiny-ja-v2.1-extend5-comparison/comparison.json",
            "outputs/model-checks/tiny-ja-v2.1-extend5-best",
            "outputs/model-checks/v21-extend5-closeout/full-validation.json",
        ]
        for relative in inputs:
            source = ROOT / relative
            if not source.exists():
                raise FileNotFoundError(source)
            files = (
                [source]
                if source.is_file()
                else sorted(p for p in source.rglob("*") if p.is_file())
            )
            for path in files:
                if (
                    path.is_symlink()
                    or path.name.startswith(".env")
                    or path.name in {".netrc", "credentials"}
                ):
                    raise ValueError(f"Unexpected input: {path}")
                target = payload / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        subprocess.run(
            ["git", "bundle", "create", str(stage / "repository.bundle"), branch],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            ["git", "bundle", "verify", str(stage / "repository.bundle")],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        manifest = {
            "format": "vimeml_windows_to_mac_v2",
            "source_commit": git("rev-parse", "HEAD"),
            "source_branch": branch,
            "mac_branch": "codex/mac-v21-coreml",
            "remote": git("remote", "get-url", "origin"),
            "model": metadata,
            "blind_included": False,
            "coreml_conversion_done": False,
            "payload_files": [
                {"path": p.relative_to(payload).as_posix(), "bytes": p.stat().st_size}
                for p in sorted(payload.rglob("*"))
                if p.is_file()
            ],
            "verification": "Exact tensor equality once; Git bundle verify; ZIP entry paths/counts/sizes. No per-file SHA256 or archive reread.",
        }
        write_json(stage / "handoff-manifest.json", manifest)
        shutil.copy2(ROOT / "scripts/deployment/setup_mac.py", stage / "setup_mac.py")
        shutil.copy2(ROOT / "docs/mac-v21-preparation.md", stage / "README.md")
        output.parent.mkdir(parents=True, exist_ok=True)
        expected = {}
        with zipfile.ZipFile(
            output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3
        ) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    name = path.relative_to(stage).as_posix()
                    archive.write(path, name)
                    expected[name] = path.stat().st_size
        with zipfile.ZipFile(output) as archive:
            entries = archive.infolist()
            if (
                len(entries) != len(expected)
                or {p.filename: p.file_size for p in entries} != expected
            ):
                raise ValueError("ZIP directory differs from packaged files.")
        write_json(
            output.with_suffix(".json"),
            {
                "zip": output.name,
                "bytes": output.stat().st_size,
                "file_count": len(expected),
                "source_commit": manifest["source_commit"],
                "model": metadata,
            },
        )
        print(
            json.dumps(
                {
                    "zip": str(output),
                    "bytes": output.stat().st_size,
                    "file_count": len(expected),
                    "source_commit": manifest["source_commit"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
