"""Inference-only bundles and portable provenance; never rewrite source artifacts."""
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import sentencepiece as spm
import torch

from vimeml.training.data import file_sha
from vimeml.training.infer import JapaneseLM, ROOT
from vimeml.training.model import GPTConfig, TinyGPT

FORMAT = "vimeml_inference_bundle_v1"
CORE = (
    "src/vimeml/training/model.py", "src/vimeml/training/data.py",
    "src/vimeml/training/train.py", "src/vimeml/training/infer.py",
    "src/vimeml/training/evaluate_ime.py", "src/vimeml/benchmarks/evaluate_ajimee.py",
    "src/vimeml/tools/phrase_demo.py",
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fresh_directory(path):
    # Even empty preexisting directories are rejected; no overwrite switch.
    Path(path).mkdir(parents=True, exist_ok=False)


def environment():
    packages = {}
    for name in ("torch", "numpy", "sentencepiece", "coremltools"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages}


def source_fingerprints():
    paths = [*CORE, *(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "src/vimeml/deployment").glob("*.py")))]
    # Raw hashes preserve the original record; LF hashes allow Git CRLF/LF checkouts.
    return {name: {"sha256": file_sha(ROOT / name), "lf_sha256": hashlib.sha256(
        (ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()} for name in paths}


def tree_inventory(path):
    path = Path(path)
    return {p.relative_to(path).as_posix(): {"bytes": p.stat().st_size, "sha256": file_sha(p)}
            for p in sorted(path.rglob("*")) if p.is_file()}


def export_bundle(checkpoint, tokenizer, token_manifest, output):
    checkpoint, tokenizer, token_manifest, output = map(Path, (checkpoint, tokenizer, token_manifest, output))
    if output.exists():
        raise ValueError("Bundle output already exists; choose a new versioned directory.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "vimeml_tiny_gpt_v1":
        raise ValueError("Unsupported checkpoint.")
    if file_sha(token_manifest) != saved["signatures"]["tokens"]:
        raise ValueError("Token manifest is not bound to this checkpoint.")
    manifest = read_json(token_manifest)
    expected = [value for name, value in manifest["input_sha256"].items()
                if name.replace("\\", "/").rsplit("/", 1)[-1] == "tokenizer.model"]
    if expected != [file_sha(tokenizer)]:
        raise ValueError("Tokenizer does not match checkpoint.")
    fingerprints = source_fingerprints()
    for name, digest in saved["signatures"]["training_code"].items():
        if file_sha(ROOT / "src/vimeml/training" / name) != digest:
            raise ValueError(f"Frozen training code fingerprint changed: {name}")
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    special = {name: getattr(processor, f"{name}_id")() for name in ("pad", "unk", "bos", "eos")}
    config = GPTConfig(**saved["model_config"])
    if special != manifest["special_ids"] or processor.vocab_size() != config.vocab_size:
        raise ValueError("Vocabulary/special IDs mismatch.")
    state = saved["model"]
    if not torch.equal(state["lm_head.weight"], state["token_embedding.weight"]):
        raise ValueError("Expected identical tied embedding and LM head.")
    model = TinyGPT(config).eval()
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite frozen weight.")
    # Keep only unique inference weights as FP32; TinyGPT re-establishes tying.
    weights = {name: tensor.detach().cpu().float().contiguous() for name, tensor in state.items()
               if name != "lm_head.weight"}
    fresh_directory(output)
    torch.save(weights, output / "weights.pt")
    write_json(output / "config.json", model.configuration())
    shutil.copyfile(tokenizer, output / "tokenizer.model")
    shutil.copyfile(token_manifest, output / "token-manifest.json")
    result = {"format": FORMAT, "status": "complete", "parameter_count": model.parameter_count(),
              "precision": "fp32", "special_ids": special, "tied_lm_head": True,
              "checkpoint_sha256": file_sha(checkpoint), "checkpoint_step": saved["step"],
              "training_signatures": saved["signatures"], "source_fingerprints": fingerprints,
              "environment": environment(), "files": tree_inventory(output),
              "policy": "Weights only; no optimizer, RNG or training state. LM-only contextual suffix logP sum; no extra EOS; no KV cache."}
    write_json(output / "manifest.json", result)
    return result


def verify_bundle(path, check_code=True):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if manifest.get("format") != FORMAT or manifest.get("status") != "complete":
        raise ValueError("Unsupported/incomplete inference bundle.")
    if set(manifest["files"]) != {"weights.pt", "config.json", "tokenizer.model", "token-manifest.json"}:
        raise ValueError("Unexpected inference bundle file list.")
    for name, entry in manifest["files"].items():
        if file_sha(path / name) != entry["sha256"] or (path / name).stat().st_size != entry["bytes"]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    if check_code:
        for name, entry in manifest["source_fingerprints"].items():
            digest = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if digest != entry["lf_sha256"]:
                raise ValueError(f"Bundle code changed: {name}; re-export as a new bundle.")
    return manifest


class BundleLM(JapaneseLM):
    """Use frozen inference/scoring/search methods without a training checkpoint."""
    def __init__(self, bundle):
        bundle = Path(bundle)
        manifest = verify_bundle(bundle)
        self.device = torch.device("cpu")
        self.processor = spm.SentencePieceProcessor(model_file=str(bundle / "tokenizer.model"))
        self.special = {name: getattr(self.processor, f"{name}_id")() for name in ("pad", "unk", "bos", "eos")}
        self.model = TinyGPT(GPTConfig(**read_json(bundle / "config.json"))).eval()
        if self.special != manifest["special_ids"] or self.processor.vocab_size() != self.model.config.vocab_size:
            raise ValueError("Bundle tokenizer contract mismatch.")
        state = torch.load(bundle / "weights.pt", map_location="cpu", weights_only=True)
        if "lm_head.weight" in state:
            raise ValueError("Bundle must store the shared embedding only once.")
        loaded = self.model.load_state_dict(state, strict=False)
        if loaded.missing_keys != ["lm_head.weight"] or loaded.unexpected_keys:
            raise ValueError(f"Unexpected inference state: {loaded}")
        if self.model.parameter_count() != manifest["parameter_count"]:
            raise ValueError("Parameter count mismatch.")
        self.forbidden = tuple(self.special[name] for name in ("pad", "unk", "bos"))
        self.metadata = {"checkpoint_sha256": manifest["checkpoint_sha256"],
                         "tokenizer_sha256": manifest["files"]["tokenizer.model"]["sha256"],
                         "bundle_manifest_sha256": file_sha(bundle / "manifest.json"),
                         "model": self.model.configuration(), "precision": "fp32", "device": "cpu",
                         "environment": environment()}
