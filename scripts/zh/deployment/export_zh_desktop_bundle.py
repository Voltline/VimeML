"""FP32 inference bundle of a zh desktop checkpoint (vimeml-zh-v2-200m / -800m) for the Hugging Face release.

  python scripts/zh/deployment/export_zh_desktop_bundle.py \
      artifacts/models/tiny-zh200m-12k-pinyin-pt-d1024/final.pt \
      artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model \
      artifacts/releases/vimeml-zh-v2-200m/inference --release vimeml-zh-v2-200m

`model.pt` keeps the checkpoint header that `zh13_lm_scores.load` reads (format, architecture, model_config,
tokenizer_identity) and the FP32 weights, without optimizer, scheduler or RNG state. The bundle is checked by
loading it back and comparing logits with the source checkpoint.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.benchmarks.zh13_lm_scores import load  # noqa: E402
from vimeml.training.data import file_sha  # noqa: E402
from vimeml.training.zh_desktop import desktop_family  # noqa: E402

FORMAT = "vimeml_inference_bundle_zh_desktop_v1"
KEEP = ("format", "architecture", "model_config", "tokenizer_identity", "step", "epoch")


def export(checkpoint, tokenizer, output, release):
    checkpoint, tokenizer, output = Path(checkpoint), Path(tokenizer), Path(output)
    if output.exists():
        raise SystemExit(f"{output} exists already")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if desktop_family(saved) is None:
        raise SystemExit("not a zh desktop checkpoint")
    if file_sha(tokenizer) != saved["tokenizer_identity"]["tokenizer_model_sha256"]:
        raise SystemExit("tokenizer differs from the checkpoint's tokenizer identity")
    source_sha = file_sha(checkpoint)
    bundle = {k: saved[k] for k in KEEP if k in saved}
    bundle["model"] = {k: v.detach().float().contiguous() for k, v in saved["model"].items()}
    bundle["config"] = {"arm": (saved.get("config") or {}).get("arm")}
    bundle.update(release_name=release, source_checkpoint_sha256=source_sha, deployment_only=True)
    output.mkdir(parents=True)
    torch.save(bundle, output / "model.pt")
    shutil.copyfile(tokenizer, output / "tokenizer.model")

    # check: identical logits from the source checkpoint and the bundle
    src, sp, _ = load(checkpoint, tokenizer)
    out, _, _ = load(output / "model.pt", output / "tokenizer.model")
    ids = torch.tensor([[2, 4] + sp.encode("今天中午我们一起去吃饭吧") + [5] + sp.encode("women") + [6]])
    with torch.no_grad():
        a, b = src.float()(ids), out(ids)
        a, b = (a[0] if isinstance(a, tuple) else a), (b[0] if isinstance(b, tuple) else b)
    diff = (a - b).abs().max().item()
    if diff != 0.0:
        raise SystemExit(f"bundle logits differ from the source checkpoint: {diff}")
    manifest = {
        "format": FORMAT, "status": "complete", "release_name": release, "architecture": saved["architecture"],
        "checkpoint_format": saved["format"], "arm": bundle["config"]["arm"], "precision": "fp32",
        "checkpoint_sha256": source_sha, "checkpoint_step": saved["step"],
        "parameter_count": sum(p.numel() for p in out.parameters()),
        "model_config": saved["model_config"], "tokenizer_identity": saved["tokenizer_identity"],
        "logits_check": {"max_abs_diff": diff, "positions": ids.shape[1]},
        "files": {name: {"bytes": (output / name).stat().st_size, "sha256": file_sha(output / name)}
                  for name in ("model.pt", "tokenizer.model")},
        "loader": "vimeml.benchmarks.zh13_lm_scores.load(model.pt, tokenizer.model)",
        "policy": "FP32 inference weights only; no optimizer, scheduler or RNG state.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf-8", newline="\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("tokenizer")
    parser.add_argument("output")
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    m = export(args.checkpoint, args.tokenizer, args.output, args.release)
    print(json.dumps({k: m[k] for k in ("release_name", "arm", "checkpoint_sha256", "checkpoint_step",
                                        "parameter_count", "logits_check")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
