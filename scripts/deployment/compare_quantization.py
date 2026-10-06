"""Distribution-level check of a Core ML package against the FP32 bundle.

Distribution diagnostics supplement the strict logits, PAD and ranking checks.
They do not establish exact conversion equivalence. This reports KL(FP32 || model),
NLL on real text, top-1 agreement and, as a control, the same metrics for a
PyTorch simulation of symmetric INT8 block quantization. Read-only.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from vimeml.deployment.bundle import BundleLM, environment
from vimeml.deployment.coreml import verify_package
from vimeml.training.data import file_sha


def simulate_int8(model, block):
    """Mirror the conservative recipe: FP16 storage except QKV/positions, then INT8 per block."""
    if block <= 0 or any(p.ndim == 2 and p.shape[1] % block for p in model.parameters()):
        raise ValueError("Block size must be positive and divide every matrix input dimension.")
    result = copy.deepcopy(model)
    with torch.no_grad():
        for name, parameter in result.named_parameters():
            if parameter.ndim != 2:
                continue
            weight = parameter if ("qkv" in name or "position" in name) else parameter.half().float()
            rows, columns = weight.shape
            blocks = weight.reshape(rows, columns // block, block)
            scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
            parameter.copy_((torch.round(blocks / scale).clamp(-127, 127) * scale).reshape(rows, columns))
    return result


def texts(benchmark):
    items = json.loads((benchmark / "evaluation_items.json").read_text(encoding="utf-8"))
    return [(item.get("context_text") or "") + item["expected_output"][0] for item in items]


def compare(lm, model, mlmodel, sequences):
    metrics = {key: [] for key in ("kl_coreml", "kl_simulated", "nll_fp32", "nll_coreml",
                                   "top1_agreement", "max_logprob_diff")}
    for ids in sequences:
        tensor = torch.tensor([ids])
        with torch.no_grad():
            reference = lm.model(tensor)[0].float().log_softmax(-1)
            simulated = model(tensor)[0].float().log_softmax(-1)
        logits = mlmodel.predict({"input_ids": np.array([ids], dtype=np.int32)})["logits"][0]
        coreml = torch.tensor(logits).log_softmax(-1)
        probabilities = reference.exp()
        metrics["kl_coreml"].append((probabilities * (reference - coreml)).sum(-1).mean().item())
        metrics["kl_simulated"].append((probabilities * (reference - simulated)).sum(-1).mean().item())
        targets = tensor[0, 1:, None]
        metrics["nll_fp32"].append(-reference[:-1].gather(-1, targets).mean().item())
        metrics["nll_coreml"].append(-coreml[:-1].gather(-1, targets).mean().item())
        metrics["top1_agreement"].append((reference.argmax(-1) == coreml.argmax(-1)).float().mean().item())
        metrics["max_logprob_diff"].append((reference - coreml).abs().max().item())
    return {key: {"mean": float(np.mean(values)), "max": float(np.max(values))} for key, values in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "artifacts/deployment/tiny-ja-v1-inference-v1")
    parser.add_argument("--model", type=Path, required=True, help="Core ML experiment directory with model.mlpackage")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "artifacts/benchmarks/ajimee-jwtd-v2-v1")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--compute-units", choices=("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"), default="CPU_ONLY")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; must not exist.")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output exists.")
    lm = BundleLM(args.bundle)
    manifest = verify_package(args.model)
    if manifest["bundle"]["bundle_manifest_sha256"] != lm.metadata["bundle_manifest_sha256"]:
        raise ValueError("Core ML model and FP32 bundle identities differ.")
    if sys.platform != "darwin":
        parser.error("Run Core ML prediction manually on Mac.")
    import coremltools as ct
    torch.set_num_threads(4)
    mlmodel = ct.models.MLModel(str(args.model / "model.mlpackage"),
                                compute_units=getattr(ct.ComputeUnit, args.compute_units))
    simulated = simulate_int8(lm.model, args.block_size).eval()
    limit = lm.model.config.context_length - 1
    encoded = [lm.processor.encode(text) for text in texts(args.benchmark)]
    if not encoded or any(not ids for ids in encoded):
        raise ValueError("Benchmark must have at least one nonempty target token per example.")
    real = [[lm.special["bos"], *ids[:limit]] for ids in encoded]
    rng = np.random.default_rng(0)
    random = [[lm.special["bos"], *rng.integers(4, lm.model.config.vocab_size, limit).tolist()] for _ in range(5)]
    report = {"format": "vimeml_distribution_diagnostic_v2", "model": str(args.model),
              "compute_units": args.compute_units, "environment": environment(),
              "bundle_manifest_sha256": lm.metadata["bundle_manifest_sha256"],
              "coreml_manifest_sha256": file_sha(args.model / "manifest.json"),
              "benchmark_items_sha256": file_sha(args.benchmark / "evaluation_items.json"),
              "simulation": {"block_size": args.block_size,
                  "note": "FP32 scales and symmetric +/-127 rounding; not exact Core ML scale storage or kernels."},
              "sampling": {"examples": len(encoded), "content_token_limit": limit,
                  "truncated_examples": sum(len(ids) > limit for ids in encoded),
                  "aggregation": "Mean of each example's position mean; not token-weighted corpus mean.",
                  "kl_top1_include_last_position": True, "nll_excludes_last_position": True,
                  "random_sequences": 5, "random_seed": 0},
              "benchmark_text": compare(lm, simulated, mlmodel, real),
              "random_tokens": compare(lm, simulated, mlmodel, random)}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
