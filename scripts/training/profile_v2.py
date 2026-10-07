"""Representative CUDA profile and optional norm/backbone compilation comparison."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from vimeml.training.data import write_json
from vimeml.training.model_factory import model_from_checkpoint
from vimeml.training.runtime_v2 import fuse_rmsnorm, compile_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("norm", "backbone"), default="norm")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a fresh output; preserve previous measurements.")
    args.output.mkdir(parents=True, exist_ok=True)
    optimize = fuse_rmsnorm if args.variant == "norm" else compile_backbone
    torch.set_num_threads(4)
    torch.manual_seed(42)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = model_from_checkpoint(saved).cuda()
    del saved
    batch, width = 256, 24
    inputs = torch.randint(4, base.config.vocab_size, (batch, width), device="cuda")
    lengths = torch.randint(14, width + 1, (batch,), device="cuda")
    labels = torch.roll(inputs, -1, 1)
    labels[torch.arange(width, device="cuda")[None, :] >= lengths[:, None]] = -100
    tokens = int((labels != -100).sum())
    weights = copy.deepcopy(base.state_dict())

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(inputs, labels)["loss_sum"] / tokens
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        return float(loss)

    results = {}
    optimized_name = "fused_rmsnorm" if args.variant == "norm" else "compiled_backbone"
    for name in ("eager", optimized_name):
        model = model_from_checkpoint(
            {
                "format": "vimeml_tiny_gpt_v2",
                "architecture": "tiny_gpt_v2",
                "model_config": base.configuration(),
                "model": weights,
            }
        ).cuda()
        if name != "eager":
            results["fusion"] = optimize(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        start = time.perf_counter()
        for _ in range(5):
            step(model, optimizer)
        initialization_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(60):
            value = step(model, optimizer)
        duration = time.perf_counter() - start
        results[name] = {
            "steps": 60,
            "seconds": duration,
            "milliseconds_per_update": duration / 60 * 1000,
            "effective_tokens_per_second": tokens * 60 / duration,
            "initialization_seconds": initialization_seconds,
            "final_fixture_loss": value,
        }
        if name == "eager":
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                step(model, optimizer)
            events = profile.key_averages()
            (args.output / "profile.txt").write_text(
                events.table(sort_by="self_cpu_time_total", row_limit=25)
                + "\n"
                + events.table(sort_by="self_cuda_time_total", row_limit=25),
                encoding="utf-8",
            )
        del optimizer, model

    # Check loss and every parameter gradient on the exact same batch/weights.
    reference = model_from_checkpoint(
        {"format": "vimeml_tiny_gpt_v2", "model_config": base.configuration(), "model": weights}
    ).cuda()
    candidate = model_from_checkpoint(
        {"format": "vimeml_tiny_gpt_v2", "model_config": base.configuration(), "model": weights}
    ).cuda()
    optimize(candidate)
    losses = []
    for model in (reference, candidate):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(inputs, labels)["loss_sum"] / tokens
        loss.backward()
        losses.append(float(loss))
    torch.testing.assert_close(
        torch.tensor(losses[0]), torch.tensor(losses[1]), rtol=5e-4, atol=5e-4
    )
    maximum_gradient_difference = 0.0
    relative_gradient_l2 = {}
    for (name, left), (other, right) in zip(
        reference.named_parameters(), candidate.named_parameters()
    ):
        assert name == other
        maximum_gradient_difference = max(
            maximum_gradient_difference, float((left.grad - right.grad).abs().max())
        )
        relative_gradient_l2[name] = float(
            torch.linalg.vector_norm(left.grad - right.grad)
            / torch.linalg.vector_norm(left.grad).clamp_min(1e-12)
        )
    # BF16 fusion changes rounding; compare whole-gradient error rather than
    # relative errors of individual near-zero elements, then prove FP32 parity.
    assert max(relative_gradient_l2.values()) < 0.02, relative_gradient_l2
    for model in (reference, candidate):
        model.zero_grad(set_to_none=True)
        (model(inputs, labels)["loss_sum"] / tokens).backward()
    for (name, left), (other, right) in zip(
        reference.named_parameters(), candidate.named_parameters()
    ):
        torch.testing.assert_close(
            left.grad, right.grad, rtol=2e-4, atol=2e-5, msg=lambda m: name + " " + m
        )
    results.update(
        status="passed",
        batch_size=batch,
        padded_width=width,
        valid_tokens=tokens,
        padding_fraction=1 - tokens / (batch * width),
        gradient_max_absolute_difference=maximum_gradient_difference,
        bf16_max_parameter_gradient_relative_l2=max(relative_gradient_l2.values()),
        fp32_all_parameter_gradient_parity=True,
        fusion_speedup=results["eager"]["milliseconds_per_update"]
        / results[optimized_name]["milliseconds_per_update"],
        note="Fixed representative batch; no data loader. Speed excludes compilation and is not a full-epoch estimate.",
        formal_training_started=False,
    )
    write_json(args.output / "report.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
