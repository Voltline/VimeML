"""One compact V2 architecture acceptance run; no corpus training or W&B run."""

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.nn import functional as F

from vimeml.training.model_factory import (
    checkpoint_format,
    configuration_for,
    create_model,
    model_from_checkpoint,
)
from vimeml.training.train import atomic_checkpoint, optimizer_for, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train-v2.toml")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/model-checks/tiny-ja-v2-phase-b"
    )
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    architecture = config["architecture"]
    model_config, settings = validate_config(config)
    device = torch.device(settings["device"])
    torch.set_num_threads(settings["cpu_threads"])
    torch.manual_seed(settings["seed"])
    started = time.perf_counter()
    model = create_model(architecture, model_config).to(device)
    assert model.parameter_count() == 12_537_920
    assert model.token_embedding.weight is model.lm_head.weight
    assert all(
        layer.bias is None for layer in model.modules() if isinstance(layer, torch.nn.Linear)
    )
    inputs = torch.randint(
        4, model_config.vocab_size, (4, model_config.context_length), device=device
    )
    labels = inputs.roll(-1, dims=1)
    labels[:, 96:] = -100
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        assert logits.shape == (4, 128, 16384)
        expected_loss = F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
        )
        result = model(inputs, labels)
        torch.testing.assert_close(result["loss_sum"], expected_loss, rtol=1e-5, atol=1e-4)
        changed = inputs.clone()
        changed[:, 64:] = (changed[:, 64:] + 7) % model_config.vocab_size
        causal_logits = model(changed)
        causal_error = float((logits[:, :64] - causal_logits[:, :64]).abs().max())
        assert causal_error == 0.0
        del logits, causal_logits
    model.train()
    optimizer = optimizer_for(model, settings, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        result = model(inputs, labels)
        loss = result["loss_sum"] / result["token_count"]
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), settings["grad_clip"], error_if_nonfinite=True
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "one-update.pt"
    atomic_checkpoint(
        checkpoint_path,
        {
            "format": checkpoint_format(architecture),
            "architecture": architecture,
            "model": model.state_dict(),
            "model_config": model.configuration(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": 1,
            "purpose": "Architecture acceptance only; random initialization plus one synthetic update.",
        },
    )
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored = model_from_checkpoint(saved).to(device).eval()
    model.eval()
    assert restored.token_embedding.weight is restored.lm_head.weight
    with torch.no_grad():
        reference = model(inputs[:1])
        actual = restored(inputs[:1])
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    restored_optimizer = optimizer_for(restored, settings, device)
    restored_optimizer.load_state_dict(saved["optimizer"])
    assert len(restored_optimizer.state) == len(optimizer.state)
    # The default architecture continues to construct the frozen V1 model.
    v1 = create_model("tiny_gpt_v1", configuration_for("tiny_gpt_v1", {}))
    assert v1.parameter_count() == 7_386_624
    report = {
        "status": "passed",
        "architecture": architecture,
        "parameters": model.parameter_count(),
        "model": model.configuration(),
        "device": str(device),
        "precision": "bf16",
        "torch_version": str(torch.__version__),
        "forward_shape": [4, 128, 16384],
        "causal_prefix_max_error": causal_error,
        "masked_loss_matches_full_logits": True,
        "weight_tying": True,
        "linear_bias_count": 0,
        "loss_one_update": float(loss.detach()),
        "grad_norm": float(grad_norm),
        "checkpoint_roundtrip_exact": True,
        "optimizer_state_restored": True,
        "v1_parameters": v1.parameter_count(),
        "elapsed_seconds": time.perf_counter() - started,
        "validation_scope": "One synthetic optimizer update; not corpus training or an accuracy evaluation.",
        "deployment_validated": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
