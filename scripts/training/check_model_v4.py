"""Random-weight V4 validation; synthetic updates only, never corpus training."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from vimeml.training.model_v4 import TinyGPTV4, PARAMETERS
from vimeml.training.model_factory_v4 import FORMAT
from vimeml.training.model_factory import model_from_checkpoint
from vimeml.training.train import optimizer_for, atomic_checkpoint, tokenizer_identity_from_manifest
from vimeml.training.data import file_sha, prepare_indexes, write_json, SentenceWindowDataset, make_loader
from vimeml.training.infer_v4 import JapaneseLMV4
from vimeml.tokenizer.v4 import read_json


def run(output, batch_size=128):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh validation directory")
    output.mkdir(parents=True, exist_ok=True)
    tests = subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-p", "test_model_v4.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    (output / "tests.log").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    if tests.returncode:
        raise RuntimeError(f"V4 tests failed; inspect {output / 'tests.log'}")
    torch.set_num_threads(4)
    torch.manual_seed(949)
    model = TinyGPTV4().eval()
    token_dir = ROOT / "artifacts/token-data/corpus-v4-8k"
    index_dir = ROOT / "artifacts/training-data/corpus-v4-c128"
    tokens = read_json(token_dir / "manifest.json")
    prepare_indexes(token_dir, index_dir, 128, splits=("train", "validation"))
    saved = {"format": FORMAT, "architecture": "tiny_gpt_v4", "model_config": model.configuration(),
        "model": model.state_dict(), "step": 0, "random_initialization_test_only": True,
        "config": {"architecture": "tiny_gpt_v4", "token_dir": "artifacts/token-data/corpus-v4-8k"},
        "signatures": {"tokens": file_sha(token_dir / "manifest.json")},
        "tokenizer_identity": tokenizer_identity_from_manifest(tokens)}
    checkpoint = output / "random-initialized.pt"
    atomic_checkpoint(checkpoint, saved)
    restored = model_from_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=True)).eval()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
    lm = JapaneseLMV4(checkpoint, ROOT / "artifacts/tokenizers/v4-a/ja-unigram-8192")
    lm.next_pieces("今日は", 5)
    gpu = {"available": torch.cuda.is_available(), "corpus_training_started": False}
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        gpu_model = TinyGPTV4().cuda().train()
        device = torch.device("cuda")
        settings = {"learning_rate": .0006, "weight_decay": .1, "beta1": .9, "beta2": .95}
        optimizer = optimizer_for(gpu_model, settings, device)
        # Worst context width and proposed batch size, entirely synthetic IDs.
        inputs = torch.randint(7, 8192, (batch_size, 128), device=device)
        labels = torch.randint(7, 8192, (batch_size, 128), device=device)
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = gpu_model(inputs, labels)
            loss = result["loss_sum"] / result["token_count"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(gpu_model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or any(
                p.grad is None or not torch.isfinite(p.grad).all() for p in gpu_model.parameters()):
            raise ValueError("Nonfinite CUDA BF16 loss or gradients")
        optimizer.step()
        torch.cuda.synchronize()
        # Validate exact serialization/restoration of the optimizer and weights.
        synthetic = output / "synthetic-update.pt"
        atomic_checkpoint(synthetic, {"model": gpu_model.state_dict(), "optimizer": optimizer.state_dict(), "synthetic_test_only": True})
        state = torch.load(synthetic, map_location=device, weights_only=True)
        other = TinyGPTV4().to(device)
        other.load_state_dict(state["model"])
        second = optimizer_for(other, settings, device)
        second.load_state_dict(state["optimizer"])
        for left, right in zip(gpu_model.parameters(), other.parameters(), strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
            for key, value in optimizer.state[left].items():
                actual = second.state[right][key]
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, actual.to(value.device), atol=0, rtol=0)
        gpu.update(status="passed", device=torch.cuda.get_device_name(0), precision="bf16", batch_size=batch_size,
            context_length=128, loss=float(loss.detach()), grad_norm=float(grad_norm),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            optimizer_roundtrip="exact", validation_scope="One synthetic AdamW update; not corpus training or model evaluation")
        del optimizer, second, other, gpu_model, state, inputs, labels, result, loss
        torch.cuda.empty_cache()
    # Read real training tensors for shape/label/finite-loss acceptance, no update.
    dataset = SentenceWindowDataset(token_dir, index_dir, "train")
    loader = make_loader(dataset, batch_size=8, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    with torch.no_grad():
        actual = model(batch["input_ids"], batch["labels"])
    if not torch.isfinite(actual["loss_sum"]):
        raise ValueError("Nonfinite real-data forward")
    dataset.close()
    report = {"status": "passed", "parameters": PARAMETERS, "model": model.configuration(),
        "local_model_tests": 5, "torch_version": str(torch.__version__), "gpu": gpu,
        "checkpoint_roundtrip": "exact", "real_data_forward": "passed; no optimizer update",
        "random_checkpoint_sha256": file_sha(checkpoint), "kv_shape": [7, 1, 6, 128, 64],
        "kv_fp32_bytes": 2_752_512, "fp32_atol": 3e-4, "fp32_rtol": 3e-4,
        "coreml_export": "not run; Apple runtime unavailable on Windows",
        "int8_block32_export": "not run; Apple runtime unavailable on Windows",
        "ios_latency_and_memory": "not measured", "formal_training_started": False}
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/model-checks/v4-a-20261014")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    run(args.output, args.batch_size)
