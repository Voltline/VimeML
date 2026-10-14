# V4-a model implementation and completed local checks

2026-10-14. [Chinese report](../../zh-CN/reports/v4-20261014/model.md). Local V4-a preparation is complete. The next step is direct base-LM training using [the V4 training guide](../../training-v4.md). There is no preliminary training run or additional test stage. Apple deployment is a later, separate phase and does not gate this local training.

## Fixed architecture

[Model](../../../src/vimeml/training/model_v4.py), [V4 registration](../../../src/vimeml/training/model_factory_v4.py), [inference](../../../src/vimeml/training/infer_v4.py).

Vocabulary8192, context128, d_model384, seven layers, six heads/head_dim64, SwiGLU960, pre-norm RMSNorm eps1e-5, no linear bias, dropout0, learned absolute positions, causal attention, and tied embeddings. Configuration changes to these fixed values are rejected.

| Component | Parameters |
| --- | ---: |
| Each Transformer block | 1,696,512 |
| Seven blocks | 11,875,584 |
| Shared token embedding/output head | 3,145,728 |
| Position embedding | 49,152 |
| Final RMSNorm | 384 |
| **Total** | **15,070,848** |

V4 reduces embedding allocation and increases backbone capacity. Relative to V2.1, token embedding decreases from 5,242,880 to 3,145,728 parameters, while Transformer blocks increase from 7,253,760 to 11,875,584. Model quality remains to be measured during and after training.

The fixed V4 class reuses proven V2 operators. `model.py`, `model_v2.py`, and `model_factory.py` remain unchanged. Shared training orchestration registers the V4 identity and explicitly selects train/validation. Index preparation accepts optional splits; legacy defaults remain three splits. Older synthetic fixtures were aligned with existing tokenizer-identity and validation interfaces without relaxing production checks.

## Completed checks

30 relevant tests passed: five V4 model tests, one V4 trainer/resume test, nine legacy model/trainer tests, and fifteen data/inference/V2 deployment regression tests.

Coverage includes exact parameter counts, tied weights/no biases, causality, padding and shifted masked loss, context128 forward, KV increments versus full recomputation, cache overflow, A/B/A rollback, joint-tokenized candidate scoring, next-token/beam paths, local TorchScript graphs, and exact committed training resume. FP32 comparison tolerance is atol=rtol=3e-4. These were synthetic/developer checks, not model-quality benchmark scores.

K/V each have shape `[7,1,6,128,64]`; two FP32 buffers total 2,752,512 bytes (2.625 MiB). Snapshot restore replaces both complete buffers and their length. The functional single-token KV graph has been checked locally in PyTorch.

Windows environment: PyTorch2.10.0+cu128, NVIDIA RTX5060 Laptop GPU, BF16 support. Synthetic AdamW updates, weight/optimizer serialization and restoration, and a real-data forward without an optimizer update all passed.

| Synthetic batch × context | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| 256 × 128 | 6.820 GiB | 7.424 GiB |
| 128 × 128 | 3.457 GiB | 3.772 GiB |

Training uses batch128. The V4 interruption/resume fixture produced bit-identical final weights and matching committed cursors. Checkpoint format: `vimeml_tiny_gpt_v4`.

Evidence is retained locally under `outputs/model-checks/`, including `v4-a-20261014-b128/report.json`, random-initialization and synthetic-update fixtures, and regression logs. Test fixtures are not formal training initializers. No formal V4 corpus training started on 2026-10-14.

## Execution decision

On 2026-10-15 morning, start [train-v4-b.toml](../../../configs/train-v4-b.toml) directly from random weights: BF16, batch128, accumulation1, two epochs, AdamW, LR6e-4 with1000-step LR warmup and cosine decay. This warmup is part of the training schedule, not a separate test run. The default V4 entry selects this configuration. Validation and checkpointing occur during normal training. Resume with the same configuration and `--resume`.
