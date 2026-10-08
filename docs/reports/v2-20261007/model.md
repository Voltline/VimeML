# V2 architecture and training entry — 2026-10-07

TinyGPTV2 contains 12,537,920 parameters: vocabulary 16,384, context 128, hidden width 320, six layers, five heads of width 64, SwiGLU width 832, pre-RMSNorm with FP32 variance and epsilon 1e-5, causal SDPA, learned positions, bias-free linear layers, and tied embeddings.

`model_factory.py` selects V1/V2 for training and inference; old configurations without an architecture field retain V1 behavior. V2 checkpoints use format `vimeml_tiny_gpt_v2`. Training computes vocabulary logits only for valid labels, avoiding unnecessary padding work.

[train-v2.toml](../../../configs/train-v2.toml) specifies BF16, batch 256, four epochs, learning rate 1e-3 → 1e-4, warmup 1,000, AdamW betas 0.9/0.95, weight decay 0.1, clip 1, and 30% prefix crop. A zero maximum-step setting derives the budget from windows/epochs.

The original targeted model check records parameter count, shapes, causality, masked loss, tying, finite gradients, save/load, and optimizer state on CUDA/BF16. Its fixture is not a trained model. Original evidence resides in `outputs/model-checks/tiny-ja-v2-phase-b/report.json`.

Optional online tracking routes the normal entry through `train_wandb.py`; metrics are synchronized while checkpoints/corpora remain local. Account/run metadata remains in ignored tracking records. [Evaluation](evaluation.md) reports the completed V2.0 run.
