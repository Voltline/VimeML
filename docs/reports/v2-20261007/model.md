# V2.0 模型与训练入口

2026-10-07。TinyGPTV2 参数量 **12,537,920**；V1 `model.py` 保持原实现。

| 结构 | 配置 |
| --- | --- |
| 词表 / context | 16,384 / 128 |
| Hidden / layers / heads | 320 / 6 / 5，head_dim=64 |
| FFN | SwiGLU，d_ff=832 |
| Norm | Pre-Norm RMSNorm，eps=1e-5，FP32 方差 |
| Attention / position | causal SDPA / learned embedding |
| Bias / tying | 无 Linear bias / 共享 token embedding 与 LM head |

`model_factory.py` 在训练和推理中选择 V1/V2，未指定 architecture 的旧配置仍使用 V1。V2 checkpoint 格式为 `vimeml_tiny_gpt_v2`。训练仅对有效 label 计算词表 logits，避免 PAD 的无效计算。

[train-v2.toml](../../../configs/train-v2.toml) 最终配置为 BF16、batch 256、4 epochs、LR 1e-3 → 1e-4、warmup 1000、AdamW β=.9/.95、decay=.1、clip=1、prefix crop 30%。`max_steps=0` 根据窗口与 epoch 预算推导 update 数。

## 已有验证

`check_model_v2.py` 在 PyTorch 2.10.0+cu128 / CUDA / BF16 上用一次 synthetic update 检查：参数量、forward shape、causal masking、masked loss、weight tying、有限梯度、保存/重载与 optimizer state。耗时 18.49 秒，报告为 `outputs/model-checks/tiny-ja-v2-phase-b/report.json`；fixture checkpoint 不是正式模型。V2 Core ML 尚未验证。

## W&B

配置 `[tracking] enabled=true, mode="online", project="vimeml"` 时，普通训练入口转入 `train_wandb.py`。TensorBoard 指标同步至 W&B，实际 run ID/URL 写入 `artifacts/tracking/<模型名>-live.json`；checkpoint、语料和代码不上传。恢复会新建同 group 的 run。

正式 V2.0 run 已完成：[`lvfu0bix`](https://wandb.ai/voltline233/vimeml/runs/lvfu0bix)。最终结果见 [评测](evaluation.md)。
