# V2 训练

2026-10-07：V2.0 已完成；V2.1 保存的 checkpoint 为 step 0，旧实例新增产物已迁回本地。当前没有运行中的训练，监控暂停。

## 配置

| 项目 | V2.0 | V2.1 |
| --- | --- | --- |
| 配置 | `configs/train-v2.toml` | `configs/train-v21.toml` |
| 起点 | 随机初始化 | 冻结 V2.0 best，step 375000，仅模型权重 |
| 预算 | 4 epochs / 393,704 updates | 2 epochs / 196,852 updates |
| 学习率 | 1e-3 → 1e-4，warmup 1000 | 3e-4 → 3e-5，warmup 500 |
| 优化器 | AdamW β=.9/.95，decay=.1，clip=1 | 新 AdamW，相同参数 |
| 数据轮次 | 0–3 | offset=4，使用轮次 4/5 |
| 编译 | eager | 整体 hidden-state stack；token head 与 CE 保持 eager |

两版均使用 TinyGPTV2：16K tokenizer、context 128、320×6、5 heads、SwiGLU 832、RMSNorm、无 Linear bias、共享 embedding。BF16、batch 256、4 workers、30% prefix crop，至少保留 8 个预测对。

每轮包含 25,196,843 个训练窗口。每 5,000 updates 做固定 validation 子集评测并保存 checkpoint；每个 epoch 做完整 validation BPC 与两套现有 IME 评测。连续两轮 BPC 比历史最佳高超过 0.01 时提前停止。best 按固定子集最低 NLL 选择，last 与 epoch checkpoint 分别保留。

## AutoDL 运行

项目和大文件目录为 `/root/autodl-tmp/vimeml`。镜像记录：Python 3.12.3、PyTorch 2.7.0+cu128、NumPy 2.2.6、TensorBoard 2.19.0；其余依赖见 [requirements-autodl.txt](../requirements-autodl.txt)。实例镜像用于环境复用，数据盘文件需单独确认迁移。

```bash
cd /root/autodl-tmp/vimeml
screen -dmS vimeml-v21 bash -c 'bash scripts/training/autodl_v21.sh > /root/autodl-tmp/vimeml-v21.log 2>&1'
```

这是原 V2.1 启动入口。迁移后新实验使用独立 output/log 路径，保留旧 step 0 目录。`--resume` 适用于配置、数据和训练代码一致的 checkpoint；step 0 迁移采用从 V2.0 best 重新初始化。

启动器读取 `/root/autodl-tmp/vimeml-wandb.env`，启用 W&B online（项目 `vimeml`）和编译缓存。run URL 写入 `artifacts/tracking/<模型名>-live.json`；日志包括 loss、BPC、IME、有效 tokens/s、数据等待、裁剪率和显存。W&B 仅同步指标。恢复入口会创建同组新 run。

V2 的 `verification.mode="metadata"` 复用冻结数据指纹，检查计数、文件大小和小型 manifest，减少重复大文件 SHA256。原 V1 模型、配置和产物保持冻结。

## 结果与记录

V2.0 总用时 4 小时 38 分钟，best BPC 3.2598221，比 V1 降低 6.16%。现有 IME 与新 2,000 条 development 均未证明 Top-1 显著优于 V1；3,000 条真实候选已导入，标签仍为草稿，blind 尚未 LM 计分。

- [结构与实验约定](plan_v2.md)
- [V2.0 最终评测](reports/v2-20261007/evaluation.md)
- [V2.1 配置与性能基准](reports/v2-20261007/v21-plan.md)
- [扩大开发集初评](reports/v2-20261007/expanded-ime-evaluation.md)
- [中断检查与迁出](reports/v2-20261007/v21-recovery.md)
