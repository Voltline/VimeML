# V2 训练

2026-10-07在AutoDL RTX4090 D完成V2.0、V2.1 restart1及extend5训练。最终选用extend5 best step40000：FP32 BPC3.0802686、AJIMEE144/200、扩大草稿1487/2000；2026-10-08完成INT8部署。总体结论见[V2总结](reports/v2-summary.md)。首次中断的step0仅作恢复记录。

restart1于18:54–19:56完成两轮。extend5于20:15–21:54运行3.273轮，在step161095结束；后续完整BPC未优于早期best，SIGINT保存last与优化器状态。配置和模型选择见[追加训练](reports/v2-20261007/v21-extend.md)。

## 配置

| 项目 | V2.0 | V2.1 |
| --- | --- | --- |
| 配置 | `configs/train-v2.toml` | `configs/train-v21-restart.toml` |
| 起点 | 随机初始化 | 冻结 V2.0 best，step 375000，仅模型权重 |
| 预算 | 4 epochs / 393,704 updates | 2 epochs / 98,426 updates |
| 学习率 | 1e-3 → 1e-4，warmup 1000 | 3e-4 → 3e-5，warmup 500 |
| 优化器 | AdamW β=.9/.95，decay=.1，clip=1 | 新 AdamW，相同参数 |
| 数据轮次 | 0–3 | offset=4，使用轮次 4/5 |
| 编译 | eager | 整体 hidden-state stack；token head 与 CE 保持 eager |

两版均使用TinyGPTV2：16K tokenizer、context128、320×6、5heads、SwiGLU832、RMSNorm、无Linear bias、共享embedding。BF16、4workers、30% prefix crop，至少保留8个预测对。V2.0及原V2.1配置batch256，restart1为batch512；restart1保留原续训的两轮窗口预算，optimizer updates减半。固定验证子集为64 batches，样本数随batch增大，初始subset loss不直接与旧子集比较；完整BPC口径不变。

extend5配置为[train-v21-extend.toml](../../configs/train-v21-extend.toml)：沿用batch512与编译backbone，加载restart1 best及45份AdamW状态，epoch offset6；最多5轮，前3轮LR3e-5，后2轮余弦降至1e-5，无warmup。实际只进入部分衰减阶段。

每轮包含 25,196,843 个训练窗口。每 5,000 updates 做固定 validation 子集评测并保存 checkpoint；每个 epoch 做完整 validation BPC 与两套现有 IME 评测。连续两轮 BPC 比历史最佳高超过 0.01 时提前停止。best 按固定子集最低 NLL 选择，last 与 epoch checkpoint 分别保留。

## AutoDL 运行

项目和大文件目录为 `/root/autodl-tmp/vimeml`。镜像记录：Python 3.12.3、PyTorch 2.7.0+cu128、NumPy 2.2.6、TensorBoard 2.19.0；其余依赖见 [requirements-autodl.txt](../../requirements-autodl.txt)。实例镜像用于环境复用，数据盘文件需单独确认迁移。

```bash
cd /root/autodl-tmp/vimeml
screen -dmS vimeml-v21-r1 bash -c 'bash scripts/training/autodl_v21.sh --config configs/train-v21-restart.toml > /root/autodl-tmp/vimeml-v21-restart1.log 2>&1'
```

restart1 使用独立 output/log 路径，保留原 `continue` 的 step0。`--resume` 适用于配置、数据和训练代码一致的 checkpoint；本次从 V2.0 best 重新初始化。

启动器读取 `/root/autodl-tmp/vimeml-wandb.env`，启用 W&B online（项目 `vimeml`）和编译缓存。run URL 写入 `artifacts/tracking/<模型名>-live.json`；日志包括 loss、BPC、IME、有效 tokens/s、数据等待、裁剪率和显存。W&B 仅同步指标。恢复入口会创建同组新 run。

V2的`verification.mode="metadata"`复用冻结数据指纹，检查计数、文件大小和小型manifest。V1模型、配置和产物保持冻结。

## 结果与记录

V2.0用时4小时38分钟，best BPC3.2598221，比V1降低6.16%，IME增益有限。restart1的吞吐提升到273k tokens/s，extend5最佳BPC进一步降至3.0802686，AJIMEE144/200；追加训练后期完整BPC未继续改善。扩大2000条development标签仍为草稿，1000条blind未计分。

- [结构与实验设计](plan_v2.md)
- [V2系列总结](reports/v2-summary.md)
- [追加训练与最终checkpoint](reports/v2-20261007/v21-extend.md)
- [V2.0 最终评测](reports/v2-20261007/evaluation.md)
- [V2.1 配置与性能基准](reports/v2-20261007/v21-plan.md)
- [V2.1 最终评测与模型选择](reports/v2-20261007/v21-evaluation.md)
- [扩大开发集初评](reports/v2-20261007/expanded-ime-evaluation.md)
- [中断与恢复记录](reports/v2-20261007/v21-recovery.md)
