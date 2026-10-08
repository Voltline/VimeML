# V2.1 真实数据性能对照

2026-10-07，原AutoDL 4090 D。真实数据对照后，restart1采用batch512与compiled hidden stack；两轮已完成，完整结果见 [最终评测](v21-evaluation.md)。

## 测量

同一冻结 V2.0 best 起点，真实 token store、epoch4 shuffle/crop、AdamW、BF16。每组 32 warmup + 128 measured updates；有效 tokens/s 包含 loader、H2D、更新与同步耗时，首次使用和 warmup 单独计时。

| 方案 | batch | 有效 tokens/s | step p50 / p95 ms | loader wait | 峰值 allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager | 256 | 112,738 | 44.96 / 56.34 | 2.2% | 6.15 |
| compiled | 256 | 152,335 | 33.72 / 39.69 | 3.0% | 5.22 |
| eager | 512 | 215,837 | 43.68 / 64.70 | 3.1% | 12.12 |
| compiled | 512 | **278,818** | 33.86 / 51.13 | 4.2% | 10.26 |

compiled 在 batch256/512 分别提升 35%/29%；compiled512 比 compiled256 快 83%。观测到 9 种 padded width（8–64、128），无 OOM。4 workers 等待占比低，未增加 worker。

各组 measured 区间仅约 4–6 秒，不能代表完整 epoch 或长期热稳定性。measured 区间保留重编译耗时，现有磁盘编译缓存未删除。每秒 GPU 快照跨不同阶段，不能据此归属某方案的平均 GPU 利用率。

## 正式配置与进度

[train-v21-restart.toml](../../../../configs/train-v21-restart.toml) 使用 batch512、2 epochs / 98,426 updates；学习率仍为 3e-4 → 3e-5，warmup500。数据覆盖仍为两轮，batch 改变了优化器更新次数与梯度统计，最终收益由完整 BPC 与固定候选 IME 评测判断。

最终完整运行：98,426 updates，998,854,924有效tokens，总耗时3739.8秒（约1小时2分）；排除验证、保存和首次100步的纯训练吞吐273,368 tokens/s，3650.2秒，约为V2.0的2.26×。短测279k与正式运行量级一致，完整混合长度性能已确认。

固定子集保留64 batches，batch512使验证样本数增加，初始 subset loss 不直接和旧子集比较；完整 BPC 口径不变。screen `vimeml-v21-r1`，W&B本地记录。

## 入口与原始记录

性能工具为 `scripts/training/benchmark_v21.py`，入口为 `scripts/training/autodl_v21_benchmark.sh`，仅更新内存中的模型副本，不保存 checkpoint 或创建正式 W&B run。

- GPU 对照：`outputs/model-checks/v21-real-data-performance-restart1/{report.json,gpu.csv,benchmark.log}`。
- 正式启动与快照：`outputs/autodl-v21-restart1-{launch,status}.json`。
- 本地 CPU 输入检查：`outputs/model-checks/v21-real-data-plan-20261007/report.json`，仅验证数据形状，不用于 GPU 吞吐比较。

原固定形状性能调查见 [V2.1 实验配置](v21-plan.md)。
