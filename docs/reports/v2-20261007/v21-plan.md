# V2.1 续训配置与性能基准

2026-10-07。原 run 于 15:39 启动，停机后保存的 checkpoint 为 step 0，已迁出。模型权重等于冻结 V2.0 best；没有可恢复的新增训练进度。详情见 [中断与迁出](v21-recovery.md)。

## 实验配置

| 项目 | 配置 |
| --- | --- |
| 起点 | V2.0 best，step 375000，仅权重；新 AdamW |
| 预算 | 2 epochs / 196,852 updates，25,196,843 windows/epoch |
| 学习率 | warmup 500，3e-4 → 3e-5 cosine |
| 数据 / 模型 | 原 16K tokenizer、320×6、context128、batch256、BF16 |
| 裁剪 / 随机性 | prefix crop .30、最少 8 对；epoch offset=4（轮次 4/5） |
| 编译 | full hidden-state stack；变长 token head / CE eager |
| 评测 | 每 5,000 updates 子集验证，每轮完整 BPC 与原两套 IME |
| 运行 | AutoDL、screen、W&B online，独立输出目录 |

配置：[train-v21.toml](../../../configs/train-v21.toml)。best 位于第四轮结束前，因此这是“从 best 额外扫描两轮”，不是从 last 接到第六轮。原优化器、调度器与数据游标不继承。初始化已验证模型配置、数据身份、权重相同与 fresh optimizer。

## GPU 调度调查

原 eager profiler 记录约 661 次 kernel launch，CPU self time 43–56ms、累计 GPU 算子约 11.2ms；真实 loader 等待约 2.5%。这些证据指向 CPU 调度与细粒度 kernel 开销，profiler 时间不等同 GPU 利用率。

同一 4090 D、同一冻结模型、batch256、width24、4,827 有效 token，5 次预热 + 60 次测量：

| 方案 | ms/update | 有效 tokens/s | 相对 eager |
| --- | ---: | ---: | ---: |
| Eager | 39.14 | 123,327 | 1.00× |
| 编译 hidden stack | 25.79 | 187,176 | **1.52×** |

单独编译 RMSNorm 更慢（39.98→48.36ms），未采用。整体编译通过 FP32 全参数梯度一致性；BF16 每参数梯度相对 L2 最大约 1.49%。state_dict 保持普通模型格式。最初近零单元素 BF16 相对误差检查失败，记录保留。

固定 batch 不含 loader、首次编译、变长重编译、评测和保存，不能据此推算整轮快 52%；重复更新同一 batch 的 loss 不用于质量评测。

原始记录：`outputs/model-checks/v2-backbone-profile-3/`、`outputs/model-checks/v21-preparation/`。扩大 benchmark 的候选与初评见 [3,000 条数据](ime-3000-handoff.md)和[开发集结果](expanded-ime-evaluation.md)。
