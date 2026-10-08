# V1阶段记录（2026-10-06）

记录V1训练及首轮离线评测完成、Core ML实测开始前的阶段状态。模型为冻结的`tiny-ja-v1`，后续部署结果见[V1 Core ML实测](../reference/coreml-v1.md)。

| 阶段 | 结果 | 实现与资料 |
| --- | --- | --- |
| 原始数据检查 | 完成 | `scripts/tools/inspect_data.py` |
| 清洗与合并 | 完成 | `scripts/corpus/`、[数据流程](../reference/data-pipeline.md) |
| 存疑审核与抽检 | 完成，638条隔离块排除 | `scripts/review/` |
| Tokenizer与编码 | 完成 | `scripts/tokenizer/`、[V1训练](../training.md) |
| DataLoader与GPU检查 | 通过 | `scripts/training/check_data.py` |
| 1 epoch训练 | 完成 | `configs/train-v1.toml`、[FP32结果](../reference/results.md) |
| AJIMEE与开发集评测 | 完成 | `scripts/benchmarks/`、[组合排序](../reference/hybrid-ranking.md) |
| 联想与自然度复核 | 完成，质量尚不稳定 | [联想参考](../reference/phrase-demo.md) |
| Core ML与设备 | 当时处于转换准备阶段 | [原部署设计](deployment.md) |

## 实验组织

配置、模型、候选池和提示词变更使用独立版本目录。原审核材料、生成草稿、开发集、公开评测和训练语料分别保存，模型辅助审核保留其方法与限制。

部署采用纯LM排序，融合策略作为独立对照：开发集选择λ，固定后对照AJIMEE。压缩模型另评估数值、任务质量和设备成本，不直接继承FP32结果。

训练计时与设备计时、有效预测目标与padding、模型包体与驻留内存分别记录。原模型和已保存分数保持冻结，目录与迁移方式见[本地产物](../artifacts.md)。
