# 文档索引

当前任务以冻结的 tiny-ja-v1 为基线。正文记录已完成的工作与可手动执行的工具；更大模型是未来独立实验，不自动训练。

## 当前指南

| 文档 | 用途 |
| --- | --- |
| [数据](data.md) | 来源、构建与审核 |
| [训练](training.md) | 环境、tokenizer、模型结构与推理 |
| [评测](evaluation.md) | 纯LM评分、开发集、AJIMEE与联想 |
| [Core ML](coreml.md) | Windows→Mac→iOS的手动步骤、结果与限制 |
| [本地产物](artifacts.md) | 迁移范围、归档、指纹与哈希 |
| [Hugging Face发布草案](publishing.md) | 发布候选内容、模型卡和许可选择 |

## 实测与交接

- [Windows合并校验](reports/mac-20261006/merge-verification.md)
- [Mac阶段与双仓库交接](reports/mac-20261006/handoff.md)
- [模拟器与早期内存审计](reports/mac-20261006/simulator-memory.md)
- [真实iPhone键盘扩展内存记录](reports/mac-20261006/iphone-keyboard-memory.md)

这些报告分别标明测试宿主、模拟器、真实扩展、指标口径与来源身份。原始JSON/trace保存在本地 `outputs/`，不以重新编辑的文档替换原始结果。

## 详细参考

- [语料流水线](reference/data-pipeline.md)
- [AJIMEE候选导出](reference/ajimee-benchmark.md)
- [开发集生成与复核](reference/development-generation.md)
- [FP32组合排序](reference/hybrid-ranking.md)
- [联想演示](reference/phrase-demo.md)
- [FP32基线结果](reference/results.md)

## 历史计划

[原部署计划](history/deployment.md)、[原Core ML逐步指南](history/coreml-guide.md)、[原工作流程](history/workflow.md) 保留实测前的设计。当前状态与执行入口请使用上面的当前指南。
