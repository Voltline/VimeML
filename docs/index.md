# 文档索引

V1 为冻结基线，V2.0 已完成，V2.1 中断产物已迁出。当前状态见 [V2 训练](training-v2.md)，实验约定见 [V2 计划](plan_v2.md)。

## 当前指南

| 文档 | 用途 |
| --- | --- |
| [数据](data.md) | 来源、构建与审核 |
| [训练](training.md) | 环境、tokenizer、模型结构与推理 |
| [V2 训练](training-v2.md) | 版本配置、AutoDL、W&B 与当前状态 |
| [评测](evaluation.md) | 纯LM评分、开发集、AJIMEE与联想 |
| [Core ML](coreml.md) | 转换入口、设备结果与限制 |
| [本地产物](artifacts.md) | 迁移范围、归档、指纹与哈希 |

## 实测与交接

- [V2.0 Phase A：Tokenizer](reports/v2-20261007/tokenizer.md)
- [V2.0 Phase B：TinyGPTV2 与 W&B](reports/v2-20261007/model.md)
- [V2.0 Phase C：Token store 与窗口索引](reports/v2-20261007/token-data.md)
- [V2.0 Phase D：Deterministic prefix crop](reports/v2-20261007/prefix-crop.md)
- [V2.0 Phase E：AutoDL 部署与正式训练](reports/v2-20261007/autodl.md)
- [V2.0 Phase F：完成评测与最佳 checkpoint](reports/v2-20261007/evaluation.md)
- [V2.1：续训配置、GPU 性能调查与扩大评测准备](reports/v2-20261007/v21-plan.md)
- [137 条标签复核与全部 V1/V2 错误分类](reports/v2-20261007/label-error-audit.md)
- [3,000 条 IME 输入与 Mac 候选交接](reports/v2-20261007/ime-3000-handoff.md)
- [真实候选验收与 2,000 条开发集 V1/V2 初评](reports/v2-20261007/expanded-ime-evaluation.md)
- [V2.1 无卡恢复、checkpoint 检查与服务器迁出](reports/v2-20261007/v21-recovery.md)
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

[原部署计划](history/deployment.md)、[原Core ML逐步指南](history/coreml-guide.md)、[原工作流程](history/workflow.md) 为实测前的历史资料，不代表当前状态。
