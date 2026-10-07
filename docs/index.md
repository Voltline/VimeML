# 文档索引

训练已停止。当前离线模型为V2.1 extend5 best / step40000；2026-10-08在Mac进行Core ML适配与量化，V1部署资源保留作对照。

## 当前指南

| 文档 | 用途 |
| --- | --- |
| [Mac准备与Git协作](mac-v21-preparation.md) | 单一ZIP、模型接口、明天的范围和双仓库PR流程 |
| [V2训练](training-v2.md) / [实验约定](plan_v2.md) | 版本配置与训练口径 |
| [评测](evaluation.md) / [Core ML](coreml.md) | 同池评分、量化与设备范围 |
| [数据](data.md) / [V1训练](training.md) | 原数据流程与冻结基线复现 |
| [本地产物](artifacts.md) | 目录、归档与迁移范围 |

## 实验报告

| 阶段 | 记录 |
| --- | --- |
| V2.0准备 | [Tokenizer](reports/v2-20261007/tokenizer.md)、[模型](reports/v2-20261007/model.md)、[Token store](reports/v2-20261007/token-data.md)、[Crop](reports/v2-20261007/prefix-crop.md)、[AutoDL](reports/v2-20261007/autodl.md) |
| 训练结果 | [V2.0](reports/v2-20261007/evaluation.md)、[restart1](reports/v2-20261007/v21-evaluation.md)、[extend5与最终选择](reports/v2-20261007/v21-extend.md) |
| 性能与恢复 | [续训配置](reports/v2-20261007/v21-plan.md)、[实测性能](reports/v2-20261007/v21-performance.md)、[旧step0恢复](reports/v2-20261007/v21-recovery.md) |
| 数据与标签 | [3000条候选交接](reports/v2-20261007/ime-3000-handoff.md)、[扩大集初评](reports/v2-20261007/expanded-ime-evaluation.md)、[标签与错误审核](reports/v2-20261007/label-error-audit.md) |
| V1 Mac历史 | [原交接](reports/mac-20261006/handoff.md)、[合并记录](reports/mac-20261006/merge-verification.md)、[模拟器](reports/mac-20261006/simulator-memory.md)、[真实键盘](reports/mac-20261006/iphone-keyboard-memory.md) |

报告按日期与实验保存；原始JSON、分数和trace保存在`outputs/`，报告不替代原始证据。

## 详细参考与历史

[语料流水线](reference/data-pipeline.md)、[AJIMEE导出](reference/ajimee-benchmark.md)、[开发集生成](reference/development-generation.md)、[组合排序](reference/hybrid-ranking.md)、[联想](reference/phrase-demo.md)、[V1 FP32结果](reference/results.md)、[V1 Core ML实测](reference/coreml-v1.md)。

`history/`中的[原部署计划](history/deployment.md)、[原Core ML指南](history/coreml-guide.md)、[原流程](history/workflow.md)是历史资料；当前执行范围以Mac准备文档为准。
