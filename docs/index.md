# 文档索引

V2系列实验已完成，当前部署版本为V2.1 extend5 step40000 / INT8 block32；V1保留作基线和回退。总体结果与实验取舍见[V2系列总结](reports/v2-summary.md)。

## 使用与复现

| 文档 | 内容 |
| --- | --- |
| [数据](data.md) | 冻结语料、来源、清洗和split |
| [V2实验设计](plan_v2.md) / [V2训练](training-v2.md) | 模型、tokenizer、裁剪、配置和运行方式 |
| [V1训练](training.md) | 冻结基线复现 |
| [评测](evaluation.md) | 评分语义、同池结果、统计与标签限制 |
| [Core ML](coreml.md) | 导出、量化、客户端接口与设备范围 |
| [本地产物](artifacts.md) / [跨平台管理](reference/artifact-exchange.md) | 路径、归档、Git与ZIP协作 |

## V2实验记录

| 阶段 | 报告 |
| --- | --- |
| 数据与模型设计 | [Tokenizer](reports/v2-20261007/tokenizer.md)、[模型](reports/v2-20261007/model.md)、[Token store](reports/v2-20261007/token-data.md)、[Prefix crop](reports/v2-20261007/prefix-crop.md) |
| V2.0 | [运行环境](reports/v2-20261007/autodl.md)、[完整评测](reports/v2-20261007/evaluation.md) |
| V2.1 restart1 | [配置](reports/v2-20261007/v21-plan.md)、[性能对照](reports/v2-20261007/v21-performance.md)、[中断恢复](reports/v2-20261007/v21-recovery.md)、[评测](reports/v2-20261007/v21-evaluation.md) |
| V2.1 extend5 | [追加训练与模型选择](reports/v2-20261007/v21-extend.md) |
| 扩大IME数据集 | [候选构建](reports/v2-20261007/ime-3000-handoff.md)、[初评](reports/v2-20261007/expanded-ime-evaluation.md)、[标签与错误分析](reports/v2-20261007/label-error-audit.md) |
| Core ML与iPhone | [转换和量化](reports/mac-20261008/v21-coreml.md)、[误差分析](reports/mac-20261008/v21-quantization-review.md)、[宿主与真实扩展](reports/mac-20261008/v21-iphone.md) |

## V1参考与历史

[FP32结果](reference/results.md)、[Core ML实测](reference/coreml-v1.md)、[模拟器内存](reports/mac-20261006/simulator-memory.md)、[真实扩展](reports/mac-20261006/iphone-keyboard-memory.md)、[快照归档](reports/mac-20261006/handoff.md)、[导入核对](reports/mac-20261006/merge-verification.md)。

详细方法：[语料流水线](reference/data-pipeline.md)、[AJIMEE导出](reference/ajimee-benchmark.md)、[开发集生成](reference/development-generation.md)、[组合排序](reference/hybrid-ranking.md)、[联想](reference/phrase-demo.md)。

`history/`保存V1实测前的[部署设计](history/deployment.md)、[命令参考](history/coreml-guide.md)和[阶段记录](history/workflow.md)，仅描述当时状态。各报告注明日期、版本与测量范围；原始JSON、逐候选分数和trace保存在`outputs/`。
