# 文档索引

英文主要文档见[Documentation index](../index.md)，当前中文概览见[README.zh-CN.md](../../README.zh-CN.md)与[模型卡](model-card.md)。此目录保留对应方法和历史实验的中文记录；英文入口汇总发布与当前实验状态。

V2系列实验已完成，当前部署版本为V2.1 extend5 step40000 / INT8 block32；V1保留作基线和回退。总体结果与实验取舍见[V2系列总结](reports/v2-summary.md)。V3 A/B、V4-B、V4.1未形成稳定替换收益；V4.2第20轮在Standard开发集领先，固定reference仅比V2.1多6条且不显著，已完成Core ML、分段KV与iPhone测试宿主测量，作为可选变体保留，未替换V2.1。

## 使用与复现

| 文档 | 内容 |
| --- | --- |
| [数据](data.md) | 冻结语料、来源、清洗和split |
| [V4 基础训练](training-v4.md) | 固定 8K 架构、已完成 A/B 调度、结果与恢复 |
| [V4 后训练](posttraining-v4.md) / [转换数据](data-v4-c1.md) | 已完成 C1/C1r2、冻结数据、条件评分与后训练设置 |
| [V3数据实验](plan_v3.md) / [A评测](reports/v3-20261008/a-evaluation.md) / [B联合重训](reports/v3-20261008/b-plan.md) / [B评测](reports/v3-20261008/b-evaluation.md) | A与B训练、评测、归档完成；B未形成稳定替换收益，保留V2.1部署 |
| [V2实验设计](plan_v2.md) / [V2训练](training-v2.md) | 模型、tokenizer、裁剪、配置和运行方式 |
| [V1训练](training.md) | 冻结基线复现 |
| [评测](evaluation.md) | 评分语义、同池结果、统计与标签限制 |
| [Standard IME基准](benchmarks/standard-ime.md) / [首次结果](benchmarks/standard-ime-results-20261008.md) | WRIME/JMultiWOZ独立来源、旧网页回归、审核与版本协议；V2.1/V3 A固定比较 |
| [IME 指标 v2](benchmarks/metrics-v2.md) | 书写宽松与池条件 Top-1、作者簇检验、判定规则与轨道角色 |
| [Standard v2 blind](benchmarks/standard-ime-v2-blind.md) / [同音词挑战](benchmarks/homophone-ime.md) / [分段输入原型](benchmarks/incremental-ime.md) | 新确认集合、书面语同音词集合与分段协议原型；均已冻结，尚未评分 |
| [确认评分计划 v2](benchmarks/confirmation-v2-plan.md) | Standard v2 blind 与同音词集合首次评分的预注册假设、模型、判定规则与功效 |
| [候选导出 v2](benchmarks/candidates-v2-handoff.md) | n-best 20/50、确定性与左文依赖对照的 Mac 交接 |
| [Core ML](coreml.md) | 导出、量化、客户端接口与设备范围 |
| [本地产物](artifacts.md) / [跨平台管理](reference/artifact-exchange.md) | 路径、归档、Git与ZIP协作 |

## V4 实验记录

| 阶段 | 报告 |
| --- | --- |
| 基础 LM | [A 续训实验](reports/v4-20261015/evaluation.md)、[B 完整十二轮对照](reports/v4-20261015/b12cosine-evaluation.md) |
| 转换后训练 | [C1r2 数据修订、完整结果与 Reading 消融](reports/v4-20261016/c1r2.md) |
| V4.1 混合预训练 | [训练设置](training-v41.md)、[十六轮完整评估](reports/v41-20261016/evaluation.md) |
| V4.2 LM／Chat 预训练 | [训练设置](training-v42.md)、[二十轮完整评估](reports/v42-20261020/evaluation.md)、[固定 reference 与退化分析](reports/v42-20261020/reference-regressions.md) |
| V5 规模扩展预训练 | [设计、V4.3 饱和证据、结果与结论](training-v5.md)：BPC −0.10，IME Top-1 无提升，不部署 |
| IME 错误分解 | [瓶颈不在 LM 容量](reports/analysis-20261021/ime-errors.md)：池外 48%、书写形式 35–44%、语义 8% |
| 免训练重打分 | [PMI、长度与引擎分融合](reports/analysis-20261021/rescoring.md)：均未通过独立检验，部署保持纯 LM 打分 |
| V4.2 Apple 部署 | [Core ML、INT8、分段 KV、Mac 与 iPhone 测量（英文）](../reports/mac-20261020/v42-coreml.md) |
| 外部日语模型 | [TinySwallow](reports/v4-20261016/external-1p5b.md)、[RakutenAI](reports/v4-20261016/external-rakuten-1p5b.md)、[Sarashina](reports/v4-20261016/external-sarashina-1p4b.md) |

V4-a 本地工作：[Tokenizer 初始化、训练与词表对照](reports/v4-20261014/tokenizer.md)、[8K token 数据与窗口](reports/v4-20261014/token-data.md)。
模型与训练入口见 [固定架构、本地验收和启动配置](reports/v4-20261014/model.md)。

## V2 实验记录

| 阶段 | 报告 |
| --- | --- |
| 数据与模型设计 | [Tokenizer](reports/v2-20261007/tokenizer.md)、[模型](reports/v2-20261007/model.md)、[Token store](reports/v2-20261007/token-data.md)、[Prefix crop](reports/v2-20261007/prefix-crop.md) |
| V2.0 | [运行环境](reports/v2-20261007/environment.md)、[完整评测](reports/v2-20261007/evaluation.md) |
| V2.1 restart1 | [配置](reports/v2-20261007/v21-plan.md)、[性能对照](reports/v2-20261007/v21-performance.md)、[中断恢复](reports/v2-20261007/v21-recovery.md)、[评测](reports/v2-20261007/v21-evaluation.md) |
| V2.1 extend5 | [追加训练与模型选择](reports/v2-20261007/v21-extend.md) |
| 扩大IME数据集 | [候选构建](reports/v2-20261007/ime-3000-handoff.md)、[初评](reports/v2-20261007/expanded-ime-evaluation.md)、[标签与错误分析](reports/v2-20261007/label-error-audit.md) |
| Core ML与iPhone | [转换和量化](reports/mac-20261008/v21-coreml.md)、[误差分析](reports/mac-20261008/v21-quantization-review.md)、[宿主与真实扩展](reports/mac-20261008/v21-iphone.md) |

## V1参考与历史

[FP32结果](reference/results.md)、[Core ML实测](reference/coreml-v1.md)、[模拟器内存](reports/mac-20261006/simulator-memory.md)、[真实扩展](reports/mac-20261006/iphone-keyboard-memory.md)、[快照归档](reports/mac-20261006/handoff.md)、[导入核对](reports/mac-20261006/merge-verification.md)。

详细方法：[语料流水线](reference/data-pipeline.md)、[AJIMEE导出](reference/ajimee-benchmark.md)、[开发集生成](reference/development-generation.md)、[组合排序](reference/hybrid-ranking.md)、[联想](reference/phrase-demo.md)。

`history/`保存V1实测前的[部署设计](history/deployment.md)、[命令参考](history/coreml-guide.md)和[阶段记录](history/workflow.md)，仅描述当时状态。各报告注明日期、版本与测量范围；原始JSON、逐候选分数和trace保存在`outputs/`。
