> 历史文档：这是 Mac 实测前的 Windows 计划；当前状态与操作入口以 [Core ML 指南](../coreml.md) 和 [文档索引](../index.md) 为准。

# 工作流程

截至 2026-10-06：数据、审核、tokenizer、token 编码、完整训练和第一轮评测均完成。当前使用冻结的 `tiny-ja-v1`，不需要重复清洗或继续训练。下一步准备 Core ML 转换。

## 阶段与入口

| 阶段 | 状态 | 入口 / 说明 |
| --- | --- | --- |
| 原始数据检查 | 完成 | `scripts/tools/inspect_data.py` |
| 并行清洗和合并 | 完成 | `scripts/corpus/`；[数据处理](../reference/data-pipeline.md) |
| 存疑审核与分层抽检 | 完成，638 条隔离块继续排除 | `scripts/review/`；[数据处理](../reference/data-pipeline.md) |
| SentencePiece / 全量编码 | 完成 | `scripts/tokenizer/`；[训练指南](../training.md) |
| DataLoader / GPU 检查 | 通过 | `scripts/training/check_data.py` |
| 完整 1 epoch 训练 | 完成 | `configs/train-v1.toml`；[基线结果](../reference/results.md) |
| AJIMEE 与独立开发集评测 | 完成 | `scripts/benchmarks/`；[组合排序](../reference/hybrid-ranking.md) |
| 联想网页与自然度复核 | 完成，质量尚不稳定 | `scripts/tools/phrase_demo.py`；[演示说明](../reference/phrase-demo.md) |
| Core ML / 4bit / iPhone | 独立工具已实现，真实转换/设备测试待手动启动 | `scripts/deployment/coreml.py`；[操作指南](coreml-guide.md) |

## 日常操作

体验联想：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/tools/phrase_demo.py
```

查看固定评测结果，无需重新推理或调用 API：

- `outputs/ime-eval/tiny-ja-v1-ajimee/results.md`
- `outputs/ime-eval/tiny-ja-v1-ajimee-hybrid-dev-v2/results.md`
- `outputs/phrase-demo/tiny-ja-v1/review.md`

确认语料保持原版本：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/tools/check_corpus.py
```

当前操作只需已有正式产物。新机器的最小推理文件和完整复现文件不同，见 [迁移清单](../artifacts.md)。

## 新实验约定

配置、模型、候选池或提示词改变时，使用新的输出目录并保留旧基线；多数入口拒绝覆盖非空目录。不要把已完成的 300 步短跑配置直接延长为正式训练。

审核材料、生成草稿、开发集、公开评测和训练语料分别保存。当前部署默认纯 LM 排序；若恢复融合，开发集用于选 λ，AJIMEE 用固定策略对照，不按测试集结果自动挑选 λ。AI 审核保留其身份与局限，不标记为母语者裁定。

所有耗时分清开发机与 iPhone；所有统计分清真实预测目标、padding 和子词。Core ML 压缩后的模型重新验证数值和效果，不直接继承 FP32 性能或质量结论。
