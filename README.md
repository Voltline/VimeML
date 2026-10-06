# VimeML

为 iOS 日语输入法 Vime 实验本地 Tiny Japanese LM。AzooKey 负责假名转换检索，LM 负责句内候选重排和短语联想：

```text
Romaji → Kana → AzooKey candidates → Tiny Japanese LM rerank → final candidates
```

当前权重已冻结，完整语料、16K SentencePiece、第一轮 Tiny GPT 训练、真实候选评测和本地联想演示均已完成。下一阶段是 Mac 上的 Core ML 转换、权重压缩和 iPhone 键盘扩展验证；尚未生成 Core ML 模型。

## 当前基线

| 项目 | 结果 |
| --- | --- |
| 语料 | 25,713,003 条唯一句子；train 25,185,368 条 |
| Tokenizer | unigram 16,384 词表，byte fallback；全量无 UNK / roundtrip 错误 |
| Transformer | 7,386,624 参数；4 层、4 头、d_model 256、FFN 1024、context 128 |
| 训练 | 1 epoch，573,295,237 个预测目标，57 分 50 秒 |
| 完整 validation | loss 4.5416，PPL 93.84；test 未用于训练或调参 |
| AJIMEE Top-1 | 原排序 87/200，纯 LM 124/200，固定 λ=2 组合 118/200 |
| 联想演示 | 20 个前缀中 10 个有明确自然建议；仍有重复、语义和截断问题 |

结果与限制见 [基线结果](docs/results.md)。上述评测不是 Vime 线上准确率或 iPhone 性能。

## 使用当前模型

在仓库根目录执行：

```powershell
# 联想网页：打开 http://127.0.0.1:8765/
.\.venv\Scripts\python.exe -X utf8 -u scripts/tools/phrase_demo.py

# 命令行 greedy / sample 续写
.\.venv\Scripts\python.exe -X utf8 scripts/training/infer.py

# 检查正式语料的元数据、文件大小和核心指纹
.\.venv\Scripts\python.exe -X utf8 scripts/tools/check_corpus.py
```

模型和数据不在 Git 中。新机器安装依赖后，按 [本地产物与迁移](docs/artifacts.md) 复制所需文件。每个入口支持 `--help`；训练和 API 请求由使用者手动启动。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [工作流程](docs/workflow.md) | 阶段状态与操作入口 |
| [数据处理](docs/data-pipeline.md) | 并行清洗、跨机分区、合并、审核与来源追踪 |
| [Tokenizer 与训练](docs/training.md) | 环境、编码、数据加载、训练复现和 TensorBoard / W&B |
| [AJIMEE](docs/ajimee-benchmark.md) | 固定公开评测集、Mac 构建和真实候选导出 |
| [开发数据](docs/development-generation.md) | DeepSeek 生成、缓存导出、审核；可选合成诊断 |
| [组合排序](docs/hybrid-ranking.md) | 开发集评分、选 λ、固定策略评测 |
| [联想演示](docs/phrase-demo.md) | 本地网页、固定场景和自然度复核 |
| [部署计划](docs/deployment.md) | FP16 Core ML → 4bit → iPhone 验证 |
| [本地产物](docs/artifacts.md) | 保存范围、磁盘用途与迁移清单 |

## 目录

```text
scripts/{corpus,review,tokenizer,training,benchmarks,tools}/  手动运行入口
src/vimeml/                                               实现
configs/                                                  版本化配置和诊断前缀
annotations/                                              清洗批准记录
tests/                                                   回归测试
docs/                                                     操作指南和基线说明
datasets/                                                 原始数据（本地）
outputs/                                                  语料、审核、评测与维护记录（本地）
artifacts/                                                模型、tokenizer、token、候选与策略（本地）
runs/                                                     TensorBoard 事件（本地）
```

Python ≥3.11，当前使用 3.13。依赖分为 `requirements-data.txt`、`requirements-tokenizer.txt`、`requirements-training.txt` 和可选 `requirements-tracking.txt`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

Git 保存代码、配置、测试、文档和少量批准记录；不保存语料、模型、API 回复、运行日志、虚拟环境或密钥。SJTU key 仅通过本地环境变量提供。正式语料与 checkpoint 绑定核心代码指纹，相关模块保留原路径和内容；清理目录不改变模型或评分规则。
