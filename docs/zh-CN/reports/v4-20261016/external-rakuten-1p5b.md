# RakutenAI-2.0-mini-instruct 本地重排对照

2026-10-16 复用 [TinySwallow 对照](external-1p5b.md)的四组冻结开发／回归集和原始文本概率评分接口，在本地 RTX 5060 Laptop 8GB 完成 1,565 条评测。Rakuten 的 Standard development 为 462/728，未复现 TinySwallow 的 527/728；其余三组与 V2.1 接近。此次结果用于比较候选重排适用性，不代表模型整体日语或对话能力。

## 模型与固定协议

选用官方 [Rakuten/RakutenAI-2.0-mini-instruct](https://huggingface.co/Rakuten/RakutenAI-2.0-mini-instruct)，revision `6d902489587d324b7d5e201299e4e1a169f3a40b`。官方说明将其定位为 RakutenAI-2.0-mini 的指令微调版本；其[基础模型](https://huggingface.co/Rakuten/RakutenAI-2.0-mini)为从头训练的日语语言模型。本轮使用指令微调权重，而不是基础权重。

实际模型为 Mistral 架构，1,534,683,136 参数、48,000 词表、22 层、hidden 2048、原生 BOS 1。使用 BF16 权重和 forward、FP32 full-vocabulary CE，无量化或训练更新。沿用已隔离的 Transformers 4.57.6 与 PyTorch 2.10.0+cu128，加载 safetensors，禁用 remote code。

原候选、参考答案、左文与输入文件 hash 保持不变。主分数仍为 `context + candidate` 联合分词后，公共 token 前缀之后的完整词表 logP sum；原生 BOS 开始，不附加候选末尾 EOS。窗口固定 128，稳定并列保留 AzooKey 原顺序，不裁剪上下文；任何不适配本应整条回退，本轮全部四组均无回退。Native 128K context 未用于扩展评分窗口。

没有加入 chat template、指令提示或 READ 条件。因而这里测量的是指令微调权重在既有原始文本 LM 重排接口上的迁移表现，不能直接外推到指令式转换。Mean 与空左文分数预先列为辅助诊断，未据结果调参。

## 完整结果

以下数字为 Top-1 正确条数，V2.1、V4-B 与 TinySwallow 均复用身份已核对的历史逐条结果。

| 数据组 | 样本 | AzooKey | V2.1 | V4-B | TinySwallow | Rakuten Instruct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard development | 728 | 453 | 505 | 498 | **527** | 462 |
| Legacy regression | 500 | 317 | 374 | **377** | **377** | 375 |
| AJIMEE | 200 | 87 | **144** | **144** | **144** | **144** |
| Historical development | 137 | 111 | 122 | 120 | **127** | 122 |

Standard 为 63.46%，相比 V2.1 低 5.91 个百分点，相比 TinySwallow 低 8.93 个百分点。相对原 AzooKey 纠正 91 条、改坏 82 条，净增九条；同一候选池覆盖正确答案的样本仍为 630/728。

| Standard 来源 | 样本 | V2.1 | V4-B | TinySwallow | Rakuten Instruct |
| --- | ---: | ---: | ---: | ---: | ---: |
| WRIME | 483 | 298 | 297 | 319 | 282 |
| JMultiWOZ | 245 | 207 | 201 | 208 | 180 |

两项来源都低于现有普通 LM 基线。对照未改变参数量以外的变量：架构、词表、数据和训练过程均不同，不能把差异单独归因于指令微调或 Mistral 架构。

## 配对比较与辅助诊断

相对 TinySwallow，Standard 上新纠正 31 条、损失 96 条，净少 65 条。相对 V2.1 为 48／91，净少 43 条；相对 V4-B 为 46／82，净少 36 条。未经多重比较校正的双侧 exact McNemar 描述 p 分别为 `6.33e-9`、`0.000333`、`0.001862`。这些仍属于重复使用开发／回归集的探索性比较。

相对 TinySwallow，Legacy 的改善／退化为 20／22，AJIMEE 为 10／10，Historical 为 5／10。AJIMEE 总分打平仍存在逐条差异。

| 数据组 | 原左文 sum，主分数 | 空左文 sum | 原左文 mean，辅助 |
| --- | ---: | ---: | ---: |
| Standard development | 462 | 416 | 337 |
| Legacy regression | 375 | 333 | 335 |
| AJIMEE | 144 | 122 | 102 |
| Historical development | 122 | 116 | 69 |

左文在四组上均有正的总收益，但未弥补主开发集与 TinySwallow 的差距。未根据这些结果更换主评分、追加其他 prompt 或重新挑选模型版本。

## 运行与审计

完整评测含模型加载耗时 168.10 秒，不含权重下载、依赖准备和结果核对。PyTorch 权重 allocated 为 2,928.18 MiB，峰值 allocated 为 3,000.22 MiB，峰值 reserved 为 3,050 MiB，约 2.98 GiB。记录不含桌面及其他进程，也不是物理 footprint 或部署延迟。

评测器 `scripts/benchmarks/evaluate_external_lm.py` 已增加显式固定模型注册项，评分实现不变。各次入口源码原始字节独立保存在相应结果目录，旧 TinySwallow 评测计划与源代码 SHA256 仍可核对。投影分块与完整 logits 的数学一致性检查通过；Mistral 原生 forward 的输出投影路径已核对。

产物位于 `outputs/evaluation-rakuten-instruct-20261016/`。`model-source.json`、`plan.json` 记录来源、模型及数据 SHA256 和协议；`environment.json`、`dependencies.json` 记录环境；各组 `scores.jsonl`、`metrics.json`、`paired-*.json` 记录逐 token 分数和相对三套基线的逐条比较；`summary.json`、`comparison-summary.json`、`acceptance.json` 记录结束及审计。1,565 条的候选集合、分数有限性、目标计数和概率加总检查均通过，评测输入及三套基线文件均未变化。

没有读取 test 或 blind，没有修改标签，没有使用评测答案训练模型或编写任务提示，没有操作远端 V4.1 训练。外部预训练／微调文本与公开评测内容的重叠未知，原开发／回归参考标签及快照来源限制继续适用。

本次结果支持优先保留 TinySwallow 作为当前概率重排任务的教师候选。日语微调标签与通用对话指标不足以替代实际 IME 评测；能否将 TinySwallow 的收益蒸馏到 15M 模型，仍是独立待验证问题。
