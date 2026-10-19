# V4.1 混合预训练

V4.1 从随机权重开始，将 V2.1 原始 LM 语料、V3 使用过的自然聊天，以及 C1r2 读音转换训练集纳入同一次训练。实验检验联合训练是否比“基础 LM 预训练后再做转换后训练”更适合固定的 15M 模型。此前后训练结果见 [C1r2 报告](reports/v4-20261016/c1r2.md)。

2026-10-16 11:34（北京时间）在 RTX 4090 正式启动，PID 5367；现已正常完成 16 轮、865,557 次更新，关键权重与逐轮记录下载核对完成。普通重排最高 509/728（第十三轮），条件重排最高 461/728（第十六轮）；开发集收益与回归取舍见[完整评估](reports/v41-20261016/evaluation.md)，训练曲线见 [W&B](https://wandb.ai/voltline233/vimeml/runs/5uuh8yfb)。

## 架构与随机性

模型保留 V4 的 8,192 词表、context 128、d384、七层、六头、SwiGLU 960，共 15,070,848 参数。SentencePiece 模型、标记 ID 与评分语义保持冻结。

本次 seed 为 **1219827974**，由 Python `secrets.randbelow(2147483646) + 1` 从操作系统随机源生成，随后固定，用于权重初始化、训练顺序、prefix crop 和聊天采样。记录位于 `outputs/remote/v41-20261016/randomization.json`。已有 train/validation 划分、转换标签和评测候选池不随新 seed 重划。没有加载 V4-B 或后训练 checkpoint，AdamW 状态从零开始。

## 数据组成

| 数据池 | 训练规模 | 用法 |
| --- | ---: | --- |
| V2.1 FineWeb/Tatoeba，采用 V4 8K 编码 | 原始 25,185,368 句，排除 97 条与已知验证文本重合的句子 | 每个 LM epoch 完整遍历其余窗口 |
| V3 RealPersonaChat / MRMP | 清理后 477,334 条自然发言；验证 4,986 条 | 占 prefix crop 后实际 LM 目标 token 的约 5% |
| C1r2 转换训练集 | 982,853 条；验证 18,675 条 | 每个 LM epoch 遍历一次，独立计算转换 loss |

聊天沿用原对话级划分，以独立自然发言训练普通 LM。排除 17,882 条与旧训练池重复的发言、1,050 条与已知验证文本重合的发言。转换池额外排除一条母句与聊天验证文本重合的训练样本。原数据文件和参考答案不修改，通过新索引与排除掩码实现。

5% 是实际有效 LM token 配额，不是句数或 loss 权重。第一个 epoch 聊天约 3,026 万个有效目标，相当于该聊天池约 5.22 次有效遍历；其余 epoch 随 prefix crop 略有变化。这个重复率是采样设计的结果，需要结合聊天验证曲线判断是否过拟合。

此次仅使用明确的 train/validation 文件和冻结的非盲开发集，没有运行 V3 原有的全量混合脚本，没有访问 test 或 blind。数据整理只做编码、既有划分检查和文本重合排除，不训练 tokenizer，不引入 V3 的 JpnMix 网页扩充。

## 目标函数与轮次定义

普通 LM 对有效 next-token 目标计算 loss；转换输入为 `BOS <CTX> context <READ> reading <OUT> surface EOS`，仅监督 OUT 与 EOS。转换数据保留 C1r2 已冻结的读音与空 READ 视图。

每次更新先分别按两项任务的实际监督 token 数归一化，再计算：

`L = 0.8 × L_lm + 0.2 × L_conversion`

LM 逻辑 batch 为 512；转换配额按当轮 LM 更新数均匀分配，每次为 18 或 19 条，整轮恰好 982,853 条。转换加载器的 batch 64 仅用于分桶和预取，不代表每次优化都重复消费 64 条。这样避免小转换池因双流等量 batch 而被意外重复数百轮。

预算为 **16 个完整 LM 混合 epoch、865,557 次更新**，共 9,683,477,884 个有效 LM 训练目标，其中聊天 484,173,786 个；另完整遍历转换池 16 次。各 epoch 的窗口数及 SHA256 冻结在混合 manifest 中。最后不足整批的 LM 窗口保留。

## 训练设置

| 项目 | 设置 |
| --- | --- |
| 配置 | [train-v41.toml](../../configs/train-v41.toml) |
| 环境 | AutoDL RTX 4090；BF16 AMP；FP32 权重和 optimizer state |
| 优化器 | CUDA fused AdamW，betas (0.9, 0.95)、weight decay 0.1、clip 1.0 |
| 学习率 | 1,000 次更新 warmup；完整 16 轮 cosine，6e-4 至 6e-5 |
| 数据加载 | 四个 workers、长度分桶、pinned memory、persistent workers |
| 编译 | 动态形状 Transformer 主干；有效目标词表头与 CE 保持 eager |
| Prefix crop | 概率 0.30，至少保留八个目标 |
| 保存 | 每 5,000 次更新、每轮、完成及正常中断时 |
| 中途验证 | 每 2,000 次更新，固定 4,096 条转换验证样本 |
| 完整轮次验证 | 原 LM BPC、聊天 BPC、完整转换 loss、冻结 Standard development 条件评分及普通 LM 评分 |

本轮从一开始按 16 轮设置调度，不沿用后训练的 LM 退化早停或转换 plateau 早停。验证用于观察与保留各维度最佳权重；训练异常会停止并保留最近安全 checkpoint，不自动改变预算。

每项完整验证使用未编译 FP32 路径。固定子集 loss 与完整转换 loss 分开记录，普通 LM 和条件评分分开记录；不同 tokenizer 的 token loss 不直接比较。Standard development 是开发集，按其选择的模型仍需后续独立验证，不能视为盲测收益。

## 产物与恢复

| 路径 | 内容 |
| --- | --- |
| `artifacts/token-data/corpus-v41-chat-8k/` | 冻结聊天 token bank 与来源指纹 |
| `artifacts/training-data/corpus-v41-chat-c128/` | 聊天窗口索引 |
| `artifacts/training-data/corpus-v41-mixed-c128/` | 数据请求、排除掩码及十六轮映射 |
| `artifacts/models/tiny-ja-v41-e8k-d384-l7/` | 权重、optimizer/RNG/游标、各任务累计曝光、验证和环境记录 |
| `outputs/handoff-v41/` | 源码包及逐文件 SHA256 清单 |
| `outputs/remote/v41-20261016/` | 随机源记录及本次远端启动凭据之外的审计信息 |

映射在本地冻结，在远端从已有 token bank 重建并逐轮核对 SHA256，避免重复传输约 1.8 GB 的索引。正式入口在读取数据前校验配置、源码、tokenizer、数据和全部映射身份；恢复保留两条流已经提交的游标、AdamW、RNG 及验证历史。

checkpoint 分别保留 `best-lm.pt`、`best-conversion.pt`、`best-rerank.pt`、`best-ordinary.pt` 和 `last.pt`；`best.pt` 是条件评分开发集选择。数据和权重不纳入 Git。

## 结论边界

混合比例与任务权重是本轮实验设置，不是已证明的最优配方。相对 V4-B，本轮同时改变训练数据、任务、随机种子和预算，因此最终差异不能单独归因于某一项。判断依据是分别观察原 LM、聊天、转换和固定候选重排的变化，再决定是否值得继续。
