# V4 转换任务后训练

V4-C1 在已完成的 V4-B 语言模型上加入读音条件转换和普通 LM 重放。2026-10-16 首轮在第 2 轮结束触发 LM 保留保护；数据与停止策略修订实验 [C1r2](reports/v4-20261016/c1r2.md)完成 11 轮后按转换验证 plateau 正常早停，最佳条件评分为 473/728，仍低于 V4-B 的 498/728。下文保留首轮 C1 的原设计与配置；最新实验以 C1r2 记录为准。[英文文档](../posttraining-v4.md)。

C1 验证读音监督能否改善现有 AzooKey 候选池上的排序，同时保留普通语言建模能力。C2 为后续候选 Ranking 实验，以 C1 取得收益为前提。此处阶段编号与 [V4-B 评估](reports/v4-20261015/b12cosine-evaluation.md)中的 A/B 基础 LM 调度实验分开。

## 初始化与冻结数据

| 项目 | 身份与设置 |
| --- | --- |
| 源模型 | `artifacts/models/tiny-ja-v4-b12cosine-e8k-d384-l7/best.pt`，step 584,000 |
| 源模型 SHA256 | `18c49f1d4de5b2852f98c9bc30bf48ca7c84077cfb09a25e3c9002cedf5d1106` |
| 架构 | 15,070,848 参数；d384、7 层、6 头、SwiGLU 960、context 128 |
| Tokenizer | 原有 8K Unigram，identity normalization、byte fallback |
| 标记 ID | `<CTX>` = 4、`<READ>` = 5、`<OUT>` = 6；PAD/UNK/BOS/EOS = 0/1/2/3 |
| 转换数据 | 2,841,003 条训练、18,956 条验证 |
| LM 重放 | 原冻结 V4 train token store 与 context-128 窗口 |
| LM 验证 | 原完整 260,337 句 |

初始化载入全部源权重，使用新的 AdamW 状态和独立的后训练步数；源 step 与 token 曝光量另行记录。Tokenizer 和模型维度保持固定。

[转换数据](data-v4-c1.md)从原 train/validation 文本中每个母句选取一个片段，经双词典比对、原文对齐、假名正字法处理及保守歧义过滤，形成自动标注的 silver 数据。已检查母句及精确任务样例隔离，尚不能确认文档级或语义独立性。Benchmark 文本不进入训练，数据制作和评测均不访问 blind。转换验证母集曾用于基础 LM 选模，因此不属于整个研发流程的独立测试集。

## 输入与训练目标

```text
[BOS]<CTX>昨日友達と<READ>はなした<OUT>話した[EOS]
```

文本字段分别编码，结构标记按 ID 插入；原始字段中的保留标记字符串拒收。转换 labels 只监督 OUT 和 EOS，包含从 `<OUT>` 位置预测首个表记 token 的目标。Labels 只 shift 一次；CTX、READ、结构标记和 padding 作为预测目标时被 mask，条件输入的 embedding 仍参与梯度更新。

完整序列最多 129 tokens，对应 128 个输入/目标位置。制作数据时仅按文字边界左裁 CTX，保留完整 READ/OUT；后两者本身超窗的样例排除。固定约 10% 的训练样例提供空 READ 视图，验证保留完整读音。普通 LM 沿用无标记格式及 30% 的合格首窗口 prefix crop，至少保留八个目标，不新增 `<LM>`。

```text
L = 0.7 * mean_token_loss(OUT + EOS)
  + 0.3 * mean_token_loss(ordinary LM)
```

每次 optimizer 更新处理 256 条转换和 256 条 LM 样例。两任务分别按实际监督 token 数归一化，加权反向传播后统一 clip 和 step；不再将 loss 权重叠加为采样概率。保留最后不足整批的转换样例；LM 尾批跨轮补足 256 条。拆分 microbatch 时仍按整组任务的监督 token 数归一化。

## 首轮配置

[C1 配置](../../configs/posttrain-v4-c1.toml)冻结以下初始实验设置，其任务效果尚待验证。

| 项目 | 设置 |
| --- | --- |
| 环境 | Linux、RTX 4090 |
| 精度与运行路径 | BF16、fused AdamW、主干编译、有效 token 输出头保持 eager |
| Batch / microbatch | 转换 256 ＋ LM 256；每任务一个 microbatch |
| AdamW | Betas (0.9, 0.95)、weight decay 0.01、gradient clipping 1.0 |
| 学习率 | 单次 cosine，1e-4 → 1e-5；warmup 1,332 次更新 |
| 最大预算 | 转换集完整遍历 12 次，133,176 次更新 |
| 转换曝光量 | 137,842,152 个 OUT/EOS 目标；LM 曝光另计 |
| 数据加载 | 两任务独立长度分桶；每个训练流四个 workers，验证零 workers |
| 子集验证与保存 | 每 2,000 次更新；固定 4,096 条转换验证子集 |
| 完整验证 | 每轮完整转换 NLL、原 LM BPC、固定 Standard 开发集 |
| 转换早停 | 相对停止参考值累计改善超过 0.2%；连续三轮未达到则停止 |
| LM 保留保护 | 完整 BPC 连续两轮超过起点的 1% 恶化上限时停止 |

完整 LM 与 IME 验证采用原始 FP32 路径，不复制第二份模型。起始 LM 参考值为 V4-B 完整 FP32 BPC 3.0277770232964114；转换初始 NLL 在更新前计算。固定子集 loss 与完整验证指标分开记录。两任务分别报告监督 token、条件输入 token、padding 与吞吐。转换集的 12 次遍历不同于原 2,500 万句 LM 语料的 12 轮；C1 稳定吞吐和完整训练耗时尚未测量。

## 评分与模型选择

主要条件分数为 OUT tokens 的 full-vocabulary logP sum，不计 EOS；EOS 参与训练以学习终止。READ 仅来自原始 query，经过 NFKC 与片假名转平假名处理，不从答案或候选反推。候选池、参考答案、并列顺序及完整评测分母保持固定。评测不裁 CTX；任一候选超出结构化输入窗口时，整条样例回退 AzooKey 原序，另行报告回退数量。

对照包括 V2.1 普通 LM、V4-B 普通 LM、C1 普通 LM、C1 条件评分，以及同一 C1 checkpoint 的空 READ、空 CTX 诊断。后两项用于观察条件依赖，不属于单独训练的等算量对照。

选模先要求完整 LM BPC ≤ 3.058054794，即起点相对恶化 1% 的上限。合格模型按 Standard contextual Top-1 排序，并列选择完整转换 NLL 更低者，再选择更早 step。普通 LM、转换 loss 和重排最优分别保存。

| 评测项目 | 首轮目标 |
| --- | --- |
| Standard 开发集 | 至少 505/728，追平 V2.1；V4-B 起点为 498/728 |
| 工程目标 | 513/728；不作为统计显著性阈值 |
| 来源回归 | WRIME、JMultiWOZ 各自相对 V2.1 下降不超过一个百分点 |
| 历史回归 | 至少 369/500；V2.1 为 374/500 |
| 普通 LM | 完整 FP32 BPC ≤ 3.058054794 |
| 联想 | 现有 14 例保留作定性诊断 |

以上为实验验收条件，尚无 C1 实测结果。评估报告纠正、改坏、净变化、来源分组和配对不确定性。反复使用的开发集提供探索证据；正式替换结论仍需要独立评测。

## 实现与恢复

| 模块 | 职责 |
| --- | --- |
| `src/vimeml/posttraining/readings.py` | 双词典片段对齐与读音规则 |
| `src/vimeml/posttraining/data.py` | 冻结存储、编码及 OUT/EOS mask |
| `src/vimeml/posttraining/streams.py` | 长度分桶、microbatch、两任务已提交游标 |
| `src/vimeml/posttraining/runtime.py` | 源模型身份、任务归一化、早停与恢复检查 |
| `src/vimeml/posttraining/train.py` | 专用训练、验证、保存与选模 |
| `src/vimeml/posttraining/infer.py` | 条件编码与固定候选池评分 |
| `scripts/posttraining/` | 数据制作、审核、过滤、训练、评测与交接入口 |

跨阶段初始化只恢复权重。同阶段 resume 要求配置、数据、代码、精度与任务/评分身份一致，并恢复 AdamW、RNG、两数据流已提交游标和早停计数。游标只在更新成功后提交；完整验证前保存安全状态，恢复时完成待执行验证，不重复 optimizer 更新。原 V4 预训练和 V2 初始化保护保持有效。

产物目录为 `artifacts/models/tiny-ja-v4-c1-e8k-d384-l7/`。`best-rerank.pt` 及其 `best.pt` 别名要求 LM 保留合格；`best-conversion.pt`、`best-lm.pt`、`epoch-N.pt`、`last.pt` 分别保存其他选择与恢复状态。没有合格重排模型时不生成对应角色。

15 项 CPU 测试覆盖编码/mask、源权重一致与空 AdamW 状态、microbatch 梯度等价、预取数据流恢复、optimizer/RNG 续接、早停、读音来源及评分/回退合同。构造数据的完整训练循环在验证中途失败后恢复，最终权重与不中断运行逐位一致。这些检查验证程序行为；Linux 编译、GPU 稳定吞吐与任务效果仍待正式执行。

`outputs/handoff-v4-c1/` 保存源码包与 SHA256 交接清单，明确列出转换二进制、manifest/acceptance、LM token/窗口、tokenizer、源 checkpoint 及允许的非 blind 快照。远端已有文件可核对 hash 后复用。原始语料文本和标注词典不属于 GPU 训练依赖。

## 本地容量记录

RTX 5060 Laptop 8 GB 的 BF16 eager 容量检查采用 microbatch 32、长度 128，覆盖两任务 mask，并预留两份 FP32 AdamW moment。峰值 allocated/reserved 为 1,150.78/1,340 MiB，optimizer 更新为零。该记录支持进一步验证本地可行性，未测完整训练器或编译模式的显存与速度。原始记录为 `outputs/posttraining-v4-c1-local-memory.json`；已准备的正式配置面向 RTX 4090。

## C2 候选 Ranking

C1 取得收益后，可在独立实验中从训练 context/reading 生成真实 AzooKey 候选列表，固定引擎与词典版本。正确答案不在候选池时跳过 ranking 项；合理替代表记采用多正例或隔离。Listwise loss 使用与推理相同的 OUT-sum 分数：

```text
Lrank = logsumexp(all scores) - logsumexp(positive scores)
```

同时保留 LM 和转换目标。Ranking 权重与预算在 C1 结果及候选规模明确后另行冻结。若 C1 未取得收益，先分析读音质量、片段分布、超窗、LM 遗忘及评分口径，再确定后续实验。
