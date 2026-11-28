# vimeml-ja-v6-200m test 确认计划（预注册）

[English version](../../../ja/benchmarks/confirmation-ja-v6-200m-plan.md)

2026-11-28 起草。本计划和执行代码提交后才算注册；提交之前，200M 不在 `ime-kana-ja-v1` 和 `ime-kana-ja-colloq-v1` 的 test 上运行。发布门槛的原则见 [v6 发布范围](../release-ja-v6-plan.md#发布门槛)，本计划给出具体的检验。

执行入口是 [confirm_ja_v6_200m.py](../../../../scripts/ja/benchmarks/confirm_ja_v6_200m.py)，所有固定项都写在 [ja_v6_200m_confirm.py](../../../../src/vimeml/benchmarks/ja_v6_200m_confirm.py) 中。搜索、合并和评分与 [20M 确认](confirmation-ja-gen-plan.md)相同（`kana_gen_ja.evaluate_arm`，常量取自 `ja_gen_confirm`）。

## 要回答的问题

200M 是 2.2 在开发集上评测的桌面模型（[2.2 报告](../reports/ja-gen-stage2-200m-20261128/report.md)），只用于桌面端。本计划在 test 上回答：

- 它的 kana 转换宽松 Top-1 是否高于现在部署的 V4.2 方式（AzooKey N-best 20 + V4.2 重排）？
- 它在 kana 和口语两个 test 上的宽松 Top-1 是否显著低于手机端的 `vimeml-ja-v6-20m`？
- 它是否显著高于 `vimeml-ja-v6-20m`？

开发集上 200M 对 `vimeml-ja-v6-20m` 的差距很小（见“怎么读”），所以主假设不设成“高于 20M”，而是与 20M 确认相同的对 V4.2 的检验，再加上不低于 20M 的守门。中文 v2-200m 的主假设是高于 v2-20m，这里是有意的不同。

## 模型

| 角色 | 模型 | checkpoint SHA256 |
| --- | --- | --- |
| 候选 | `vimeml-ja-v6-200m`：2.2 联合读音预训练 200M（`configs/ja/train-ja-kana-200m-joint.toml`，193,625,088 参数，845,000 次更新） | `47721ddd357cb3a720d7108e4aeaa60f671d83167504fd9f649365b37d70a843` |
| 对照（H1） | V4.2 重排，用 20M 确认 run-01 在两个 test 上写出的结果 | （V4.2 `0e83ea74…`，不重新运行） |
| 守门与 H2 | `vimeml-ja-v6-20m`（`colloq-c40`），用 20M 确认 run-01 在两个 test 上写出的结果 | （`19b04ddfe0f98041ffe50511cab9d64d2f2a20b2918f5e78cc78c0768981d15f`，不重新运行） |

200M 的设置与 20M 完全相同：分词器 `ja-unigram-12288`（`61d71460…`），候选索引 `b60c485d…`，按键位分桶的受限 beam 16，前 10，与 AzooKey N-best 20 合并后按 gen + λ·LM 排序，λ = 1.6，用 200M 自己的分数。FP32，TF32 关。

**对照用 run-01 的逐条结果**：V4.2 和 c40 在 test 上不再运行。脚本检查 20M 确认的消耗记录只有一次已完成的 run-01（发布 c40）、`summary.json` 的 SHA256 与记录一致，并且四个逐条结果文件（V4.2 与 c40，各两个 test）重新汇总后与 `summary.json` 的表格完全相同；这些文件的 SHA256 写入本次的 `preflight.json` 和 `summary.json`。

## 测试集

与 20M 确认相同，登记的 SHA256 也相同：

- **`ime-kana-ja-v1` test**：600 条（WRIME 400 + JMultiWOZ 200，59 个作者 / 对话簇），候选池 manifest `b11df321…`。
- **`ime-kana-ja-colloq-v1` test**：600 条口语片段，候选池 manifest `667e12ce…`。

两个 test 在 2026-11-28 的 20M 确认中第一次使用。这是它们的第二次使用，只用于 200M，只运行一次。

## 假设与判定

kana test 按作者 / 对话聚类做符号翻转检验；口语 test 按条目配对。双侧，α = 0.05。

**主假设 H1**：kana test，V4.2 重排 → 200M，宽松 Top-1，差值 > 0 且 p < 0.05。

**守门**（都要满足）：

1. kana test，`vimeml-ja-v6-20m` → 200M，宽松 Top-1，不出现差值 < 0 且 p < 0.05；
2. 口语 test，`vimeml-ja-v6-20m` → 200M，宽松 Top-1，同上。

**H2**（固定顺序，只在 H1 通过后检验，α = 0.05）：kana test，`vimeml-ja-v6-20m` → 200M，宽松 Top-1，差值 > 0 且 p < 0.05。它与守门 1 是同一组配对，只是方向不同。

H1 与 H2 按固定顺序检验，不另做多重比较校正。不使用 Standard blind。

**怎么读**：

- 开发集上，200M 对 V4.2 重排宽松 Top-1 +25 条（536 对 511，p 0.0002）。对 `vimeml-ja-v6-20m`：kana 宽松 536 对 529（+7），严格 432 对 440（−8）；口语宽松 538 对 533（+5），严格 451 对 451。
- test 上，20M 确认的结果是：kana 宽松 V4.2 510、c40 528；口语宽松 V4.2 509、c40 537。
- **H1 通过且守门满足**：200M 作为 `vimeml-ja-v6-200m` 用于桌面端。H2 不影响发布：H2 通过时，发布说明写明 200M 在 test 上的宽松 Top-1 高于 20M；不通过时，只报告数字，写明桌面端与手机端在 test 上没有显著差别。
- **否则**：200M 不以 `vimeml-ja-v6-200m` 发布，桌面端使用 `vimeml-ja-v6-20m`，v6 只发布 20M。两个 test 不为其他 200M checkpoint 重新打开。
- Standard 开发集重排 517，已满足发布门槛第 3 条（不低于 501），不在 test 上另做检验。

## 描述性报告（不作判定）

- 两个 test 上 V4.2、c40、200M 的宽松 / 严格 Top-1 与 Top-3，按长度档、有无左文、来源分开；仅生成的 Top-1；D5 条目；口语 rule band。
- 两个 test 上 200M 对 V4.2 的其余配对差（kana 严格，口语宽松与严格），200M 对 c40 的严格 Top-1。
- 生成回退次数。

## 执行

单 GPU，一个进程顺序完成，在 20M 确认所在的同一检出目录中运行（需要 `outputs/benchmarks/ja-gen-confirmation/run-01`）。

1. 提交本计划和执行代码。运行时，脚本检查它们已提交且未修改，所有登记文件的 SHA256 一致，然后按上文检查 run-01 的对照结果。
2. **预检**（只用开发集）：200M 重新生成 kana 和口语开发集，宽松和严格 Top-1 命中数与 2.2 判定结果（kana 536 / 432，口语 538 / 451）相差都不超过 3 条，且没有生成回退。不满足时 test 不打开。预检可以单独运行（`--preflight-only`）。
3. 写入消耗记录 `outputs/benchmarks/ja-v6-200m-confirmation/consumption.json`，然后 200M 运行两个 test，计算判定，写出 `summary.json`。
4. 只运行一次。技术故障在 `summary.json` 写出前中断运行时，可以按同一计划重跑，并记录在消耗记录中；不得为改变结果而重跑。

逐条结果只在内部传输，传完即删除中转副本。报告只公开汇总数字和题目 id，不公开 test 题目文本和候选。

## 局限

- 两个 test 已用于 20M 的确认；本计划在看到 20M 的 test 结果之后起草。主假设和守门的设计依据是开发集上 200M 与 20M 的差距，没有使用 test 上 200M 的任何结果。
- 预检比较的是命中数，不是逐条命中：2.2 的逐条结果不在本运行的检出目录中。
- 口语集的读音是自动读音，经 LLM 确认，没有人工审核。
