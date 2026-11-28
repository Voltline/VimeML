# 日文生成路线 20M 候选 colloq-c40 test 确认计划（预注册）

[English version](../../../ja/benchmarks/confirmation-ja-gen-plan.md)

2026-11-28 起草。本计划和执行代码提交后才算注册；提交之前，`ime-kana-ja-v1` 和 `ime-kana-ja-colloq-v1` 的 test 都不打开。对应 [ja-gen 计划](../training/training-ja-gen-plan.md) 的第 4 阶段。

执行入口是 [confirm_ja_gen.py](../../../../scripts/ja/benchmarks/confirm_ja_gen.py)，所有固定项都写在 [ja_gen_confirm.py](../../../../src/vimeml/benchmarks/ja_gen_confirm.py) 中。搜索、重排和评分与 2.3 / 3.x 的开发集评测相同（[evaluate_ja_gen_stage2.py](../../../../scripts/ja/benchmarks/evaluate_ja_gen_stage2.py) 使用的 `kana_gen_ja.evaluate_arm`）。

## 要回答的问题

`colloq-c40` 是第 3 阶段在开发集上选出来的（[3.4 报告](../reports/ja-gen-stage3-seeds-20261128/report.md)）。本计划在 test 上回答：

- 它的 kana 转换宽松 Top-1 是否高于现在部署的 V4.2 方式（AzooKey N-best 20 + V4.2 重排）？
- 它的严格 Top-1 是否高于 `kd-cont`（3.4 的主指标在 test 上是否复现）？
- 它在 kana 和口语两个 test 上的宽松 Top-1 是否显著低于 `kd-cont`？

通过后，`colloq-c40` 作为 `vimeml-ja-v2-20m` 进入 Mac 对齐和 iOS 集成，替换 V4.2。

## 模型

| 角色 | 模型 | checkpoint SHA256 |
| --- | --- | --- |
| 候选 | `colloq-c40`（s1，从 `kd-cont` 继续 30,000 步，口语 208/512，seed 20261205） | `19b04ddfe0f98041ffe50511cab9d64d2f2a20b2918f5e78cc78c0768981d15f` |
| 对照（H1） | V4.2 第 20 轮 `best-lm.pt`，重排 AzooKey N-best 20（上下文求和分数） | `0e83ea7417b9816b37183044468343b151f4e610da0764c31f1f780c5fa5796d` |
| 对照（H2 与守门） | `kd-cont`（2.3 蒸馏 20M） | `5a44839657569b34dea57fae4a45a2050506f892a441e38715756727fe46a92f` |

两个 20M 的设置：分词器 `ja-unigram-12288`（`61d71460…`），候选索引 `b60c485d…`，按键位分桶的受限 beam 16，每个模型取前 10，与 AzooKey N-best 20 合并后按 gen + λ·LM 排序，λ = 1.6（2.3 冻结值）。V4.2 的设置与第 0 阶段相同（FP32，TF32 关，同分保持 AzooKey 顺序）。

## 测试集

- **`ime-kana-ja-v1` test**：600 条（WRIME 400 + JMultiWOZ 200，59 位作者 / 用户），manifest SHA256 `8d51fc00…`。AzooKey 候选在 2026-11-25 与开发集一起导出（候选池 manifest `b11df321…`），之后没有被任何模型打分。
- **`ime-kana-ja-colloq-v1` test**：600 条口语片段（rule band zero 142 / low 136 / mid 148 / high 174），manifest SHA256 `6538b128…`。AzooKey 候选在 2026-11-28 按开发集同样的设置导出（转换器 `d59a28e4c7ca`，N-best 20，typo 关，Zenzai 关），候选池 SHA256 登记在执行代码中。导出日志没有导入，也没有被读取。

## 假设与判定

kana test 的检验按作者 / 对话聚类做符号翻转检验；口语 test 每篇文档至多 1 条，按条目配对。双侧，α = 0.05。

**主假设 H1**：kana test，V4.2 重排 → `colloq-c40`，宽松 Top-1，差值 > 0 且 p < 0.05。

**H2**（固定顺序，只在 H1 通过后检验，同样 α = 0.05）：kana test，`kd-cont` → `colloq-c40`，严格 Top-1，差值 > 0 且 p < 0.05。

**守门**（都要满足）：

1. kana test，`kd-cont` → `colloq-c40`，宽松 Top-1，不出现差值 < 0 且 p < 0.05；
2. 口语 test，`kd-cont` → `colloq-c40`，宽松 Top-1，同上。

H1 与 H2 按固定顺序检验，不另做多重比较校正。不使用 Standard blind。

**怎么读**：

- 开发集上，`colloq-c40` 对 V4.2 宽松 Top-1 +18 条（529 对 511，p 0.013），对 `kd-cont` 严格 Top-1 +12 条（440 对 428，p 0.032）；对 `kd-cont` 的宽松 Top-1 在 kana 和口语开发集上都是 −2（p 0.77 / 0.73）。
- **H1 通过且守门满足**：`colloq-c40` 作为 `vimeml-ja-v2-20m` 发布，替换 V4.2。H2 不影响发布：H2 通过时，发布说明写明口语后训练在 test 上提高了严格 Top-1；H2 不通过时，只报告数字，不作这一说法。
- **H1 通过、守门不满足**：不发布 `colloq-c40`。若 `kd-cont` 在 kana test 上对 V4.2 的宽松 Top-1 也是差值 > 0 且 p < 0.05，则 `kd-cont` 作为 `vimeml-ja-v2-20m` 发布；否则保留 V4.2。
- **H1 不通过**：保留 V4.2 部署。两个 test 此后不再称为未使用的集合，也不为其他模型重新打开做确认。

## 描述性报告（不作判定）

- 两个 test 上 AzooKey 原始顺序、V4.2、`kd-cont`、`colloq-c40` 的宽松 / 严格 Top-1 与 Top-3，按长度档、有无左文、来源和口语 rule band 分开；仅生成的 Top-1；D5 条目。
- 两个 test 上 V4.2 对 AzooKey、各 20M 对 V4.2 的其余配对差，口语 test 上 `colloq-c40` 对 `kd-cont` 的严格 Top-1。
- 生成回退次数。

## 执行

单 GPU，一个进程顺序完成。

1. 提交本计划和执行代码。运行时，脚本检查它们已提交且未修改，所有登记文件的 SHA256 一致。
2. **预检**（只用开发集）：V4.2 重新重排 `ime-kana-ja-v1` 开发集，与第 0 阶段的顺序比较，相同的条目不少于 99%；`colloq-c40` 重新生成 kana 和口语开发集，逐条与 3.3 判定用的结果比较宽松和严格 Top-1 命中，翻转不超过 1%（6 条）且命中数变化不超过 0.5%（3 条）。不满足时 test 不打开。预检可以单独运行（`--preflight-only`）。
3. 写入消耗记录 `outputs/benchmarks/ja-gen-confirmation/consumption.json`，然后对两个 test 做 V4.2 重排、运行两个 20M，计算判定，写出 `summary.json`。
4. 只运行一次。技术故障在 `summary.json` 写出前中断运行时，可以按同一计划重跑，并记录在消耗记录中；不得为改变结果而重跑。

逐条结果只在内部传输，传完即删除中转副本。报告只公开汇总数字和题目 id，不公开 test 题目文本和候选。

## 局限

- `colloq-c40` 是在 kana 和口语开发集上多次比较后选出的；本确认是对这个选择的独立检验。
- 口语集的读音是自动读音，经 LLM 确认，没有人工审核。
- V4.2 只重排 AzooKey 候选，不生成；H1 比较的是两种完整部署方式，而不是两个模型在同一候选上的打分。
