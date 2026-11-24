# 口语后训练候选 colloq-c40 test 确认计划（预注册）

> 已于 2026-11-23 执行（run-02），判定通过，结果见[确认报告](../reports/zh-colloq-confirmation-20261123/report.md)。

2026-11-23 起草。本计划和执行代码提交后才算注册；提交之前，`ime-pinyin-zh-colloq-v1-r1` 的 test 不打开。本文是参考文本，英文版见 [docs/zh/benchmarks/confirmation-zh-colloq-plan.md](../../../zh/benchmarks/confirmation-zh-colloq-plan.md)。

执行入口是 [confirm_zh_colloq.py](../../../../scripts/zh/benchmarks/confirm_zh_colloq.py)，所有固定项都写在 [zh_colloq_confirm.py](../../../../src/vimeml/benchmarks/zh_colloq_confirm.py) 中。搜索、重打分、上下文和 Standard 打分都与 [v2 确认计划](confirmation-zh-v2-plan.md) 相同。

## 要回答的问题

`colloq-c40` 是在 development 上选出来的（[口语后训练计划](../training/training-zh-colloq-plan.md)，见“偏离”一节）。本计划在 test 上回答：

- 它在口语整句拼音转写上，是否高于已确认的 `vimeml-zh-v2-20m`（`kd-cont`）？
- 它在公开拼音集、全拼和 Standard 重排上有没有显著退步？

通过后，`colloq-c40` 取代 `kd-cont` 成为 `vimeml-zh-v2-20m`，进入 Mac 对齐和 iOS 集成。

## 模型

| 角色 | 实验臂 | checkpoint SHA256 |
| --- | --- | --- |
| 候选 | `colloq-c40`（从 `kd-cont` 继续训练 30,000 步，口语 208/512，seed 20261122） | `0ec5ffb110fc3959534858fe1a931f4c177878daa8db33bea861c843ca1c1027` |
| 对照 | `kd-cont`（`vimeml-zh-v2-20m`，2026-11-20 确认通过） | `412aee1d1c2d73c9f2ff52533188055e4655575472fe3be136e048fd240268a4` |
| 描述 | `colloq-c20`（第一轮胜出臂） | `658e6c0e600eec400657a9f61546f48994596a42ddc585c33509d1672d6f0e29` |

判定用的 development 结果：`colloq-c40` 口语集自重打分运行，`rows.jsonl` SHA256 `70a1f5be08aa8fc38e52131ed6f0cb4b8d3f4aca51b14ade473560faf0d2036d`，Top-1 42.3%。

## 测试集

- **`ime-pinyin-zh-colloq-v1-r1` test**：775 句，manifest SHA256 `f94a76df…`。冻结后一直没有打开过，各轮训练都只读 development。上下文用冻结时生成的 `test-contexts.jsonl`（SHA256 `8275d961…`）。
- **`ime-pinyin-zh-v1-r1` test**：576 句，3,436 条输入。这是**第二次使用**：2026-11-20 的 v2-20m 确认（run-02）已经打开过它，`kd-cont` 在其中的结果对模型选择没有影响，选 `colloq-c40` 时也没有用到它。这里只作守门，复用 run-02 的上下文文件和 `kd-cont` 的 test 结果（`rows.jsonl` SHA256 `6f6a31d2…`）。
- **`ime-standard-zh-v1-r1` blind 全拼重排**：与 v2 确认相同，v1 一侧用 2026-11-11 确认运行的分数。

## 假设与判定

检验是按句子聚类的符号翻转检验，20,000 次置换，双侧，α = 0.05；Standard 用精确 McNemar。

**主假设 H1**：口语集 test 全部输入，`kd-cont` → `colloq-c40`（两边都用自己的分数重打分），精确 Top-1，差值 > 0 且 p < 0.05。

**守门**（都要满足）：

1. 公开拼音集 test 全部输入，`kd-cont` → `colloq-c40`，不出现差值 < 0 且 p < 0.05；
2. 只看全拼输入，同上；
3. Standard blind 全拼重排，v1 → `colloq-c40`，不出现净值 < 0 且精确 McNemar p < 0.05。

只有一个主假设，不做多重比较校正。

**怎么读**：

- development 上，`colloq-c40` 比 `kd-cont` 在口语集上高 2.7 个点（42.3 对 39.6），公开拼音集高 1.5（p 0.011），Standard 对 v1 净 0。
- **通过**：`colloq-c40` 成为 `vimeml-zh-v2-20m`，发布说明报告两个 test 上的 Top-1。
- **不通过**：`kd-cont` 继续作为 `vimeml-zh-v2-20m`。口语集 test 此后不再称为未使用的集合，也不为其他实验臂重新打开。

## 描述性报告（不作判定）

- 三个模型在口语集 test 上的 Top-1、Top-3、Top-10 和重打分 Top-1，按输入类型分开。
- `colloq-c20` 对 `kd-cont`、`colloq-c40` 对 `colloq-c20` 的口语集 test 配对差。
- `colloq-c40` 和 `kd-cont` 在公开拼音集 test 上的同样表格。

## 执行

在本地运行；test 文本只留在本地被忽略的目录中。

1. 提交本计划和执行代码。运行时，脚本检查它们已提交且未修改，所有登记文件的 SHA256 一致。
2. **预检**：重新生成 `colloq-c40` 的口语集 development，逐条输入和判定用的结果比较重打分 Top-1 命中。阈值与 v2 确认相同：翻转超过 0.5% 的输入，或 Top-1 变动超过 0.2 个点，test 不打开。预检可以单独运行（`--preflight-only`）。
3. 写入消耗记录 `outputs/benchmarks/zh-colloq-confirmation/consumption.json`，然后生成两个 test 和 Standard blind 分数，计算判定，写出 `summary.json`。
4. 只运行一次。技术故障在 `summary.json` 写出前中断运行时，可以按同一计划重跑，并记录在消耗记录中；不得为改变结果而重跑。

报告只公开汇总数字和题目 id，不公开 test 题目文本和候选。

## 局限

- 口语集的读音是三种工具一致的自动读音，没有人工审核；上下文只有同一条消息内光标前的部分。
- 公开拼音集 test 是第二次使用，守门结论不如首次使用那样独立。
- `colloq-c40` 是事后改选的（见口语后训练计划的“偏离”），本确认是对这个选择的独立检验。
