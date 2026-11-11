# vimeml-tiny-zh-v1 blind 确认评分计划（预注册）

> 已于 2026-11-11 执行（run-01），判定通过，结果见 [blind 确认与对照基线](../reports/zh-v1-20261111/blind-and-baselines.md)。

2026-11-11 起草。本计划和执行脚本提交后才算注册；提交之前，blind 不打开。基准版本为 `ime-standard-zh-v1-r1`。执行入口是 [confirm_zh_v1.py](../../../scripts/benchmarks/confirm_zh_v1.py)，所有固定项都写在 [zh_v1_confirm.py](../../../src/vimeml/benchmarks/zh_v1_confirm.py) 中。

## 要回答的问题

[八臂判定](../reports/zh-v1-20261110/ime-benchmark.md)只用了 development，结论是探索性的，[实验计划](../training-zh-v1-plan.md)要求部署前在中文 blind 上预注册检验。本计划回答一个问题：选定的模型在未打开的 blind 上，精确 Top-1 是否确实高于 Rime 第一候选？同时给出 blind 上的准确率估计。

## 模型

| 项目 | 内容 |
| --- | --- |
| 发布名称 | `vimeml-tiny-zh-v1` |
| 实验臂 | `12k-fresh-s1`（12K 词表，L9，新数据） |
| checkpoint | `artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/final.pt`，step 844,730，SHA256 `44c423a1e7cb0e32161afc94d6167cabb439a3d25fd9de6d1fbb150a927542d4` |
| tokenizer | `zh-unigram-12288/tokenizer.model`，SHA256 `68bd37c45bcb018b32b2eef5b241561ab5b6dc1b77c1c3aadcc1b41c4bf47e5a` |
| 对照 | 冻结候选池的原顺序，即 librime 1.17.0 + rime-ice 的第一候选 |

**选择依据**：新数据和 12K 来自实验计划的默认规则。同一配方的三个 seed 在 development 上的 Top-1 是 1,363、1,359、1,363，差别在噪声内，所以不按分数挑选。选 s1 是因为它就是预注册对比中代表 `12k-fresh` 的一臂。这个选择在打开 blind 之前作出，没有使用 blind。

只比较这一个模型。其余七臂不进 blind：它们和这个模型的差别在 development 上都未达到有效标准，blind 也不用于重新选型。

## 假设与判定

**主假设 H1**：全拼轨道，blind 全部 1,445 题，Rime 第一候选 → `vimeml-tiny-zh-v1`，精确 Top-1。

通过需要同时满足：

1. **H1 成立**：净值 > 0，且精确 McNemar 双侧 p < 0.05。
2. **各来源不退步**：CrossWOZ、RiSAWOZ、KdConv 三个来源的 Top-1 净值都 > 0。
3. **Top-3 不下降**：Top-3 没有出现 McNemar p < 0.05 的下降。

只有这一个主假设，不做多重校正。每段对话最多抽一题，对话簇就是单题，逐题检验和簇检验等价。

**怎么解读**：

- development 上的净值是 +362/1,456，这项检验几乎必然通过，不是一道难关。它的价值在于排除“只在 development 上有效”的情况，并提供一个未经挑选的准确率估计。
- 通过：可以说 `vimeml-tiny-zh-v1` 在 blind 上相对 Rime 的 Top-1 改进得到确认。报告 blind Top-1 及其 Wilson 95% 区间，这个数字用来代替 development 的 93.6%。
- 未通过：不发布这个 checkpoint，也不换一个臂再打开 blind。原因分析写入报告，blind 不能再称为未使用过的独立集合。
- 通过也不代表可以上线。设备端性能和客户端关键词屏蔽另行验证。

## 描述性报告（不做判定）

- 全拼：Top-3、Top-5、Top-10、池条件 Top-1、MRR；各来源 micro、三来源 macro；按目标长度分层；McNemar 与 bootstrap 区间。
- 简拼轨道（964 题）的同样指标。候选池只覆盖约四成答案，不并入全拼汇总。
- 去掉重叠筛查命中条目后的敏感性子集。
- development 与 blind 的差异只作描述。blind 的 Rime Top-1（72.3%）本来就高于 development（68.8%），两者的绝对分数不宜直接相减解读。

## 评分

沿用 development 的方法：`BOS + 左文 + 候选` 联合分词，FP32，全词表 logP 之和，无 EOS，TF32 关闭，分数并列时保持候选池原顺序。候选池里没有答案或候选池为空的题仍留在分母里。个别题若出现 roundtrip 或上下文窗口问题，这一题回退到候选池原顺序并计入 `fallbacks`（[基准说明](standard-ime-zh.md)已规定）。development 上这类问题为 0。

## 执行

1. 提交本计划和脚本。`confirm_zh_v1.py` 运行时检查：计划和代码已提交且没有改动；checkpoint、tokenizer、题目和候选池的 SHA256 与登记值一致；没有已完成的使用记录。
2. **预检**：先重打 development 的全拼和简拼，每一题的 rank 必须和 2026-11-11 判定时的结果完全相同，否则不打开 blind。
3. 在 `outputs/benchmarks/zh-v1-confirmation/consumption.json` 写入使用记录，然后对 blind 的全拼和简拼打分，校验 Rime 基线（全拼 1,045、简拼 147），生成 `summary.json`。
4. 只运行一次。如果因技术故障在生成 `summary.json` 之前中断，可以在同一计划下重跑，并记入使用记录；不为改变结果而重跑。
5. 首次运行后的任何修改都作为偏离报告，原输出保留。

报告只公开汇总数字和题目 id，不公开 blind 的题目文本和候选。

## 局限

- 标签经过 AI 专家审核和人工抽查，没有经过母语者正式标注。
- blind 和 development 的来源相同，都是对话数据，不代表真实键盘流量。
- 结果衡量的是“给定 Rime 候选池时的重排效果”，不是完整输入法的准确率。
- 重叠检查只查得出逐字重叠。[泄露检查](../reports/zh-v1-20261110/ime-benchmark.md#泄露检查)只覆盖了 development；blind 这边只有原有的逐字筛查结果（44 条连续汉字段命中、27 条整条消息命中）。
