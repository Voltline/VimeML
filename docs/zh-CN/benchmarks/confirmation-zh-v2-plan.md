# vimeml-zh-v2 test 确认评分计划（预注册）

> v2-20m 已于 2026-11-20 执行（run-02），判定通过，结果见[确认报告](../reports/zh-v2-confirmation-20261120/report.md)。v2-200m 尚未执行。

2026-11-20 起草。本计划和执行代码提交后才算注册；提交之前，`ime-pinyin-zh-v1-r1` 的 test 不打开。本计划覆盖 `vimeml-zh-v2-20m` 和 `vimeml-zh-v2-200m` 两个模型，每个模型各运行一次，先 20M 后 200M。执行入口是 [confirm_zh_v2.py](../../../scripts/benchmarks/confirm_zh_v2.py)，所有固定项都写在 [zh_v2_confirm.py](../../../src/vimeml/benchmarks/zh_v2_confirm.py) 中。发布门槛的原则见 [v2 发布范围](../release-zh-v2-plan.md#发布门槛)，本计划给出具体的数字和检验。

## 要回答的问题

两个模型都是在 development 上选出来的：20M 经过四轮蒸馏判定（[蒸馏计划](../training-zh-kd-plan.md)），200M 经过[联合拼音预训练评测](../reports/zh-200m-pinyin-eval-20261116/report.md)；λ = 1.6、beam 16 也是在 development 和私有基准上定下来的。本计划在未打开的公开 test 上回答：

- **v2-20m**：手机上只装这一个模型，用自己的分数重打分。它的拼音生成 Top-1 是否高于当前线上逻辑（Rime 16 句 + v1 重排，带框内上下文）？全拼和 Standard 重排有没有显著退步？
- **v2-200m**：桌面模型，用自己的分数重打分。它的拼音生成 Top-1 是否高于 v2-20m？

同时给出两个模型在 test 上的准确率估计。

## 模型

| 项目 | v2-20m | v2-200m |
| --- | --- | --- |
| 实验臂 | `kd-cont`（从 r2-cont 出发，200M 老师蒸馏 30,000 步，α 0.5，T 1，r1 数据） | `zh200m-12k-pinyin-pt`（12K 联合拼音预训练） |
| checkpoint | `artifacts/models/tiny-zh13-12k-pinyin-kd-cont/final.pt`，step 30,000，SHA256 `412aee1d1c2d73c9f2ff52533188055e4655575472fe3be136e048fd240268a4` | `artifacts/models/tiny-zh200m-12k-pinyin-pt-d1024/final.pt`，SHA256 `90e48fe4cbaeacf5c58774ded243e57d94c86923a51d191555fbbfc7954150f8` |
| 判定用的 development 结果 | 自重打分运行，`rows.jsonl` SHA256 `0003a012b4b750bccc26c8013ada2aec089b15b8a6dacf0d0f92256b2f9b6a46`，Top-1 54.82% | 自重打分运行，`rows.jsonl` SHA256 `9e1be73704483e0b298c2b1a18c03de24ea88de1bdb7b32e48b3e0e32b7a397c`，Top-1 58.77% |
| H1 的对照 | 线上逻辑：Rime 16 句 + v1 重排，带框内上下文 | v2-20m 在 test 上的结果（本计划的第一次运行） |

两个模型共用：

- tokenizer `zh-unigram-12288/tokenizer.model`，SHA256 `68bd37c45bcb018b32b2eef5b241561ab5b6dc1b77c1c3aadcc1b41c4bf47e5a`；
- 读音词典 `vime_pinyin.dict.yaml`，SHA256 `3a1654cd5ce20ecacd88046186c34ad87e748a494f7078f23c1ad15492eb0c8a`；
- 冻结集 `ime-pinyin-zh-v1-r1` 的 manifest，SHA256 `19133b0afaed162034277fe3bbe72b195b46a101462c65697a1a1ec64589248f`。

**选择依据**：

- `kd-cont` 在 2026-11-17 的[蒸馏评测](../reports/zh-kd-eval-20261117/report.md)中成为候选：公开 development 自重打分 54.8%，比 v1 单独（LM 模式）高 2.1（p 0.041）。之后第 3、4 轮（[蒸馏计划](../training-zh-kd-plan.md)）没有一臂替换它。
- 200M 的选择在 2026-11-16 定下：联合预训练，自己的分数重打分。
- 两个选择都在打开 test 之前作出，没有使用 test。其余实验臂不进 test，test 也不用于重新选型。

## 假设与判定

所有比较都在公开 test 的全部输入上进行：576 句，3,436 条输入（全拼 576、混合简拼 2,284、全首字母 576）。检验是按句子聚类的符号翻转检验，20,000 次置换，双侧，α = 0.05。

### v2-20m

**主假设 H1**：全部输入，线上逻辑 → `vimeml-zh-v2-20m`（自重打分），精确 Top-1。

通过需要同时满足：

1. **H1 成立**：差值 > 0 且 p < 0.05。
2. **全拼不显著退步**：只看全拼输入，同一检验下不出现差值 < 0 且 p < 0.05。
3. **Standard 重排守卫**：`ime-standard-zh-v1-r1` blind 全拼重排，v1 → v2-20m，不出现净值 < 0 且精确 McNemar p < 0.05。v1 一侧用 2026-11-11 确认运行的分数（run-01），不重新计算。

只有一个主假设，不做多重比较校正。条件 2、3 是守卫，不是额外的假设。手机上只装一个模型，Standard 重排退步时没有 v1 可以退回，所以条件 3 是硬性条件。

### v2-200m

**主假设 H1**：全部输入，v2-20m 的 test 结果 → `vimeml-zh-v2-200m`（自重打分），精确 Top-1。通过条件：差值 > 0 且 p < 0.05。

**Standard 重排**：和 v2-20m 的条件 3 用同一个检验，但不决定是否通过。按[发布门槛](../release-zh-v2-plan.md#发布门槛)第 2 条，桌面端可以同时装 v1：如果 200M 显著退步（净值 < 0 且 p < 0.05），桌面端的重排继续用 v1，发布说明写明这一点。

200M 只在 v2-20m 的运行完成之后执行。它不设全拼守卫：对照是 v2-20m，而全拼守卫保护的是线上逻辑。

### 怎么读

- development 上（按本计划的代码路径计算），v2-20m 比线上逻辑高 22.5 个点（54.8 对 32.4，+869/−95，p 5e-5），全拼 85.8 对 81.8（+40/−17，p 0.003）；Standard 重排对 v1 是 −4 题（p 0.63）。所以 v2-20m 的 H1 和全拼守卫几乎一定通过，Standard 守卫是三项里唯一有实际风险的一项。
- development 上，200M 比 v2-20m 高 3.9 个点（+292/−156，p 1e-4），Standard 重排对 v1 是 +23 题（p 0.0002）。H1 大概率通过，重排大概率不需要退回 v1。
- **通过**：模型在 test 上确认，发布说明报告 test Top-1 和置信区间，替换 development 上的数字。
- **不通过**：这个 checkpoint 不以该名称发布，test 不为其他实验臂重新打开。分析写入报告，此后 test 不再称为未使用的独立集。v2-20m 不通过时，手机端继续使用 v1；200M 不通过时，计划另行讨论。
- 通过不等于可以发布。iPhone 延迟（v2-20m）、客户端关键词屏蔽等发布门槛另行验证。

## 描述性报告（不作判定）

- 每个模型的 Top-1、Top-3、Top-10 覆盖率，按输入类型（全拼、混合简拼、全首字母）分开，重打分开和关，以及生成顺序。
- 按句长分层（5–7、8–12、13–20 字）和按来源（CrossWOZ、RiSAWOZ、KdConv、CPED）分层。
- 对照一侧的同样表格。
- development 和 test 之间的差别只作描述，不做检验。
- v1 单独（LM 模式）不在 test 上运行。v2-20m 对 v1 单独的比较只有 development 上的 +2.1（p 0.041）。

## 评分设置

- 搜索：[zh_search.py](../../../src/vimeml/deployment/zh_search.py) 的分桶搜索，每个键位 beam 16，task 模式，带框内上下文。
- 重打分：前 10 个候选按 genScore + λ × prior 排序，λ = 1.6，prior 是模型自己的 LM 分数。
- 框内上下文：test 的上下文由 `contexts_pinyin_ime_zh.py` 在确认运行中现场重建，规则和 development 相同（Vime 的 `sentenceContext`：最后 64 个字符，截在最后一个句末标点或换行之后）。
- 线上逻辑：Rime 16 句加 v1 重排（Vime #70 的逻辑），用同样的上下文，由本地的 Rime 测试工具生成。这个工具不在本仓库中；它读取确认运行写出的 test 题目和上下文，写出每条输入的候选顺序。
- FP32，TF32 关闭。Standard 守卫的打分和 v1 确认计划相同：`BOS + 左侧上下文 + 候选` 联合切分，全词表对数概率求和，不加 EOS，并列时保持候选池原顺序。

## 执行

在本地运行；test 文本只留在本地被忽略的目录中。

1. 提交本计划，并在 `zh_v2_confirm.MODELS` 中登记两个模型的 checkpoint、判定用的 development 结果及其 SHA256。运行时，脚本检查本计划和相关代码已经提交且未修改，所有登记文件的 SHA256 一致。
2. **预检**：用登记的设置重新生成 development，逐条输入和判定用的结果比较重打分 Top-1 命中。翻转超过 0.5% 的输入，或 Top-1 变动超过 0.2 个点，test 不打开。预检可以单独运行（`--preflight-only`），不打开 test。
3. 写入消耗记录 `outputs/benchmarks/zh-v2-confirmation/consumption.json`，然后重建 test 上下文，生成 test。
4. v2-20m：本地 Rime 测试工具生成线上逻辑在 test 上的结果后，运行 `finish`，计算 H1、全拼守卫和 Standard 守卫，写出 `summary.json`。v2-200m 直接以 v2-20m 已完成的 test 结果为对照，计算 H1 和 Standard 重排。
5. 每个模型只运行一次。如果技术故障在 `summary.json` 写出前中断了运行，可以按同一计划重跑，并记录在消耗记录中；不得为改变结果而重跑。已完成的模型不会再次运行。
6. 首次运行之后的任何改动都作为偏离报告，原始输出保留。

报告只公开汇总数字和题目 id，不公开 test 题目文本和候选。

## 局限

- test 的读音是三种工具一致的自动读音，没有人工审核；键位是合成的，不是真实打字。
- 来源都是对话语料，和真实键盘输入有差别。
- Standard blind 已在 2026-11-11 用于 v1 确认，在这里是固定的参考测试集，不是新的 blind。
- 私有拼音基准用于定 λ，只作描述，不进入判定。
