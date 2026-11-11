# vimeml-tiny-zh-v1 对照基线计划（预注册）

2026-11-11 起草。基准版本为 `ime-standard-zh-v1-r1`。本计划和执行脚本提交后才在 blind 上运行。执行入口是 [compare_zh_baselines.py](../../../scripts/benchmarks/compare_zh_baselines.py)，实现在 [zh_baselines.py](../../../src/vimeml/benchmarks/zh_baselines.py)，Rime 配置在 [rime_candidates.py](../../../scripts/benchmarks/rime_candidates.py) 的 `VARIANTS` 中。

## blind 的状态

blind 已在[确认评分](confirmation-zh-v1-plan.md)中使用过一次（run-01，通过）。从那以后它只作为固定参考测试集，不再是未使用的独立集合。本计划在 blind 上的结果只用于描述 v1 和其他系统的相对位置，**不用于**选择、调参或改动 v1，也不改变确认评分的结论。报告只公开汇总数字。

## 要回答的问题

给 v1 的成绩找参照：

1. 更强的 Rime 配置（加词库和语法模型）能把 Rime 自身的第一候选提高多少，v1 是否仍然领先？
2. Rime 默认方案（朙月拼音）的水平如何？
3. 0.5B 级别的通用中文 LM 用同样的方法重排同一候选池，能做到多少？v1（约 20M 参数）和它们差多少？

## 对照系统

| 名称 | 类型 | 内容 |
| --- | --- | --- |
| `rime-ice` | 引擎 | librime 1.17.0 + rime-ice `99662ba`，配置同冻结候选池（关用户词典、去掉 Lua 翻译器、关 emoji 和繁体） |
| `rime-ice-gram` | 引擎 | 上述配置 + 万象语法模型 `wanxiang-lts-zh-hans.gram`（418,071,596 字节，SHA256 `3457b69f5fe0fc4f5d211c878bbfa2e2361b90639952c9845c8e31e9d047cde6`），参数照抄 rime-ice 的 `others/recipes/grammar.recipe.yaml`：`collocation_max_length` 6、`collocation_min_length` 3、`collocation_penalty` −14、`non_collocation_penalty` −6、`weak_collocation_penalty` −100、`rear_penalty` −20、`contextual_suggestions` false、`max_homophones` 8 |
| `rime-luna` | 引擎 | librime 1.17.0 + `luna_pinyin_simp`（rime-luna-pinyin `56b934b`、rime-prelude `082425e`、rime-essay `054920d`、rime-stroke `1e8fff9`，librime 自带 OpenCC 数据），只关用户词典 |
| `qwen2.5-0.5b` | 外部 LM | `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| `qwen3-0.6b-base` | 外部 LM | `Qwen/Qwen3-0.6B-Base` revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| `vimeml-tiny-zh-v1` | 被比较模型 | development 用 2026-11-11 判定的 `12k-fresh-s1` 结果，blind 用确认评分 run-01 的结果，不重新打分 |

Qwen 选 base 版本，不用指令微调版，因为任务是给续写打分。transformers 4.57.6。

## 怎么打分

**引擎**：每个配置独立运行，取它对整串输入的候选里全是汉字、且字数等于目标长度的前 20 个，按引擎顺序排序，不用 lattice 补足，也不看左文（Rime 本来就没有左文）。这和冻结候选池的构成规则相同，只是去掉了补足步骤。因此 `rime-ice` 在全拼上和候选池原顺序完全一致（development 1,001）。简拼上有 3 题 Rime 没给出候选、答案只在 lattice 补足里，引擎单独评分时为 139，候选池原顺序为 142，判定和确认评分里的 Rime 基线仍指候选池原顺序。

引擎排序可以用到候选池以外的候选，所以引擎之间、引擎和 v1 之间比较的是“整个系统的第一候选”，而不是同一个池子里的重排。

**外部 LM**：在冻结候选池上重排，方法和 v1 相同：`<|endoftext|>` 作为起始 token，`起始 + 左文 + 候选` 联合分词，从任一候选和单独左文首次不同的 token 开始，FP32 全词表 logP 相加，无 EOS，TF32 关闭，分数并列时保持池顺序。要求 roundtrip 成立。不成立或出错的题回退到池顺序并计入 `fallbacks`。空池题留在分母里。这样 LM 和 v1 之间的差别只来自语言模型本身。

## 比较与判定

主要比较是全拼轨道的精确 Top-1：v1 对五个对照系统逐一配对，各做精确 McNemar 双侧检验，五个 p 值做 Holm 校正，同时报告 bootstrap 净值区间。简拼轨道、Top-3、各来源、按长度分层、池条件 Top-1、MRR 只作描述。

development 是开放集，可以随时重跑和查看。blind 每个对照系统只运行一次。如因技术故障中断，可以在同一计划下重跑；不为改变结果重跑。结果不论方向都照实报告：如果某个对照系统胜过 v1，就写明这一点，不重新挑选对照或改动打分方法。

## 执行

1. 部署三个 Rime 配置：`rime_candidates.py deploy --variant {ice,ice-gram,luna}`。
2. development：`compare_zh_baselines.py engine|lm <系统> OUT --track {full,abbrev}`，再用 `score_standard_ime_zh.py compare` 和 v1 配对。
3. 提交本计划和脚本。
4. blind：同样的命令加 `--split blind --blind-plan docs/zh-CN/benchmarks/baselines-zh-v1-plan.md`。评分脚本会检查计划已提交、没有改动、写明 `ime-standard-zh-v1-r1`。和确认评分 run-01 的 v1 结果配对。
5. 输出写在 `outputs/benchmarks/zh-v1-baselines/{development,blind}/`。

## 局限

- 引擎只看拼音，不看左文；v1 和外部 LM 看左文。这是系统差异本身，不是要消除的偏差。[泄露检查](../reports/zh-v1-20261110/ime-benchmark.md#泄露检查)里的空左文对照（development 81.9%）可以作为“不看左文的 v1”参考。
- 外部 LM 只在冻结候选池上重排，池的上限（全拼约 97.5%、简拼约 41%）同样限制它们。
- 我们不知道 Qwen 的训练数据是否包含 CrossWOZ、RiSAWOZ、KdConv 这些公开数据集，无法排除泄露。如果 Qwen 分数偏高，需要按这个可能解读。
- 万象语法模型的参数直接取 rime-ice 推荐值，没有调参。
- 只比较速度以外的准确率，0.5B 模型能否在设备端运行不在本计划内。
