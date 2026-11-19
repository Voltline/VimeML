# 公开中文键盘错字集 ime-typo-zh-v1

**是什么。** [键盘错字计划](../training-zh-typo-plan.md)中的纠错评测集，版本 `ime-typo-zh-v1-r1`，只有开发集。英文版见 [docs/benchmarks/typo-ime-zh.md](../../benchmarks/typo-ime-zh.md)。

**代码与文本。** 代码是 [typo_zh.py](../../../src/vimeml/benchmarks/typo_zh.py)，命令行入口 [scripts/benchmarks/typo_zh.py](../../../scripts/benchmarks/typo_zh.py)，测试在 [test_benchmark_typo_zh.py](../../../tests/test_benchmark_typo_zh.py)。评测集由 [ime-pinyin-zh-v1](pinyin-ime-zh.md) 的开发集文本和 Vime 的读音词典生成，所以和它们一样只放在忽略目录（`outputs/benchmarks/ime-typo-zh-v1/`），不提交。

## 条目

`ime-pinyin-zh-v1-r1` 开发集的全部 578 句，只用全拼输入，每句三个版本，共 1,734 条输入。

| 版本 | 输入 |
| --- | --- |
| `clean` | 句子的全拼 |
| `typo1` | 一处错字 |
| `typo2` | 两处错字，落在不同音节 |

错字规则与训练数据相同（[zh_typo.py](../../../src/vimeml/posttraining/zh_typo.py)，权重取自 [data-zh-pinyin-typo.toml](../../../configs/data-zh-pinyin-typo.toml)）：QWERTY 邻键替换 40、漏字母 25、多字母 20（一半是重复字母）、相邻交换 15。随机种子为 `ime-typo-zh-v1/<id>/<count>`，与训练数据无关。每条保留拼音集里的上下文。

## 冻结候选池

由 vimeml-tiny-zh-v1（`tiny-zh13-12k-fresh-s1-d384-l9/final.pt`）一次生成：Vime 的分桶搜索，lm 模式，beam 16，带输入框上下文。所有模型都在同一个池上打分。

- **base**：输入本身的前 10 个结果。
- **corrections**：把数据规则里的一处错字撤销、且结果仍是合法全拼的所有单步编辑（删一个字母、插入任意字母、交换相邻字母、换成 QWERTY 邻键），每个拼写取搜索第 1 名。与 `base[0]` 相同的文本和重复文本去掉。干净输入也同样生成 corrections，误显示率就是在它们上面测的。

抽 40 句估算：一处错字的输入平均约 65 个纠正拼写，正确拼写每次都在其中。每条输入约 140 秒，所以构建分片运行（`--shard I/N`，按句子序号取模），再由 `merge` 按拼音集顺序合并。

## 打分规则

每条输入对 `base[0]` 和所有纠正文本打分：

| 规则 | 分数 | 适用模型 |
| --- | --- | --- |
| `lm` | 上下文后的 LM 分（`zh13_lm_scores.score_item`，线上规则） | 全部 |
| `task` | `log P(output + EOS \| [2,4] + 上下文 + [5] + 实际按键 + [6])` | 三任务模型 |

上下文按每条输入裁剪一次，从左侧截断，保证最长的候选放得下。显示规则：`delta = 最佳纠正 − base[0]`，`delta ≥ margin` 时纠正结果显示在第 2 位，否则列表为 `base[0], base[1]`。纠正结果从不排第 1。

## 指标

- **错字输入**：@2（参考句在前两位）、纠正命中（显示的纠正结果就是参考句）和 base Top-1，分 `typo1`、`typo2` 和合计。
- **干净输入**：误显示率（出现了纠正结果）和 @2。
- **margin**：`lm` 规则报告线上 margin 3 的结果。所有规则另报校准 margin 下的结果：在干净输入误显示率 ≤ 10% 的前提下使错字 @2 最高的 margin，并列时取较大者。这个值写进导出的 `calibration.json`。`calibration` 命令用模型的 `lm` 打分结果生成这个文件：格式 `vimeml_zh_calibration_v1`，记录模型 checkpoint 的 sha256 和步数、搜索设置（束宽 16，取前 10）、重打分（λ 和先验，`self` 或 `v1`），以及纠错规则和校准 margin，并附上该 margin 和 margin 3 下的错字 @2 与干净误显示率（没有 margin 满足上限时 `enabled` 为 false）。
- **候选池覆盖**：参考句是 `base[0]`、在 `base[:2]` 中、在纠正结果中的比例。
- **检验**：错字 @2 配对比较，按句子聚类的 sign-flip，20,000 次置换，种子 1118。

主指标是 `typo1` 的 @2。单步编辑撤销不了两处错字，所以 `typo2` 主要反映第 2 个候选里是否已经有原句，只作描述。

## 命令

```
python -X utf8 scripts/benchmarks/typo_zh.py build --out OUT/shard-I --shard I/N
python -X utf8 scripts/benchmarks/typo_zh.py merge OUT/shard-0 ... OUT/shard-(N-1)
python -X utf8 scripts/benchmarks/typo_zh.py score --checkpoint CKPT --out DIR [--rules lm task]
python -X utf8 scripts/benchmarks/typo_zh.py report NAME=DIR:RULE ... [--base NAME/fixed|NAME/calibrated] --out FILE
python -X utf8 scripts/benchmarks/typo_zh.py calibration --rows DIR --lambda 1.6 --prior self --out calibration.json
```

`ime-pinyin-zh-v1` 的测试集从不读取。
