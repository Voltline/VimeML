# vimeml-tiny-zh-v1 blind 确认与对照基线（2026-11-11）

本文报告两件事：一是[预注册确认评分](../../benchmarks/confirmation-zh-v1-plan.md)在 blind 上的结果，二是按[对照基线计划](../../benchmarks/baselines-zh-v1-plan.md)和更强的 Rime 配置、通用中文 LM 的比较。基准版本为 `ime-standard-zh-v1-r1`。八臂判定和泄露检查见 [IME 打分与判定](../zh-v1-20261110/ime-benchmark.md)。

blind 只公开汇总数字，不公开题目文本和候选。

## 结论

- **确认评分通过**：blind 全拼精确 Top-1 为 93.8%（1,355/1,445，Wilson 95% [92.4%, 94.9%]），Rime 第一候选为 72.3%。三个来源都是正净值，Top-3 没有下降。
- **v1 胜过所有对照系统**：两个 Qwen（0.5B / 0.6B）、三种 Rime 配置。在 development 和 blind 上，全拼精确 Top-1 的五项配对比较经 Holm 校正后都显著。
- **最接近的是 Qwen2.5-0.5B**：blind 91.8%，v1 净 +28（p = 0.005）。它的参数量约是 v1 的 25 倍。在 blind 的 CrossWOZ 上它比 v1 多对 2 题，是唯一一处 v1 没有领先的来源。
- **最强的 Rime 配置**是雾凇加万象语法模型：82.2%，比默认雾凇高约 10 个百分点，比 v1 仍低 11.6 个百分点。朙月拼音（Rime 默认方案）为 68.9%。
- blind 在确认评分中已被使用，对照比较是在它作为参考测试集的情况下进行的，结果不用于选择或改动 v1。

## blind 确认评分

| 项目 | 结果 |
| --- | --- |
| 运行 | run-01，2026-11-11 15:38，计划 commit `d457a99`，只运行一次 |
| 预检 | development 全拼和简拼每题 rank 与判定时完全相同 |
| 全拼 Top-1 | v1 1,355/1,445 = 93.8%，Rime 1,045 = 72.3% |
| H1（McNemar） | +329/−19，净 +310，p = 3.6e-74，簇 bootstrap 净值 [278, 343] |
| 各来源 | CrossWOZ 94.4% / 63.5%，RiSAWOZ 94.2% / 76.8%，KdConv 92.8% / 76.6%（v1 / Rime） |
| Top-3 | 97.2% / 89.9%，+107/−1 |
| 判定 | **通过**（三项条件全部满足） |

描述性指标（不参与判定）：

| 指标 | v1 | Rime |
| --- | ---: | ---: |
| 2–3 字 / 4–6 字 / 7–12 字 Top-1 | 97.8% / 94.6% / 89.0% | 82.8% / 75.8% / 58.6% |
| 池条件 Top-1 | 96.3% | 74.3% |
| MRR | 0.954 | 0.818 |
| 去掉重叠条目（1,381 题） | 93.7% | 71.8% |
| 简拼 Top-1（964 题） | 35.0% | 15.3%（+205/−15） |

回退 0 题，全拼 4 题候选池为空（留在分母里）。development 上 v1 为 93.6%，blind 为 93.8%，没有出现 development 偏高的迹象。

使用记录在 `outputs/benchmarks/zh-v1-confirmation/consumption.json`，summary SHA256 `8554fe722fa2c0fda3dc8f0f568c228009e4487229cd873a1afcad71bf013936`。

## 对照基线

### 系统

| 系统 | 怎么排序 |
| --- | --- |
| 雾凇 | librime 1.17.0 + rime-ice `99662ba`，配置同冻结候选池 |
| 雾凇 + 万象 | 上述配置 + `wanxiang-lts-zh-hans.gram` 语法模型，参数取 rime-ice 推荐值 |
| 朙月 | librime 1.17.0 + `luna_pinyin_simp`，只关用户词典 |
| Qwen2.5-0.5B | 冻结候选池上重排，打分方法与 v1 相同 |
| Qwen3-0.6B-Base | 同上 |

Rime 各配置独立排序、不看左文，取整串输入的同长度汉字候选，不用 lattice 补足。所以雾凇单独评分时略低于候选池原顺序（blind 全拼 1,043 对 1,045，简拼 141 对 147）。Qwen 和 v1 在同一候选池、用同一方法打分，差别只来自模型本身。版本和全部设置见[计划](../../benchmarks/baselines-zh-v1-plan.md)。

### 全拼精确 Top-1

| 系统 | 参数 | development | blind | blind Top-3 | blind MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| **vimeml-tiny-zh-v1** | 约 20M | **93.6%** | **93.8%** | **97.2%** | **0.954** |
| Qwen2.5-0.5B | 494M | 91.3% | 91.8% | 96.1% | 0.940 |
| Qwen3-0.6B-Base | 596M | 89.8% | 89.2% | 95.4% | 0.924 |
| 雾凇 + 万象 | — | 79.0% | 82.2% | 83.7% | 0.829 |
| 雾凇 | — | 68.8% | 72.2% | 73.6% | 0.729 |
| 朙月 | — | 68.3% | 68.9% | 70.5% | 0.696 |

### v1 对每个系统的配对比较（全拼 Top-1）

| 对照 | split | v1 多对 / 多错 | 净值 | 簇 bootstrap 95% | Holm p |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | development | +63 / −29 | +34 | [16, 53] | 5.1e-4 |
| | blind | +61 / −33 | +28 | [9, 48] | 5.1e-3 |
| Qwen3-0.6B-Base | development | +81 / −25 | +56 | [36, 77] | 9.0e-8 |
| | blind | +96 / −30 | +66 | [44, 89] | 6.1e-9 |
| 雾凇 + 万象 | development | +242 / −29 | +213 | [184, 244] | 1.5e-42 |
| | blind | +202 / −35 | +167 | [138, 196] | 3.0e-29 |
| 雾凇 | development | +377 / −15 | +362 | [328, 397] | 3.8e-91 |
| | blind | +331 / −19 | +312 | [280, 345] | 4.0e-74 |
| 朙月 | development | +380 / −12 | +368 | [334, 402] | 2.4e-95 |
| | blind | +376 / −16 | +360 | [326, 395] | 1.1e-89 |

Holm 校正在每个 split 内对五个对照进行。Top-3 上 v1 对每个系统都是正净值，且都显著。

### blind 各来源与长度（全拼 Top-1）

| 系统 | CrossWOZ | RiSAWOZ | KdConv | 2–3 字 | 4–6 字 | 7–12 字 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 94.4% | 94.2% | 92.8% | 97.8% | 94.6% | 89.0% |
| Qwen2.5-0.5B | 94.8% | 90.7% | 90.1% | 92.2% | 92.5% | 90.9% |
| Qwen3-0.6B-Base | 91.6% | 87.2% | 88.8% | 89.1% | 90.5% | 87.9% |
| 雾凇 + 万象 | 79.7% | 84.7% | 82.2% | 86.1% | 86.9% | 73.6% |
| 雾凇 | 63.5% | 76.6% | 76.4% | 82.6% | 75.8% | 58.4% |
| 朙月 | 60.8% | 75.4% | 70.4% | 84.3% | 71.2% | 51.6% |

- v1 对 Qwen2.5 的优势主要在短词：2–3 字高 5.6 个百分点。7–12 字 Qwen2.5 高 1.9 个百分点，CrossWOZ 上 Qwen2.5 净多 2 题。development 上 7–12 字 v1 高 1.5 个百分点，这一层的差别在两个 split 上方向不一致，可能只是噪声，但在长句上 v1 至少没有明显优势。
- 万象语法模型主要改善长句：雾凇的 7–12 字从 58.4% 升到 73.6%，但仍远低于 v1 和 Qwen。

### 简拼精确 Top-1

| 系统 | development | blind |
| --- | ---: | ---: |
| v1 | 34.0% | 35.0% |
| Qwen2.5-0.5B | 30.9% | 31.0% |
| Qwen3-0.6B-Base | 29.4% | 30.6% |
| 雾凇 + 万象 | 14.7% | 15.7% |
| 雾凇 | 14.0% | 14.6% |
| 朙月 | 12.1% | 12.9% |

简拼的候选池只覆盖约四成答案，LM 重排的上限约 41%。万象语法模型对简拼几乎没有帮助（+1 个百分点）。

## 怎么解读

- **和 Qwen 的比较**是对 v1 模型本身最有信息量的参照：同一候选池、同一打分方法，v1 以约 1/25 的参数量略胜 Qwen2.5-0.5B。v1 的训练数据包含聊天语料，任务分布更接近；Qwen 是通用模型，也没有针对输入法调整。所以这不说明 v1 在一般中文建模上强于 Qwen，只说明在这个任务上，20M 的专用模型已经达到 0.5B 通用模型的水平。
- 我们无法确定 Qwen 是否训练过 CrossWOZ、RiSAWOZ、KdConv。如果训练过，它的分数会偏高，v1 的相对优势只会被低估。
- **和 Rime 的比较**是系统间比较：Rime 不看左文，v1 看左文。[泄露检查](../zh-v1-20261110/ime-benchmark.md#泄露检查)里 v1 在空左文时 development 为 81.9%，和雾凇 + 万象（79.0%）接近。v1 对 Rime 的大部分优势来自对左文的利用。

## 局限

- 确认评分的局限见[计划](../../benchmarks/confirmation-zh-v1-plan.md#局限)：标签不是母语者正式标注，数据来自对话而非真实键盘流量，衡量的是候选池重排而非完整输入法。
- Qwen 只在冻结候选池上重排，没有测它能否在设备端运行。万象语法模型参数没有调。
- 对照比较时 blind 已经被使用过一次，结果只作参考。

## 复现

```bash
# 部署三种 Rime 配置（一次）
python -I -X utf8 scripts/benchmarks/rime_candidates.py deploy --variant {ice,ice-gram,luna}
# 每个系统、每条轨道一次；blind 加 --split blind --blind-plan docs/zh-CN/benchmarks/baselines-zh-v1-plan.md
python -I -X utf8 scripts/benchmarks/compare_zh_baselines.py engine {ice,ice-gram,luna} OUT --track {full,abbrev}
python -I -X utf8 scripts/benchmarks/compare_zh_baselines.py lm {qwen2.5-0.5b,qwen3-0.6b-base} OUT --track {full,abbrev}
# 和 v1 配对：development 用判定时的 12k-fresh-s1，blind 用确认评分 run-01 的 v1-{full,abbrev}
python -I -X utf8 scripts/benchmarks/score_standard_ime_zh.py compare OUT/result V1 CMP.json
```

输出在 `outputs/benchmarks/zh-v1-baselines/` 和 `outputs/benchmarks/zh-v1-confirmation/`，不进 Git。
