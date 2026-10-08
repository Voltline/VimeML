# V2.1 restart1 最终评测

2026-10-07，AutoDL RTX4090 D。18:54—19:56完成两轮、98,426 updates。选择**best.pt / step98426**作为追加训练起点，第一轮checkpoint保留作对照。

| 模型 | 完整 validation BPC ↓ | AJIMEE /200 | 原 development /137 | 扩大 development 草稿 /2000 |
| --- | ---: | ---: | ---: | ---: |
| V1 | 3.4736559 | 124 | 122 | 1453（72.65%） |
| V2.0 best，step375000 | 3.2598221 | 125 | 124 | 1463（73.15%） |
| V2.1 epoch1，step49213 | 3.1859122 | 136 | **125** | 1462（73.10%） |
| V2.1 epoch2 / best / last，step98426 | **3.0811788** | **137** | 121 | **1476（73.80%）** |

BPC 相对 V2.0 best 降低 **5.48%**，相对 V1 降低 **11.30%**。best 按固定子集最低 loss 选择，最终完整 BPC 也优于第一轮；best 与 last 为同一最终训练步，复用第二轮 IME 分数。评测使用该轮保存的完整validation和IME结果。

## IME 判读

冻结候选、原始参考标签、FP32、联合分词 suffix logP sum，无 EOS/截断。下表为 V2.0 best → V2.1 最终模型的逐例配对；p 为未校正多重比较的 exact two-sided McNemar，作为探索性结果。

| 数据集 | 改善 | 退化 | 净增 | p |
| --- | ---: | ---: | ---: | ---: |
| AJIMEE 200 | 15 | 3 | +12 | .00754 |
| 原 development 137 | 2 | 5 | -3 | .45313 |
| 扩大 development 草稿 2000 | 75 | 62 | +13 | .30524 |

AJIMEE 有明确的同集增益；扩大集相对 V1 为 81 改善 / 58 退化，净增23，p=.06165，仍未达到 .05。扩大集标签尚未正式审核，近重复与原公开集训练重叠限制仍在，不能据此宣称已证明生产收益。1,000 条 blind 未 LM 计分，未使用后验词典别名选模型。

第一轮 → 第二轮：原137条为3改善/7退化，p=.34375；2000条为63改善/49退化，p=.21913。原137条的五条退化涉及 `昼ごはん/昼ご飯`、`つぶす/潰す`、`つかる/浸かる`、`襟/衿`、`捲り/まくり` 表记；另两条为 `中村区→中村九`、`花がさく→花が作` 实质错选。这是逐例诊断，未改标签或重新计算口径，尚不足以认定整体 IME 已过拟合。

扩大集的最终 Top-5 为1750/2000，MRR=.79921，coverage=1756/2000；相对引擎修正314/损坏62，fallback=0。第一轮/最终模型均在本机5060 Laptop以同池FP32计分，复用V1/V2.0缓存。

## 性能与推理

扫描50,393,686窗口，训练998,854,924有效预测对，crop移除114,070,390；覆盖2 passes，有效token passes=1.795。summary总耗时3739.8秒，纯训练3650.2秒、**273,368 tokens/s**，约为V2.0的 **2.26×**。batch512与backbone编译同时改变，不能单独归因于其中一项。allocated峰值10.42GiB，reserved峰值21.18GiB。

12条固定prompt已完成：`今日はとても暑いですね。`、`お忙しいところ、ご来場いただき、誠にありがとうございました。`；`明日の朝は` 的greedy续写仍有重复。开放生成样例与IME、设备评测分别解释。

## 模型选择与归档

最终模型兼有最低BPC和较高的扩大集草稿Top-1，作为继续预训练起点；第一轮在原137条更高，保留对照。独立标签审核仍是质量证据的限制。

- 本地模型：`artifacts/models/tiny-ja-v2.1-e16k-d320-l6-restart1/`，best、last、epoch-1及两轮完整报告。
- 扩大集评分：`outputs/ime-eval/expanded-v21-dev-draft-restart1-{epoch1,epoch2}/`。
- 配对与全部分歧：`outputs/ime-eval/tiny-ja-v2.1-restart1-comparison/comparison.json`。
- 推理：`outputs/model-checks/tiny-ja-v2.1-e16k-d320-l6-restart1/samples.json`。
- 报告归档：`handoff/vimeml-v21-restart1-reports.tar.gz`；启动源码快照与旧step0产物保留。

后续extend5追加3.27轮后停止，保留早期step40000；结果见[追加训练](v21-extend.md)。本报告的restart1结果保持冻结。
