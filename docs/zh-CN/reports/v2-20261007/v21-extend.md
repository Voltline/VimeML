# V2.1 追加训练结果

配置：[train-v21-extend.toml](../../../../configs/train-v21-extend.toml)。起点为冻结restart1最佳权重step98426（完整BPC3.0811788），携带AdamW动量与参数step计数；使用独立模型、日志和W&B run。

2026-10-07 20:15—21:54在4090 D运行，step161095结束，共3.273轮。SIGINT保存last权重与优化器状态，训练会话正常退出。初始固定子集loss4.1511，恢复45份AdamW状态。

| 项目 | 设置 |
| --- | --- |
| 上限 | 5 epochs / 246,065 updates，49,213 updates/epoch |
| 学习率 | 前147,639步保持3e-5，后98,426步余弦降至1e-5；无warmup |
| 数据 | 相同冻结token store与窗口；epoch offset6，使用数据轮次6–10 |
| 模型/批次 | 16K、320×6、context128、batch512、BF16、compiled backbone |
| 优化器 | 同一AdamW β=.9/.95、decay=.1、clip1，携带45份状态，源参数step98426 |
| 检查点/验证 | 每5000步固定子集与保存；每轮完整BPC、AJIMEE200、原development137 |
| 提前停止 | 从起点BPC3.0811788开始记录历史最佳；连续两轮高于最佳超过.01时停止 |

模型结构、有效batch、语料和prefix crop保持一致。新run的step从0计，AdamW内部参数step从98426继续；实际只进入少量衰减，未跑完预定后两轮。

## 评测与选择

| 模型 | 完整 validation BPC ↓ | AJIMEE /200 | 原 development /137 | 扩大 development 草稿 /2000 |
| --- | ---: | ---: | ---: | ---: |
| V1 | 3.4736559 | 124 | 122 | 1453 |
| V2.0 best | 3.2598221 | 125 | 124 | 1463 |
| restart1 best | 3.0811788 | 137 | 121 | 1476 |
| extend5 best，step40000 | **3.0802686** | **144** | 122 | **1487（74.35%）** |
| extend5 epoch1，step49213 | 3.0815054 | 140 | 125 | — |
| extend5 epoch2，step98426 | 3.0878445 | 142 | 122 | — |
| extend5 epoch3，step147639 | 3.0829648 | 135 | 122 | — |
| extend5 last，step161095 | 3.0920557 | — | — | — |

按固定子集最低loss选出**best.pt / step40000**，完整BPC仅比restart1低0.000910（0.0295%），后续完整epoch和停止时last均未改善。相同语料继续追加已呈收益递减，训练在3.273轮结束；未完成衰减阶段的效果未知。该checkpoint随后用于INT8部署，见[V2总结](../v2-summary.md)。

FP32、冻结候选与原始标签、联合分词suffix logP sum，无EOS或截断。相对restart1：AJIMEE为8改善/1退化（p=.03906），原137条2/1（p=1），2000条草稿39/28（净增11，p=.22155）。相对V1的2000条为91/57（净增34，p=.00648）；相对V2.0为84/60（p=.05490）。均为探索性、未校正多重比较；草稿标签未经正式审核，不能据此证明生产收益。未修改标签、使用后验别名或对1000条blind计分。

纯训练280,712 tokens/s，总耗时5923秒，扫描82,480,001窗口。12条固定推理示例完成，礼貌句续写正常，部分greedy仍重复。best/last和全部现有评测报告已下载，旧模型及启动源码快照保持冻结。训练收益与设备部署效果需分别验证。

- 模型与三轮原IME：`artifacts/models/tiny-ja-v2.1-e16k-d320-l6-extend5/`（epoch权重仍保留远端，best/last在本地）。
- 停止时完整验证：`outputs/model-checks/v21-extend5-closeout/full-validation.json`。
- 最佳模型原IME：`outputs/ime-eval/tiny-ja-v2.1-extend5-best-{ajimee,development}/`。
- 扩大集与配对：`outputs/ime-eval/expanded-v21-dev-draft-extend5-best/`、`outputs/ime-eval/tiny-ja-v2.1-extend5-comparison/comparison.json`。
- 推理：`outputs/model-checks/tiny-ja-v2.1-extend5-best/samples.json`；报告归档：`handoff/vimeml-v21-extend5-reports.tar.gz`。

优化器衔接记录与模型评测独立保存；远端IME和推理共用模型加载，归档复用checkpoint来源身份。
