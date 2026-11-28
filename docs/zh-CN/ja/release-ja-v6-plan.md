# vimeml-ja-v6 发布范围

[English version](../../ja/release-ja-v6-plan.md)

2026-11-28 起草。本文规定日文 v6 系列发布哪几个模型、哪些工作只算研发记录，以及每个模型发布前要过的门槛。结构与 [vimeml-zh-v2 发布范围](../zh/release-zh-v2-plan.md)一致，只是没有 800M。各模型的训练规则仍以 [ja-gen 计划](training/training-ja-gen-plan.md)为准；test 的确认评分另写预注册计划，本文只定原则。

## 命名

日文线此前用过 V1–V4.x（V2.1 和 V4.2 先后部署）和 V5（规模扩展预训练，没有部署，见 [V5 记录](training/training-v5.md)）。ja-gen 路线这一批模型统一称为 **vimeml-ja-v6**，接续这个编号。ja-gen 文档早先写的 `vimeml-ja-v2-20m` 就是 `vimeml-ja-v6-20m`。

## v6 包含什么

v6 的模型都能直接**从假名读音生成文本**：输入假名，模型直接输出转换结果，不再只给 AzooKey 的候选重新排序。同一个模型既生成候选，也给 AzooKey 和生成的候选统一打分（`gen_score + 1.6 × lm_score`）。所有模型都用 12K 分词器 `ja-unigram-12288`，都带 KV 缓存。

### 发布模型

| 发布名称 | 端 | 参数 | 来源 | 状态 |
| --- | --- | ---: | --- | --- |
| `vimeml-ja-v6-20m` | 手机 | 20,036,736 | `colloq-c40`：200M 蒸馏的 `kd-cont` 再做口语后训练（[3.4 报告](reports/ja-gen-stage3-seeds-20261128/report.md)），checkpoint `19b04ddf…` | 2026-11-28 test 确认通过：kana test 宽松 Top-1 88.00%，V4.2 部署方式 85.00%（[报告](reports/ja-gen-confirmation-20261128/report.md)）；Mac / Vime 交接 2026-11-28（[任务说明](mac/mac-ja-v6-20m.md)）；待 Mac 与真机结果，然后发布 |
| `vimeml-ja-v6-200m` | 桌面 | 193,625,088 | 2.2 联合读音预训练，用自己的分数重打分，checkpoint `47721ddd…` | 开发集评测：kana 宽松 Top-1 89.33，口语 89.67，Standard 重排 517（[报告](reports/ja-gen-stage2-200m-20261128/report.md)）；待 test 确认和桌面延迟测试，然后发布 |

- **20M** 是手机端模型，替换 V4.2：Core ML INT8、批量 KV v4、按读音位置分桶的束搜索。手机上只加载一个模型。
- **200M** 是桌面端模型，不考虑手机，不做 Mac INT8 测试，只测 FP32/FP16 精度和 Windows PyTorch 上的桌面延迟，与中文 v2 的桌面模型相同。
- **没有 800M**。中日英多语 800M 是 ja-gen 计划第 6 阶段的单独实验，不属于 v6。

### 不发布、只作为研发记录的工作

| 工作 | 结论 | 记录 |
| --- | --- | --- |
| 2.1 20M 纯 LM 组与读音后训练组（各 4 个 seed） | 读音组成为蒸馏前的基线 | [2.1 报告](reports/ja-gen-stage2-20m-20261127/report.md) |
| 2.3 蒸馏 20M `kd-cont` 及 seed 复现 | 被 `colloq-c40` 取代；在第 4 阶段作为对照 | [2.3 报告](reports/ja-gen-stage2-kd-20261128/report.md) |
| 3.3 `colloq-ctrl`、`colloq-c40-p85`，3.4 的 seed 复现 | 用于选择和判定，不发布 | [3.3 报告](reports/ja-gen-stage3-colloq-20261128/report.md)、[3.4 报告](reports/ja-gen-stage3-seeds-20261128/report.md) |

## 发布门槛

下面是原则。具体数字和检验方法写进各模型的确认计划，先提交再打开 test。

1. **假名转换是主指标。** 在 `ime-kana-ja-v1` test 上测宽松 Top-1，最终列表为 AzooKey N-best 20 加生成的前 10，用模型自己的分数排序。严格 Top-1 和口语集 `ime-kana-ja-colloq-v1` test 一并报告。
2. **对照。** 20M 对照现在部署的 V4.2 方式（AzooKey N-best 20 + V4.2 重排）。200M 对照 `vimeml-ja-v6-20m`。
3. **重排守门。** Standard 开发集重排不低于 V4.2 配方 seed 带下沿 501（ja-gen 第 2 阶段的护栏）。
4. **延迟。** 20M 在 iPhone 上逐键实测（见交接说明）；200M 在 Windows 上用 PyTorch 测完整流程的 p50/p95。
5. **只确认一次。** 开发集上定下的设置（λ 1.6、beam 16、前 10、AzooKey 前 20）冻结。两个 test 已在 20M 的确认中用过一次；200M 的确认同样在这两个 test 上只运行一次，与中文 v2-200m 在 20M 之后用同一公开 test 确认的做法相同。

## 发布方式

- Hugging Face 仓库 `Voltline/vimeml-ja-v6`，按目录收录 `20m/` 和 `200m/`，文件布局与 `Voltline/vimeml-zh-v2` 相同：推理包、tokenizer、读音索引和模型卡。
- `NOTICE.md` 列出训练数据的来源和许可。WRIME（CC BY-NC-ND 4.0）和 JMultiWOZ（CC BY-SA 4.0）只用于评测，条目文本不随包发布。
- 20M 在 Mac 与真机结果通过后发布；200M 在 test 确认和桌面延迟测试之后发布。

## 未决事项

- `vimeml-ja-v6-200m` 的 test 确认计划：起草、提交后运行一次。
- `vimeml-ja-v6-20m` 的 Mac 与 Vime 结果。
