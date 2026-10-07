# 评测

当前离线候选模型为V2.1 extend5 best / step40000；部署结果仍以V1为基线。模型结果见[追加报告](reports/v2-20261007/v21-extend.md)，Core ML验收见[Mac准备](mac-v21-preparation.md)。

## 评分定义

AzooKey提供冻结的真实候选池，LM仅重排，不注入答案。context+candidate联合SentencePiece编码，候选token序列公共前缀后的完整词表logP求和；不加EOS、不截断，稳定并列保持原序。超出128-token窗口时整条回退引擎，召回失败和回退保留在分母。mean只作次要诊断；这是token后缀似然代理，不是精确字符串条件概率。

Top-1/5按任一冻结可接受答案精确匹配，不做宽度正规化。报告候选召回、MinCER、MRR、纠正/改坏及回退；同池差异按case ID、读音、左文、答案和候选顺序配对。完整validation BPC使用所有预测目标（含EOS）与原Unicode字符数，test split未用于本轮。

## 数据与结果

| 集合 | 范围与限制 |
| --- | --- |
| AJIMEE JWTD_v2/v1 | 200条，公开集、83条多答案；已用于错误分析，不是blind。网页训练重叠未知 |
| 原development | 137条，AI生成/复核，用于原策略开发；不是母语gold |
| 扩大development | 2000条、冻结真实候选；参考标签仍为草稿 |
| 扩大blind | 1000条，冻结且尚未LM计分，不用于选模型或量化参数 |

| 模型 | 完整BPC ↓ | AJIMEE /200 | 原dev /137 | 扩大草稿 /2000 |
| --- | ---: | ---: | ---: | ---: |
| V1 | 3.4736559 | 124 | 122 | 1453 |
| V2.0 best | 3.2598221 | 125 | 124 | 1463 |
| V2.1 restart1 | 3.0811788 | 137 | 121 | 1476 |
| V2.1 extend5 best | 3.0802686 | 144 | 122 | 1487 |

extend5相对restart1的扩大集净增11，exact two-sided McNemar p=.22155；相对V1净增34，p=.00648。均为开发集探索性、未校正多重比较；草稿标签、公开集重叠和模型选择限制仍在，不代表已证明生产收益。原标签和词典别名口径没有根据模型结果修改。

## 入口与缓存

`scripts/benchmarks/evaluate_ajimee.py`接受`--benchmark/--checkpoint/--tokenizer/--device/--output`，输出metrics、逐候选scores和变化样例；已有结果优先复用，每次新模型使用独立输出。

- 最新FP32：`outputs/ime-eval/tiny-ja-v2.1-extend5-best-{ajimee,development}/`。
- 扩大集：`outputs/ime-eval/expanded-v21-dev-draft-extend5-best/`。
- 配对：`outputs/ime-eval/tiny-ja-v2.1-extend5-comparison/comparison.json`。
- Core ML量化与设备报告：`outputs/deployment/`，不能把严格logits对齐、Top-1命中或设备时延混为同一验收。

纯LM sum为当前接入路线；历史AzooKey+λ×LM组合在原开发集选λ=2，量化后未重新验证，不能沿用为当前最优。联想以固定前缀进行自然度与重复诊断，不作为唯一答案准确率。

详细参考：[AJIMEE导出](reference/ajimee-benchmark.md)、[开发集生成](reference/development-generation.md)、[组合排序](reference/hybrid-ranking.md)、[联想](reference/phrase-demo.md)、[V1结果](reference/results.md)、[扩大数据初评](reports/v2-20261007/expanded-ime-evaluation.md)。
