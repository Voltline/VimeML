# 评测

当前模型为V2.1 extend5 best / step40000，部署采用INT8 block32；V1为冻结基线。总体结论见[V2总结](reports/v2-summary.md)，训练和量化分别见[追加报告](reports/v2-20261007/v21-extend.md)与[Core ML评测](reports/mac-20261008/v21-coreml.md)。

## 评分定义

AzooKey提供冻结的真实候选池，LM仅重排，不注入答案。context+candidate联合SentencePiece编码，候选token序列公共前缀后的完整词表logP求和；不加EOS、不截断，稳定并列保持原序。超出128-token窗口时整条回退引擎，召回失败和回退保留在分母。mean只作次要诊断；这是token后缀似然代理，不是精确字符串条件概率。

Top-1/5按任一冻结可接受答案精确匹配，不做宽度正规化。报告候选召回、MinCER、MRR、纠正/改坏及回退；同池差异按case ID、读音、左文、答案和候选顺序配对。完整validation BPC使用所有预测目标（含EOS）与原Unicode字符数，test split未用于本轮。

## 数据与结果

新基准`ime-standard-ja-v1-ai-expert-r1`包含WRIME 958条、JMultiWOZ 495条和旧FineWeb回归500条。AI专家审核、冻结、真实候选导入与首次固定模型评分已完成；AI审核不等同母语人工gold。新来源blind为V2.1 535/725、V3 A 546/725，旧网页回归为374/500、368/500。独立来源和旧回归单独报告；方法见[Standard IME协议](benchmarks/standard-ime.md)，来源分项与配对见[首次结果](benchmarks/standard-ime-results-20261008.md)。该blind后续作为已暴露reference test使用。

[V3 B联合重训](reports/v3-20261008/b-evaluation.md)完成4轮，按Standard development选定第2轮step132915：507/728，V2.1为505/728；选型冻结后的固定参考为533/725、旧网页回归364/500。开发集近似持平，旧能力未形成稳定替换收益，部署继续保留V2.1。全部参考和候选沿用冻结版本，旧500仍为草稿标签，725结果属于已暴露reference，不作为新的盲测证据。

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
| V2.1 extend5 INT8 | — | 144 | 122 | 1481 |

extend5相对restart1的扩大集净增11，exact two-sided McNemar p=.22155；相对V1净增34，p=.00648。均为开发集探索性、未校正多重比较；草稿标签、公开集重叠和模型选择限制仍在，不代表已证明生产收益。原标签和词典别名口径没有根据模型结果修改。

上表前四行为FP32，INT8 BPC未测。量化后扩大草稿8条改对、14条改错，净少6条，配对p=.28628；未发现显著退化，未建立统计等效性。严格logits失败与任务质量分别记录，当前量化损失在实验接受范围内。细节见[误差分析](reports/mac-20261008/v21-quantization-review.md)。

## 入口与缓存

`scripts/benchmarks/evaluate_ajimee.py`接受`--benchmark/--checkpoint/--tokenizer/--device/--output`，输出metrics、逐候选scores和变化样例；已有结果优先复用，每次新模型使用独立输出。

- 最新FP32：`outputs/ime-eval/tiny-ja-v2.1-extend5-best-{ajimee,development}/`。
- 扩大集：`outputs/ime-eval/expanded-v21-dev-draft-extend5-best/`。
- 配对：`outputs/ime-eval/tiny-ja-v2.1-extend5-comparison/comparison.json`。
- Core ML量化与设备报告：`outputs/deployment/`，不能把严格logits对齐、Top-1命中或设备时延混为同一验收。

纯LM sum为当前接入路线；历史AzooKey+λ×LM组合在原开发集选λ=2，量化后未重新验证，不能沿用为当前最优。联想以固定前缀进行自然度与重复诊断，不作为唯一答案准确率。

详细参考：[AJIMEE导出](reference/ajimee-benchmark.md)、[开发集生成](reference/development-generation.md)、[组合排序](reference/hybrid-ranking.md)、[联想](reference/phrase-demo.md)、[V1结果](reference/results.md)、[扩大数据初评](reports/v2-20261007/expanded-ime-evaluation.md)。
