# Standard IME v1：首次固定模型比较

2026-10-08。Mac真实候选导入及V2.1/V3 A的FP32评分完成。V3 A在新来源blind上净增11/725，在旧网页回归上净少6/500；WRIME在development与blind上均改善。结果支持领域适应存在收益与取舍，不能把旧基准的下降解释为所有输入场景都退化，也不足以认定V3 A全面优于当前部署版本。

## 冻结范围

基准为`ime-standard-ja-v1-ai-expert-r1`，release ID `d9c60e0e8cb44481ae97c2cea16cfdca`。全部新标签在候选收集与LM评分前逐条AI专家审核、冻结，不属于母语人工gold；旧网页500条仍为历史草稿标签。构建、分组、许可和重叠筛查范围见[协议](standard-ime.md)。

比较模型在任何新LM评分之前登记：V2.1 extend5 best step40000与V3 A best step25146，plan ID `8ac20582a3ca4dce9c79eb09342b836c`。没有依据development更换模型、评分或标签；两模型按同一plan首次打开blind。本轮以后，该blind作为已暴露的固定reference test使用，后续调参的独立证据需要另建版本。

AzooKey converter固定为`d59a28e4c7ca049aef04f29a91eae9677a7753f2`，主字典`4d418525b090cf49c219819d05a7e3cc2a4346eb`、emoji字典`67b822603391b01238d7b80b8b61b63f966cf357`。n-best20、typo关闭、Zenzai关闭、stable关闭。Mac返回输入、ID、顺序、参考和字典版本与冻结包一致，无需重导或修改参考。

主评分采用FP32 context+candidate联合分词公共前缀后的logP sum，不加EOS、不截断、并列保留原序。所有候选缺失和回退保留分母；两模型四次新评分的回退均为0。没有评测INT8或Mac设备时延，本地批量评分耗时不作UI延迟证据。

## Top-1结果

| 集合 | 数量 | AzooKey原序 | V2.1 | V3 A | V3 A净变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development WRIME | 483 | 262 | 298（61.70%） | 310（64.18%） | +12 |
| Development JMultiWOZ | 245 | 191 | 207（84.49%） | 198（80.82%） | -9 |
| Development micro | 728 | 453 | 505（69.37%） | 508（69.78%） | +3 |
| Blind WRIME | 475 | 265 | 309（65.05%） | 318（66.95%） | +9 |
| Blind JMultiWOZ | 250 | 202 | 226（90.40%） | 228（91.20%） | +2 |
| Blind micro | 725 | 467 | 535（73.79%） | 546（75.31%） | +11 |
| 旧FineWeb2 Edu regression | 500 | 317 | 374（74.80%） | 368（73.60%） | -6 |

新来源等权macro Top-1：development为V2.1 73.09%、V3 A 72.50%；blind为77.73%、79.07%。development的micro微升而macro微降，原因是WRIME样本更多，不能只选有利的汇总口径。旧网页不混入新来源主分数，development与blind也不合并用于选型。

## 候选上限与其他指标

| 集合 | WRIME候选召回 | JMultiWOZ候选召回 | Micro候选召回 |
| --- | ---: | ---: | ---: |
| Development | 397/483（82.19%） | 233/245（95.10%） | 630/728（86.54%） |
| Blind | 405/475（85.26%） | 245/250（98.00%） | 650/725（89.66%） |

WRIME的召回限制更明显；缺失参考的题目无法仅靠LM重排改对。召回使用冻结的多参考精确匹配，参考并非穷尽所有自然表记，不能把全部未命中直接认定为字典缺词。

| Micro指标 | Dev V2.1 | Dev V3 A | Blind V2.1 | Blind V3 A |
| --- | ---: | ---: | ---: | ---: |
| Top-5 | 618/728 | 618/728 | 636/725 | 639/725 |
| MRR | 0.75934 | 0.76814 | 0.79875 | 0.81147 |
| MinCER ↓ | 0.04255 | 0.03871 | 0.04061 | 0.03642 |
| 已召回题目Top-1 | 80.16% | 80.63% | 82.31% | 84.00% |
| 相对引擎改对 / 改坏 | 100 / 48 | 104 / 49 | 107 / 39 | 120 / 41 |

## 同池配对

| V3 A相对V2.1 | 改对 | 改错 | 净变化 | Case-level exact McNemar双侧p |
| --- | ---: | ---: | ---: | ---: |
| Development WRIME | 40 | 28 | +12 | 0.1818 |
| Development JMultiWOZ | 10 | 19 | -9 | 0.1360 |
| Development micro | 50 | 47 | +3 | 0.8392 |
| Blind WRIME | 36 | 27 | +9 | 0.3135 |
| Blind JMultiWOZ | 9 | 7 | +2 | 0.8036 |
| Blind micro | 45 | 34 | +11 | 0.2604 |
| 旧网页regression | 14 | 20 | -6 | 0.3915 |

配对已核对ID、参考、读音、给定左文、来源、provenance和实际候选内容/次序。p值按case计算，未处理同作者/用户相关性与多重比较，仅作探索性诊断；未检出差异不证明统计等效。标签为AI专家版且公开来源的训练重叠仅做限定范围筛查，不能宣称绝对无污染。

## 实验判断与复用

此前旧IME基准和旧BPC的下降仍是有效的旧能力诊断，但不是日常口语总体退化的充分证据。WRIME两组Top-1、blind MRR与MinCER的方向一致，支持V3 A的口语领域收益；JMultiWOZ两组方向不同，尚未形成稳定的任务对话优势。当前仍保留V2.1部署，V3 A作为领域适应候选，后续优先做较低续训学习率或旧语料回放的单变量对照。此结果不要求立即从头启动B。

固定题库与候选可以长期用于回归，版本与标签保持冻结。后续结果分别列development、已暴露reference test与旧网页，不反复把同一blind称作新盲测；不根据错误样例追加参考后沿用原版分数。

原始结果保存在忽略目录：

- 候选：`artifacts/benchmarks/ime-standard-ja-v1-candidates/`，含blind consumption记录。
- 新评分：`outputs/ime-eval/standard-v1-{v21,v3-a}-{development,blind}/`。
- 旧500条缓存：`outputs/benchmarks/ime-standard-ja-v1/legacy-{v21,v3-a}/`，未重复推理。
- 导入、固定plan、配对与阶段：`outputs/benchmarks/ime-standard-ja-v1/{candidate-import,blind-plan,paired-development,paired-blind,paired-regression,stages}.json`。

新增`compare_standard_ime.py`直接读取已完成分数并核对同池身份。四次必要的新评分完成后没有重复LM推理、全文件SHA256或额外冒烟/回归测试；受许可限制的文本和候选不进入Git。
