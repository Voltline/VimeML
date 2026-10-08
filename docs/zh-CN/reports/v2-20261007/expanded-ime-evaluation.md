# 扩大 IME 开发集初评

2026-10-07。3,000 条真实 AzooKey 候选已验收，2,000 development 完成 V1/V2.0 best FP32 CUDA 评分。参考标签仍是草稿；1,000 blind 只验收导出，未执行 LM 计分。

## 主要结果

| 同池排序 | Top-1 | Top-5 | MRR | 修正 / 损坏引擎 |
| --- | ---: | ---: | ---: | ---: |
| AzooKey | 1,224/2,000（61.20%） | 1,711 | .71462 | — |
| V1 | 1,453（72.65%） | 1,745 | .79248 | 306 / 77 |
| V2.0 best | **1,463（73.15%）** | **1,750** | **.79542** | 323 / 84 |

V1→V2：92 改善、82 退化，净增 10（.50 个百分点），exact McNemar **p=.49517**，无显著优势。两模型相对引擎在草稿标签下均 p<1e-30。精确答案 coverage **1,756/2,000（87.80%）**，244 条失败保留；covered Top-1 为 82.74% / 83.31%。空池与长度/roundtrip 回退均为 0。

## 导出与评分

全部输入、原位置、query/context/answers、映射一致；候选非空、去重、分数有限，逐例 rank 与 CLI aggregate 一致。Converter `d59a28e4c7ca049aef04f29a91eae9677a7753f2`，主字典 `4d418525b090cf49c219819d05a7e3cc2a4346eb`，emoji `67b822603391b01238d7b80b8b61b63f966cf357`；Swift 6.4 / arm64 macOS 27.2。n-best20、typo/Zenzai/stable off，两套 `cli_exit=0`。Flags 来自导出脚本记录。

实际开发候选数量：1–5 个 918 条，6–10 个 727 条，11–20 个 355 条。Blind 引擎 Top-1 585/1,000、coverage 862/1,000，仅作导出核验。主评分固定联合分词 suffix logP sum，无 EOS/截断，并列保持原序。

| Development 子组 | 条数 | V1 Top-1 | V2 Top-1 |
| --- | ---: | ---: | ---: |
| 有左文 | 1,439 | 1,066 | 1,076 |
| 无左文 | 561 | 387 | 387 |
| 读音 ≤16 | 749 | 606 | 618 |
| 读音 17–32 | 689 | 515 | 515 |
| 读音 >32 | 562 | 332 | 330 |

同 2,000 条移除左文诊断为 1,420/1,425；token mean logP 仅 1,057/1,082。描述性分层不证明因果。两模型工具耗时约 32/39 秒，含加载和评分，不是 iOS 延迟或 AutoDL 吞吐。

## 标签敏感性

草稿漏收 `事/こと`、`為/ため`、`無い/ない` 等表记。后验 UniDic 代理要求相同 lemma、词汇/实际假名读音、POS 和活用；双方含汉字时保留汉字序列，OOV/空白精确匹配，保留标点与边界。共 1,393 条有 2,934 个待审别名。

| 后验词典代理，非 gold | Top-1 |
| --- | ---: |
| AzooKey | 1,532/2,000（76.60%） |
| V1 | 1,755（87.75%） |
| V2 | 1,769（88.45%） |

代理口径 43 改善 / 29 退化，p=.12492，仍不显著。该诊断在看到结果后执行，可能误收或漏收，未修改原标签/分数，不用于正式模型选择。2,000 条隐藏排序/分数的待审包已准备，逐条母语审核尚未完成；网页噪声与近重复限制仍在。

## 产物

- 候选及验收：`artifacts/benchmarks/ime-expanded-v21-candidates-v1/`。
- 完整评分：`outputs/ime-eval/expanded-v21-dev-draft-{v1,v2}/`。
- 配对、分层、逐例、vs-engine：`outputs/ime-eval/expanded-v21-dev-draft-comparison/`。
- 后验诊断：`outputs/ime-eval/expanded-v21-dev-draft-label-sensitivity-v2/`。
- 待审包：`outputs/ime-eval/expanded-v21-label-review/development-score-hidden-v2.json`。

入口为 `scripts/benchmarks/evaluate_ajimee.py --benchmark artifacts/benchmarks/ime-expanded-v21-candidates-v1/development`；模型、tokenizer和新output由参数指定。V2.0 best保留冻结记录，V1保留基线；V2.1后续两轮与扩大集对照见 [最终评测](v21-evaluation.md)，中断记录见 [迁出报告](v21-recovery.md)。
