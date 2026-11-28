# ja-gen stage 2.2: 200M joint reading pretraining and development evaluation

> [Chinese version](../../../zh-CN/ja/reports/ja-gen-stage2-200m-20261128/report.md)

2026-11-28. Follows the 2.2 registration (2026-11-26) in the [ja-gen plan](../../training/training-ja-gen-plan.md). It also adds the baselines of the 2.1 arms and the 200M on the colloquial development set `ime-kana-ja-colloq-v1`. Development sets only: neither the `ime-kana-ja-v1` test split nor the `ime-kana-ja-colloq-v1` test split was read.

**Result**: the 200M is the best model so far on all three development sets and serves as the teacher for the 2.3 distillation.

- kana development lenient Top-1 is 89.33, 2.2 points above the 2.1 reading arm's four-seed mean.
- Colloquial development lenient Top-1 is 89.67, 2.6 points above the 2.1 reading arm's four-seed mean.
- Standard reranking is 517, back at the level of the 20M LM control. Joint pretraining does not show the reranking cost seen in 20M reading post-training.

2.2 was registered without a decision rule. Its purpose is to be the 2.3 teacher.

## Training

- **Configuration**: `configs/ja/train-ja-kana-200m-joint.toml`, 12K / L15 / c128, 193,625,088 parameters, random initialization, 845,000 updates of 832 samples, mix as registered.
- **Run**: all 845,000 updates in one run, with no interruption and no resume from last.pt.
- **Time**: update time alone was 110,230 seconds (30.6 hours), excluding startup, compilation, 86 validations and checkpoint saves.
- **final.pt**: SHA256 `47721ddd357cb3a720d7108e4aeaa60f671d83167504fd9f649365b37d70a843`. Only final.pt is used, as registered.

Validation loss (for reading kinds, teacher-forced whole-sample accuracy in parentheses):

| Step | plain | chat | web | span | sentence | web_span | web_sentence |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 4.2122 | 4.2241 | 4.2652 | 0.1644 | 0.1989 | 0.1951 | 0.2089 |
| 100,000 | 3.9213 | 4.0165 | 3.9252 | 0.1230 | 0.1482 | 0.1415 | 0.1565 |
| 400,000 | 3.7878 | 3.9130 | 3.7681 | 0.1031 | 0.1207 | 0.1186 | 0.1319 |
| 800,000 | 3.5634 | 3.7952 | 3.5090 | 0.0777 | 0.0942 | 0.0850 | 0.0989 |
| 845,000 | **3.5573** | **3.8374** | **3.5044** | **0.0783** (82.36%) | **0.0942** (71.35%) | **0.0826** (82.54%) | **0.0971** (72.83%) |

- `plain` and `web` validation loss falls until the last step.
- `chat` reaches its minimum at step 740,000 (3.7796) and then rises to 3.8374. With the registered mix, `chat` gets about 43 passes in 845,000 updates and overfits late. It is only 24 of 832 samples per update and was not changed.

## Evaluation setup

- Script `scripts/ja/benchmarks/evaluate_ja_gen_stage2.py`; candidate index `kana-index.json`, SHA256 prefix `b60c485d7ba2`, as in 2.1.
- **kana and colloquial development**: the final list is AzooKey N-best 20 ∪ constrained beam-16 generation bucketed by key position (top 10), ordered by the 200M's own generation score + λ × LM score with λ = 1.6. Constrained search never fell back on either set (fallbacks = 0).
- **Standard development**: the 200M's LM score reranks the AzooKey pool.
- **Statistics**: paired differences use author/dialogue-clustered sign-flip tests. The colloquial set has one item per document, so each item is its own cluster and the test is item-level.
- **Check**: every number in the tables was recomputed from the per-row result files: Top-1, the λ sweep, the colloquial bands, and the paired differences against the 2.1 arms.

## `ime-kana-ja-v1` development and Standard development

| Metric | 200M | 2.1 reading mean | 2.1 LM control mean | V4.2 reranking |
| --- | ---: | ---: | ---: | ---: |
| kana lenient Top-1 (primary) | **89.33** | 87.17 | 85.79 | 85.2 |
| kana exact Top-1 | 72.00 | 71.25 | 66.67 | 66.3 |
| kana lenient Top-3 | 93.00 | 91.88 | 90.63 | 90.5 |
| D5 items, lenient Top-1 | 75.00 (18/24) | 69.79 | 70.83 | |
| Standard exact Top-1 (/728) | **517** | 504.50 | 515.25 | 523 |

Generated candidates alone (not merged with the AzooKey pool) reach kana lenient Top-1 68.00, exact 46.50 and lenient Top-3 73.67.

λ sweep, kana development lenient Top-1 (items):

| λ | 0 | 0.8 | 1.2 | 1.6 | 2.0 | 2.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lenient Top-1 | 88.17 (529) | 89.17 (535) | 89.17 (535) | **89.33 (536)** | 89.00 (534) | 89.83 (539) |

Across 0.8–2.5 the spread is at most 5 items, so λ stays at 1.6. 2.5 is the highest point of the sweep (+3 items) and is not adopted.

Paired differences of the 200M against each baseline (lenient Top-1 on kana, exact Top-1 on Standard, items):

| Comparison | kana net (gain / loss) | kana p | Standard net (gain / loss) | Standard p |
| --- | ---: | ---: | ---: | ---: |
| V4.2 reranking | +25 (+32 / −7) | 0.0002 | −6 (+36 / −42) | 0.54 |
| reading s1 | +12 (+18 / −6) | 0.016 | +16 (+51 / −35) | 0.074 |
| reading s2 | +10 (+18 / −8) | 0.052 | +12 (+47 / −35) | 0.24 |
| reading s3 | +14 (+19 / −5) | 0.003 | +9 (+47 / −38) | 0.40 |
| reading s4 | +16 (+19 / −3) | 0.0009 | +13 (+49 / −36) | 0.16 |
| LM control s1 | +19 (+29 / −10) | 0.004 | +10 (+44 / −34) | 0.35 |
| LM control s2 | +18 (+30 / −12) | 0.009 | −1 (+39 / −40) | 1.00 |
| LM control s3 | +28 (+33 / −5) | <0.0001 | +5 (+41 / −36) | 0.66 |
| LM control s4 | +20 (+26 / −6) | 0.0005 | −7 (+34 / −41) | 0.47 |

## `ime-kana-ja-colloq-v1` development (colloquial, 600 items)

V4.2 was not scored on this set. The baselines are AzooKey order and the eight 2.1 arms. Band sizes: zero 151, low 134, mid 141, high 174.

| Model | lenient Top-1 | lenient Top-3 | exact Top-1 | exact Top-3 | lenient zero / low / mid / high | exact zero / low / mid / high |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| AzooKey order | 72.83 | 86.50 | 55.50 | 71.67 | 72.19 / 72.39 / 70.21 / 75.86 | 62.25 / 55.97 / 44.68 / 58.05 |
| LM control s1 | 83.83 | 91.17 | 64.67 | 78.83 | 83.44 / 82.09 / 82.27 / 86.78 | 70.86 / 63.43 / 57.45 / 66.09 |
| LM control s2 | 85.17 | 91.33 | 65.67 | 79.83 | 82.78 / 85.07 / 85.82 / 86.78 | 70.86 / 67.16 / 56.74 / 67.24 |
| LM control s3 | 83.33 | 91.67 | 64.83 | 79.17 | 82.12 / 80.60 / 83.69 / 86.21 | 68.21 / 66.42 / 57.45 / 66.67 |
| LM control s4 | 85.33 | 92.17 | 65.17 | 79.50 | 83.44 / 83.58 / 84.40 / 89.08 | 70.86 / 64.93 / 57.45 / 66.67 |
| **LM control mean** | **84.42** | **91.58** | **65.08** | **79.33** | 82.95 / 82.84 / 84.04 / 87.21 | 70.20 / 65.49 / 57.27 / 66.67 |
| reading s1 | 87.33 | 93.17 | 73.33 | 86.50 | 90.73 / 84.33 / 87.94 / 86.21 | 78.81 / 72.39 / 68.79 / 72.99 |
| reading s2 | 87.50 | 93.50 | 73.17 | 85.83 | 88.74 / 84.33 / 87.94 / 88.51 | 78.15 / 71.64 / 69.50 / 72.99 |
| reading s3 | 86.83 | 93.67 | 71.33 | 86.33 | 90.73 / 82.84 / 87.94 / 85.63 | 78.15 / 68.66 / 67.38 / 70.69 |
| reading s4 | 86.83 | 93.17 | 72.50 | 86.50 | 88.74 / 85.07 / 85.82 / 87.36 | 76.82 / 71.64 / 67.38 / 73.56 |
| **reading mean** | **87.12** | **93.38** | **72.58** | **86.29** | 89.73 / 84.14 / 87.41 / 86.93 | 77.98 / 71.08 / 68.26 / 72.56 |
| **200M** | **89.67** | **95.33** | **75.17** | **88.17** | 91.39 / 85.82 / 88.65 / 91.95 | 77.48 / 71.64 / 70.92 / 79.31 |

- **Generated candidates alone**: on the colloquial set the 200M reaches lenient Top-1 74.33, exact 55.83 and lenient Top-3 81.00. Lenient Top-1 is already above AzooKey order.
- **λ sweep** (lenient Top-1): 87.67 at 0, 89.67–90.33 across 0.8–2.5.

Paired differences in lenient Top-1 (items, item-level sign-flip):

| Comparison | s1 | s2 | s3 | s4 |
| --- | ---: | ---: | ---: | ---: |
| reading − LM control | +21 (p 0.004) | +14 (p 0.038) | +21 (p 0.005) | +9 (p 0.24) |
| 200M − reading | +14 (p 0.018) | +13 (p 0.025) | +17 (p 0.005) | +17 (p 0.005) |
| 200M − LM control | +35 (p <0.0001) | +27 (p 0.0002) | +38 (p <0.0001) | +26 (p 0.0003) |

The 200M against AzooKey order: +101 (+109 / −8).

## Observations

- **Joint pretraining has no reranking cost**: 20M reading post-training reranks Standard about 11 items worse than the LM control. The jointly pretrained 200M scores 517 on Standard, on par with the 20M LM control mean (515.25). It is 9–16 items above each of the four reading arms, but no single seed's difference is significant. Against V4.2 reranking it is −6 (p 0.54), inside the seed band.
- **On kana development, the 200M's gain is mostly lenient**: lenient Top-1 is 10–16 items above the reading arm at every seed, while exact Top-1 is only 0.75 points higher and lenient Top-3 1.1 points higher.
- **The colloquial set ranks the models the same way as kana**: LM control < reading arm < 200M. As on kana development, the reading arm's gain over the LM control is larger in exact Top-1 (+7.5 points).
- **Bands**:
  - The reading arm gains most over the LM control in the zero band (lenient +6.8 points). This band holds the lowest-rule-score colloquial text, without sentence-final particles or markers such as 笑.
  - The 200M gains most over the reading arm in the high band (lenient +5.0 points, exact +6.8).
- **Generation helps more on colloquial text**: generated candidates alone reach 74.33 on the colloquial set, above AzooKey order (72.83). On kana development they reach 68.00, below AzooKey order (74.3).
- **chat overfitting**: the 200M's `chat` validation loss rises over the last 100,000 updates. Its colloquial development result is still the best of the three model groups, so the mix is unchanged here. A multilingual zh-ja-en 800M reusing this mix needs a cap on `chat` passes.

## Next

- **2.3**: distill the 20M reading arm with the kd-cont recipe, using the 200M final.pt as the teacher.
  - The teacher SHA256 is pinned in `configs/ja/train-ja-kana-20m-kd-cont.toml`.
  - The student start and the decision rules are registered in the plan before any 2.3 training result is available.
- **3.3**: continues from the 2.3 kd-cont, with the three colloquial arms run together. As registered, λ is the value frozen in 2.3.
