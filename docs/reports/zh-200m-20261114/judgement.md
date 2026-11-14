# zh-200m six-arm prediction and reranking scores and decision (2026-11-14)

The six arms' final checkpoints were scored and decided under the evaluation rules frozen in the [200M training plan](../../training-zh-200m-plan.md). All on development; blind was not read. Training is described in the [six-arm training record](../zh-200m-20261113/training.md).

**Conclusion: none of the three contrasts is effective, so by the plan's defaults the choice is the 24K vocabulary with context 512, and the official checkpoint is the final checkpoint of `24k-c512-s1`.** The reranking gate passes: no arm is below vimeml-tiny-zh-v1 (20M).

## Scoring setup

- Prediction: `ime-predict-zh-v1-r1`, all four tracks of `scripts/benchmarks/predict_ime_zh.py`. Decoding: beam 16, candidates of 2–6 characters, top 5 candidates, context at the model's limit (512 for c512 arms, 128 for c128), FP32.
- Reranking: `ime-standard-zh-v1-r1` development full pinyin, 1,456 items, scored with `scripts/benchmarks/score_zh13_lm.py` (its loader supports zh-200m checkpoints), then counted with `score_standard_ime_zh.py`.
- The six checkpoint SHA256s match the training output manifests. The checkpoint SHA in the scoring outputs was also checked per arm: the prediction and reranking outputs used the same file.
- The primary metric, Top-3, was recomputed locally from per-item results and matches the scorer's `score.json` to four decimals.

## Primary metric: track A Top-3 (micro, 7,200 items)

| Arm | Top-1 | **Top-3** | Top-5 |
| --- | ---: | ---: | ---: |
| 16k-c512 | 0.1932 | 0.3135 | 0.3744 |
| 24k-c512-s1 | 0.1971 | **0.3164** | 0.3728 |
| 24k-c512-s2 | 0.1782 | 0.2999 | 0.3525 |
| 24k-c512-s3 | 0.1858 | 0.3051 | 0.3632 |
| 32k-c512 | 0.1892 | 0.3014 | 0.3606 |
| 24k-c128 | 0.1910 | 0.3099 | 0.3674 |

The range of the three 24K seeds is **0.0165** (0.3164 − 0.2999).

## Decision

Every contrast uses `24k-c512-s1` as the reference: the plan gives the other arms s1's seed, so s1 is the same-seed control. Tests:

- Paired tests with `vimeml.benchmarks.statistics.paired`, clustered by dialogue (2,400 clusters), p from the cluster sign-flip test.
- Holm correction over the three contrasts.
- Effective = absolute difference above the seed range, and Holm p < 0.05.

| Contrast | Difference | Net (fixed/broken) | Cluster p | Holm p | 95% cluster bootstrap (items) | Effective |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 16K vs 24K | −0.0029 | −21 (+430/−451) | 0.503 | 0.503 | [−78, 37] | No |
| 32K vs 24K | −0.0150 | −108 (+372/−480) | 0.00035 | 0.00105 | [−166, −50] | No |
| c128 vs c512 | −0.0065 | −47 (+414/−461) | 0.119 | 0.238 | [−104, 11] | No |

- **32K**: significantly worse than the reference, but the 0.0150 difference does not exceed the 0.0165 seed range, so by the rule it is not effective. The direction is 32K worse, so no reading would choose 32K.
- **16K**: level with 24K.
- **c128**: 0.0065 behind, within noise.
- The reference s1 happens to have the highest Top-3 of the three seeds. Against the three-seed mean of 0.3071, 16K (+0.0064) and c128 (+0.0028) are slightly higher and 32K (−0.0057) slightly lower. This is descriptive only and takes no part in the decision.

## Descriptive tracks (not used for the decision)

| Arm | C first-syllable Top-3 | C initials Top-3 | B keystroke savings | B pick rate | D top-20% confidence Top-1 | D top-50% Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16k-c512 | 0.2133 | 0.7417 | 0.2056 | 0.1619 | 0.3389 | 0.2556 |
| 24k-c512-s1 | 0.2137 | 0.7300 | 0.2062 | 0.1621 | 0.3486 | 0.2575 |
| 24k-c512-s2 | 0.2096 | 0.7092 | 0.1949 | 0.1526 | 0.3194 | 0.2275 |
| 24k-c512-s3 | 0.2183 | 0.7375 | 0.1995 | 0.1562 | 0.3326 | 0.2450 |
| 32k-c512 | 0.2200 | 0.7358 | 0.2030 | 0.1584 | 0.3312 | 0.2358 |
| 24k-c128 | 0.2167 | 0.7300 | 0.1968 | 0.1551 | 0.3431 | 0.2447 |

On tracks B, C and D, differences between arms fall within or near the range of the three 24K seeds, consistent with the primary metric.

## Reranking gate: Standard development full-pinyin Top-1

| Arm | Top-1 | vs v1 `12k-fresh-s1` (1,363) | Cluster p |
| --- | ---: | ---: | ---: |
| 16k-c512 | 1,378 (94.6%) | +15 (+31/−16) | 0.038 |
| 24k-c512-s1 | 1,370 (94.1%) | +7 (+22/−15) | 0.328 |
| 24k-c512-s2 | 1,383 (95.0%) | — | — |
| 24k-c512-s3 | 1,371 (94.2%) | — | — |
| 32k-c512 | 1,382 (94.9%) | +19 (+34/−15) | 0.010 |
| 24k-c128 | 1,380 (94.8%) | +17 (+28/−11) | 0.012 |

- Every arm's pool covers 1,420 items; 0 fallbacks; 160–225 items rescored at a merged boundary. **No arm is significantly below v1; the gate passes.**
- The three 24K seeds differ by 13 items (s1 vs s2: +26/−13, p 0.053). Differences among 200M arms on reranking are the same size as seed noise; this benchmark is near saturation (the v1 report's conclusion).

## BPC and memorization sample (20th point, full FP32, from the training record)

| Arm | Web BPC | Chat BPC | Memorization sample BPC | Sample − web |
| --- | ---: | ---: | ---: | ---: |
| 16k-c512 | 3.8251 | 5.3731 | 3.8083 | −0.0168 |
| 24k-c512-s1 | 3.8641 | 5.4210 | 3.8469 | −0.0171 |
| 24k-c512-s2 | 3.8750 | 5.4133 | 3.8577 | −0.0172 |
| 24k-c512-s3 | 3.8767 | 5.4046 | 3.8601 | −0.0166 |
| 32k-c512 | 3.9153 | 5.4446 | 3.8995 | −0.0158 |
| 24k-c128 | 3.8910 | 5.4037 | 3.8752 | −0.0158 |

- The gap between the memorization sample and web validation is −0.016 to −0.017 for all six arms; no arm memorizes noticeably more.
- The BPC vocabulary trend (16K < 24K < 32K) matches the primary metric's direction, but on prediction the differences do not exceed seed noise.

## Not yet done

- The plan asks each arm to report "the rate at which the model's Top-1 prediction appears verbatim in the training data". That needs a search of the P1 pool and LCCC for the candidate strings; not done this time, to be reported later.
- The desktop INT8 measurements are dropped (200M and larger sizes are not deployed on mobile; plan deviation 3).
- The 800M gate (200M default arm against v1) is met on all three conditions; recorded in the [roadmap](../../benchmarks/predict-zh-plan.md).

## How the rules were followed

- Development only; no `--split blind`; no blind items read.
- The checkpoint is the preregistered final checkpoint, not picked among the three seeds by results.
- Benchmark files and per-item results are not in the repository; this page contains only statistics, no items, candidates or answers.
- The scorers and decision rules were frozen before scoring and not changed afterwards.
