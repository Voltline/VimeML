# zh-800m arch1 evaluation (2026-11-20)

The 800M joint-pinyin model finished training and was evaluated under the rules frozen in the [800M training plan](../../training-zh-800m-plan.md) and its deviation 1. All on development; blind was not read.

**Conclusion.** The 800M is ahead of both references on every registered metric:

| Metric | 800M | Reference |
| --- | --- | --- |
| Prediction (primary metric), track A Top-3 | **0.3394** | 200M default arm 0.3164 (+0.0230, more than twice the 200M seed range of 0.0165); v1 0.2760 (+0.0634) |
| Pinyin generation, rescored Top-1 | **65.3%** | joint-pinyin 200M 57.3% (+8.0); v1 LM mode 52.6% (+12.7) |
| Rerank guard (standard full pinyin) | top 1 net +26 against v1 | passes |

All paired differences are significant. The memorization checks find no long verbatim recall: with 8 characters of context, 2.4% of its Top-1 strings occur in the training text, against 2.3% for the human continuations (verbatim section below).

Question 1 of the plan (scaling) now has three points: track A Top-3 is 0.276 for 20M, 0.316 for 200M and 0.339 for 800M. Question 2 (the 800M as distillation teacher) is answered by distillation round 4, which is running.

## Model

- Config `configs/train-zh800m-12k-c512-pinyin.toml` (arch1: 12K, d1536, L27, 784M parameters, context 512).
- Data: r1 + r2 in one pass, plus joint pinyin.
- 1,198,964 updates in 58.07 h; finished 2026-11-20.
- The evaluated checkpoint is the preregistered `final.pt`, sha256 `ec3e90df4c113f84990918a2e7d3c54cb5affd64c325100a0d4121b58f85d80f`.
- Every scoring output records this sha.

## BPC and memorization sample (20th milestone, full FP32)

| Model | Web BPC | Chat BPC | Memorization sample BPC | Sample − web |
| --- | ---: | ---: | ---: | ---: |
| 800M arch1 | 3.2451 | 5.1726 | 3.2374 | −0.0077 |
| 200M `24k-c512-s1` | 3.8641 | 5.4210 | 3.8469 | −0.0171 |

- BPC is per character, so the two tokenizers compare directly.
- The memorization sample stays below web validation by a smaller margin than the 200M's. The 800M has seen the sample documents no more than any other web text.

Final pinyin validation loss: full 0.153, mixed 0.464, initials 0.936, error 0.292.

## Prediction: `ime-predict-zh-v1-r1`, track A (7,200 items)

- Decoding is identical for all three runs: beam 16, candidates of 2–6 characters, top 5, context 512 (v1: 128, its limit), FP32.
- Paired tests use `vimeml.benchmarks.statistics.paired`, clustered by dialogue (2,400 clusters), with p from the cluster sign-flip test and Holm correction over the two primary contrasts.

| Model | Top-1 | **Top-3** | Top-5 |
| --- | ---: | ---: | ---: |
| 800M arch1 | 0.2107 | **0.3394** | 0.3929 |
| 200M `24k-c512-s1` | 0.1971 | 0.3164 | 0.3728 |
| vimeml-tiny-zh-v1 (20M) | 0.1647 | 0.2760 | 0.3317 |

| 800M against | Metric | Net (fixed/broken) | Cluster p | Holm p | 95% cluster bootstrap (items) |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 | Top-3 | +457 (+833/−376) | 5e-5 | 1e-4 | [390, 525] |
| 200M `24k-c512-s1` | Top-3 | +166 (+598/−432) | 5e-5 | 1e-4 | [102, 230] |
| v1 | Top-1 | +331 (+669/−338) | 5e-5 | – | [269, 394] |
| 200M `24k-c512-s1` | Top-1 | +98 (+486/−388) | 0.0011 | – | [39, 157] |

- Against the 200M, the Top-3 gain (+0.0230) is more than the 200M's three-seed range (0.0165), so it is not seed noise. This model has one seed (plan limitation).
- By source, Top-3 is 0.220 on CPED, 0.349 on CrossWOZ, 0.377 on KdConv and 0.412 on RiSAWOZ.

## Pinyin generation: `ime-pinyin-zh-v1-r1` development (3,444 inputs)

- Bucketed search with beam 16 per key position, rescored with the model itself as prior (λ 1.6).
- Paired tests use sentence-cluster sign-flip.

| Model | Kind | Top-1 | Top-3 | Top-10 | Rescored Top-1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 800M | all | 56.2 | 71.1 | 79.9 | **65.3** |
| 800M | full / mixed / initials | 88.6 / 54.3 / 31.5 | 97.1 / 70.5 / 47.6 | 99.3 / 80.7 / 57.4 | 91.7 / 65.0 / 40.3 |
| joint-pinyin 200M | all | 47.4 | 62.3 | 71.5 | 57.3 |
| v1 LM mode | all | 52.7 | 64.0 | 68.5 | 52.6 |

| 800M against | Order | Diff (points) | +/− | p |
| --- | --- | ---: | --- | ---: |
| v1 LM mode | generation | +3.5 | +394/−272 | 0.0021 |
| v1 LM mode | rescored | +12.7 | +585/−146 | 5e-5 |
| joint-pinyin 200M | generation | +8.9 | +498/−193 | 5e-5 |
| joint-pinyin 200M | rescored | +8.0 | +430/−153 | 5e-5 |

- The joint-pinyin 200M's 57.3 is today's rerun of the same pipeline. The plan's 58.8 came from an earlier search version.

## Rerank guard: `ime-standard-zh-v1-r1` development, full pinyin (1,456 items)

- Top-1 is 95.40% (1,389 of 1,456), Top-3 97.32%.
- Against v1, top 1 has net +26 (+37/−11): exact McNemar p 2.2e-4, cluster sign-flip p 1e-4, 95% CI [13, 39].
- The guard passes.

## Typo correction (descriptive; not a registered metric)

`ime-typo-zh-v1-r1` development, LM display rule, calibrated with `typo_zh calibration` (λ 1.6, prior self):

| Model | Margin | typo@2 | Clean false display |
| --- | ---: | ---: | ---: |
| 800M | 3.5048 | 34.08 | 9.0% |
| joint-pinyin 200M | 3.6705 | 34.43 | 8.65% |
| 20M kd-cont | 3.8559 | 33.48 | 9.52% |

Typo correction does not grow with model size. This agrees with the closed typo line: the gap is sentence-level correction, not model capacity.

## Top-1 predictions found verbatim in the training data

Method as in the [200M decision](../zh-200m-20261114/judgement.md#top-1-predictions-found-verbatim-in-the-training-data), `scripts/benchmarks/verbatim_predict_zh.py`:

- Each Top-1 candidate is prefixed with the last k Han characters of the prefix and searched verbatim in `export-r1/train`.
- The reference is the human continuation of the same length.
- The 800M also trained on export-r2, which is not searched here. Its rates cover the r1 part of its data.

| Model | k = 4 (3,966 items) model / human | Difference (cluster p) | k = 8 (1,369 items) model / human | Difference (cluster p) |
| --- | --- | --- | --- | --- |
| 800M | 64.3 / 49.2 | +15.1 (5e-5) | 2.4 / 2.3 | +0.1 (1) |
| 200M `24k-c512-s1` | 62.1 / 48.3 | +13.8 (5e-5) | 2.0 / 2.3 | −0.4 (0.33) |
| v1 (20M) | 64.3 / 49.3 | +15.0 (5e-5) | 1.8 / 2.4 | −0.6 (0.14) |

At k = 0 every model is at 100%, which does not discriminate.

Paired between models, on the same items:

| 800M against | k = 4 | k = 8 |
| --- | --- | --- |
| 200M | +2.2 (+322/−236, p 0.0002) | +0.4 (+12/−6, p 0.24) |
| v1 | 0.0 (+286/−288, p 0.97) | +0.6 (+10/−2, p 0.039) |

- **k = 4.** The 800M sits at v1's level, 2.2 points above the 200M. This is the usual pull of a language model towards frequent phrasing, which every size shows (+14 to +15 over human text).
- **k = 8.** The 800M finds 2.4% of its strings in the training text, against 2.3% for the human continuations (p 1). That is no sign of long verbatim recall. The +0.6 against v1 rests on 10 against 2 items.

## How the rules were followed

- Development only; no blind items read.
- The checkpoint is the preregistered final checkpoint.
- The evaluation files (sha256 above in every output) were read from the training run's upload, not rescored here.
  - The paired prediction tests and the verbatim search were computed locally from the per-item `track-a.jsonl`.
  - The 200M and v1 per-item files are the ones behind the [200M decision](../zh-200m-20261114/judgement.md).
- Benchmark files and per-item results are not in the repository. This page contains only statistics.
