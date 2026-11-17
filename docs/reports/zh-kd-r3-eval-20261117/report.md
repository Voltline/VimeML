# Development evaluation of the round 3 v2-20m distillation arms (2026-11-17)

This report applies the round 3 decision rule of the [distillation plan](../../training-zh-kd-plan.md) to four arms: `r3-ctrl`, `r3-data`, `r3-alpha` and `r3-plain`. They are compared with:

- `kd-cont`, the candidate chosen in round 2;
- `vimeml-tiny-zh-v1` in LM mode.

Every evaluation uses development only; blind and the public test split were not read. Pinyin generation, Standard and prediction come from `cluster_eval_zh.sh`. Self-rescoring was computed with `pinyin_gen_zh.py reprior`, using each arm's own final.pt as the prior. The `kd-cont` and v1 numbers come from the [round 2 evaluation](../zh-kd-eval-20261117/report.md), with the same backend and the same inputs.

## Conclusions

- **No arm replaces `kd-cont`, which stays the `vimeml-zh-v2-20m` candidate.**
  - The primary metric is public-set Top-1 after self-rescoring (λ = 1.6):
    - `r3-plain` 54.8, `r3-data` 54.7 and `r3-ctrl` 54.6 all tie `kd-cont` at 54.8 (p ≥ 0.69);
    - `r3-alpha` scores 53.5, significantly below `kd-cont` (−1.4, p = 0.016).
  - The second replacement condition is prediction track A Top-3 reaching v1's 27.6. No arm reaches it; the best is `r3-plain` at 26.5.
  - The guard is Standard full-pinyin rerank. All four arms tie v1 (net +1 to +9, p ≥ 0.19).
- **With the 200M as teacher, the 20M has stopped improving.**
  - `r3-ctrl` trains 30,000 more updates on the `kd-cont` recipe, and the primary metric does not move (−0.2, p = 0.69).
  - Adding r2 data (`r3-data` vs `r3-ctrl`) and adding plain samples (`r3-plain` vs `r3-data`) leave it unchanged too.
- **α = 0.9 hurts.** Fitting the teacher more closely lowers both generation order and rescoring significantly (−1.2 and −1.3 against `r3-data`).
- **Prediction rises slightly but stays below v1.** Track A Top-3 goes from 25.6 to 25.8–26.5, highest for `r3-plain`, still 1.1 below v1.
- **Next steps:**
  - Round 4 uses the `kd-cont` recipe (r1 only, α = 0.5, T = 1) and changes only the teacher, to the 800M.
  - The two typo-plan arms (with and without `typo`) start from `kd-cont`.

## Setup

- **Arms** (details in round 3 of the distillation plan):
  - Each starts from the `kd-cont` final checkpoint and trains 30,000 updates.
  - Peak LR 1e-4 → 1e-5, warmup 1000, seed `20261117`.
  - The teacher is the 12K joint-pinyin 200M, T = 1.

  | Arm | Data per update | α | final.pt sha256 |
  | --- | --- | ---: | --- |
  | `r3-ctrl` | r1 only: plain 160 / full 96 / mixed 128 / initials 96 / error 32 | 0.5 | `b4b771f5…` |
  | `r3-data` | each kind half r1, half r2, same totals | 0.5 | `f07d377c…` |
  | `r3-alpha` | as `r3-data` | 0.9 | `cc3d3160…` |
  | `r3-plain` | as `r3-data`, but plain 256 (r1 128 + r2 128) | 0.5 | `7cd2ce8d…` |

- **Pinyin generation**: public set `ime-pinyin-zh-v1` development, 3,444 inputs.
  - Bucketed beam search, beam 16, with in-field context, task mode.
  - The top 10 candidates are reranked by genScore + λ × prior, with λ = 1.6 and the model itself as the prior.
- **Paired tests**: sentence-cluster sign-flip, 20,000 permutations.
- **Standard**: `ime-standard-zh-v1-r1` development, full-pinyin rerank, 1,456 items.
- **Prediction**: `ime-predict-zh-v1-r1` development, tracks A to D.

## Results

### Pinyin generation (public development, Top-1 %)

| Model | Generation order | Self-rescored | Full | Mixed | All initials | Top-10 coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 (LM mode) | 52.7 | — | 83.6 | 51.7 | 25.6 | 68.5 |
| kd-cont | 44.2 | **54.8** | 85.8 | **53.7** | 28.2 | 70.0 |
| r3-ctrl | **44.6** | 54.6 | 86.7 | 53.0 | **28.7** | **70.4** |
| r3-data | 43.7 | 54.7 | 87.0 | 53.5 | 27.2 | 69.9 |
| r3-alpha | 42.5 | 53.5 | 86.0 | 52.0 | 26.6 | 67.9 |
| r3-plain | 43.2 | **54.8** | **87.4** | 53.1 | 28.5 | 69.3 |

The three breakdown columns are generation order for v1 and rescored for every other row.

Paired comparisons (Top-1, rescored; generation order on the v1 side):

| Comparison | Diff | Wins/losses | p |
| --- | ---: | --- | ---: |
| r3-ctrl − v1 | +1.9 | 290/224 | 0.063 |
| r3-data − v1 | +2.1 | 296/225 | 0.047 |
| r3-alpha − v1 | +0.8 | 253/226 | 0.44 |
| r3-plain − v1 | +2.1 | 291/219 | 0.038 |
| r3-ctrl − kd-cont | −0.2 | 81/89 | 0.69 |
| r3-data − kd-cont | −0.1 | 99/102 | 0.91 |
| r3-alpha − kd-cont | −1.4 | 77/124 | 0.016 |
| r3-plain − kd-cont | −0.1 | 110/112 | 0.96 |
| r3-data − r3-ctrl | +0.1 | 85/80 | 0.81 |
| r3-alpha − r3-data | −1.3 | 57/101 | 0.008 |
| r3-plain − r3-data | 0.0 | 58/57 | 1 |

In generation order:
- `r3-alpha` is 1.2 below `r3-data` (p = 0.014) and 1.7 below `kd-cont` (p = 0.005).
- No other arm differs significantly from `kd-cont` (−1.0 to +0.3).

### Standard full-pinyin rerank (development, 1,456 items)

| Model | Wins/losses vs v1 | Net | McNemar p | Cluster sign-flip p | Cluster bootstrap 95% |
| --- | --- | ---: | ---: | ---: | --- |
| kd-cont | 17/21 | −4 | 0.627 | 0.627 | [−16, +8] |
| r3-ctrl | 17/16 | +1 | 1.0 | 1.0 | [−10, +12] |
| r3-data | 20/18 | +2 | 0.871 | 0.874 | [−10, +14] |
| r3-alpha | 23/14 | +9 | 0.188 | 0.191 | [−3, +21] |
| r3-plain | 20/17 | +3 | 0.743 | 0.745 | [−9, +15] |

All four arms tie v1, so the guard passes.

### Prediction (development, descriptive)

| Model | A Top-1 | A Top-3 | A Top-5 | B savings | D first 20% | D first 50% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 16.5 | 27.6 | 33.2 | 16.7% | 28.9 | 21.9 |
| kd-cont | 15.2 | 25.6 | 31.0 | 15.7% | 26.3 | 20.3 |
| r3-ctrl | 15.1 | 25.8 | 30.9 | 15.7% | 25.8 | 19.8 |
| r3-data | 15.3 | 26.0 | 31.2 | 15.9% | 26.3 | 20.1 |
| r3-alpha | 15.5 | 26.2 | 31.3 | 15.8% | 26.9 | 20.6 |
| r3-plain | **15.6** | **26.5** | **31.3** | **16.0%** | **27.0** | 20.4 |

The arms are within one point of each other, and all are below v1.

### Validation at the end of training (r1 validation set, loss)

| Arm | plain | full | mixed | initials | error |
| --- | ---: | ---: | ---: | ---: | ---: |
| r3-ctrl | 4.240 | **0.246** | **0.802** | **1.407** | **0.528** |
| r3-data | 4.235 | 0.248 | 0.806 | 1.414 | 0.529 |
| r3-alpha | 4.253 | 0.265 | 0.837 | 1.453 | 0.546 |
| r3-plain | **4.194** | 0.249 | 0.808 | 1.413 | 0.531 |

`r3-plain` has the lowest plain loss, consistent with its lead on prediction, but the margin does not change the decision.

## Consequences for v2

- The `vimeml-zh-v2-20m` candidate is still `kd-cont`, scored by its own rescoring with λ = 1.6.
- The release-notes line that prediction is below v1 stays.
- Round 4 uses the `kd-cont` recipe and changes only the teacher (the 800M). r2 data is not added, because with the 200M as teacher it brought no gain.
- The typo-plan arms start from `kd-cont`.
- The private benchmark was not run for round 3.
