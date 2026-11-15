# Prediction benchmark step 0: scaling survey

2026-11-15. Step 0 of the [prediction benchmark and desktop roadmap](../../benchmarks/predict-zh-plan.md) measures existing systems on the development split of `ime-predict-zh-v1-r1`, to see how prediction scores change with model size.

**Schedule.** The step was meant to come before 200M training. It was deferred because the development machine was too slow (deviation 4) and was later run on the training machines. Systems, decoding and adapter rules are unchanged (deviation 3).

**What this report adds.** The 800M deferral condition was already checked once track A was in. This report adds the full results for all four tracks.

The registered text is the Chinese version, [docs/zh-CN/reports/zh-predict-step0-20261115/survey.md](../../zh-CN/reports/zh-predict-step0-20261115/survey.md).

## Systems

| System | Parameters | Precision | Context | Version |
| --- | ---: | --- | ---: | --- |
| Character 5-gram (fixed sample of `export-r1/train`, interpolated Kneser-Ney) | — | — | 4 characters | — |
| `vimeml-tiny-zh-v1` | 20M | FP32 | 128 | release |
| zh-200m default arm `24k-c512-s1` | 200M | FP32 | 512 | 2026-11-13 training record |
| Qwen2.5-0.5B (base) | 0.49B | FP32 | 512 | `060db649` |
| Qwen2.5-1.5B (base) | 1.5B | BF16 | 512 | `8faed761` |
| Qwen2.5-3B (base) | 3.1B | BF16 | 512 | `3aab1f19` |

**Not in the original step-0 table:**

- **librime's prediction plugin is skipped.** There is no public prediction database for it (deviation 3).
- **zh-200m is added for comparison.** Its scores come from the 200M judgement.

**Track A rerun check.** The three Qwen models reran track A along with the other tracks. All 7,200 ranks match the earlier track-A-only run item by item.

## Results

All values are %.

| System | A Top-1 | A Top-3 | A Top-5 | C first syllable Top-3 | C initials Top-3 | B keystroke savings | B click rate | D @20% | D @50% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Character 5-gram | 5.92 | 10.89 | 13.58 | 11.08 | 32.37 | 5.30 | 4.46 | 9.24 | 7.08 |
| v1 (20M) | 16.47 | 27.60 | 33.17 | 18.54 | 70.08 | 16.74 | 13.37 | 28.89 | 21.92 |
| zh-200m | 19.71 | 31.64 | 37.28 | 21.37 | 73.00 | 20.62 | 16.21 | 34.86 | 25.75 |
| Qwen2.5-0.5B | 26.00 | 39.87 | 45.40 | 24.29 | 79.67 | 27.72 | 21.78 | 44.03 | 34.42 |
| Qwen2.5-1.5B | 28.82 | 44.01 | 50.78 | 25.42 | 84.42 | 31.08 | 24.58 | 50.76 | 38.19 |
| Qwen2.5-3B | 31.21 | 47.06 | 53.33 | 28.04 | 86.42 | 32.89 | 26.13 | 53.96 | 40.69 |

**Track B threshold.** The plan picks the threshold from a grid. All six systems chose "no threshold": they always show predictions.

### Paired comparisons

**How to read the tables:**

- **Tracks A and C** use a cluster sign-flip test and a cluster bootstrap, clustered by dialogue. The bootstrap interval is in items.
- **Track B** uses a dialogue bootstrap over 2,400 dialogues. The interval is in points.

**Tracks A and C (Top-3):**

| Track | Comparison | Difference (points) | Gains/losses | Cluster p | Bootstrap 95% (items) |
| --- | --- | ---: | --- | ---: | --- |
| A | 1.5B vs 0.5B | +4.14 | +590/−292 | 5e-5 | [+241, +357] |
| A | 3B vs 1.5B | +3.04 | +472/−253 | 5e-5 | [+165, +272] |
| A | 3B vs 0.5B | +7.18 | +765/−248 | 5e-5 | [+454, +579] |
| A | v1 vs 0.5B | −12.28 | +289/−1,173 | 5e-5 | [−956, −810] |
| A | zh-200m vs 0.5B | −8.24 | +366/−959 | 5e-5 | [−663, −521] |
| C first syllable | 3B vs 0.5B | +3.75 | +116/−26 | 5e-5 | [+67, +113] |
| C initials | 3B vs 0.5B | +6.75 | +194/−32 | 5e-5 | [+134, +191] |
| C first syllable | v1 vs 0.5B | −5.75 | +37/−175 | 5e-5 | [−166, −111] |
| C initials | v1 vs 0.5B | −9.58 | +108/−338 | 5e-5 | [−271, −189] |

**Track B (keystroke savings):**

| Comparison | Difference (points) | Bootstrap 95% (points) |
| --- | ---: | --- |
| 1.5B vs 0.5B | +3.36 | [+3.13, +3.60] |
| 3B vs 1.5B | +1.81 | [+1.60, +2.02] |
| 3B vs 0.5B | +5.17 | [+4.91, +5.42] |
| v1 vs 0.5B | −10.97 | [−11.37, −10.58] |
| zh-200m vs 0.5B | −7.10 | [−7.44, −6.75] |

## Reading under the plan

- **800M is not deferred.**
  - On track A Top-3, 3B is 7.18 points above 0.5B, and the cluster test is significant.
  - That is far above the 2-point deferral line.
  - It was decided on track A alone on 2026-11-14; all four tracks agree.
- **All four tracks rise monotonically with size.**
  - Each step from 0.5B to 1.5B to 3B is significant on A and B.
  - The steps shrink: A Top-3 rises +4.14, then +3.04; savings rise +3.36, then +1.81.
- **v1 is below Qwen2.5-0.5B.**
  - The plan says to note it if v1 is not below 0.5B. That does not hold: v1 is significantly lower on every track (A Top-3 −12.28).
  - Under the plan, this does not change the 200M decision.
- **External models show the trend only.** Qwen's training data is unknown and may include these public dialogue sets, so its scores are not a target line.

## Observations outside the plan

**zh-200m vs Qwen2.5-0.5B.**

- zh-200m closes only about a third of the gap between v1 and 0.5B: A Top-3 goes from 27.60 to 31.64, against 39.87 for 0.5B.
- 0.5B has 2.5 times the parameters, but the two differ by orders of magnitude in training tokens, and their data sources differ.
- So the gap cannot be put down to size alone, and Qwen having seen the test dialogues cannot be ruled out.

**Track C.**

- In initials mode, every neural model is above 70%, and the gaps are smaller than on track A.
- First-syllable mode is the hardest: even 3B reaches only 28%.

**Track B threshold.**

- In this simulation, showing predictions every time is always best: only three candidates are shown, and a wrong one costs nothing.
- This does not mean a real keyboard needs no confidence threshold.

## Files

Run outputs stay local in `outputs/benchmarks/ime-predict-zh-v1/runs/predict-qwen/` and are not committed. Each model's directory holds:

- `meta.json`;
- `score.json`;
- per-item records for the three decoded tracks.
