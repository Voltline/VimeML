# Seed-ensemble replication results (second use of v2 blind, 2026-11-05)

The [V4.4 ablation](../training-v44.md) found post hoc on development that an ensemble of the four V4.2-recipe seeds (V4.2 epoch 20 and V4.4 `s1`–`s3` epoch 20), formed by summing their in-context log-probabilities, lifts Standard from a single-seed mean of 513 to 535. This report replicates that effect on Standard v2 blind and the homophone set, both used once before by [confirmation v2](confirmation-v2-results-20261021.md).

- The plan [`ime-ensemble-replication-v2.json`](../../configs/ime-ensemble-replication-v2.json) and script [`replicate_ensemble_v2.py`](../../scripts/benchmarks/replicate_ensemble_v2.py) were committed and pushed before the sets were opened (`87fbd33`). The ensemble rule, members, primary hypothesis and decision rule are fixed in the plan.
- `s1`–`s3` first reproduced their recorded rankings on v1 development (728/728 each) before the two sets were scored. V4.2's scores reuse the record files of confirmation v2, with their SHA-256 pinned in the plan.
- Scoring ran once, locally, FP32 with TF32 off. The consumption record is a separate `ensemble-replication.json`; the existing `consumption.json` is unchanged.
- **Set status**: this is the second use of both sets. After the first use no model or ensemble rule was tuned on them, so this is a genuine out-of-sample replication. Per the plan, however, the conclusion is descriptive evidence only; a deployment decision based on it still needs a new blind set.
- Only summary numbers are published. Output is local in `outputs/ensemble-replication-v2-20261105/`.

## Deviation

After scoring, the report step failed on the homophone set: 71 items (73 at n50) have a left context longer than the 128-token window, so those rows have no in-context scores. On these items every single model already falls back to AzooKey order (`fallbacks.lm_context_sum = context_length`); the four members share one tokenizer, so they fall back on exactly the same items. With the fix, the ensemble also falls back to AzooKey order on these items, and the members' fallbacks are checked for agreement. The fix changed only the report code; nothing was rescored. Standard has no fallbacks, so the primary hypothesis is unaffected. `summary.json` records the SHA-256 of the fixed report script.

## Decisions

| Hypothesis | Contrast | Track (n20) | Exact gain/loss (net) | Author-cluster p | Cluster bootstrap 95% | Decision |
| --- | --- | --- | --- | ---: | --- | --- |
| **R1** | V4.2 → four-seed ensemble | Standard v2 blind | 30/9 (+21) | **0.0012** | [10, 33] | **Replicated** |
| R2 | V4.2 → four-seed ensemble | Homophones | 17/10 (+7) | 0.25 | [−3, 17] | None |

Notation-lenient Top-1 drops on neither track, so the guard does not trigger: Standard +11 (p = 0.042), homophones +7 (p = 0.21).

**R1 replicates.** On development the ensemble beat V4.2 by 12 items (p = 0.076). On a blind set that took part in no selection it wins by 21, author-cluster p = 0.0012, with an interval excluding 0. The gain appears in both sources: WRIME 372 → 386, JMultiWOZ 226 → 233.

## Per model

Standard v2 blind (729 items, 62 author clusters; 677 items in the n20 pool, 52 pool misses for every model):

| Model | Exact | Notation-lenient | WRIME /486 | JMultiWOZ /243 | Semantic errors | Notation errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AzooKey first | 516 | 567 | 309 | 207 | 65 | 96 |
| V4.2 epoch 20 | 598 | 639 | 372 | 226 | 18 | 61 |
| `s1` epoch 20 | 595 | 637 | 370 | 225 | 14 | 68 |
| `s2` epoch 20 | 591 | 636 | 371 | 220 | 17 | 69 |
| `s3` epoch 20 | 587 | 638 | 365 | 222 | 18 | 72 |
| **Four-seed ensemble** | **619** | **650** | **386** | **233** | 11 | 47 |

Homophone challenge (619 items, 320 homophone groups; 526 items in the n20 pool):

| Model | Exact | Notation-lenient | More frequent | Less frequent | Both correct /299 groups | Semantic errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AzooKey first | 305 | 319 | 194 | 111 | 58 | 184 |
| V4.2 epoch 20 | 438 | 452 | 248 | 190 | 160 | 65 |
| `s1` / `s2` / `s3` | 432 / 433 / 431 | 445 / 446 / 449 | 246 / 247 / 246 | 186 / 186 / 185 | 155 / 151 / 150 | 72 / 67 / 69 |
| **Four-seed ensemble** | **445** | **459** | 248 | 197 | 162 | 57 |

## Descriptive results

- **Seed band.** The four single seeds score 587–598 on Standard blind (SD 4.8) and 431–438 on homophones (SD 3.1), against 501–523 (SD 9.5) on development. V4.2 is the best single seed on both sets.
- **Ensemble against each single seed (exact; author-cluster / homophone-group cluster p):**

  | Ensemble vs | V4.2 | `s1` | `s2` | `s3` |
  | --- | --- | --- | --- | --- |
  | Standard n20 | +21 (0.0012) | +24 (0.011) | +28 (0.0001) | +32 (0.0001) |
  | Standard n50 | +23 (0.0025) | +28 (0.0036) | +30 (0.0004) | +37 (0.0001) |
  | Homophones n20 | +7 (0.25) | +13 (0.028) | +12 (0.038) | +14 (0.017) |
  | Homophones n50 | +10 (0.089) | +15 (0.015) | +17 (0.006) | +19 (0.003) |

- **Ensemble size** (mean over all subsets, Standard n20 exact Top-1): 1 member 592.8, 2 members 609.2, 3 members 616, 4 members 619. On development: 513 → 524 → 528.5 → 535. Gains diminish with size; going from 3 to 4 members adds only about 3 items.
- **Where the gain comes from.** On Standard, semantic errors fall from 18 to 11 (single seeds 14–18) and notation errors from 61 to 47 (single seeds 61–72). On homophones almost all of the gain is on the less frequent word (190 → 197); the more frequent word is unchanged (248). This matches the development explanation: a single 15M model has unstable preferences among close candidates, and averaging cancels part of that.
- **n50 pool.** Standard 617, homophones 452; same conclusions as n20.

## Conclusions

1. The ensemble's Standard gain was not picked out of development: on a set that took part in no selection it beats the deployed V4.2 by 21 items, significant by author cluster, with no notation-lenient drop. On written-text homophones the direction is the same but not significant.
2. The ensemble needs four times the parameters and compute and cannot be deployed directly in the keyboard extension. The next step, as planned in the [V4.4 report](../training-v44.md), is to distill the ensemble into a 15M student ([V4.5 plan](../training-v45-plan.md)). This replication sets the distillation target: +21 over V4.2 on Standard blind.
3. Both sets have now been used twice. V4.5 model selection uses development only; confirming a deployment needs a new blind set.
