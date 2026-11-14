# V4.4 single-factor ablation results

Six arms trained and evaluated locally under the [preregistered plan](training-v44-plan.md) (commit `a2c4f07`, pushed before training). Three conclusions:

1. **None of the three treatment arms meets the preregistered "effective" standard.** Removing chat (`chat0`) is clearly worse; raising chat to 10% or prefix crop to 0.60 brings no gain. V4.2 epoch 20 stays deployed, and the pretraining recipe search at 15M / 20 epochs is closed.
2. **Seed noise is large, and the deployed V4.2 happened to draw the best seed.** Rerunning the same recipe with other seeds gives Standard exact Top-1 between 501 and 523 (SD 9.5), and V4.2's 523 is the highest of the four points. Every earlier single-seed comparison on Standard (V2.1, V4.3, V5, WSD) falls within this range.
3. **Exploratory finding (not preregistered)**: summing the log-probabilities of the four same-recipe seeds into an ensemble reaches 535 on Standard and is no worse than the best single seed on any of the four tracks. Turning this variance into a deployable gain is more promising than changing the recipe further.

## Runs and provenance

- **Hardware.** Each arm trained on its own GPU, the same GPU type as V4.2.
- **Code and environment.** torch 2.7.0+cu128, code at `a2c4f07`. The uploaded source matches the repository byte for byte.
- **Data maps.** `control` was rebuilt first on the training machines: its first 16 epochs match V4.1 and all 20 match V4.2. All six arms' mixture manifests reproduce the SHAs pinned in the configs.
  - Before the check, the V4.2 mixture directory was found to have been extended to 50 epochs by the WSD experiment, so it was restored to the frozen 20 epochs. The 30 extra maps were moved to a backup directory, not deleted. They must be restored before `train-v42-50ep.toml` is rerun.
- **W&B.** The six runs recorded offline and were later synced to `voltline233/vimeml`, all in the group `v44-ablation`.
- **HF archive.** Outputs are in the private HF dataset `Voltline/vimeml-v44-training-output`, revision `a77fd281…`. Each arm uploaded only its epoch 20 weights, per-epoch records, map manifests and source, 246 files in all, every one checked against the arm's `SHA256SUMS`. Receipt in `outputs/training-v44-20261105/download-receipt.json`. Intermediate checkpoints were kept only on the training machines.

## Training and BPC

| Run | Updates | Update time | LM BPC (epoch 20) | Chat BPC (epoch 20) |
| --- | ---: | ---: | ---: | ---: |
| V4.2 | 1,081,942 | 3.21 h | 3.045152 | 3.122060 |
| `s1` | 1,081,944 | 3.23 h | 3.048590 | 3.114598 |
| `s2` | 1,081,940 | 3.17 h | 3.045774 | 3.123937 |
| `s3` | 1,081,937 | 3.17 h | 3.050546 | 3.115061 |
| `chat0` | 984,700 | 2.92 h | **3.028382** | 4.334714 |
| `chat10` | 1,189,984 | 3.44 h | 3.074296 | 3.148587 |
| `crop60` | 1,077,365 | 3.01 h | 3.077708 | 3.148474 |

- **Seed band.** The four seeds' LM BPC range is 0.0054 (SD 0.0025); chat BPC range 0.0093.
- **Every treatment arm's BPC lies outside the seed band.**
  - `chat0` has the lowest web LM BPC but never saw chat text, so its chat BPC reaches 4.33.
  - `chat10`'s chat BPC is actually worse than at 5% (3.149 against a seed band of 3.115–3.124). The chat corpus has only about 5.8M tokens, so at 10% it repeats about 11 times per epoch, and the extra repetition only overfits.
  - `crop60` truncates more than half of its training windows, and both LM and chat BPC are worse on the untruncated validation sets.
- **BPC and reranking decouple again.** `chat0` has the lowest LM BPC of any run at V4.2 scale and the lowest Standard Top-1.

## IME results (preregistered)

Scored exactly as in the V4.2 evaluation: FP32, TF32 off, frozen n-best 20 pools, epoch 20 only for each arm.

| Model | Standard /728 | Lenient | Legacy /500 | AJIMEE /200 | Historical dev /137 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V4.2 epoch 20 | 523 | 594 | 370 | 139 | 123 |
| `s1` | 501 | 583 | 370 | 138 | 122 |
| `s2` | 518 | 586 | 377 | 137 | 121 |
| `s3` | 511 | 590 | 378 | 136 | 124 |
| **Seed band** (mean ± SD, range) | 513.3 ± 9.5 (501–523) | 588.3 ± 4.8 (583–594) | 373.8 ± 4.4 (370–378) | 137.5 ± 1.3 (136–139) | 122.5 ± 1.3 (121–124) |
| `chat0` | 495 | 577 | 374 | 139 | 127 |
| `chat10` | 501 | 582 | 376 | 141 | 125 |
| `crop60` | 514 | 583 | 371 | 140 | 125 |

**Paired tests between seeds.** On Standard, among the six same-recipe seed pairs, V4.2 → `s1` gives 27 gains / 49 losses (net −22), author-cluster p = 0.019, per-item McNemar p = 0.015. `s1` → `s2` is net +17, p = 0.057. So rerunning the same recipe with another seed can produce a "significant" difference under the author-cluster test. That test handles only the variance from item sampling, not from the training seed, so a single-seed p < 0.05 cannot show that two recipes differ.

**Seed noise is mostly notation.** On Standard the 98 pool misses are the same for every model. The four seeds have 14–22 semantic errors and 90–107 notation errors (case variants included). The lenient SD is only half the exact SD: seeds disagree mainly by flipping between acceptable notation variants.

**Treatment-arm decisions** (primary track Standard exact Top-1, two-sided):

| Arm | Standard | Net vs V4.2 | Author-cluster p | Holm p | Outside seed range | Lenient regression (p < 0.05) | Decision |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `chat0` | 495 | −28 | 0.0013 | 0.0040 | Yes (below 501) | Standard (−17, p = 0.017) | Not effective: worse |
| `chat10` | 501 | −22 | 0.013 | 0.027 | No (equals lower bound) | None | Not effective |
| `crop60` | 514 | −9 | 0.29 | 0.29 | No | None | Not effective |

- **These differences are inflated.** They are computed against V4.2 at the top of the seed band. Against the seed mean, `chat0` is −18 (about 1.9 SD), `chat10` −12 and `crop60` +1.
- **Only `chat0` meets both conditions, and in the worse direction.** Removing chat hurts the conversational Standard set, consistent with its chat BPC. Even so, the drop only just exceeds the seed band.
- **The other tracks are within noise.** No treatment arm leaves the seed band on Legacy or AJIMEE; the smoke check did not flag.

Scripts: [`evaluate_v44.py`](../scripts/benchmarks/evaluate_v44.py) (per-arm scoring) and [`report_v44.py`](../scripts/benchmarks/report_v44.py) (preregistered analysis); output locally in `outputs/evaluation-v44-20261105/`.

## Earlier single-seed comparisons revisited

| Model (single seed) | Standard | vs V4.2-recipe seed band (501–523) | AJIMEE | vs seed band (136–139) |
| --- | ---: | --- | ---: | --- |
| V2.1 | 505 | Inside | 144 | Above |
| V4.3 epoch 30 | 519 | Inside | 143 | Above |
| V5 epoch 30 | 514 | Inside | 145 | Above |
| WSD V4.2-50ep | 515 | Inside | 140 | Just above |
| WSD V5-50ep | 506 | Inside | 143 | Above |

- **Standard development.** Every recipe difference since V2.1 falls inside the V4.2 recipe's own seed band. "V4.2 is best" looks more like a lucky seed draw than an advantage of the recipe. The deployment choice still holds: no recipe has been shown to beat V4.2.
- **v2 blind.** In [confirmation v2](benchmarks/confirmation-v2-results-20261021.md), V2.1 → V4.2 was +22, also a single-seed result. The SD of the difference between two single runs is about √2 × 9.5 ≈ 13, so that difference is only about 1.6 SD.
- **AJIMEE.** This written-text track has a narrow seed band (SD 1.3), and V2.1, V4.3 and V5 are all 4–6 items above its top. This matches the v2 confirmation finding that a larger model improves written-text homophones (S3). Written-text differences exceed seed noise; conversational ones do not.

## Exploratory: seed ensemble (not preregistered)

The idea came only after seeing the results above and uses the same development tracks, so it is an observation only.

Method: for each item, sum the four seeds' in-context log-probabilities (full-vocabulary suffix log-probability sum) for each candidate and rerank, with ties keeping AzooKey order. No model has to be rerun ([`ensemble_v44.py`](../scripts/benchmarks/ensemble_v44.py)).

| Ensemble size | Standard (mean, range) | Legacy | AJIMEE | Historical dev |
| --- | --- | --- | --- | --- |
| 1 (single seed) | 513.3 (501–523) | 373.8 | 137.5 | 122.5 |
| 2 (6 combinations) | 524.3 (521–529) | 379.8 | 143.2 | 122.7 |
| 3 (4 combinations) | 528.5 (527–530) | 380.5 | 145.3 | 122.8 |
| 4 | **535** | 380 | 143 | 123 |

- **Author-cluster tests of the four-seed ensemble against each single seed on Standard:**
  - vs V4.2: +12 (p = 0.076)
  - vs `s1`: +34 (p < 0.001)
  - vs `s2`: +17 (p = 0.026)
  - vs `s3`: +24 (p = 0.001)
- **Notation-lenient.** The ensemble scores 599 and regresses on no track.
- **Fewer errors.** Standard semantic errors fall to 12 (single seeds 14–22) and notation errors to 83 (single seeds 90–107).
- **Against a larger model.** An ensemble of two 15M models (30M) averages 524, above the single V5 of similar size (28M) at 514. On AJIMEE it trails V5 (143 vs 145).

This suggests that a 15M model often has no clear preference among competing notation variants, with different seeds leaning different ways; averaging cancels part of that variance. But an ensemble multiplies parameters and compute by the number of members. Memory and latency in the keyboard extension have never been measured, so the ensemble itself cannot be deployed yet.

## Next steps

1. **Close the pretraining recipe search.** At this scale and budget, chat fraction, prefix crop, schedule and extra web data have produced no gain beyond seed noise.
2. **Compress the ensemble gain into one 15M model.** Use the four seeds' averaged distribution as teacher and distill a 15M student (provisionally V4.5) with the same deployment cost as V4.2.
   - Train at least 3 student seeds so they can be compared with the seed band.
   - Preregister before training, as here.
   - The teachers are the four existing epoch 20 weights on HF; nothing needs retraining.
3. **Before distilling, replicate the ensemble effect on the used v2 blind and homophone sets.** This only reranks existing scores. Both sets were used once in the confirmation scoring, so this is a second use that needs a committed plan, with descriptive results only. This step needs the owner's approval. A final confirmation still needs a new blind set. (Added after the report: done, see the [ensemble replication results](benchmarks/ensemble-replication-v2-results-20261105.md): Standard blind +21, p = 0.0012, replicated; the distillation plan is [V4.5](training-v45-plan.md).)
4. **Rules for later comparisons.** On Standard, differences under about 20 items between single seeds are not interpreted. Recipe comparisons need at least 3 seeds per side, or compare ensembles directly. AJIMEE and Legacy have small seed noise and can be reported on their own.

## Limits of the conclusions

- All IME results come from development sets used many times. The ensemble analysis was proposed after the preregistered analysis and also run on these sets.
- The seed band has only 4 points; the SD estimate itself is uncertain, with only 3 degrees of freedom.
- Each treatment arm changes one variable at one value only. `chat10` may be worse only because of excessive repetition; more non-repeated chat data might still help but was not tested.
- Treatment arms differ from V4.2 in update count (`chat0` about 9% fewer, `chat10` about 10% more), without compensation.
- Memory and latency in the keyboard extension are still unmeasured, so ensemble cost is compared by parameter count only.
