# WSD schedule runs (V4.2-50ep, V5-50ep)

Two runs replace the cosine schedule of V4.2 and V5 with WSD (Warmup-Stable-Decay), to test whether the schedule alone, with data and model unchanged, can lower BPC further, and whether lower BPC brings an IME gain. Status: both runs completed 50 epochs and were evaluated locally. Conclusion: LM BPC reaches the best value yet at both scales, but IME Top-1 stays within noise. V4.2 epoch 20 stays deployed; no Core ML conversion or device measurement.

## Design

| Item | Cosine (V4.2 / V5-30ep) | WSD (this experiment) |
| --- | --- | --- |
| Peak lr | 6e-4 | 3e-4 |
| Schedule | 1,000 warmup, then cosine to 6e-5 throughout | 1,000 warmup → constant 3e-4 to 85% → cosine to 6e-5 over the last 15% (`stable_decay`) |
| Epochs | 20 / 30 | 50 |

Everything else is identical to the controls: data pools, seed 1219827974, batch 512, BF16, fused AdamW (wd 0.1, clip 1.0), prefix crop 0.30, 5% chat, architecture.

| Run | Config | Parameters | Data | Updates | stable_steps | Training time (single GPU) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| V4.2-50ep | [`train-v42-50ep.toml`](../configs/train-v42-50ep.toml) | 15,070,848 | V4.2 pool, 50-epoch maps | 2,704,835 | 2,299,110 | 8.1 h |
| V5-50ep | [`train-v5-50ep.toml`](../configs/train-v5-50ep.toml) | 28,385,792 | V4.3 pool, 50-epoch maps | 5,376,440 | 4,569,974 | 19.2 h |

Decay starts at epoch 42.5 in both. The originally planned V4.2-30ep is identical to V4.2-50ep before epoch 25.5 (same data and lr); it was stopped midway and left no outputs.

**Same data order.** The 50-epoch mixture requests differ from the original 20-epoch (V4.2) and 30-epoch (V4.3) requests only in the `epochs` field; each epoch's map is determined by seed + epoch, and per-epoch window counts in overlapping epochs are equal one by one. So in overlapping epochs WSD and cosine see the same data each epoch and can be compared epoch by epoch. The 50-epoch map build script was not uploaded with the runs (the repository's `prepare_v42.py` / `prepare_v43.py` hard-code 20 / 30 epochs); the dataset identities are `8d62f498…` (V4.2) and `e6df0f46…` (V5).

**The basis for 3e-4 is weak.** It was chosen from the "per-epoch BPC change bucketed by lr" of earlier cosine runs, but per-epoch changes are driven mainly by data order: the per-epoch LM changes of V4.3-30ep and V5-30ep correlate at 0.98, and in this experiment WSD and the matching cosine run, with completely different schedules over the overlapping epochs, still correlate at 0.81–0.87. The so-called "stall" in the 6e-4 range (V5 epochs 4–5) also appears in the same epochs of V4.3; the rebound at epoch 29 appears both in cosine at lr 6e-5 and in WSD at lr 3e-4. Whether 3e-4 is optimal cannot be answered by these two runs.

## Provenance

Run outputs are in the private HF dataset `Voltline/vimeml-wsd-training-output` (revision `9bc0f378…`): V5-50ep at the root, V4.2-50ep under `v42-50ep/`. After the 2026-10-24 evaluation the repository kept only epochs 42 and 50 and the stable-phase fork points (V5 epoch 25, V4.2 epochs 17 and 25, matching same-budget decays of 30 / 20 / 30 epochs); other checkpoints were deleted and the history squashed, now `ab78c2c2…`. Records, source and the epoch 42 and 50 checkpoints of both runs were downloaded locally, all checked against the remote `SHA256SUMS`; receipt in `outputs/training-wsd-20261024/download-receipt.json`.

- The recorded `code_sha256` covers 90 files. `joint_v42.py` (relaxed to at most 50 epochs, records the schedule name) and `joint_v5.py` (records the schedule name) were imported from the run source and now match the records byte for byte. `expanded_ime.py` is the version before `54d7618` (an evaluation module, not used in training). All other files match the repository.
- Both TOMLs equal the recorded configs exactly; their header comments were rewritten (the originals were copied from other configs). The V4.2 launch script gained the same `--config` parsing as V5's, so it can launch non-default configs.
- The IME line in both READMEs is identical ("best Standard/728 522 (epoch 29), AJIMEE/200 145 (epochs 20/29/30)") and matches V5-30ep cosine's numbers: it was copied, not a result of these runs. The per-epoch IME records from the training side were not uploaded either.

## Validation BPC

| Run | Updates | LM BPC | Extended web BPC | Chat BPC |
| --- | ---: | ---: | ---: | ---: |
| V4.2, 20-epoch cosine | 1,081,942 | 3.045152 | — | 3.122060 |
| V4.3, 30-epoch cosine | 3,225,888 | 3.043831 | 3.049611 | 3.128712 |
| **V4.2-50ep WSD** | 2,704,835 | **3.032766** | — | **3.106215** |
| V5 controlled, 10-epoch cosine | 1,075,287 | 2.948747 | 2.952379 | 3.084419 |
| V5, 30-epoch cosine | 3,225,888 | 2.942180 | 2.945641 | 3.108036 |
| **V5-50ep WSD** | 5,376,440 | **2.931708** | **2.936232** | **3.085076** |

Both runs reach their best LM BPC at epoch 50. The three earlier cosine runs of the 15M model all stopped at 3.044–3.049; WSD is the first to go below that plateau.

**The gap cannot all be credited to the schedule.** Both comparisons change schedule, peak lr and budget at once: V4.2-50ep has 2.5 times the updates of V4.2-20ep (0.84 times V4.3-30ep, which uses a different data pool); V5-50ep has 1.67 times V5-30ep. Extra budget gives little under cosine (V5 10→30 epochs −0.0065, V4.3 10→30 epochs −0.0052), which suggests most of this experiment's −0.010 to −0.012 comes from the schedule or peak lr; but with no same-budget cosine control and a single seed, that is only a guess.

### Curve shape

Paired same-epoch difference (WSD − cosine, LM BPC):

| Epoch | V4.2 (vs 20-epoch cosine) | V5 (vs 30-epoch cosine) |
| ---: | ---: | ---: |
| 1 | −0.050 | −0.066 |
| 5 | −0.045 | −0.070 |
| 10 | +0.004 | −0.049 |
| 15 | +0.076 | −0.007 |
| 16 | +0.091 | +0.005 |
| 20 | +0.122 | +0.045 |
| 30 | — | +0.122 |

- **The early lead only reflects the lower lr.** WSD leads by 0.05–0.07 in the first epochs because its lr is half of cosine's, giving lower instantaneous loss. Once cosine's lr drops below 3e-4 (V4.2 epoch 10, V5 epoch 16), the lead reverses, to a maximum deficit of about 0.12–0.13. The early lead does not predict the end point.
- **The stable phase reaches its floor quickly.** Over the 29 epochs from 13 to 42, V4.2 drops only 0.022 (3.1812 → 3.1589) and V5 only 0.0125 (3.0739 → 3.0614), less than 0.001 per epoch and smaller than the per-epoch fluctuation from data order (detrended residual about 0.0035). This is expected at constant lr and does not indicate a problem with 3e-4.
- **The decay recovers the whole gap.** From epoch 42 to 50 (7.5 epochs of decay), V4.2 drops 0.126 and V5 0.130. For reference, cosine from lr about 3e-4 to the end: V4.2 epochs 11 to 20 drop 0.118 (9 epochs), V5 epochs 16 to 30 drop 0.124 (14 epochs). WSD gets the same recovery from a shorter anneal. Comparing against cosine's last 15% is wrong: by then its lr is below 1e-4 and it recovers only 0.02–0.03.

### Chat domain

Chat is resampled every epoch under a 5% token quota. Cosine V5-30ep's chat BPC bottoms out at epoch 25 (3.0771), then rises to 3.1080 while LM is still falling, widening the chat–LM gap from 0.104 to 0.166. WSD does not show this: V5's chat BPC falls throughout the decay to 3.0728 at epoch 49 and rises to 3.0851 at epoch 50, an end gap of 0.153, still below cosine; V4.2's chat BPC is lowest at epoch 50. One possible explanation: WSD spends far less time at low lr (about 7.5 epochs) than cosine (lr < 3e-4 for roughly the second half), and memorization of repeated data happens mainly at low lr. The chat validation set has only 4,986 entries and its per-epoch fluctuation is also driven by data order, so this is descriptive only.

## IME Top-1 (local FP32)

Evaluation entry point [`evaluate_wsd.py`](../scripts/benchmarks/evaluate_wsd.py), with the same semantics as V4.2/V4.3/V5: full-vocabulary suffix log-probability sum, joint tokenization of context and candidate, FP32, TF32 off, stable ordering. The fixed reference, blind sets and both confirmation-v2 sets were not read. Results in `outputs/evaluation-wsd-20261024/`.

By the rule fixed in advance, the primary candidate is epoch 50, best by BPC. Epoch 42, the last full stable-phase epoch, was chosen before the local IME evaluation as an in-run control: the same training trajectory as epoch 50 with BPC about 0.13 higher, a larger BPC gap than any earlier between-model comparison. It is descriptive only.

| Model | Standard /728 | Legacy /500 | AJIMEE /200 | Historical dev /137 |
| --- | ---: | ---: | ---: | ---: |
| V2.1 | 505 | 374 | 144 | 122 |
| V4.2 epoch 20 (deployed) | 523 | 370 | 139 | 123 |
| V4.3 epoch 30 | 519 | 367 | 143 | 123 |
| V5 controlled epoch 10 | 504 | 372 | 145 | 123 |
| V5 epoch 30 | 514 | 365 | 145 | 124 |
| V4.2-50ep WSD epoch 42 (end of stable phase) | 503 | 376 | 135 | 121 |
| **V4.2-50ep WSD epoch 50 (primary)** | **515** | **378** | **140** | **126** |
| V5-50ep WSD epoch 42 (end of stable phase) | 504 | 372 | 134 | 121 |
| **V5-50ep WSD epoch 50 (primary)** | **506** | **374** | **143** | **123** |

Paired exact McNemar (net, with flips and two-sided p in parentheses):

| Comparison | Standard | Legacy | AJIMEE | Historical dev |
| --- | --- | --- | --- | --- |
| V5 WSD epoch 50 vs V5 cosine epoch 30 (same scale and data) | −8 (+36/−44, p=0.43) | +9 (+26/−17, p=0.22) | −2 (+6/−8, p=0.79) | −1 (+4/−5, p=1.00) |
| V4.2 WSD epoch 50 vs V4.2 epoch 20 (same scale and data, deployed) | −8 (+32/−40, p=0.41) | +8 (+21/−13, p=0.23) | +1 (+13/−12, p=1.00) | +3 (+6/−3, p=0.51) |
| V4.2 WSD epoch 50 vs V4.3 epoch 30 | −4 (+42/−46, p=0.75) | +11 (+20/−9, p=0.061) | −3 (+9/−12, p=0.66) | +3 (+6/−3, p=0.51) |
| V5 WSD epoch 50 vs epoch 42 (same run, decay) | +2 (+39/−37, p=0.91) | +2 (+18/−16, p=0.86) | +9 (+15/−6, p=0.078) | +2 (+4/−2, p=0.69) |
| V4.2 WSD epoch 50 vs epoch 42 (same run, decay) | +12 (+43/−31, p=0.20) | +2 (+15/−13, p=0.85) | +5 (+14/−9, p=0.41) | +5 (+7/−2, p=0.18) |

- In both same-scale, same-data schedule comparisons, no track is significant and the directions disagree.
- The decay lowers BPC by about 0.13, and all 8 in-run cells are positive (+2 to +12), but none is significant and the tracks are not independent. The decay may bring a small IME gain, no more than 1–2 points per track, which the current sets cannot detect.
- Across 88 paired comparisons of the 4 checkpoints against all baselines, 5 are nominally significant: stable-phase epoch 42 vs V4.2 epoch 20 on Standard (V4.2 run −20, p=0.027; V5 run −19, p=0.037), V5 epoch 42 vs V5 epoch 30 and V5 epoch 10 on AJIMEE (−11, p=0.019 and 0.027), and V4.2 WSD epoch 50 vs V5 epoch 30 on Legacy (+13, p=0.041). At the 0.05 level, 88 comparisons should give about 4 false positives anyway; none of the comparisons between the two primary candidates and the deployed version or the same-scale cosine runs is significant.

### Checkpoint selection

The training-side note that "both runs' best Standard is 521/728" is the maximum picked by dev score over 50 epochs, a selection bias on the dev set. In the V5 report, adjacent epochs already flip about 60 Standard items. The locally reproduced primary candidates score 506 (V5) and 515 (V4.2).

## Conclusions

1. **WSD lowers BPC but does not turn that into IME accuracy.** V4.2 reaches 3.0328 and V5 2.9317, the best at each scale; but schedule is confounded with budget and peak lr, with a single seed, so the gain cannot be attributed precisely. IME Top-1 stays within Standard 503–523 and AJIMEE 134–145, no different from the cosine runs. This is the 7th confirmation that BPC and reranking decouple, and the one with the largest BPC gap (0.13 within a run).
2. **V4.2 epoch 20 stays deployed.** Neither primary candidate is significantly better; no Core ML conversion, device measurement or new preregistered confirmation.
3. **Training lessons.** The early lead and the stable-phase stall are inherent to WSD and are no basis for tuning; compare schedules at the post-decay end point and at equal budget. WSD stable-phase checkpoints can be forked into decays, the cheapest way to build same-budget controls in future, but the current continuation mechanism only accepts cosine sources and requires an extended budget, so it needs rework.
4. **Direction unchanged.** As in the V5 conclusion, with the current "fixed AzooKey candidate pool + suffix log-probability sum" reranking, lowering pretraining BPC further is not a route to better IME.

## Limits of the conclusions

Both comparisons change schedule, peak lr and budget at once, so they only answer "is this WSD configuration better with a larger budget": yes on LM, no on IME reranking. They cannot answer whether WSD beats cosine at equal budget, or whether 3e-4 is the best peak. All IME numbers come from non-blind development and regression sets and are used only for description and negative conclusions.
