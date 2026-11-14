# V4.2 LM/chat pretraining evaluation

2026-10-20. V4.2 completed 20 epochs and 1,081,942 updates normally. The last epoch, preselected by full LM validation BPC, reaches **523/728 (71.84%)** on the fixed Standard development set, 18 more than V2.1 and 14 more than V4.1's best development weights. The old web set and AJIMEE are 4 and 5 lower than V2.1, so the result is not yet a consistent improvement across all regression sets.

The training records, actual source and `best-lm.pt` were downloaded: 18 files, 183,647,412 bytes, checked against a pinned Hugging Face revision, remote file identities and the release checksums. Raw records are kept in `outputs/training-v42-20261020/`; the four local FP32 rerank sets, pairings and acceptance records in `outputs/evaluation-v42-20261020/`. No test/blind data was read, no references adjusted and nothing retrained.

## Experiment identity

Source: [Voltline/vimeml-v42-training-output](https://huggingface.co/datasets/Voltline/vimeml-v42-training-output), pinned revision `99d914882ae7410bbd99d08090b7d59b3becbded` (after the 2026-10-24 cleanup the history was squashed and that revision no longer exists; the `best-lm.pt` hash is unchanged and is now at `9b0830d5…`).

| Item | Record |
| --- | --- |
| Model | 8K, context 128, d384, 7 layers, 6 heads, SwiGLU 960, 15,070,848 parameters |
| Initialization | From random weights, seed 1219827974 |
| Objective | Plain LM on mixed web/sentence and chat text; V4.1's conversion objective and structured input removed |
| Data | V4.1's LM and chat databases, exclusion masks, cropping and 5% chat supervised-token quota |
| Training | BF16, compiled backbone, fused AdamW, batch 512, 20 epochs |
| Learning rate | Full 20-epoch cosine, 6e-4 → 6e-5, warmup 1,000 |
| Other settings | Weight decay 0.1, betas (0.9, 0.95), clip 1.0, prefix crop 0.30 |
| Completed exposure | 12,104,389,994 LM supervised targets, mixed chat targets included |
| Device / runtime | Single GPU, PyTorch 2.7.0+cu128 |
| Status | `complete`, step 1,081,942, 20 epochs |

The local V4.1 manifest matches the remote training plan on the two databases, exclusion masks, seed, cropping and chat quota; the release hashes of the first 16 epochs' LM maps match one by one. The model definition, factory and V4 inference module used locally also match the remote plan's source hashes. These are record-consistency checks; the training corpus was not rebuilt or scanned locally.

Against V4.1, the conversion objective was removed and the learning-rate budget changed from 16 to 20 epochs at the same time. Against V4-B, chat mixing, cropping, seed and exposure also changed. This run supports the complete recipe as effective, but cannot show that all of the gain comes from removing READ/conversion.

The remote `training_seconds=11564.9709` is about 3.21 hours. The source accumulates only update-phase time into this field, with validation and saving outside the timed section, so it is not the job's wall-clock time.

## Full validation trend

From the remote full FP32 validation records, with the same denominator in every epoch. Plain LM: 11,110,489 characters, 6,627,568 supervised targets; chat: 110,814 characters, 68,562 supervised targets. BPC was recomputed as `nll_sum / ln(2) / characters`.

| Epoch | Step | Full LM BPC ↓ | Full chat BPC ↓ |
| --- | ---: | ---: | ---: |
| 1 | 54,098 | 3.331562 | 3.300638 |
| 4 | 216,389 | 3.261786 | 3.259810 |
| 8 | 432,771 | 3.212882 | 3.221593 |
| 12 | 649,163 | 3.147780 | 3.201754 |
| 16 | 865,557 | 3.082979 | 3.146115 |
| 18 | 973,749 | 3.053533 | 3.125537 |
| 19 | 1,027,844 | 3.049784 | **3.120972** |
| 20 | 1,081,942 | **3.045152** | 3.122060 |

The best full LM is the last epoch; `best-lm.pt`'s step, config, signature and full epoch history match the release record. The last epoch still lowers LM BPC by about 0.00463, while chat rises by about 0.00109 over epoch 19. This small divergence cannot be read as every ability still improving together, nor as proof that more epochs always help.

FP32 results on the same original LM validation text:

| Model / weights | Full LM BPC ↓ |
| --- | ---: |
| V2.1 extend5 step40000 | 3.080191 |
| V4-B step584000 | **3.027777** |
| V4.1 epoch 13 | 3.164086 |
| V4.1 epoch 16 | 3.121224 |
| V4.2 epoch 20 | 3.045152 |

V4.2 is about 1.14% lower than V2.1 and about 2.44% lower than V4.1's last epoch, still about 0.57% above V4-B. V4.2's final chat BPC is also below V4.1's final 3.172254 and its epoch-15 best of 3.153930. Chat validation was not rerun for V2.1/V4-B, so no cross-model conclusion is drawn about their chat ability.

## Fixed candidate-pool reranking

Before looking at the local IME results, the evaluation was fixed to `best-lm.pt`, the best full LM BPC; no other epoch was searched on these four sets. Fixed input SHA256, candidates, order, references and scoring semantics are reused: full-vocabulary log-probability sum after the jointly tokenized common prefix, no final EOS, no truncation, stable ties. Local RTX 5060 Laptop, uncompressed FP32, TF32 off.

| Set | AzooKey | V2.1 | V4-B | V4.1 epoch 13 | V4.1 epoch 16 | V4.2 epoch 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard development /728 | 453 | 505 | 498 | 509 | 505 | **523** |
| Legacy regression /500 | 317 | 374 | **377** | 370 | 376 | 370 |
| AJIMEE /200 | 87 | **144** | **144** | 132 | 137 | 139 |
| Historical development /137 | 111 | 122 | 120 | **124** | **124** | 123 |

Standard accuracy rises from V2.1's 69.37% to 71.84%, **+2.47 points**. Both sources improve: WRIME 311/483, 13 more than V2.1's 298; JMultiWOZ 212/245, 5 more than 207.

Against AzooKey, V4.2 fixes 110 Standard items and breaks 40, net +70; V2.1's net is +52. Standard Top-5 is 624/728 and MRR 0.774969, against V2.1's 618/728 and 0.759341. None of the four sets had length fallbacks; all denominators are kept.

Per-item pairing against V2.1:

| Set | Gain | Loss | Net | Exact McNemar two-sided p |
| --- | ---: | ---: | ---: | ---: |
| Standard | 46 | 28 | +18 | 0.047393 |
| Legacy | 15 | 19 | −4 | 0.607591 |
| AJIMEE | 8 | 13 | −5 | 0.383310 |
| Historical | 4 | 3 | +1 | 1.000000 |

On Standard, against V4-B: 50 gains, 25 losses, net +25; against V4.1 epoch 13: 43 and 29, net +14. The p values above are exploratory, uncorrected for multiple sets, models and the history of repeated experiments. Standard's borderline p must not be written up as passing an independent test, and the absence of significant differences elsewhere must not be written up as no regression.

The fixed Standard set has been used repeatedly in earlier experiments; Legacy is an old-web regression diagnostic; AJIMEE is public and exposed; Historical contains synthetic/AI-assisted labels. The four are reported separately and not pooled as independent samples. The independent benchmark under construction is still needed.

## Context and judgement

With its left context, V4.2 scores 523/728 on Standard; with the left context cleared, 510/728. The whole gain comes from the 407 items that have a left context (281 vs 268). AJIMEE: 139 with context, 130 without; Legacy 370 and 369; Historical 123 and 124. Context helps, but the effect varies by set.

The structured READ path was not evaluated; V4.2 was not trained on that objective. No new formal prediction evaluation was added, and no iOS deployment test was exported or run.

This run produces a stronger plain-LM reranking candidate than V4.1: the main development set improves, and plain LM and chat validation improve again. The recommendation is to freeze epoch 20 and let an independent benchmark decide whether it replaces V2.1 (the later fixed-reference result is in the [reference and regression analysis](reference-regressions.md)), keeping the negative records on the old web set and AJIMEE. The current data is not enough to recommend switching the official deployment directly, nor to show that reading conditioning as a research direction is ineffective in general.

## Review outputs

- Local weights: `outputs/training-v42-20261020/remote/checkpoints/best-lm.pt`. The other epoch and last weights were deleted from the private repository on 2026-10-24.
- Raw records: `outputs/training-v42-20261020/remote/records/`; download identities and per-file checks in `download-receipt.json`.
- Evaluation plan: `outputs/evaluation-v42-20261020/plan.json`; a single checkpoint chosen by full LM BPC.
- Per-item scores, metrics and pairings against four older models: `outputs/evaluation-v42-20261020/<track>/`.
- Acceptance: `outputs/evaluation-v42-20261020/acceptance.json`. The 18 downloaded files, complete 20-epoch records, finite values in 10,819 update log lines, checkpoint identity, all fixed inputs/candidates/references and scoring denominators passed.
- Local execution sources: `outputs/training-v42-20261020/evaluate_local.py`, `audit_local.py`; the evaluation script's SHA256 is recorded in the evaluation plan. The downloaded remote training source was only checked statically and never executed.
