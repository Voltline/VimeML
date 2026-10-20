# V4.2 LM/chat pretraining

V4.2 completed twenty epochs and 1,081,942 updates on 2026-10-20. It trained the fixed 8K, 15,070,848-parameter V4 architecture from random weights with the V4.1 seed **1219827974**, on the frozen V4.1 LM and chat pools with the conversion objective and structured READ input removed. V2.1 remains the deployed model. [Detailed training record (Chinese)](zh-CN/training-v42.md) · [Full evaluation (Chinese)](zh-CN/reports/v42-20261020/evaluation.md).

## Data and objective

| Pool | Frozen population | Role |
| --- | --- | --- |
| Original V2.1 FineWeb/Tatoeba, encoded with V4 8K | V4.1 retained windows and exclusion mask | Traverse retained LM windows once per epoch |
| V3 RealPersonaChat/MRMP | V4.1 chat bank and dialogue-level splits | Approximately 5% of actual cropped LM target tokens, with repeated sampling |
| C1r2 conversion | — | Removed |

[`prepare_v42.py`](../scripts/corpus/prepare_v42.py) derives the mixture from the frozen V4.1 request: identical joined banks, exclusion masks, seed, 30% prefix crop and 5% chat quota, with the conversion keys removed and the budget raised from 16 to 20 epochs. Epoch maps 0–15 must reproduce the V4.1 maps byte-identically; the script refuses otherwise. No test or blind data is read.

The objective is ordinary unmarked next-token loss over all valid targets, normalized by the total supervised tokens per update. Each epoch has about 27.7 million windows and 54,095–54,098 updates at batch 512; the final partial batch is retained. Total exposure is 553,949,647 windows and 12,104,389,994 supervised targets, including mixed chat.

## Training settings

| Setting | Value |
| --- | --- |
| Configuration | [train-v42.toml](../configs/train-v42.toml) |
| Environment | One NVIDIA L20C; PyTorch 2.7.0+cu128 |
| Precision and runtime | BF16 AMP, compiled backbone with dynamic shapes, eager valid-token head and CE |
| Optimizer | Fused AdamW, betas (0.9, 0.95), weight decay 0.1, clip 1.0 |
| Schedule | 1,000 warmup updates; cosine over all twenty epochs, 6e-4 → 6e-5 |
| Validation | Subset every 2,000 updates; complete primitive-FP32 LM and chat BPC after each epoch |
| Checkpoints | Every 5,000 updates, every epoch, `best-lm.pt` and `last.pt` |

No IME scoring runs during training; epoch checkpoints are kept for post-hoc candidate-pool evaluation. The recorded `training_seconds` of 11,565 (about 3.21 hours) sums update phases only and excludes validation and saving, so it is not job wall-clock time.

## Results

Final complete LM BPC is **3.045152** at epoch 20, the best epoch, versus V2.1 **3.080191**, V4.1 epoch 16 **3.121224** and V4-B **3.027777** on the same original validation text. Best chat BPC is **3.120972** at epoch 19; final chat BPC is **3.122060**.

`best-lm.pt` (epoch 20) was selected on complete LM BPC before any local IME scoring. All local comparisons use the same frozen pools, primitive FP32, disabled TF32, full-vocabulary suffix log-probability sums, stable ties, and no truncation or final EOS.

| Metric | V2.1 | V4-B | V4.1 epoch 13 | V4.1 epoch 16 | V4.2 epoch 20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard development /728 | 505 | 498 | 509 | 505 | **523** |
| Legacy regression /500 | 374 | **377** | 370 | 376 | 370 |
| AJIMEE /200 | **144** | **144** | 132 | 137 | 139 |
| Historical development /137 | 122 | 120 | **124** | **124** | 123 |

Against V2.1, Standard improves 46 cases and regresses 28 (+18; descriptive exact McNemar p=0.047), with gains in both WRIME (+13) and JMultiWOZ (+5). Legacy is −4 (15/19) and AJIMEE −5 (8/13). These p-values are uncorrected for repeated sets, models and experiments; Standard has been reused across experiments and is not an independent test.

A preregistered fixed-reference evaluation (plan `81512f147d3749ef9a36e61470fce173`, registered before scoring, run once) gives **541/725** (WRIME 315, JMultiWOZ 226) versus V2.1 535 and V3 A 546: +6 against V2.1 with 40 improvements and 34 regressions (p=0.56). The 725 cases were consumed in the original V2.1/V3 A comparison, so this is a fixed reference, not a new blind test; the historical 1,000-case blind set remains unscored. [Reference and regression analysis (Chinese)](zh-CN/reports/v42-20261020/reference-regressions.md).

Of all 210 Top-1 flips against V2.1 across the five sets, 61% are notation changes of the same word (kana/kanji/okurigana) where the frozen references list one form. V4.2 shifts toward hiragana (82 versus 51 flips); this costs Legacy formal web text 14 cases while fixing 7, which accounts for its −4. AJIMEE losses are mostly genuine encyclopedic homophone errors (11 of 13). Meaning-level gains on Standard development (+13) shrink to +3 on the reference.

Clearing left context lowers Standard from 523 to 510 and AJIMEE from 139 to 130, so context contributes, though unevenly across sets. Relative to V4.1, V4.2 removes the conversion objective and lengthens the schedule together, so the improvement is not attributable to removing READ alone.

## Source and artifacts

| Component | Path |
| --- | --- |
| Mixture derivation | `scripts/corpus/prepare_v42.py` |
| Formal trainer | `scripts/training/train_v42.py`, `src/vimeml/training/joint_v42.py` |
| Cluster launcher | `scripts/training/cluster_v42.sh`; set `VIMEML_PYTHON` for the training environment |
| Local comparison | `scripts/benchmarks/evaluate_v42.py`; requires a separate empty output directory |
| Download receipt, records and `best-lm.pt` | `outputs/training-v42-20261020/` |
| Frozen local evaluation and acceptance | `outputs/evaluation-v42-20261020/` |

The trainer, entry point and configuration are byte-identical to the archived run source and match its recorded `code_sha256`. Only the launcher's site-specific interpreter default was replaced by `VIMEML_PYTHON`. The evaluation entry preserves the archived local script's scoring logic and adds an explicit output argument; the completed evaluation retains its original script hash. Other epoch checkpoints and `last.pt` remain in the private training-output repository.

V4.2 is a sound ordinary-LM reranker and is not worse than V2.1 on the fixed reference, but the evidence does not support replacing V2.1: the reference gain is small and not significant, the main differences are notation style, and AJIMEE shows a few real encyclopedic regressions. No Core ML export or device validation has been performed.
