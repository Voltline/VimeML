# V4.4 single-factor ablation (preregistered)

Registered 2026-11-04 and committed before any arm started training. V4.4 is not a new recipe but six parallel runs of the V4.2 recipe: three change only the seed, three change one data variable each. It has two goals: measure for the first time how much IME reranking varies when the same recipe is rerun with another seed, and test against that noise band whether the chat fraction and prefix crop are worth changing further. Results: [V4.4 ablation results](training-v44.md) (link added after registration; the text keeps the registered wording).

## Why

- V4.2 epoch 20 is the deployed version, but its leads over V4.3, V5 and WSD are all within noise (for example WSD V4.2-50ep scores 515 vs 523 on Standard). Every one of these comparisons has a single seed and run-to-run variation has never been measured, so "within noise" has no scale.
- The [error decomposition](reports/analysis-20261021/ime-errors.md) shows that of V4.2's 205 Standard errors, 98 are pool misses, 71–90 notation, and only 17 in-pool semantic; lowering BPC further has failed to bring a rerank gain seven times. What pretraining might still affect is mainly notation, which depends on the register of the training text (chat fraction) and the context-length distribution (prefix crop).
- WSD has already been tried with no IME gain ([WSD runs](training-wsd.md)) and is not repeated.

## Six arms

The shared recipe is exactly V4.2's: tiny_gpt_v4 d384/L7 (15,070,848 parameters), 8K vocabulary, 20 epochs, cosine 6e-4 → 6e-5 (1,000 warmup), batch 512, BF16 + compile, fused AdamW (wd 0.1, clip 1.0), the V4.2 data pools and exclusion masks. Each arm changes only one variable from the table and has its own 20-epoch mixture map ([`prepare_v44.py`](../scripts/corpus/prepare_v44.py)).

| Arm | Seed | Chat fraction | Prefix crop | Config | Mixture manifest SHA-256 |
| --- | ---: | ---: | ---: | --- | --- |
| V4.2 (existing, not retrained) | 1219827974 | 0.05 | 0.30 | [`train-v42.toml`](../configs/train-v42.toml) | `b1db1c1f…` |
| `s1` | 137218780 | 0.05 | 0.30 | [`train-v44-s1.toml`](../configs/train-v44-s1.toml) | `d375bd20…` |
| `s2` | 299822247 | 0.05 | 0.30 | [`train-v44-s2.toml`](../configs/train-v44-s2.toml) | `6a713a9b…` |
| `s3` | 2097573425 | 0.05 | 0.30 | [`train-v44-s3.toml`](../configs/train-v44-s3.toml) | `38f8d943…` |
| `chat0` | 1219827974 | **0** | 0.30 | [`train-v44-chat0.toml`](../configs/train-v44-chat0.toml) | `3bb2e433…` |
| `chat10` | 1219827974 | **0.10** | 0.30 | [`train-v44-chat10.toml`](../configs/train-v44-chat10.toml) | `b08df1a9…` |
| `crop60` | 1219827974 | 0.05 | **0.60** | [`train-v44-crop60.toml`](../configs/train-v44-crop60.toml) | `26dada4c…` |

- **Seed arms change the whole seed**: initialization, crop decisions, chat sampling and data order all change, so they measure real run-to-run noise. The three treatment arms also change data order, because they change the crop or chat quota, so they compare directly with the seed arms. The three seeds were drawn at once with `secrets.randbelow(2**31)` and lie far apart, so the `seed + epoch` chat sampling streams do not overlap.
- **The chat corpus has only about 5.8M tokens**: at 5% it already repeats 5.2 times per epoch; `chat10` about 11 times per epoch, about 220 times over 20 epochs. 15% would reach about 350 repetitions and mostly measure overfitting, so it is not used.
- **Update counts differ somewhat**: chat windows are added on top of web windows and are shorter, so the per-epoch update count, counted in windows, changes more than the token fraction. Epoch 1 windows: V4.2 27,697,799, `chat0` 25,208,203 (about −9%), `chat10` 30,463,978 (about +10%), `crop60` 27,579,336. No compensation; reported as is.
- **Mixture build check**: `control` rebuilds the map with V4.2's three values; its first 16 epochs match the frozen V4.1 map byte for byte (checked locally 2026-11-04), and wherever the V4.2 mixture directory exists, all 20 epochs must match. The six arms' manifests are built locally and their SHAs written into the configs; a rebuild on the training machines must reproduce them byte for byte, or training does not start. To keep the SHAs independent of the operating system, `request.json` and `manifest.json` always use LF line endings.

Training entry point [`train_v44.py`](../scripts/training/train_v44.py), with launch scripts for one arm and for all six. W&B does not use any machine login: credentials come only from `WANDB_API_KEY` in a private file, and each arm's W&B config, cache and netrc point to its own directory; without credentials, or if online initialization fails, it records offline and runs `wandb sync` later. The six runs share the group `v44-ablation`.

## Evaluation (fixed before training ends)

- **Checkpoint**: epoch 20 of each arm, not chosen by BPC or any IME result. Only epoch 20 (optimizer state removed) and the per-epoch records are uploaded to HF.
- **Scoring**: exactly as in the V4.2 evaluation: local FP32, TF32 off, full-vocabulary suffix log-probability sum, joint tokenization of left context and candidate, ties keep AzooKey order, frozen n-best 20 pools. Four non-blind tracks: Standard /728, Legacy /500, AJIMEE /200, historical development /137. No blind or test set is read. Statistics follow [metrics v2](benchmarks/metrics-v2.md).
- **Noise band**: four points, V4.2 epoch 20 and `s1`–`s3`; for each track, mean, range and standard deviation (n − 1) of exact and notation-lenient Top-1; for the six seed pairs, author-cluster sign-flip p and per-item McNemar (descriptive: how often two runs of the same recipe come out "significant").
- **Treatment-arm decision** (`chat0`, `chat10`, `crop60`; primary track Standard /728 exact Top-1, two-sided). "Exploratory effective" requires all three:
  1. Top-1 outside the range of the four seed points;
  2. an author-cluster sign-flip test against V4.2 epoch 20, Holm-corrected across the three treatment arms, family α = 0.05;
  3. the metrics v2 guard: no notation-lenient Top-1 regression with p < 0.05 (author-cluster test) on Standard, Legacy or AJIMEE.
- **Smoke**: flagged when exact Top-1 on historical development /137 is 3 or more below V4.2; not used for comparison.
- **Descriptive reports**: exact Top-1, pool-conditional Top-1 and per-item McNemar on Legacy and AJIMEE (for comparison with v1 reports), LM and chat validation BPC (per-epoch curves), and the dose-response direction `chat0` → V4.2 → `chat10`. These take no part in the decision.

## How the conclusions are used

- Both blind sets were consumed in [confirmation v2](benchmarks/confirmation-v2-results-20261021.md), so this round's development conclusions can only be exploratory. If a treatment arm is effective, the next step is to build a new blind set and preregister a confirmation on it; only after confirmation could that arm be deployed as V4.4.
- If no treatment arm is effective, V4.2 epoch 20 stays deployed, "changing the pretraining recipe" is closed at this scale and budget, and work moves to candidate pool coverage (48% of Standard errors).
- Whatever the outcome, the seed noise band is applied back to the reading of the earlier single-seed comparisons (whether V4.3, V5 and WSD differ from V4.2 by more than the band).
