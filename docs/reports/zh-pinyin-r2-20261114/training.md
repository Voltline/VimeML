# zh three-task round 2: training record (2026-11-14)

This is a translation; the Chinese file [zh-CN/reports/zh-pinyin-r2-20261114/training.md](../../zh-CN/reports/zh-pinyin-r2-20261114/training.md) is the primary record.

**Round 1** (`configs/train-zh-pinyin-s1-*.toml`, v1 data) was trained and evaluated earlier.

**Round 2 changes:**

- It switches to the v2 data, whose readings come from Vime's Rime dictionaries.
- It raises the share of pinyin samples.
- It runs two arms: one continues from round-1 `s1-task`, one restarts from the 12K base.

This page records only training facts and validation curves. For the prediction, rerank and generation results see the [evaluation](evaluation.md).

## Data

`artifacts/training-data/zh-pinyin-v2-12k` holds 33 files:

- five stores (plain / full / mixed / initials / error) of six files each;
- `manifest.json`, `preview.jsonl` and `run.json`.

**Integrity checks:**

- The configs pin the manifest SHA256 to `c449439b…05cea472`.
- At start-up the trainer verifies the SHA256 of every data file and exits on any mismatch.

## The two arms

| Arm | Start | Updates | Learning rate | Warmup | Normalization |
| --- | --- | ---: | --- | ---: | --- |
| r2-cont | Round-1 `pinyin-s1-task` `final.pt` (step 30000, `788e9fbd…b7d87a`) | 60,000 | 1.5e-4 → 1.5e-5 (cosine) | 1,000 | `per_kind` |
| r2-base | 12K base `12k-fresh-s1` `final.pt` (`44c423a1…27542d4`) | 90,000 | 3e-4 → 3e-5 (cosine) | 2,000 | `per_kind` |

**Mix.** Both arms use the same mix. Each update has 512 samples:

| Store | Samples per update | Share |
| --- | ---: | ---: |
| plain | 160 | 31% |
| full | 96 | 19% |
| mixed | 128 | 25% |
| initials | 96 | 19% |
| error | 32 | 6% |

**Other hyperparameters (same as round 1):**

| Group | Settings |
| --- | --- |
| Architecture | d_model 384, 6 heads, 9 layers, d_ff 960, SwiGLU, RMSNorm |
| Input | context 128, vocabulary 12,288 |
| Optimization | bf16, weight decay 0.1, β 0.9/0.95, clip 1.0, prefix crop 0.30 |
| Schedule | validation every 2,000 steps; `last.pt` every 1,000 steps; `step-*.pt` every 6,000 steps |

## Store repetition

The v2 stores are smaller than v1 and the runs are longer, so both arms repeat the pinyin stores. Passes per store, computed from the manifest's train sample counts and the mix:

| Arm | plain | full | mixed | initials | error |
| --- | ---: | ---: | ---: | ---: | ---: |
| r2-cont (60k) | 1.26 | 2.50 | **3.32** | 2.50 | 2.47 |
| r2-base (90k) | 1.88 | 3.75 | **4.99** | 3.75 | 3.70 |

These passes help explain the shape of the curves below:

| Store | r2-cont | r2-base |
| --- | ---: | ---: |
| mixed | about 3.3 passes | about 5.0 passes |
| initials | about 2.5 passes | about 3.7 passes |

At this level of repetition, both arms' validation loss is still falling at the last point, with no upturn (see "Are the curves still falling?").

The validation set comes from shards separate from the training set, and every evaluation recomputes it in full.

## Smoke runs before training

**What ran.** Before the real runs, each arm ran 200 updates on a temporary config copy. `smoke_updates` was set only in that copy and was not committed.

**What the smoke runs confirmed:**

- Training starts.
- It writes `last.pt` / `final.pt` / `step-*.pt`, `evaluations.json` and `metrics.jsonl`.
- `step-*.pt` carries the full `tokenizer_identity`:
  - `tokenizer_model_sha256`;
  - `vocab_size = 12288`;
  - `special_ids = {pad:0, unk:1, bos:2, eos:3}`.
- **The continued arm starts where round 1 ended.** Its smoke run reproduced round 1's final values at step 0:

  | plain | full | mixed | initials | error |
  | ---: | ---: | ---: | ---: | ---: |
  | 4.1872 | 0.646 | 0.278 | 0.144 | 0.414 |

## Training time

| Arm | Per step | Training only | Wall clock | Interruptions / resumes |
| --- | ---: | ---: | ---: | --- |
| r2-cont | 0.0381 s | 2,285.7 s (38.1 min) | 41 min 03 s | None |
| r2-base | 0.0375 s | 3,371.0 s (56.2 min) | 1 h 00 min 21 s | None |

The two arms trained in parallel. Each finished its 60,000 / 90,000 updates in one go:

- no pre-emption;
- no restart;
- no resume from `last.pt`.

`metrics.jsonl` step numbers are continuous.

## Validation results

Final evaluation point:

| Arm | plain loss | full loss / tf_exact | mixed loss / tf_exact | initials loss / tf_exact | error loss / tf_exact |
| --- | ---: | --- | --- | --- | --- |
| Round-1 s1-task (reference) | 4.278 | 0.314 / 0.654 | 0.933 / 0.306 | 1.592 / 0.146 | 0.671 / 0.413 |
| r2-cont (60k) | **4.288** | 0.255 / 0.690 | 0.832 / 0.331 | 1.453 / 0.177 | 0.557 / 0.478 |
| r2-base (90k) | **4.353** | 0.253 / 0.695 | 0.826 / 0.336 | 1.462 / 0.175 | 0.556 / 0.474 |

**Cross-round comparisons are only a rough guide,** because the two rounds use different validation data (v1 vs v2):

- r2-cont's step-0 plain loss on the v2 validation set is 4.262. Round 1's 4.278 was measured on the v1 set.

**Within-arm change in plain loss:**

| Arm | Start | End | Change |
| --- | ---: | ---: | ---: |
| r2-cont | 4.262 | 4.288 | +0.026 |
| r2-base | 4.167 | 4.353 | +0.186 |

Curves below are at steps 0, 2000, 4000, 6000 and every 6,000 steps after that. The full every-2,000-step series is in each arm's `evaluations.json`.

**r2-cont**

| step | plain | full l/tf | mixed l/tf | initials l/tf | error l/tf |
| ---: | ---: | --- | --- | --- | --- |
| 0 | 4.2620 | 0.3095/0.646 | 0.9353/0.278 | 1.5582/0.144 | 0.6458/0.414 |
| 2000 | 4.4398 | 0.3638/0.594 | 1.0258/0.251 | 1.6925/0.119 | 0.7228/0.375 |
| 4000 | 4.4648 | 0.3630/0.588 | 1.0170/0.261 | 1.6899/0.125 | 0.7199/0.380 |
| 6000 | 4.4726 | 0.3612/0.595 | 1.0125/0.253 | 1.6840/0.129 | 0.7097/0.388 |
| 12000 | 4.4794 | 0.3502/0.601 | 0.9933/0.265 | 1.6628/0.136 | 0.6895/0.395 |
| 18000 | 4.4617 | 0.3323/0.618 | 0.9681/0.263 | 1.6211/0.135 | 0.6657/0.419 |
| 24000 | 4.4373 | 0.3171/0.637 | 0.9447/0.280 | 1.5951/0.143 | 0.6455/0.418 |
| 30000 | 4.4081 | 0.3000/0.639 | 0.9195/0.284 | 1.5679/0.151 | 0.6217/0.427 |
| 36000 | 4.3712 | 0.2869/0.659 | 0.8921/0.304 | 1.5305/0.152 | 0.6040/0.445 |
| 42000 | 4.3411 | 0.2747/0.677 | 0.8655/0.311 | 1.4939/0.162 | 0.5873/0.448 |
| 48000 | 4.3135 | 0.2633/0.684 | 0.8474/0.320 | 1.4677/0.168 | 0.5688/0.469 |
| 54000 | 4.2964 | 0.2567/0.692 | 0.8352/0.323 | 1.4573/0.173 | 0.5600/0.483 |
| 60000 | 4.2884 | 0.2547/0.690 | 0.8316/0.331 | 1.4529/0.177 | 0.5570/0.478 |

**r2-base**

| step | plain | full l/tf | mixed l/tf | initials l/tf | error l/tf |
| ---: | ---: | --- | --- | --- | --- |
| 0 | 4.1669 | 5.8562/0.000 | 5.7315/0.000 | 5.6434/0.000 | 5.8139/0.000 |
| 2000 | 4.7866 | 0.7893/0.363 | 1.5418/0.147 | 2.2183/0.071 | 1.2291/0.212 |
| 4000 | 4.8404 | 0.5878/0.450 | 1.3454/0.182 | 2.0571/0.086 | 0.9928/0.278 |
| 6000 | 4.8142 | 0.5272/0.487 | 1.2645/0.190 | 1.9741/0.092 | 0.9282/0.296 |
| 12000 | 4.7583 | 0.4506/0.535 | 1.1674/0.219 | 1.8647/0.105 | 0.8389/0.339 |
| 18000 | 4.7106 | 0.4161/0.557 | 1.1042/0.231 | 1.8001/0.115 | 0.7607/0.369 |
| 24000 | 4.6769 | 0.3980/0.571 | 1.0706/0.245 | 1.7721/0.122 | 0.7606/0.360 |
| 30000 | 4.6443 | 0.3734/0.590 | 1.0474/0.250 | 1.7364/0.127 | 0.7339/0.367 |
| 36000 | 4.6083 | 0.3568/0.607 | 1.0095/0.261 | 1.6994/0.135 | 0.7071/0.393 |
| 42000 | 4.5737 | 0.3432/0.612 | 0.9854/0.269 | 1.6617/0.137 | 0.6830/0.401 |
| 48000 | 4.5376 | 0.3171/0.634 | 0.9545/0.275 | 1.6360/0.142 | 0.6531/0.401 |
| 54000 | 4.5053 | 0.3043/0.643 | 0.9280/0.289 | 1.5892/0.146 | 0.6319/0.436 |
| 60000 | 4.4662 | 0.2921/0.660 | 0.9068/0.303 | 1.5610/0.157 | 0.6147/0.435 |
| 66000 | 4.4309 | 0.2797/0.664 | 0.8779/0.305 | 1.5321/0.160 | 0.5969/0.452 |
| 72000 | 4.4020 | 0.2640/0.682 | 0.8576/0.325 | 1.5040/0.160 | 0.5780/0.465 |
| 78000 | 4.3796 | 0.2597/0.685 | 0.8419/0.328 | 1.4838/0.167 | 0.5704/0.469 |
| 84000 | 4.3617 | 0.2560/0.693 | 0.8303/0.332 | 1.4718/0.169 | 0.5616/0.476 |
| 90000 | 4.3531 | 0.2529/0.695 | 0.8259/0.336 | 1.4621/0.175 | 0.5558/0.474 |

## The plain bump near warmup

**Background.** Round 1 had a plain-loss spike at step 2000. This round lowered the peak learning rate to 1.5e-4 and lengthened warmup to 1,000 steps. The bump shrank but did not go away, and the two arms differ.

**Bump relative to step 0:**

| Arm | Step 0 | Step 2000 | Peak | End |
| --- | ---: | ---: | --- | --- |
| r2-cont | 4.2620 | +0.178 | +0.219 at steps 8000–10000 | +0.026 at step 60000 |
| r2-base | 4.1669 | +0.620 | +0.674 at step 4000 | +0.186 at step 90000 |

After the peak, both arms fall monotonically; r2-base keeps falling for a long time.

**Effect of the lower learning rate (r2-cont):**

- The bump is about half of round 1's (+0.360).
- The peak moves from step 2000 to around step 10,000.

**The restarted arm (r2-base):**

- Its peak learning rate is still 3e-4.
- Its bump is about three times the continued arm's.
- By step 90,000 it has not returned to baseline.

## Are the curves still falling?

**r2-cont is close to saturation.** Last three points:

| Measure | Last three points | Trend |
| --- | --- | --- |
| plain | 4.2928 → 4.2901 → 4.2884 | Still falling slowly |
| full tf_exact | 0.692 / 0.692 / 0.690 | Flat |
| mixed tf_exact | 0.323 → 0.331 | Rising slightly, noisy |
| initials tf_exact | 0.176 → 0.177 | Rising slightly, noisy |
| error tf_exact | 0.473 → 0.481 → 0.478 | Rising slightly, noisy |

**r2-base still gains visibly in the last 30,000 steps:**

| Measure | Last points | Trend |
| --- | --- | --- |
| plain | 4.3598 → 4.3563 → 4.3531 | Still falling |
| full tf_exact | 0.687 → 0.695 | Rising slowly |
| mixed tf_exact | 0.330 → 0.336 | Rising slowly |

The other two pinyin kinds' tf_exact is also rising slowly.

## Integrity checks

Both arms pass every item:

- `summary.json` status is `complete`, and `step` equals `max_steps` (60,000 / 90,000).
- `final.pt` exists and loads. Its header step matches the total step count, and its optimizer state is non-empty.
- **Step checkpoints:** all `step-*.pt` files are present, one every 6,000 steps (10 for cont, 15 for base). Each carries `tokenizer_identity` and `special_ids`.
- **Evaluation points:** `evaluations.json` has one point every 2,000 steps (31 for cont, 46 for base), with no gaps.
- **Start-up checks:** training verified the SHA256 of the v2 data manifest, of every data file, and of the starting checkpoint.
- **Continuity:** both arms ran in one go, with no pre-emption, restart or resume.
- **W&B:** no credentials were available, so the runs were logged offline.

## Artifacts

- `artifacts/models/tiny-zh13-12k-pinyin-r2-cont/` contains:
  - `final.pt`;
  - 10 `step-*.pt` files;
  - `evaluations.json`, `summary.json`, `metrics.jsonl`, `plan.json`, `config.json`, `environment.json`.
- `artifacts/models/tiny-zh13-12k-pinyin-r2-base/` contains `final.pt`, 15 `step-*.pt` files, and the same metadata.

A copy of both arms is archived in private storage for local evaluation.

## Next

- **Evaluation:** see [evaluation.md](evaluation.md).
- **The third arm** continues training as planned.
