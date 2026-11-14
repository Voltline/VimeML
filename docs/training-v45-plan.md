# V4.5 ensemble distillation (preregistered)

> **Status: registered, not run.** No student was trained; the Japanese line paused when work moved to the Chinese model. V4.2 epoch 20 remains the deployed Japanese model. The plan is kept as written in case the line resumes.

Registered 2026-11-05 and committed before any student started training. V4.5 asks one question: if the ensemble of four V4.2-recipe seeds is distilled into one 15M model with the same deployment cost as V4.2, how much of the ensemble's IME gain survives?

## Why

- The [V4.4 ablation](training-v44.md) showed that changing the pretraining recipe no longer helps at this scale. The seed noise band on Standard development is 501–523 (SD 9.5). Summing the log-probabilities of four same-recipe seeds reaches 535.
- The [ensemble replication](benchmarks/ensemble-replication-v2-results-20261105.md) reproduced the effect on Standard v2 blind, which took no part in any selection: ensemble 619, V4.2 598, net +21, author-cluster p = 0.0012. On written-text homophones it was +7, not significant.
- The ensemble needs four times the parameters and compute and does not fit in the keyboard extension. Distillation is the direct way to turn that gain into a deployable model.

## Design

**Teachers**: epoch 20 of the four V4.2-recipe seeds, frozen.

| Teacher | Source (private HF, pinned revision) | Step | SHA-256 |
| --- | --- | ---: | --- |
| `v42` | `Voltline/vimeml-v42-training-output` `99d91488…` `checkpoints/best-lm.pt` | 1,081,942 | `0e83ea74…` |
| `s1` | `Voltline/vimeml-v44-training-output` `a77fd281…` `s1/checkpoints/epoch-20.pt` | 1,081,944 | `31cc8120…` |
| `s2` | Same, `s2/…` | 1,081,940 | `dd8a33ac…` |
| `s3` | Same, `s3/…` | 1,081,937 | `bac6f0fe…` |

**Students**: eight, two combination methods × four twin seeds.

- Each student is an **exact twin** of one CE teacher: config, seed, initialization and the frozen 20-epoch mixture map are the same as its teacher's (`v42` uses the V4.2 mixture, `s1`–`s3` their own V4.4 mixtures). Only the training objective changes, so each student can be compared item by item with its twin.
- When building the plan, the trainer checks that the config equals the twin config field by field, except for the student fields (stage, arm, output directory, `[distillation]`, `[tracking]`, `lm_microbatch`), and that the mixture manifest SHA matches.
- Teachers are built under a forked RNG, so they do not affect the student's initialization or data stream.

| Arms | Combination | Twins | Configs |
| --- | --- | --- | --- |
| `geo-v42`, `geo-s1`, `geo-s2`, `geo-s3` | Normalized geometric mean | V4.2 / `s1` / `s2` / `s3` | [`train-v45-geo-*.toml`](../configs/train-v45-geo-v42.toml) |
| `mix-v42`, `mix-s1`, `mix-s2`, `mix-s3` | Arithmetic mean | Same | [`train-v45-mix-*.toml`](../configs/train-v45-mix-v42.toml) |

- **Combination.** `geo`: q ∝ exp(mean log p_k), the token-level counterpart of the "sum the log-probabilities" rerank ensemble. `mix`: q = mean p_k, the usual ensemble-distillation target. Both are combined in FP32.
- **Loss.** On each supervised token, 0.5 × CE(label) + 0.5 × CE(q), temperature 1. As in V4.2, normalized by the total supervised tokens of each update.
- **Everything else is exactly V4.2.** 20 epochs, cosine 6e-4 → 6e-5, batch 512, BF16 + compile, 5% chat, prefix crop 0.30. The only numeric change is `lm_microbatch` = 256: the loss is summed and then normalized, so this only affects memory, not the objective.
- **Cost.** Each update adds four teacher forward passes. Measured locally without compilation, an update costs about 3.4 times a CE update, so each student is expected to take about 10 hours (V4.2: 3.2 hours). Memory and speed are to be measured with `--smoke 300` on the training machines before the real runs.

Entry points: [`joint_v45.py`](../src/vimeml/training/joint_v45.py), [`train_v45.py`](../scripts/training/train_v45.py), [`prepare_v45_teachers.py`](../scripts/training/prepare_v45_teachers.py) and the launch scripts. W&B isolation is the same as in V4.4, group `v45-distillation`.

## Evaluation (fixed before training ends)

- **Checkpoint**: epoch 20 of each student only, not chosen by BPC or IME results.
- **Scoring**: exactly as in V4.4: local FP32, TF32 off, full-vocabulary suffix log-probability sum, frozen n-best 20 pools, single-model rule (fall back to AzooKey order when the left context is too long). Tracks: Standard /728, Legacy /500, AJIMEE /200, historical development /137. No blind or test set is read. Statistics follow [metrics v2](benchmarks/metrics-v2.md).
- **Primary comparison**: Standard /728 exact Top-1, decided separately for each combination method. For each twin seed s, d_s = student_s − twin_s. The twins score 523 / 501 / 518 / 511, mean 513.25. A combination method is "effective" only if all four hold:
  1. the mean of the four d_s is ≥ +10 (about one development seed SD);
  2. at least three of the four d_s are > 0;
  3. pooled test: per item, the mean correctness of the four students minus that of the four twins, author-cluster sign-flip, two-sided, Holm-corrected across the two methods, family α = 0.05;
  4. guard: the same pooled test on notation-lenient Top-1 shows no regression with p < 0.05 on Standard, Legacy or AJIMEE.
- **Smoke**: flagged when the student mean on historical development /137 is 3 or more below the twin mean; not used for the decision.
- **Descriptive reports** (not used for the decision):
  - per track, mean and range of students, twins and the four-teacher ensemble (Standard 535);
  - recovered fraction = (student mean − 513.25) / (535 − 513.25);
  - per-seed paired differences between `geo` and `mix`;
  - per-epoch LM and chat validation BPC against the twins;
  - KL and teacher CE during training;
  - error classes (semantic / notation);
  - the ensemble of the four students themselves.

## How the conclusions are used

- **If effective**: if both methods are effective, take the one with the larger mean d_s. The deployment candidate is fixed in advance as that method's student whose twin is V4.2 (`geo-v42` or `mix-v42`), not picked among the four seeds by IME score. It has exactly V4.2's architecture, so it only needs a new Core ML export, not new device latency measurements. Before deployment, its advantage over V4.2 must still be confirmed in a preregistered test on a new blind set; both v2 sets have been used twice and are no longer used for confirmation.
- **If not effective**: report the recovered fraction. Below half means 15M capacity or this training budget cannot hold the ensemble's preferences; the recipe work closes at this scale and focus moves to pool coverage and notation. Above half without meeting the standard, the next step would be soft targets only (weight 1.0) or longer training, which needs a separate registration.
- Whatever the outcome, V4.2 epoch 20 stays deployed until a new confirmation is complete.
