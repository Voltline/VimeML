# vimeml-zh-v2-20m distillation plan

Drafted 2026-11-16. The Chinese version, [docs/zh-CN/training-zh-kd-plan.md](zh-CN/training-zh-kd-plan.md), is the reference text.

The phone loads only one model (see [release scope](release-zh-v2-plan.md)), so v2-20m must rescore its candidates with its own scores. The [200M evaluation](reports/zh-200m-pinyin-eval-20261116/report.md) found:

- with its own prior, r2-cont is only +0.8 over v1 alone (p = 0.49), because pinyin post-training weakened its LM ability;
- the 12K joint-pinyin 200M both generates and scores well (58.8% self-rescored, +6.2 over v1).

This plan uses that 200M as the teacher for online distillation into the 20M. The goal is one 20M that has both generation and LM ability.

## Method

- **Teacher**: `tiny-zh200m-12k-pinyin-pt-d1024/final.pt` (step 844,730, sha256 `90e48fe4…`).
  - It is frozen, runs in BF16 without gradients, and sees the same batch as the student.
  - Teacher and student share the 12K tokenizer and the pinyin sample format, so token-level distillation applies directly.
- **Loss**: on every supervised position, `(1 − α)·CE + α·T²·KL(teacher_T ‖ student_T)`, with α = 0.5 and T = 1.
  - Plain text and all four pinyin tasks are distilled.
  - `per_kind` normalization is unchanged.
- **Data and recipe**: r2-base's, kept the same:
  - v2 pinyin data with Rime readings;
  - the same mix: plain 160 / full 96 / mixed 128 / initials 96 / error 32;
  - the same seed, `20261116`, so the data order matches r2-base exactly.

  This round uses r1 data only. r2 is left for the next round.
- **Code**: the optional `[distillation]` section of [joint_zh_pinyin.py](../src/vimeml/training/joint_zh_pinyin.py).
  - Every log line carries the KL per kind.
  - Before training, the teacher's own validation losses are written to `teacher_validation.json`.

## Arms

| Arm | Initialization | Updates | Peak LR | Warmup | Config |
| --- | --- | ---: | ---: | ---: | --- |
| `kd-base` | v1 base `tiny-zh13-12k-fresh-s1-d384-l9` | 90,000 | 3e-4 → 3e-5 | 2000 | [train-zh-pinyin-kd-base.toml](../configs/train-zh-pinyin-kd-base.toml) |
| `kd-cont` | r2-cont final (step 60,000) | 30,000 | 1.5e-4 → 1.5e-5 | 1000 | [train-zh-pinyin-kd-cont.toml](../configs/train-zh-pinyin-kd-cont.toml) |

- `kd-base` differs from r2-base only by the teacher. It measures what distillation itself adds.
- `kd-cont` asks whether the teacher can restore the LM ability r2-cont lost. It costs a third of `kd-base`.

## Decision

Everything runs on development; neither blind nor the public test split is read.

1. **Primary**: public pinyin development, self-rescored Top-1 (λ = 1.6). It must be significantly above v1 alone in LM mode, by a sentence-cluster sign-flip test.
2. **Guard**: Standard full-pinyin rerank Top-1 must not be significantly below v1.
3. **Descriptive**: prediction tracks A–D and the private benchmark. They do not decide.

If both arms pass, the one with the higher primary score becomes the v2-20m candidate. If neither passes, the next round adds r2 data or changes α and T.

## Round 3 (drafted 2026-11-17)

The [round 2 evaluation](reports/zh-kd-eval-20261117/report.md) picked `kd-cont`, with two gaps left:

- self-rescored 54.8% against the teacher's 58.8%;
- prediction track A Top-3 at 25.6 against v1's 27.6.

r2-cont plus kd-cont trained for 90,000 updates in total, so the r1 pinyin data has been seen 3.7 to 5 times (plain 1.9 times). Round 3 adds unseen r2 data and tries two changes aimed at those gaps.

### r2 pinyin data

- The config is [data-zh-pinyin-r2.toml](../configs/data-zh-pinyin-r2.toml); output goes to `zh-pinyin-r2-12k`.
- Builder, sample rules, tokenizer and Rime readings are the same as for r1. The label becomes `zh-pinyin-r2`, so document selection and every random draw are new.
- export-r2 has no pools, no memorization probe and no LCCC, so every r2 train web document is eligible, with 0.8% each for plain and pinyin.
- The size was estimated by running the same rules on one row group of export-r1:
  - about 13M plain windows;
  - about 3.3M per pinyin kind, 1.45 times r1;
  - about 1.1M error samples.
  Round 3 reads less than half a pass of it.
- It is built on the cluster, where export-r2 lives. The same rules processed 979K documents in 16.5 s on 16 local workers, with no single hotspot: pinyin sampling takes 57% and parquet reads 18%. r2 has about 83M documents, about 3,000 CPU seconds in all, and the build as a whole is bound by disk reads.
- r2 has no LCCC, so with r1 and r2 at half each the chat share of pinyin samples drops from about 25% to about 13%. That change is part of the overall effect of adding r2, measured by r3-ctrl against r3-data.

### Trainer

- `joint_zh_pinyin` gains an optional `[[data.extra]]`: each extra directory has its own mix and its own sample stream (the permutation seed gets the source index appended). Its samples follow the primary directory's samples of the same kind, so `per_kind` normalization is unchanged.
- Without extra directories every batch is byte-identical to before (checked on the v2 pinyin data).
- Validation reads the primary (r1) directory only, so the arms' validation curves compare directly.
- The keyboard-typo plan's second `typo` directory uses the same mechanism.

### Four arms

All start from `kd-cont` final (step 30,000) and train for 30,000 updates, peak LR 1e-4 → 1e-5, warmup 1000, same teacher, seed `20261117`. r2-cont and kd-cont both started at step 0 of seed `20261116`, so a new seed is needed to avoid replaying the same data order.

| Arm | Data per update (r1 + r2) | α | T | Question |
| --- | --- | ---: | ---: | --- |
| `r3-ctrl` | r1 only: plain 160 / full 96 / mixed 128 / initials 96 / error 32 | 0.5 | 1 | What 30,000 more updates alone give (control) |
| `r3-data` | Half r1, half r2 for every kind, same total as r3-ctrl | 0.5 | 1 | What new data gives |
| `r3-alpha` | As r3-data | 0.9 | 1 | Whether fitting the teacher harder restores LM ability |
| `r3-plain` | As r3-data, with plain raised from 160 to 256 (r1 128 + r2 128) | 0.5 | 1 | Prediction specifically |

T stays at 1: rescoring and prediction both use the probabilities at T = 1, so a higher T trains the student on a distribution it never uses. α is the more direct lever.

### Decision

All on development, with the round 2 evaluation methods.

1. The arm with the highest primary score replaces `kd-cont` if either holds:
   - its primary score (public-set self-rescored Top-1) is significantly above `kd-cont`;
   - its primary score is not significantly below `kd-cont`, and its prediction track A Top-3 reaches v1's 27.6.
2. The guard is unchanged: Standard full-pinyin rerank Top-1 must not be significantly below v1.
3. If `r3-ctrl` wins, more training alone is enough, and r2 is not added further.
4. The private benchmark stays descriptive.

The arm round 3 picks is the starting point of the with/without `typo` arms in the [keyboard typo plan](training-zh-typo-plan.md).

### Round 3 result (2026-11-17)

[Round 3 evaluation](reports/zh-kd-r3-eval-20261117/report.md): no arm replaces `kd-cont`.

- Primary metric: `r3-plain` 54.8, `r3-data` 54.7 and `r3-ctrl` 54.6 all tie `kd-cont` at 54.8; `r3-alpha` scores 53.5, significantly lower.
- Prediction A Top-3 is 25.8 to 26.5, short of 27.6 in every arm. Standard ties v1 in every arm.
- Conclusion: with the 200M as teacher, more training, r2 data and more plain samples bring no gain, and α = 0.9 hurts.
- `kd-cont` remains the v2-20m candidate and the starting point of the typo-plan arms.

### Round 4: the 800M as teacher

Once the 800M passes its formal evaluation, start from `kd-cont` with the `kd-cont` recipe (r1 only, α = 0.5, T = 1; round 3 found no better recipe) and swap only the teacher for the 800M (same 12K vocabulary, so token-level distillation works directly). It goes into v2-20m if it is ready before v2-20m is final, and into the next release otherwise. The cost of the 800M forward pass will be measured once the 800M is training.

The config is [train-zh-pinyin-r4.toml](../configs/train-zh-pinyin-r4.toml): r3-ctrl's recipe (continue `kd-cont` for 30,000 updates, seed 20261117, peak LR 1e-4) with only the teacher swapped, so its control is r3-ctrl. The teacher is the 800M arch1 final.pt (step 1,198,964, sha256 `ec3e90df…`), finished 2026-11-20. Measured locally on 2026-11-19 with random 800M weights: on the same batches the 800M forward pass takes 3.6 times as long as the 200M's; r3-ctrl took about 45 minutes for 30,000 updates with the 200M teacher, so about 2 hours is expected with the 800M.

Seven more arms run alongside r4; the arms are independent and run in parallel (about 2 hours each; r4-60k about 4). Each changes one thing from r4 and is compared with it on the same evaluation:

| Arm | Change from r4 | Question |
|---|---|---|
| [r4-s2](../configs/train-zh-pinyin-r4-s2.toml) | seed 20261118 | seed noise of r4 |
| [r3-ctrl-s2](../configs/train-zh-pinyin-r3-ctrl-s2.toml) | 200M teacher, seed 20261118 | second teacher comparison (with r4-s2) |
| [r4-t2](../configs/train-zh-pinyin-r4-t2.toml) | T = 2 | a larger teacher's softer distribution |
| [r4-a07](../configs/train-zh-pinyin-r4-a07.toml) | α = 0.7 | more weight on a stronger teacher |
| [r4-60k](../configs/train-zh-pinyin-r4-60k.toml) | 60,000 updates | whether r4 is still improving at 30,000 |
| [r4-data](../configs/train-zh-pinyin-r4-data.toml) | half r1, half r2 (r3-data's mix) | unseen text with the 800M teacher |
| [r4-r2](../configs/train-zh-pinyin-r4-r2.toml) | starts from `r2-cont` | the 800M teacher from `kd-cont`'s own start (pair with `kd-cont`) |

The teacher comparison counts only if r4 − r3-ctrl and r4-s2 − r3-ctrl-s2 have the same sign. A recipe arm replaces r4 only if it beats r4 by more than r4 − r4-s2.

## Deviations

(None yet.)
