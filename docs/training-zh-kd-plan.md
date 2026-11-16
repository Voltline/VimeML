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

## Deviations

(None yet.)
