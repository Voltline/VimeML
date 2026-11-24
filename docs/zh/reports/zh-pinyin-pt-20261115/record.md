# zh three-task arm 3: joint pretraining from scratch (2026-11-15)

Arm 3 (`configs/zh/train-zh-pinyin-pt.toml`, stage `zh-pinyin-pt`) tests a different route. Instead of continuing pinyin training on the 20M base, it starts from random weights and mixes pinyin samples straight into the base pretraining stream.

**References:**

- `r2-cont` from [round 2](../zh-pinyin-r2-20261114/evaluation.md), which continued from round 1.
- The 12K base `12k-fresh-s1`, which saw the same plain stream without pinyin.

All results are development results. Blind was not read, and the public set's test split was not read.

The registered text is the Chinese version, [docs/zh-CN/zh/reports/zh-pinyin-pt-20261115/record.md](../../../zh-CN/zh/reports/zh-pinyin-pt-20261115/record.md).

## Conclusions

- **Joint pretraining does not suit the 20M. The 20M stays on the `r2-cont` continuation route.**
- **Pinyin generation is below `r2-cont` on both sets.**
  - Private benchmark Top-1: −2.2 points (p 0.0004), and −4.4 on full pinyin.
  - Public set: 37.8% vs 39.6% overall, and 74.9% vs 77.3% on full pinyin.
- **The base abilities lose less than with continued training, but not enough to pay for it.**
  - Standard rerank: −4 against the base (p 0.61), against −24 for `r2-cont`.
  - Prediction track A Top-3: +1.1 points over `r2-cont` (p 0.004), still −1.7 points below vimeml-tiny-zh-v1 (p 5e-5).
- **The same numbers need not hold for the 200M.** It has many more parameters for the same training tokens, so capacity pressure is far lower. The 12K joint-pinyin 200M uses exactly this recipe and is training now.

## Setup

The base 20M recipe is unchanged:

- Data: the frozen mixture and seed of `12k-fresh-s1`.
- Learning rate: cosine 6e-4 → 6e-5, warmup 1,000.
- 512 LM windows per update, 844,730 updates.
- Model: d_model 384, 6 heads, 9 layers, d_ff 960, SwiGLU, RMSNorm, context 128, vocabulary 12,288, bf16.

**Pinyin samples on top:**

- Every update adds 192 pinyin samples: full 48, mixed 64, initials 56, error 24.
- They come from the pretraining-scale pinyin stores in `artifacts/training-data/zh-pinyin-pt-12k` (`configs/zh/data-zh-pinyin-pt.toml`). The config pins the manifest SHA256.
- The loss is normalized `per_kind`. LM carries 512/704 of the weight, and each pinyin kind carries its sample share.

## Training

- 844,730 updates in one run, with no interruption or resume. Pure training time was 35,251 s (9.79 h).
- LM: 432,501,627 windows, 14,519,248,976 supervised tokens.
- Pinyin samples:
  - full 40,547,040;
  - mixed 54,062,720;
  - initials 47,304,880;
  - error 20,273,520.

  That is about half a pass over the stores.
- A milestone every 5%, 20 in all.

The last milestone (M20) against the base (BPC, lower is better):

| Validation set | Arm 3 | Base `12k-fresh-s1` | Difference |
| --- | ---: | ---: | ---: |
| web | 4.4603 | 4.3545 | +0.106 |
| chat | 5.7230 | 5.6907 | +0.032 |
| probe | 4.4466 | 4.3404 | +0.106 |

M20 pinyin validation, teacher forced. There are 4,096 items per kind, and 1,674 for error.

| Kind | Loss | Exact match |
| --- | ---: | ---: |
| full | 0.345 | 63.9% |
| mixed | 0.891 | 31.4% |
| initials | 1.532 | 16.3% |
| error | 0.617 | 45.8% |

**These pinyin numbers do not compare across arms.** This validation set comes from the pretraining-scale stores, not round 2's v2 validation set. Compare arms on the benchmarks below.

**The pinyin samples cost the LM.** With the same number of LM updates, plain BPC is about 0.1 above the base. The pinyin samples carry about 27% of the loss weight and take part of the 20M's capacity.

## Evaluation

Scripts and settings are the same as round 2:

- beam 32, top 10, task mode;
- Rime uses Vime #70's 16-sentence configuration.

Paired tests are clustered by sentence or dialogue, with the cluster sign-flip p and the cluster bootstrap 95% interval.

### Pinyin generation (private benchmark)

The private benchmark is the author's real input and is not public: 774 sentences, 4,612 inputs. Only statistics are reported.

Top-1 (%):

| Group | n | `r2-cont` | Arm 3 | Difference | Gains/losses | Cluster p | 95% |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| all | 4,612 | 29.9 | 27.7 | −2.2 | +212/−313 | 0.0004 | [−3.4, −1.0] |
| full | 774 | 65.9 | 61.5 | −4.4 | +42/−76 | 0.003 | [−7.1, −1.7] |
| mixed | 3,492 | 23.9 | 22.1 | −1.8 | +161/−224 | 0.008 | [−3.1, −0.5] |
| initials | 346 | 10.4 | 9.2 | −1.2 | +9/−13 | 0.53 | [−3.8, +1.4] |

Top-3 goes the same way: 40.6 → 37.9 overall (−2.7, p 0.0001).

**Against the production logic** (Rime 16 sentences + v1 rerank, 26.0%), arm 3 generation is +1.8 overall (p 0.03):

- mixed +3.4 (p 5e-5);
- full −5.7 (p 0.0007).

`r2-cont` is +3.9 on the same comparison.

### Pinyin generation (public set ime-pinyin-zh-v1 development)

578 sentences and 3,444 inputs: full 578, mixed 2,288, initials 578.

| System | all | full | mixed | initials | Top-3 all |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1-prod (production logic) | 31.6 | 81.3 | 25.0 | 7.8 | — |
| `r2-cont` generation | 39.6 | 77.3 | 36.1 | 15.9 | 54.9 |
| Arm 3 generation | 37.8 | 74.9 | 33.7 | 16.6 | 51.7 |

Arm 3's generated top 10 contains the target 62.9% of the time (`r2-cont` 65.4%).

The public set agrees with the private benchmark: joint pretraining is worse on full and mixed input and ties only on initials.

### Rerank (Standard development, full pinyin)

Scored the production way: a plain LM with no keys. Against the base `12k-fresh-s1`:

- net −4 (+15/−19);
- McNemar p 0.61, cluster p 0.61;
- cluster bootstrap [−15, +7].

`r2-cont` is −24 (p 0.003).

### Prediction (ime-predict-zh-v1 development)

| Reference | Track A Top-3 ref / arm 3 | Difference | Gains/losses | Cluster p | 95% | Track B savings ref / arm 3 | Difference [95%] |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| vimeml-tiny-zh-v1 | 27.60 / 25.92 | −1.7 | +279/−400 | 5e-5 | [−2.4, −1.0] | 16.74 / 15.91 | −0.83 [−1.03, −0.63] |
| `r2-cont` | 24.78 / 25.92 | +1.1 | +415/−333 | 0.004 | [+0.4, +1.9] | 15.27 / 15.91 | +0.63 [+0.43, +0.84] |

## Reading

**The idea.** Joint pretraining lets the base cover all three tasks from the start and avoids the forgetting that continued training causes.

**What it did:**

- **Rerank:** it works. The regression drops from −24 to −4, which is not significant.
- **Prediction:** it recovers only part of the loss. Arm 3 is +1.1 over `r2-cont`, still −1.7 below v1.
- **Pinyin generation:** it drops across the board, in the same direction on the private benchmark and the public set. Full pinyin regresses most: −4.4 private, −2.4 public.

**Verdict.** For the 20M the trade does not pay. Pinyin generation is a required task, and the rerank and prediction gains are too small.

## Next steps

1. **Keep the `r2-cont` route for the 20M.** Decide on a third round of post-training once the 12K joint-pinyin 200M has been evaluated: distillation or data changes.
2. **Rescoring narrows the gap.** With inference-time rescoring (v1 prior, λ = 1.6):
   - Arm 3 goes from 27.7% to 35.3% on the private benchmark, and from 37.7% to 50.1% on the public set.
   - The gap to rescored `r2-cont` (36.4%, 51.3%) narrows to about 1 point.

   See the [inference-time rescoring report](../zh-pinyin-rescore-20261115/report.md).
3. **Evaluate the 12K joint-pinyin 200M the same way** when it arrives. It uses the same data and recipe: `configs/zh/train-zh200m-12k-pinyin-pt.toml`, d1024 / L15 / c128, 193,625,088 parameters.
