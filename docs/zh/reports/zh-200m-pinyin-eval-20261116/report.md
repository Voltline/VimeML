# Development evaluation of the 12K joint-pinyin 200M (2026-11-16)

This report evaluates `zh200m-12k-pinyin-pt`, the 200M model with a 12K vocabulary that saw pinyin samples during pretraining. It is the candidate for `vimeml-zh-v2-200m`. The references are `vimeml-tiny-zh-v1` (generating in LM mode) and the 20M three-task model `r2-cont`. All three benchmarks use development only; neither blind nor the public test split was read. The Chinese version, [docs/zh-CN/zh/reports/zh-200m-pinyin-eval-20261116/report.md](../../../zh-CN/zh/reports/zh-200m-pinyin-eval-20261116/report.md), is the reference text.

## Conclusions

- **The 200M passes all three v2 development gates** (see [release scope](../../release-zh-v2-plan.md)):
  - pinyin generation is significantly above v1;
  - Standard full-pinyin rerank is significantly above v1;
  - it leads on all four prediction tracks.

  The 200M therefore stays on joint pretraining. The fallback (200M base plus pinyin post-training) is not needed.
- **The 200M should rescore with its own scores.** On the public set:
  - Top-1 is 58.8% with its own prior and 57.3% with the v1 prior, a gain of +1.5 (p = 0.016);
  - it is +6.2 over v1 alone (p = 5e-5).

  Joint pretraining kept its LM ability, so it is a better prior than v1. The desktop therefore loads one model.
- **r2-cont does not work alone on the phone.** Rescoring with its own prior:
  - it is only +0.8 over v1 alone, which is not significant (p = 0.49);
  - it is −2.1 against rescoring with the v1 prior (p = 0.003).

  The private benchmark shows the same thing: pinyin post-training weakened r2-cont's LM ability. The choice of v2-20m still waits for the distillation line.
- **Generation alone is still below v1.** Without rescoring, the 200M is −5.3 and r2-cont is −10.3. The 200M's gap is on full pinyin (78.9 vs 83.6) and mixed abbreviations (44.7 vs 51.7); on all-initials it is slightly ahead (26.3 vs 25.6). Its top-10 coverage is the highest (71.5% vs v1's 68.5%), and rescoring turns that coverage into Top-1 gains.

## Setup

- **Models**:
  - 200M: `tiny-zh200m-12k-pinyin-pt-d1024`, d1024/L15, context 128, 194M parameters, step 844,730;
  - v1: `vimeml-tiny-zh-v1`;
  - r2-cont: 20M three-task round-2 continuation, step 60,000.
- **Pinyin generation**: `ime-pinyin-zh-v1` development, 3,444 inputs covering full pinyin, mixed abbreviations and all-initials.
  - Search uses [zh_search.py](../../../../src/vimeml/deployment/zh_search.py) with a beam of 16 per key position and the in-field context, on the FP32 `ModelBatchBackend`.
  - The 200M and r2-cont generate in task mode; v1 generates in LM mode.
  - The top 10 are reranked by genScore + λ × prior, with λ = 1.6 (from the [rescoring report](../zh-pinyin-rescore-20261115/report.md)). The prior is v1 or the model itself, using the production rule (BOS + context + candidate).
  - Paired tests use a sentence-cluster sign-flip with 20,000 permutations (minimum p 5e-5).
- **Standard**: `ime-standard-zh-v1-r1` development, full-pinyin rerank, 1,456 items.
- **Prediction**: `ime-predict-zh-v1-r1` development, tracks A–D.
- **Self-prior rescoring**: computed afterwards with `pinyin_gen_zh.py reprior`, which recomputes only the prior of the existing top 10 and does not rerun generation.
- **v1 baseline**: v1 scores 52.7% here and 52.5% in the [beam report](../zh-pinyin-beam-20261115/report.md). The difference comes from the backend (`ModelBatchBackend` vs `TorchBatchBackend`) and the input count (3,444 vs 3,428). All comparisons in this report are within one run.

## Results

### Pinyin generation (public development, Top-1 %)

The full, mixed and initials columns are rescored Top-1.

| Model | Prior | Generation order | Rescored | Full | Mixed | Initials | Top-10 coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 (LM mode) | v1 | 52.7 | 52.6 | 83.6 | 51.6 | 25.4 | 68.5 |
| r2-cont | v1 | 42.4 | 55.4 | 86.3 | 54.0 | 30.1 | 68.4 |
| r2-cont | self | 42.4 | 53.3 | 86.7 | 51.7 | 26.3 | 68.4 |
| 200M | v1 | 47.4 | 57.3 | 86.5 | 56.2 | 32.5 | 71.5 |
| **200M** | **self** | 47.4 | **58.8** | **88.4** | **57.9** | **32.5** | 71.5 |

Paired comparisons on Top-1 (sentence-cluster sign-flip):

| Comparison | Diff | Won/lost | p |
| --- | ---: | --- | ---: |
| 200M self-rescored − v1 | +6.2 | 381/168 | 5e-5 |
| 200M v1-rescored − v1 | +4.7 | 284/122 | 5e-5 |
| 200M self − 200M with v1 | +1.5 | 138/87 | 0.016 |
| 200M self − r2-cont self | +5.4 | 353/166 | 5e-5 |
| r2-cont self-rescored − v1 | +0.8 | 283/257 | 0.49 |
| r2-cont self − r2-cont with v1 | −2.1 | 90/161 | 0.003 |
| 200M generation order − v1 | −5.3 | 267/450 | 5e-5 |
| r2-cont generation order − v1 | −10.3 | 206/560 | 5e-5 |

λ scan from 0 to 4 in steps of 0.2. This is exploratory: λ was tuned on the same development set.

| Pair | λ = 1.6 | Best λ | Best |
| --- | ---: | ---: | ---: |
| 200M + self | 58.8 | 1.8 | 58.8 |
| 200M + v1 | 57.3 | 1.4 | 57.5 |
| r2-cont + v1 | 55.4 | 1.0 | 55.6 |
| r2-cont + self | 53.3 | 2.2 | 53.4 |

The curves are flat from 1.2 to 4, so λ = 1.6 stays.

**Private benchmark** (run locally the day before, for reference only). It used gen.py's shared beam of 32, task mode and no context, so its search differs from the public-set runs.

| Model | Generation | v1 prior | Self prior |
| --- | ---: | ---: | ---: |
| 200M | 30.7 | 37.8 | 39.7 |
| r2-cont | 29.9 | 36.4 | 34.9 |

Both sets point the same way: the self prior helps the 200M and hurts r2-cont.

### Standard full-pinyin rerank (development, 1,456 items)

| Model | Top-1 | Top-3 | MRR | Top-1 without overlap hits (1,392) |
| --- | ---: | ---: | ---: | ---: |
| v1 | 93.6 | 97.3 | 0.954 | 93.5 |
| 200M | **95.2** | 97.3 | 0.962 | **95.1** |

Paired against v1:
- Top-1: 30 won and 7 lost, net +23, McNemar p = 0.0002, cluster bootstrap 95% interval [11, 35].
- By source: crosswoz +9, kdconv +8, risawoz +6. All three are positive; risawoz is not significant.

The rerank guard passes.

### Prediction (development)

| Model | A Top-1 | A Top-3 | A Top-5 | B key savings | B click rate | C first syllable | C initials | D top 20% | D top 50% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 16.5 | 27.6 | 33.2 | 16.7% | 13.4% | 12.6 | 58.1 | 28.9 | 21.9 |
| r2-cont | 14.6 | 24.8 | 29.8 | 15.3% | 12.2% | 11.5 | 51.0 | 24.4 | 18.9 |
| **200M** | **20.2** | **32.3** | **37.6** | **19.9%** | **15.8%** | **14.5** | **64.1** | **35.0** | **26.7** |

Prediction is descriptive only. The 200M is best on every track. r2-cont is below v1 on every track, which again shows that pinyin post-training weakens LM ability.

## Consequences for v2

- **`vimeml-zh-v2-200m`**:
  - Candidate: `zh200m-12k-pinyin-pt`, rescoring with its own scores at λ = 1.6.
  - Still to do: latency (full-search p50/p95 on a Mac) and the confirmation run.
- **800M**:
  - The 200M result favours joint pretraining for the 800M's pinyin ability as well.
  - This is decided before 800M training starts and recorded in the 800M plan.
- **v2-20m**:
  - r2-cont alone is not significantly better than v1 on the phone, so the choice waits for the distillation line.
  - The 200M's own scores are the teacher signal distillation needs: one model that both generates and scores well as an LM.
