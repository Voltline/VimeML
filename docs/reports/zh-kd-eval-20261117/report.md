# Development evaluation of the v2-20m distillation arms (2026-11-17)

This report evaluates the `kd-base` and `kd-cont` arms under the decision rules of the [distillation plan](../../training-zh-kd-plan.md). The teacher is the 12K joint-pinyin 200M. The references are:

- `vimeml-tiny-zh-v1` in LM mode;
- the 20M three-task model `r2-cont`;
- the 200M teacher itself.

Everything uses development only; neither blind nor the public test split was read. The public-set numbers come from `cluster_eval_zh.sh` and `pinyin_gen_zh.py reprior` on the cluster. The reference numbers come from the run of the [200M evaluation](../zh-200m-pinyin-eval-20261116/report.md), with the same backend and inputs. The Chinese version, [docs/zh-CN/reports/zh-kd-eval-20261117/report.md](../../zh-CN/reports/zh-kd-eval-20261117/report.md), is the reference text.

## Conclusions

- **`kd-cont` passes and becomes the `vimeml-zh-v2-20m` candidate; `kd-base` does not pass.**
  - The primary metric is public-set Top-1 after rescoring with the model's own scores (λ = 1.6):
    - `kd-cont` reaches 54.8%, +2.1 over v1 alone (p = 0.041): pass;
    - `kd-base` reaches 53.9%, +1.2 (p = 0.23): fail.
  - The guard is Standard full-pinyin rerank: both arms lose a few items against v1, neither significantly (−4, p = 0.63; −6, p = 0.38), so the guard passes.
- **Distillation recovers some LM ability, not much.**
  - `kd-cont` self-rescored is +1.5 over `r2-cont` self-rescored (p = 0.025), and +1.8 in generation order (p = 0.008).
  - The gap to the teacher remains large: the 200M self-rescored is 58.8%.
- **Prediction is still below v1.** Track A Top-3: `kd-cont` 25.6, `r2-cont` 24.8, v1 27.6. Prediction is descriptive under the rules, but the phone loads one model, so v2-20m prediction will be weaker than today's v1. This goes into the release notes and is a target for the next round.
- **The private benchmark agrees.**
  - `kd-cont` self-rescored is +2.1 over v1 alone (p = 0.003);
  - against the production v1 path (Rime 16 sentences plus v1 rerank) it is +9.3, with full pinyin +4.4 (p = 0.003), so the release gate "no significant full-pinyin regression" passes.
  - On the private set `kd-base` is slightly ahead of `kd-cont` (35.9 vs 35.5), a small difference; under the plan the arm is chosen on the public primary metric.

## Setup

- **Arms** (see the distillation plan):
  - `kd-base`: from the v1 base, 90,000 updates, final.pt sha256 `bc819588…`;
  - `kd-cont`: from r2-cont, 30,000 updates, sha256 `412aee1d…`.
  - Both use α = 0.5 and T = 1, on r1 data only.
- **Pinyin generation**: public `ime-pinyin-zh-v1` development, 3,444 inputs.
  - Search uses [zh_search.py](../../../src/vimeml/deployment/zh_search.py) with a beam of 16 per key position, in-field context, task mode.
  - The top 10 are reranked by genScore + λ × prior, with λ = 1.6 and the model itself as prior.
- **Paired tests**: sentence-cluster sign-flip, 20,000 permutations.
- **Standard**: `ime-standard-zh-v1-r1` development, full-pinyin rerank, 1,456 items.
- **Prediction**: `ime-predict-zh-v1-r1` development, tracks A–D.

## Results

### Pinyin generation (public development, Top-1 %)

| Model | Prior | Generation order | Rescored | Full | Mixed | Initials | Top-10 coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 (LM mode) | — | 52.7 | — | 83.6 | 51.7 | 25.6 | 68.5 |
| r2-cont | self | 42.4 | 53.3 | 86.7 | 51.7 | 26.3 | 68.4 |
| kd-base | self | 44.4 | 53.9 | 86.2 | 52.3 | 27.9 | 69.0 |
| **kd-cont** | **self** | 44.2 | **54.8** | 85.8 | **53.7** | **28.2** | **70.0** |
| 200M (teacher) | self | 47.4 | 58.8 | 88.4 | 57.9 | 32.5 | 71.5 |

The v1 row's three per-kind columns are generation order; every other row's per-kind columns are rescored.

Paired comparisons (Top-1):

| Comparison | Diff | Wins/losses | p |
| --- | ---: | --- | ---: |
| kd-cont self-rescored − v1 | +2.1 | 301/227 | 0.041 |
| kd-base self-rescored − v1 | +1.2 | 268/226 | 0.23 |
| kd-cont self − r2-cont self | +1.5 | 161/110 | 0.025 |
| kd-base self − r2-cont self | +0.6 | 208/189 | 0.55 |
| kd-cont self − kd-base self | +0.9 | 181/149 | 0.23 |
| kd-cont generation order − r2-cont generation order | +1.8 | 193/130 | 0.008 |
| kd-base generation order − r2-cont generation order | +2.0 | 247/179 | 0.015 |
| kd-cont generation order − v1 | −8.4 | 208/499 | 5e-5 |

Both arms still need rescoring to beat v1; in generation order alone they are about 8 points below, mostly on full pinyin and mixed abbreviations.

### Standard full-pinyin rerank (development, 1,456 items)

| Model | Wins/losses vs v1 | Net | McNemar p | Cluster sign-flip p | Cluster bootstrap 95% |
| --- | --- | ---: | ---: | ---: | --- |
| kd-base | 13/19 | −6 | 0.377 | 0.381 | [−17, +5] |
| kd-cont | 17/21 | −4 | 0.627 | 0.627 | [−16, +8] |
| 200M (reference) | 30/7 | +23 | 0.0002 | 0.0002 | [+11, +35] |

Both arms tie with v1; the guard passes.

### Prediction (development, descriptive)

| Model | A Top-1 | A Top-3 | A Top-5 | B savings | D first 20% | D first 50% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 16.5 | 27.6 | 33.2 | 16.7% | 28.9 | 21.9 |
| r2-cont | 14.6 | 24.8 | 29.8 | 15.3% | 24.4 | 18.9 |
| kd-base | 14.9 | 24.6 | 29.7 | 15.4% | 25.3 | 19.1 |
| kd-cont | 15.2 | 25.6 | 31.0 | 15.7% | 26.3 | 20.3 |
| 200M | 20.2 | 32.3 | 37.6 | 19.9% | 35.0 | 26.7 |

`kd-cont` is slightly above `r2-cont` on every track and below v1 on every track.

### Private benchmark (local, 4,612 inputs, 774 sentences, descriptive)

Setup: bucketed search with a beam of 16, task mode, no context; self rescoring at λ = 1.6. "Production v1" means Rime 16 sentences plus v1 rerank, with in-field context.

| Model | Generation order | Self-rescored | Full | Mixed | Initials |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production v1 | — | 26.2 | 67.4 | 18.8 | 8.1 |
| v1 (LM mode) | 33.4 | — | 67.4 | 28.1 | 11.0 |
| r2-cont | 29.9 | 34.8 | 70.7 | 29.0 | 12.7 |
| kd-base | 30.3 | **35.9** | **72.2** | **30.1** | **13.3** |
| kd-cont | 30.3 | 35.5 | 71.8 | 29.8 | 11.6 |

The v1 (LM mode) row's three per-kind columns are generation order; the other model rows' per-kind columns are rescored.

| Comparison (Top-1) | Diff | Wins/losses | p |
| --- | ---: | --- | ---: |
| kd-cont self-rescored − v1 (LM mode) | +2.1 | 310/217 | 0.003 |
| kd-base self-rescored − v1 (LM mode) | +2.5 | 332/219 | 0.0008 |
| r2-cont self-rescored − v1 (LM mode) | +1.4 | 322/259 | 0.073 |
| kd-cont self-rescored − production v1 | +9.3 | 594/167 | 5e-5 |
| kd-cont full − production v1 full | +4.4 | 77/43 | 0.003 |

The private set points the same way as the public one: both distilled arms beat r2-cont and v1 alone.

## What this means for v2

- The `vimeml-zh-v2-20m` candidate is `kd-cont`, rescoring with its own scores at λ = 1.6. The phone loads only this model.
- Release gates still open:
  - iPhone latency;
  - confirmation scoring.
- Next round (round 3):
  - add r2 data;
  - tune α and T, aiming to move generation order and prediction closer to the teacher;
  - alongside that, the with/without `typo` arms of the [keyboard typo plan](../../training-zh-typo-plan.md) on the chosen arm.
