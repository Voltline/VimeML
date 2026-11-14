# vimeml-tiny-zh-v1 blind confirmation and baselines (2026-11-11)

This report covers two things: the blind result of the [preregistered confirmation](../../benchmarks/confirmation-zh-v1-plan.md), and the comparison with stronger Rime configurations and general Chinese LMs under the [baseline plan](../../benchmarks/baselines-zh-v1-plan.md). Benchmark version `ime-standard-zh-v1-r1`. The eight-arm decision and leakage check are in [IME scoring and decision](../zh-v1-20261110/ime-benchmark.md).

Blind publishes only summary numbers, never item text or candidates.

## Conclusions

- **The confirmation passed.** Blind full-pinyin exact Top-1 is 93.8% (1,355/1,445, Wilson 95% [92.4%, 94.9%]); Rime's first candidate is 72.3%. All three sources have positive net and Top-3 does not drop.
- **v1 beats every comparison system**: two Qwen models (0.5B / 0.6B) and three Rime configurations. On both development and blind, all five paired full-pinyin exact Top-1 comparisons are significant after Holm correction.
- **The closest is Qwen2.5-0.5B**: blind 91.8%, v1 net +28 (p = 0.005), with about 25 times v1's parameters. On blind CrossWOZ it gets 2 more items right than v1, the only source where v1 does not lead.
- **The strongest Rime configuration** is rime-ice with the Wanxiang grammar model: 82.2%, about 10 points above default rime-ice and still 11.6 points below v1. Luna Pinyin (Rime's default scheme) scores 68.9%.
- Blind had already been used by the confirmation; the baseline comparison treats it as a reference test set, and the results are not used to select or change v1.

## Blind confirmation

| Item | Result |
| --- | --- |
| Run | run-01, 2026-11-11 15:38, plan commit `d457a99`, run once |
| Preflight | Every development full and abbreviated item rank identical to the decision |
| Full-pinyin Top-1 | v1 1,355/1,445 = 93.8%, Rime 1,045 = 72.3% |
| H1 (McNemar) | +329/−19, net +310, p = 3.6e-74, cluster bootstrap net [278, 343] |
| Per source | CrossWOZ 94.4% / 63.5%, RiSAWOZ 94.2% / 76.8%, KdConv 92.8% / 76.6% (v1 / Rime) |
| Top-3 | 97.2% / 89.9%, +107/−1 |
| Decision | **Pass** (all three conditions met) |

Descriptive metrics (not used for the decision):

| Metric | v1 | Rime |
| --- | ---: | ---: |
| Top-1 at 2–3 / 4–6 / 7–12 characters | 97.8% / 94.6% / 89.0% | 82.8% / 75.8% / 58.6% |
| Pool-conditional Top-1 | 96.3% | 74.3% |
| MRR | 0.954 | 0.818 |
| Overlap items removed (1,381 items) | 93.7% | 71.8% |
| Abbreviated Top-1 (964 items) | 35.0% | 15.3% (+205/−15) |

0 fallbacks; 4 full-pinyin items have empty pools (kept in the denominator). v1 scores 93.6% on development and 93.8% on blind, with no sign of development inflation.

Consumption record in `outputs/benchmarks/zh-v1-confirmation/consumption.json`, summary SHA256 `8554fe722fa2c0fda3dc8f0f568c228009e4487229cd873a1afcad71bf013936`.

## Baselines

### Systems

| System | Ranking |
| --- | --- |
| rime-ice | librime 1.17.0 + rime-ice `99662ba`, configured as for the frozen pool |
| rime-ice + Wanxiang | The above + the `wanxiang-lts-zh-hans.gram` grammar model, rime-ice's recommended parameters |
| Luna | librime 1.17.0 + `luna_pinyin_simp`, only the user dictionary off |
| Qwen2.5-0.5B | Reranks the frozen pool, scored the same way as v1 |
| Qwen3-0.6B-Base | Same |

Each Rime configuration ranks on its own without left context, taking same-length Han candidates for the whole input, without lattice padding. So rime-ice scored alone is slightly below the pool's original order (blind full 1,043 vs 1,045, abbreviated 141 vs 147). Qwen and v1 score the same pool with the same method, so their differences come from the models alone. Versions and all settings are in the [plan](../../benchmarks/baselines-zh-v1-plan.md).

### Full-pinyin exact Top-1

| System | Parameters | Development | Blind | Blind Top-3 | Blind MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| **vimeml-tiny-zh-v1** | About 20M | **93.6%** | **93.8%** | **97.2%** | **0.954** |
| Qwen2.5-0.5B | 494M | 91.3% | 91.8% | 96.1% | 0.940 |
| Qwen3-0.6B-Base | 596M | 89.8% | 89.2% | 95.4% | 0.924 |
| rime-ice + Wanxiang | — | 79.0% | 82.2% | 83.7% | 0.829 |
| rime-ice | — | 68.8% | 72.2% | 73.6% | 0.729 |
| Luna | — | 68.3% | 68.9% | 70.5% | 0.696 |

### v1 paired with each system (full-pinyin Top-1)

| System | Split | v1 more right / more wrong | Net | Cluster bootstrap 95% | Holm p |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | development | +63 / −29 | +34 | [16, 53] | 5.1e-4 |
| | blind | +61 / −33 | +28 | [9, 48] | 5.1e-3 |
| Qwen3-0.6B-Base | development | +81 / −25 | +56 | [36, 77] | 9.0e-8 |
| | blind | +96 / −30 | +66 | [44, 89] | 6.1e-9 |
| rime-ice + Wanxiang | development | +242 / −29 | +213 | [184, 244] | 1.5e-42 |
| | blind | +202 / −35 | +167 | [138, 196] | 3.0e-29 |
| rime-ice | development | +377 / −15 | +362 | [328, 397] | 3.8e-91 |
| | blind | +331 / −19 | +312 | [280, 345] | 4.0e-74 |
| Luna | development | +380 / −12 | +368 | [334, 402] | 2.4e-95 |
| | blind | +376 / −16 | +360 | [326, 395] | 1.1e-89 |

Holm correction is over the five systems within each split. On Top-3, v1's net against every system is positive and significant.

### Blind by source and length (full-pinyin Top-1)

| System | CrossWOZ | RiSAWOZ | KdConv | 2–3 chars | 4–6 chars | 7–12 chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 94.4% | 94.2% | 92.8% | 97.8% | 94.6% | 89.0% |
| Qwen2.5-0.5B | 94.8% | 90.7% | 90.1% | 92.2% | 92.5% | 90.9% |
| Qwen3-0.6B-Base | 91.6% | 87.2% | 88.8% | 89.1% | 90.5% | 87.9% |
| rime-ice + Wanxiang | 79.7% | 84.7% | 82.2% | 86.1% | 86.9% | 73.6% |
| rime-ice | 63.5% | 76.6% | 76.4% | 82.6% | 75.8% | 58.4% |
| Luna | 60.8% | 75.4% | 70.4% | 84.3% | 71.2% | 51.6% |

- v1's lead over Qwen2.5 is mainly in short words: 5.6 points at 2–3 characters. At 7–12 characters Qwen2.5 is 1.9 points ahead, and on CrossWOZ Qwen2.5 nets 2 more items. On development v1 is 1.5 points ahead at 7–12 characters; this stratum disagrees in direction across the two splits and may be noise, but on long sentences v1 at least has no clear advantage.
- The Wanxiang grammar model mainly helps long sentences: rime-ice at 7–12 characters rises from 58.4% to 73.6%, still far below v1 and Qwen.

### Abbreviated-pinyin exact Top-1

| System | Development | Blind |
| --- | ---: | ---: |
| v1 | 34.0% | 35.0% |
| Qwen2.5-0.5B | 30.9% | 31.0% |
| Qwen3-0.6B-Base | 29.4% | 30.6% |
| rime-ice + Wanxiang | 14.7% | 15.7% |
| rime-ice | 14.0% | 14.6% |
| Luna | 12.1% | 12.9% |

The abbreviated pool covers only about four in ten answers, so LM reranking tops out near 41%. The Wanxiang grammar model barely helps abbreviated input (+1 point).

## How to read this

- **The Qwen comparison** is the most informative reference for the v1 model itself: same pool, same scoring, and v1 slightly beats Qwen2.5-0.5B with about 1/25 of the parameters. v1's training data includes chat text, closer to the task distribution; Qwen is a general model, not tuned for IME. So this does not show v1 is better at Chinese modeling in general, only that on this task a 20M specialized model already reaches the level of a 0.5B general model.
- Whether Qwen trained on CrossWOZ, RiSAWOZ or KdConv is unknown. If it did, its scores are inflated and v1's relative advantage is only underestimated.
- **The Rime comparison** is between systems: Rime does not see left context, v1 does. In the [leakage check](../zh-v1-20261110/ime-benchmark.md#leakage-check), v1 with empty left context scores 81.9% on development, close to rime-ice + Wanxiang (79.0%). Most of v1's advantage over Rime comes from using left context.

## Limits

- The confirmation's limits are in the [plan](../../benchmarks/confirmation-zh-v1-plan.md#limits): labels are not formal native-speaker annotation, data is dialogue rather than real keyboard traffic, and the result measures pool reranking rather than a complete IME.
- Qwen only reranks the frozen pool; whether it can run on device was not tested. The Wanxiang parameters were not tuned.
- Blind had already been used once at comparison time, so the results are for reference only.

## Reproduction

```bash
# Deploy the three Rime configurations (once)
python -I -X utf8 scripts/benchmarks/rime_candidates.py deploy --variant {ice,ice-gram,luna}
# Once per system and track; for blind add --split blind --blind-plan docs/zh-CN/benchmarks/baselines-zh-v1-plan.md
python -I -X utf8 scripts/benchmarks/compare_zh_baselines.py engine {ice,ice-gram,luna} OUT --track {full,abbrev}
python -I -X utf8 scripts/benchmarks/compare_zh_baselines.py lm {qwen2.5-0.5b,qwen3-0.6b-base} OUT --track {full,abbrev}
# Pair with v1: development uses 12k-fresh-s1 from the decision, blind uses v1-{full,abbrev} from confirmation run-01
python -I -X utf8 scripts/benchmarks/score_standard_ime_zh.py compare OUT/result V1 CMP.json
```

Outputs in `outputs/benchmarks/zh-v1-baselines/` and `outputs/benchmarks/zh-v1-confirmation/`, not in Git.
