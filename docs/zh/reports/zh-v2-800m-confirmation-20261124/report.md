# vimeml-zh-v2-800m test confirmation result (2026-11-24)

Run under the "Amendment: v2-800m" section of the [preregistered confirmation plan](../../benchmarks/confirmation-zh-v2-plan.md) (registered at 116f7dc): **passed**. The benchmark is the test split of `ime-pinyin-zh-v1-r1`: 576 sentences, 3,436 inputs. This report publishes summary numbers only, never test item text or candidates. The Chinese version, [docs/zh-CN/zh/reports/zh-v2-800m-confirmation-20261124/report.md](../../../zh-CN/zh/reports/zh-v2-800m-confirmation-20261124/report.md), is the reference text.

## Decision

| Condition | Result | Decision |
| --- | --- | --- |
| H1: all inputs, v2-200m (its 2026-11-23 test rows, run-01) → v2-800m, self-rescored Top-1 | 54.7% → 63.3%, +8.6 points (+429/−134), p 5e-5 | pass |
| Standard blind full-pinyin rerank, v1 → v2-800m (descriptive, does not decide the pass) | net +22 (+35/−13), exact McNemar p 0.0021, cluster bootstrap 95% [+9, +36] | significant gain; the 800M reranks for itself, no fallback to the 200M |

H1 passes, and arch2 (`tiny-zh800m-arch2-k4`) is confirmed on test as `vimeml-zh-v2-800m`. Its Standard rerank Top-1 is 95.3% (1,377 / 1,445), against 95.2% (1,376) for v2-200m and 93.8% for v1.

## Accuracy on test

Self-rescored (λ = 1.6) Top-1 of v2-800m: all inputs 63.3% (2,175 / 3,436, Wilson 95% 61.7–64.9), full pinyin 88.7% (511 / 576, 85.9–91.0). The Wilson intervals are per input and ignore correlation between inputs of the same sentence.

| Input kind | Inputs | Online path | v2-20m (c40) | v2-200m | **v2-800m** | v2-800m generation-order Top-1 | v2-800m Top-10 coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 3,436 | 31.0 | 51.4 | 54.7 | **63.3** | 56.5 | 80.6 |
| Full pinyin | 576 | 81.6 | 84.4 | 86.5 | **88.7** | 88.4 | 99.1 |
| Mixed abbreviation | 2,284 | 23.7 | 49.8 | 53.3 | **63.3** | 54.4 | 82.3 |
| All initials | 576 | 9.2 | 24.7 | 28.5 | **38.0** | 33.0 | 55.0 |

Full pinyin is not a guard for the 800M (same reason as the 200M). Descriptively: +2.3 points over v2-200m (+28/−15, p 0.066).

## Against the current v2-20m, colloq-c40 (descriptive)

c40's public test rows come from the guard run of its colloquial confirmation ([colloquial confirmation](../zh-colloq-confirmation-20261123/report.md)): same inputs and decoding settings, so they pair input by input.

| Input kind | c40 | v2-800m | Difference | +/− | Sentence-clustered sign-flip p |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 51.4 | 63.3 | +11.9 | +537/−127 | 5e-5 |
| Full pinyin | 84.4 | 88.7 | +4.3 | +37/−12 | 0.0005 |
| Mixed abbreviation | 49.8 | 63.3 | +13.5 | +405/−97 | 5e-5 |
| All initials | 24.7 | 38.0 | +13.4 | +95/−18 | 5e-5 |

This table is not a preregistered test and does not change the decision.

## Strata (descriptive)

| Stratum | Inputs | c40 | v2-200m | v2-800m | 800M − 200M |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5–7 characters | 1,354 | 58.1 | 60.4 | 67.1 | +6.6 |
| 8–12 characters | 1,350 | 48.2 | 52.7 | 62.7 | +10.0 |
| 13–20 characters | 732 | 44.7 | 47.8 | 57.4 | +9.6 |
| CrossWOZ | 846 | 52.7 | 56.7 | 68.9 | +12.2 |
| RiSAWOZ | 862 | 60.2 | 65.4 | 73.5 | +8.1 |
| KdConv | 874 | 50.9 | 53.7 | 60.2 | +6.5 |
| CPED | 854 | 41.6 | 43.0 | 50.6 | +7.6 |

The 800M leads the 200M in every stratum, most on medium and long sentences and on CrossWOZ. The gain comes mainly from abbreviated input: full pinyin gains 2.3 points, mixed abbreviation and all initials about 10 points each.

## Against development (descriptive)

| | development | test |
| --- | ---: | ---: |
| v2-800m rescored Top-1 | 65.9 | 63.3 |
| v2-200m rescored Top-1 | 58.8 | 54.7 |
| Difference | +7.1 | +8.6 |

Both are lower on test than on development: the 800M by 2.6 points, the 200M by 4.1. arch2 went through a single development judgement, against arch1.

## Execution record

- Consumption record written 2026-11-24 10:32; plan commit and HEAD at run time were both 116f7dc.
- Preflight: development regenerated, 0 flips over 3,438 keys, Top-1 moved 0.0 (inputs merged as in the v2-20m report); the regenerated rescored Top-1 is 65.9%, the same as the registered rows.
- Test generated locally on GPU: 3,436 inputs, 4,000 seconds. Checkpoint SHA256 `8af7dab5…d58b`, reading dictionary SHA256 `3a1654cd…0c8a`.
- `finish` compared against the v2-200m-run-01 test rows, wrote `summary.json` (SHA256 `981cc611…6989`), and marked the attempt finished. v2-800m does not run on test again.
- This is the fourth use of the `ime-pinyin-zh-v1-r1` test split: the v2-20m confirmation (2026-11-20), the guard of the c40 colloquial confirmation (2026-11-23), the v2-200m confirmation (2026-11-23), then this run. Each was registered in advance in its plan.
- No deviations.

## Next

- Released 2026-11-24 in `800m/` of the Hugging Face repository `Voltline/vimeml-zh-v2` (revision `e5296ee8`): the FP32 inference bundle (`scripts/zh/deployment/export_zh_desktop_bundle.py`, logits identical to the checkpoint) and the evaluation summary. The desktop models get no Core ML packages.
