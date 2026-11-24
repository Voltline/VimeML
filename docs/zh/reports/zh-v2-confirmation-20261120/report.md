# vimeml-zh-v2-20m test confirmation result (2026-11-20)

Run under the [preregistered confirmation plan](../../benchmarks/confirmation-zh-v2-plan.md) (registered at 31152c0): **passed**. The benchmark is the test split of `ime-pinyin-zh-v1-r1`: 576 sentences, 3,436 inputs. This report publishes summary numbers only, never test item text or candidates. The Chinese version, [docs/zh-CN/zh/reports/zh-v2-confirmation-20261120/report.md](../../../zh-CN/zh/reports/zh-v2-confirmation-20261120/report.md), is the reference text.

## Decision

| Condition | Result | Decision |
| --- | --- | --- |
| H1: all inputs, online path → v2-20m, Top-1 | 31.0% → 49.1%, +18.1 points (+702/−79), p 5e-5 | pass |
| No significant full-pinyin drop | 81.6% → 82.6%, +1.0 point (+27/−21), p 0.47 | pass |
| Standard blind full-pinyin rerank, v1 → v2-20m | net −2 (+19/−21), exact McNemar p 0.87, cluster bootstrap 95% [−14, +10] | pass |

All three hold: `kd-cont` is confirmed on test as `vimeml-zh-v2-20m`. Of the release gates, iPhone latency and client keyword blocking remain and are verified separately.

## Accuracy on test

v2-20m self-rescored (λ = 1.6) Top-1: all inputs 49.1% (1,687 / 3,436, Wilson 95% 47.4–50.8), full pinyin 82.6% (476 / 576, 79.3–85.5). The Wilson intervals are per input and ignore the correlation between inputs of the same sentence.

| Input kind | Inputs | Online path Top-1 | v2-20m generation-order Top-1 | v2-20m rescored Top-1 | v2-20m Top-10 coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 3,436 | 31.0 | 39.9 | **49.1** | 67.1 |
| Full pinyin | 576 | 81.6 | 76.6 | **82.6** | 95.7 |
| Mixed | 2,284 | 23.7 | 36.5 | **47.4** | 66.9 |
| All-initials | 576 | 9.2 | 16.7 | **22.2** | 39.2 |

The online path gives only its top 3, so its Top-10 coverage equals its Top-3 (all inputs 36.2%).

## Strata (descriptive)

| Stratum | Inputs | Online path | v2-20m | Difference |
| --- | ---: | ---: | ---: | ---: |
| 5–7 characters | 1,354 | 41.1 | 54.8 | +13.7 |
| 8–12 characters | 1,350 | 27.0 | 46.6 | +19.6 |
| 13–20 characters | 732 | 19.5 | 43.2 | +23.6 |
| CrossWOZ | 846 | 29.2 | 49.9 | +20.7 |
| RiSAWOZ | 862 | 38.3 | 59.4 | +21.1 |
| KdConv | 874 | 27.6 | 47.6 | +20.0 |
| CPED | 854 | 28.8 | 39.5 | +10.7 |

The longer the sentence, the larger v2-20m's lead. CPED (TV-series lines, the most colloquial) has the smallest lead.

## Development versus test (descriptive)

| | development | test |
| --- | ---: | ---: |
| v2-20m rescored Top-1 | 54.8 | 49.1 |
| Online path Top-1 | 32.4 | 31.0 |
| Difference | +22.5 | +18.1 |
| Full pinyin: v2-20m / online path | 85.8 / 81.8 | 82.6 / 81.6 |

On test v2-20m is 5.7 points lower and the online path only 1.4, so the gap shrinks from +22.5 to +18.1; the full-pinyin lead shrinks from +4.0 to +1.0 and is no longer significant. λ, the beam width and kd-cont were all chosen on development and the private benchmark, so some fall-back on test is expected.

## Run record

- Preflight (`--preflight-only`, run-01, test not opened): development regenerated, 0 flips over 3,438 keys, Top-1 moved 0.0. 3,438 is 6 fewer than the 3,444 inputs because 6 inputs have the same kind and keys as another input of the same sentence and merge by key; both sides merge the same way.
- Formal run run-02: consumption record written at 19:52 on 2026-11-20, plan commit 31152c0. The preflight again gave 0 flips; then test was generated.
- The online path came from the local Rime harness (librime + Vime #70's 16-sentence logic + v1 rerank, with the same in-field contexts): 3,436 inputs, 1,436 with a non-empty context.
- `finish` wrote `summary.json` and closed the attempt in the consumption record. v2-20m is not run on test again.
- No deviations.

## Next

- v2-200m runs under the same plan, against this v2-20m test result.
- v2-20m's Core ML build and Vime integration: [mac-zh-v2-20m.md](../../mac/mac-zh-v2-20m.md).
