# vimeml-zh-v2-200m test confirmation result (2026-11-23)

Run under the [preregistered confirmation plan](../../benchmarks/confirmation-zh-v2-plan.md) (registered at 511b798): **passed**. The benchmark is the test split of `ime-pinyin-zh-v1-r1`: 576 sentences, 3,436 inputs. This report publishes summary numbers only, never test item text or candidates. The Chinese version, [docs/zh-CN/reports/zh-v2-200m-confirmation-20261123/report.md](../../zh-CN/reports/zh-v2-200m-confirmation-20261123/report.md), is the reference text.

## Decision

| Condition | Result | Decision |
| --- | --- | --- |
| H1: all inputs, v2-20m (`kd-cont`, its 2026-11-20 test rows) → v2-200m, self-rescored Top-1 | 49.1% → 54.7%, +5.6 points (+343/−150), p 5e-5 | pass |
| Standard blind full-pinyin rerank, v1 → v2-200m (descriptive, does not decide the pass) | net +21 (+32/−11), exact McNemar p 0.0019, cluster bootstrap 95% [+9, +34] | significant gain; desktop rerank uses the 200M, not v1 |

H1 passes, and `zh200m-12k-pinyin-pt` is confirmed on test as `vimeml-zh-v2-200m`. Its Standard rerank Top-1 is 95.2% (1,376 / 1,445), against v1's 93.8%.

## Accuracy on test

Self-rescored (λ = 1.6) Top-1 of v2-200m: all inputs 54.7% (1,880 / 3,436, Wilson 95% 53.0–56.4), full pinyin 86.5% (498 / 576, 83.4–89.0). The Wilson intervals are per input and ignore correlation between inputs of the same sentence.

| Input kind | Inputs | Online path | `kd-cont` | v2-20m (c40) | **v2-200m** | v2-200m generation-order Top-1 | v2-200m Top-10 coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 3,436 | 31.0 | 49.1 | 51.4 | **54.7** | 42.9 | 69.2 |
| Full pinyin | 576 | 81.6 | 82.6 | 84.4 | **86.5** | 76.9 | 95.8 |
| Mixed abbreviation | 2,284 | 23.7 | 47.4 | 49.8 | **53.3** | 40.6 | 69.0 |
| All initials | 576 | 9.2 | 22.2 | 24.7 | **28.5** | 18.2 | 42.9 |

Full pinyin is not a guard for the 200M (the plan's full-pinyin guard protects only the online path). Descriptively: +3.8 points over `kd-cont` (+35/−13, p 0.002).

## Against the current v2-20m, colloq-c40 (descriptive)

The plan's H1 baseline is `kd-cont`, which was v2-20m when the plan was registered. On 2026-11-23 `colloq-c40` passed its colloquial confirmation and replaced it. c40's public test rows come from that confirmation's guard run ([colloquial confirmation](../zh-colloq-confirmation-20261123/report.md)): same inputs and decoding settings, so they pair input by input.

| Input kind | c40 | v2-200m | Difference | +/− | Sentence-clustered sign-flip p |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 51.4 | 54.7 | +3.3 | +280/−165 | 0.0002 |
| Full pinyin | 84.4 | 86.5 | +2.1 | +27/−15 | 0.09 |
| Mixed abbreviation | 49.8 | 53.3 | +3.5 | +204/−123 | 0.0007 |
| All initials | 24.7 | 28.5 | +3.8 | +49/−27 | 0.014 |

This table is not a preregistered test and does not change the decision. It shows that the 200M still leads the current 20M by more than 3 points.

## Strata (descriptive)

| Stratum | Inputs | `kd-cont` | c40 | v2-200m | 200M − c40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5–7 characters | 1,354 | 54.8 | 58.1 | 60.4 | +2.3 |
| 8–12 characters | 1,350 | 46.6 | 48.2 | 52.7 | +4.5 |
| 13–20 characters | 732 | 43.2 | 44.7 | 47.8 | +3.1 |
| CrossWOZ | 846 | 49.9 | 52.7 | 56.7 | +4.0 |
| RiSAWOZ | 862 | 59.4 | 60.2 | 65.4 | +5.2 |
| KdConv | 874 | 47.6 | 50.9 | 53.7 | +2.8 |
| CPED | 854 | 39.5 | 41.6 | 43.0 | +1.4 |

c40's colloquial post-training closes part of the gap on short sentences and on CPED (TV-drama lines). The 200M leads most on task-oriented dialogue (RiSAWOZ, CrossWOZ) and on medium and long sentences.

## Against development (descriptive)

| | development | test |
| --- | ---: | ---: |
| v2-200m rescored Top-1 | 58.8 | 54.7 |
| `kd-cont` rescored Top-1 | 54.8 | 49.1 |
| Difference | +3.9 | +5.6 |

Both are lower on test than on development. `kd-cont` falls further (5.7 points against 4.1), so the difference grows. `kd-cont` went through several development judgements, while the 200M was evaluated once (2026-11-16), so a smaller drop for the 200M is expected.

## Execution record

- Consumption record written 2026-11-23 14:53, plan commit 511b798, HEAD 993169f at run time. No confirmation file changed during the run.
- Preflight: development regenerated, 0 flips over 3,438 keys, Top-1 moved 0.0 (inputs merged as in the v2-20m report).
- Test generated locally on GPU: 3,436 inputs, 634 seconds. Checkpoint SHA256 `90e48fe4…50f8`, reading dictionary SHA256 `3a1654cd…0c8a`.
- `finish` compared against the v2-20m-run-02 test rows, wrote `summary.json` (SHA256 `d94bb49f…95a6`), and marked the attempt finished. v2-200m does not run on test again.
- This is the third use of the `ime-pinyin-zh-v1-r1` test split: the v2-20m confirmation (2026-11-20), the guard of the c40 colloquial confirmation (2026-11-23), then this run. Each was registered in advance in its plan.
- No deviations.

## Next

- Per the [release scope](../../release-zh-v2-plan.md), v2-200m goes into `200m/` of the Hugging Face repository `Voltline/vimeml-zh-v2`: the FP32 inference bundle and the evaluation summary. The desktop models get no Core ML packages.
- Desktop latency is already measured ([report](../zh-desktop-latency-20261118/report.md): 83/205 ms p50/p95 with CUDA graphs in FP16).
