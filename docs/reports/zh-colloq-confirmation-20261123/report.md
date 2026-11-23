# Test confirmation of the colloquial candidate colloq-c40 (2026-11-23)

Run under the [preregistered confirmation plan](../../benchmarks/confirmation-zh-colloq-plan.md) (registered in a1190f8): **passed**. `colloq-c40` replaces `kd-cont` as `vimeml-zh-v2-20m`. This report gives summary numbers only, never test text or candidates. The Chinese version, [docs/zh-CN/reports/zh-colloq-confirmation-20261123/report.md](../../zh-CN/reports/zh-colloq-confirmation-20261123/report.md), is the reference text.

- Preflight: the regenerated `colloq-c40` colloquial development run matches the judged run on all 4,509 inputs (0 flips, Top-1 moved 0.0).
- One run (run-02), the only finished attempt in the consumption record; `summary.json` SHA256 `760cf6b2…`.

## Decision

| Condition | Result | Verdict |
| --- | --- | --- |
| H1: colloquial test, all inputs, `kd-cont` → `colloq-c40`, self-rescored Top-1 | 39.0% → 42.3%, +3.2 points (+217/−69), p 5e-5 | pass |
| Public pinyin test, no significant drop | 49.1% → 51.4%, +2.3 points (+152/−74), p 0.0002 | pass |
| Public pinyin test, full pinyin, no significant drop | 82.6% → 84.4%, +1.7 points (+14/−4), p 0.031 | pass |
| Standard blind full-pinyin rerank, v1 → `colloq-c40` | net +7 (+21/−14), exact McNemar p 0.31, cluster bootstrap 95% [−4, +18] | pass |

This is the public pinyin test's second use (the first was the v2-20m confirmation of 2026-11-20), so it serves only as a guard. The candidate is significantly better than `kd-cont` there too, but that is descriptive only.

## Accuracy on test

Self-rescored (λ = 1.6) Top-1. Wilson 95% intervals are per input and ignore the correlation between inputs of the same sentence.

**Colloquial test** (775 sentences, 4,560 inputs)

| Input kind | n | `kd-cont` | `colloq-c20` | `colloq-c40` | `colloq-c40` Top-10 coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 4,560 | 39.0 | 41.5 | **42.3** (40.9–43.7) | 60.9 |
| full pinyin | 775 | 74.6 | 75.9 | **76.1** | 91.2 |
| mixed abbreviation | 3,010 | 35.7 | 38.7 | **39.5** | 59.5 |
| all initials | 775 | 16.3 | 17.9 | **19.1** | 36.4 |

**Public pinyin test** (576 sentences, 3,436 inputs)

| Input kind | n | `kd-cont` | `colloq-c40` |
| --- | ---: | ---: | ---: |
| all | 3,436 | 49.1 | **51.4** (49.7–53.0) |
| full pinyin | 576 | 82.6 | **84.4** |
| mixed abbreviation | 2,284 | 47.4 | **49.8** |
| all initials | 576 | 22.2 | **24.7** |

## Descriptive comparisons (not part of the decision)

- Colloquial test: `colloq-c20` vs `kd-cont` +2.4 points (+174/−63, p 5e-5); `colloq-c40` vs `colloq-c20` +0.8 points (+107/−70, p 0.030).
- On development c40 vs c20 was +0.4 (p 0.34). On test the direction is the same and the gap slightly larger, consistent with choosing c40.

## Next

`vimeml-zh-v2-20m` now uses the `colloq-c40` checkpoint (SHA256 `0ec5ffb1…`). It goes through v2-20m's existing Mac alignment and iOS integration. Its architecture is that of `kd-cont`, so the export path does not change.
