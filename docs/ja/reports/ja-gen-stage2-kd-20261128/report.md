# ja-gen stage 2.3: 20M distillation and development evaluation

> [Chinese version](../../../zh-CN/ja/reports/ja-gen-stage2-kd-20261128/report.md)

2026-11-28. Follows the 2.3 registration (2026-11-28, submitted before any distillation result) in the [ja-gen plan](../../training/training-ja-gen-plan.md). Development sets only: neither the `ime-kana-ja-v1` test split nor the `ime-kana-ja-colloq-v1` test split was read.

**Result: pass.** `20m-kd-cont` (s1) is the 3.3 start, and λ is frozen at 1.6.

- The main arm's kana development lenient Top-1 is 88.50: 7 items above its start `read-s1` (p 0.039) and 20 items above V4.2 reranking (p 0.0049). The four-seed mean is 87.92, 0.75 points above the 2.1 reading mean of 87.17, and 3 of 4 seeds are above their start.
- Standard reranking: 514 for the main arm, four-seed mean 512.00. Reading post-training in 2.1 cost about 11 items; distillation recovers 7.5 of them.
- Colloquial development (reference only): four-seed mean 88.46, with all 4 seeds above the 2.1 reading mean of 87.12.

## Training

- **Configuration**: `configs/ja/train-ja-kana-20m-kd-cont.toml` and `-s2/-s3/-s4`. Each arm continues its `read-s{n}` for 30,000 updates with the 200M final.pt (`47721ddd…`, frozen) as teacher, 0.5 × CE + 0.5 × KL (T = 1.0), mix `plain` 175 / `chat` 17 / `span` 192 / `sentence` 128, as registered.
- **Run**: each of the four arms ran all 30,000 updates in one run. Update time alone was 2,270–2,313 seconds per arm. Only final.pt is used.

| Arm | Start | final.pt SHA256 |
| --- | --- | --- |
| `20m-kd-cont` (s1, main arm) | `read-s1` `35ac5768…` | `5a44839657569b34dea57fae4a45a2050506f892a441e38715756727fe46a92f` |
| `20m-kd-cont-s2` | `read-s2` `10b16c52…` | `efb01bf9e1e01ac342139d19bb8a13f03df8b0d096d8d5485743755b5f3d38db` |
| `20m-kd-cont-s3` | `read-s3` `44cf4171…` | `63b7694f0eca8d567835bc967186e04d5319e14315600dff4be8155b1369850c` |
| `20m-kd-cont-s4` | `read-s4` `4345b8ed…` | `eb39608204ef559d8e5033bec8a2ddb5be178272f0ee8a22eb17db9a811400d0` |

Validation loss, start (step 0) → final.pt (for reading kinds, teacher-forced whole-sample accuracy in parentheses):

| Arm | plain | chat | web | span | sentence |
| --- | --- | --- | --- | --- | --- |
| s1 | 3.7765 → 3.7635 | 4.0096 → 3.9578 | 4.7271 → 4.2854 | 0.1014 → 0.0947 (79.72 → 80.83%) | 0.1227 → 0.1156 (68.04 → 68.47%) |
| s2 | 3.7889 → 3.7715 | 4.0129 → 3.9559 | 4.7675 → 4.2999 | 0.1024 → 0.0964 (79.92 → 80.54%) | 0.1228 → 0.1142 (68.29 → 69.32%) |
| s3 | 3.8050 → 3.7876 | 4.0090 → 3.9519 | 4.7439 → 4.2941 | 0.1001 → 0.0948 (79.93 → 80.42%) | 0.1209 → 0.1150 (68.53 → 69.29%) |
| s4 | 3.8016 → 3.7850 | 4.0095 → 3.9546 | 4.7393 → 4.2999 | 0.1023 → 0.0963 (79.75 → 80.43%) | 0.1245 → 0.1160 (68.38 → 68.83%) |

`web` is not in the distillation mix, yet its validation loss falls from about 4.74 to about 4.29.

## Evaluation setup

- As in 2.2: script `scripts/ja/benchmarks/evaluate_ja_gen_stage2.py`, candidate index SHA256 prefix `b60c485d7ba2`. On kana and colloquial development the final list is AzooKey N-best 20 ∪ constrained beam-16 generation bucketed by key position (top 10), ordered by the student's own generation score + λ × LM score with λ = 1.6. On Standard development the LM score reranks the AzooKey pool. Constrained search never fell back for any arm on either set (fallbacks = 0).
- **Statistics**: kana and Standard paired differences use author/dialogue-clustered sign-flip tests. The colloquial set has one item per document, so its test is item-level.
- **Check**: Top-1/Top-3, D5, generation alone, the λ sweep, the colloquial bands, the colloquial paired differences against `read-s{n}` and all paired differences against the 200M were recomputed from the per-row result files. The kana and Standard paired differences against `read-s{n}` and V4.2 come from the evaluation script's comparison output; their net values equal the differences of hit counts against the 2.1 arms.

## `ime-kana-ja-v1` development and Standard development

| Metric | s1 (main) | s2 | s3 | s4 | four-seed mean | 2.1 reading mean | 200M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kana lenient Top-1 (primary) | **88.50** | 87.50 | 88.17 | 87.50 | **87.92** | 87.17 | 89.33 |
| kana exact Top-1 | 71.33 | 70.50 | 72.83 | 71.17 | 71.46 | 71.25 | 72.00 |
| kana lenient Top-3 | 92.67 | 92.50 | 92.67 | 92.50 | 92.59 | 91.88 | 93.00 |
| D5 items, lenient Top-1 (/24) | 19 | 17 | 18 | 17 | 73.96% | 69.79% | 18 |
| generation alone, lenient Top-1 | 68.67 | 68.17 | 67.83 | 67.83 | 68.12 | 68.13 | 68.00 |
| Standard exact Top-1 (/728) | **514** | 522 | 507 | 505 | **512.00** | 504.50 | 517 |

Paired differences (lenient Top-1 on kana, exact Top-1 on Standard, items):

| Comparison | s1 | s2 | s3 | s4 |
| --- | ---: | ---: | ---: | ---: |
| kana: against own `read-s{n}` | **+7** (+11 / −4, p 0.039) | −1 (p 1.00) | +7 (+9 / −2, p 0.016) | +5 (+10 / −5, p 0.27) |
| kana: against V4.2 reranking | **+20** (+29 / −9, p 0.0049) | +14 (+26 / −12, p 0.058) | +18 (+26 / −8, p 0.0038) | +14 (+24 / −10, p 0.058) |
| kana: against 200M | −5 (+11 / −16, p 0.36) | −11 (+5 / −16, p 0.038) | −7 (+7 / −14, p 0.17) | −11 (+5 / −16, p 0.051) |
| Standard: against own `read-s{n}` | +13 (+33 / −20, p 0.085) | +17 (p 0.035) | −1 (p 1.00) | +1 (p 1.00) |
| Standard: against 200M | −3 (+32 / −35, p 0.79) | +5 (+37 / −32, p 0.60) | −10 (+30 / −40, p 0.31) | −12 (+33 / −45, p 0.20) |

λ sweep, kana development lenient Top-1:

| λ | 0 | 0.8 | 1.2 | 1.6 | 2.0 | 2.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s1 | 84.83 | 87.83 | 88.33 | **88.50** | 88.33 | 88.17 |
| s2 | 83.67 | 87.83 | 87.67 | 87.50 | 87.83 | 87.67 |
| s3 | 84.83 | 87.33 | 88.00 | 88.17 | 88.17 | 88.00 |
| s4 | 83.83 | 86.83 | 87.17 | 87.50 | 87.50 | 87.17 |
| four-seed mean | 84.29 | 87.46 | 87.79 | **87.92** | 87.96 | 87.75 |

## `ime-kana-ja-colloq-v1` development (colloquial, 600 items, reference only)

| Model | lenient Top-1 | lenient Top-3 | exact Top-1 | exact Top-3 | lenient zero / low / mid / high | exact zero / low / mid / high |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| AzooKey order | 72.83 | 86.50 | 55.50 | 71.67 | 72.19 / 72.39 / 70.21 / 75.86 | 62.25 / 55.97 / 44.68 / 58.05 |
| 2.1 reading mean | 87.12 | 93.38 | 72.58 | 86.29 | 89.73 / 84.14 / 87.41 / 86.93 | 77.98 / 71.08 / 68.26 / 72.56 |
| kd-cont s1 | 89.17 | 92.83 | 74.50 | 85.83 | 90.73 / 86.57 / 90.07 / 89.08 | 78.15 / 70.90 / 69.50 / 78.16 |
| kd-cont s2 | 88.33 | 93.67 | 73.83 | 86.67 | 90.07 / 83.58 / 89.36 / 89.66 | 76.82 / 70.90 / 70.21 / 76.44 |
| kd-cont s3 | 88.17 | 93.50 | 74.00 | 86.33 | 91.39 / 85.07 / 88.65 / 87.36 | 78.15 / 71.64 / 69.50 / 75.86 |
| kd-cont s4 | 88.17 | 93.00 | 73.67 | 87.17 | 89.40 / 85.07 / 87.94 / 89.66 | 76.82 / 71.64 / 69.50 / 75.86 |
| **kd-cont mean** | **88.46** | **93.25** | **74.00** | **86.50** | 90.40 / 85.07 / 89.01 / 88.94 | 77.48 / 71.27 / 69.68 / 76.58 |
| 200M | 89.67 | 95.33 | 75.17 | 88.17 | 91.39 / 85.82 / 88.65 / 91.95 | 77.48 / 71.64 / 70.92 / 79.31 |

- Generation alone, lenient Top-1: 72.67 / 73.83 / 72.67 / 73.00 (mean 73.04), on par with AzooKey order (72.83) and below the 200M (74.33).
- λ sweep, four-seed mean: 85.25 at 0, 87.83–88.46 across 0.8–2.5, highest at 1.6.

Paired differences in lenient Top-1 (items, item-level sign-flip):

| Comparison | s1 | s2 | s3 | s4 |
| --- | ---: | ---: | ---: | ---: |
| against own `read-s{n}` | +11 (+17 / −6, p 0.038) | +5 (+11 / −6, p 0.33) | +8 (+13 / −5, p 0.096) | +8 (+13 / −5, p 0.096) |
| against 200M | −3 (+11 / −14, p 0.68) | −8 (+8 / −16, p 0.15) | −9 (+10 / −19, p 0.14) | −9 (+14 / −23, p 0.19) |

## Decision

By the registered rule:

1. **Primary**:
   - Four-seed mean 87.92 > 87.17. Met.
   - 3 of 4 seeds are above their `read-s{n}` (s1 +7, s3 +7, s4 +5; s2 −1). Met.
   - Main arm 88.50 ≥ 87.33, and +20 against V4.2 reranking with p 0.0049 < 0.05. Met.
2. **Guard**: main arm Standard 514 ≥ 501; four-seed mean 512.00 ≥ 504.50. Met.
3. **λ**: the four-seed mean at 2.0 (87.96) is only 0.04 points above 1.6, and only s2 is higher there (s1 lower, s3 and s4 equal). This is below the change threshold, so λ is frozen at 1.6.

**Pass**: `20m-kd-cont` (s1, `5a448396…`) is the 3.3 start, and 3.3 uses λ = 1.6.

## Observations

- **Distillation recovers part of the teacher's lead**: the 200M leads the four `read-s{n}` on kana by 52 items in total (+12 / +10 / +14 / +16). After distillation the students' net gain over their starts totals 18 items, about a third. The students are still 5–11 items behind the 200M, significantly so for s2 (p < 0.05).
- **Standard recovers the reranking cost**: reading post-training was about 11 items below the LM control (504.50 against 515.25). After distillation the mean is 512.00, with 514 for the main arm and 522 for s2. The seed spread is wider than in 2.1 (505–522). No seed differs significantly from the 200M.
- **The gain is in lenient Top-1 and D5**: the kana exact Top-1 mean is only 0.2 points higher and generation alone is unchanged (68.12 against 68.13), while the D5 mean is 4.2 points higher. The change comes from how the union is ordered, not from generation itself.
- **Colloquial development**: the four-seed mean is 1.34 points above the 2.1 reading arm. By band the high band gains most (lenient +2.0 points, exact +4.0), the same direction as the 200M against the reading arm in 2.2. The lenient Top-3 mean, 93.25, is slightly below the reading arm (93.38).
- **`web` validation loss**: `web` is not in the distillation mix, yet its loss falls by about 0.45, and whole-sample accuracy on `web_span` / `web_sentence` rises by 0.7–1.9 points. The teacher's web distribution reaches the student through the KL term on `plain`.

## Next

- **3.3**: the three colloquial post-training arms (ctrl / c40 / c40-p85) start from the `20m-kd-cont` final.pt (`5a448396…`) with λ = 1.6. The start and teacher SHA256 are filled in for all three configs.
