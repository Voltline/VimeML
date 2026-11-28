# ja-gen stage 3.3: 20M colloquial post-training and development evaluation

> [Chinese version](../../../zh-CN/ja/reports/ja-gen-stage3-colloq-20261128/report.md)

2026-11-28. Follows 3.3 in the stage 3 registration (2026-11-26) of the [ja-gen plan](../../training/training-ja-gen-plan.md). Development sets only: neither the `ime-kana-ja-v1` test split nor the `ime-kana-ja-colloq-v1` test split was read.

**Result: `kd-cont` is kept.**

- Colloquial development lenient Top-1 (primary): c40 scores 88.83, the same as ctrl and 2 items below its start kd-cont (89.17). p85 is 2 items above c40, not significant.
- All three arms pass the guards.
- What the colloquial data changes is exact Top-1: on kana development c40 is 16 items above ctrl (p 0.002) and 12 above kd-cont (p 0.032). This metric is not part of the 3.3 rule, so 3.4 was registered to replicate it with more seeds.

## Training

- **Configuration**: `configs/ja/train-ja-kana-20m-colloq-{ctrl,c40,c40-p85}.toml`. All three arms continue the `20m-kd-cont` final.pt (`5a448396…`) for 30,000 updates, distilled from the 200M teacher (`47721ddd…`) with α = 0.5, T = 1.0, cosine 1e-4 → 1e-5, seed 20261205, 512 samples per update.
  - `colloq-ctrl`: `ja-kana-v1-12k` only, with the kd-cont mix.
  - `colloq-c40`: 304 samples per update from `ja-kana-v1-12k` and 208 from the colloquial store `ja-colloq-v1-12k` (`p_chat ≥ 0.70`).
  - `colloq-c40-p85`: the same, with the colloquial store `ja-colloq-v1-p85-12k` (`p_chat ≥ 0.85`).
- **Run**: each arm ran in one go. Update time alone was 2,224–2,307 seconds per arm. Only final.pt is used.

| Arm | final.pt SHA256 |
| --- | --- |
| `colloq-ctrl` | `3c0c7dd8520c8cebe1b730b0e14102549a99992a91448cd2a90e431e1a1ee52e` |
| `colloq-c40` | `19b04ddfe0f98041ffe50511cab9d64d2f2a20b2918f5e78cc78c0768981d15f` |
| `colloq-c40-p85` | `7dbca92bbde4ae056007e570564e5d47495e2783a2f8bb6fd3f8b3f5532ad3d0` |

Validation loss, step 0 (the same for all three arms) → final.pt (for reading kinds, teacher-forced whole-sample accuracy in parentheses):

| Arm | plain | chat | web | span | web_span |
| --- | --- | --- | --- | --- | --- |
| step 0 | 3.7888 | 3.9578 | 4.2672 | 0.0959 (80.25%) | 0.1495 (76.38%) |
| ctrl | 3.7741 | 3.9376 | 4.2449 | 0.0935 (80.88%) | 0.1443 (76.54%) |
| c40 | 3.8397 | 3.9020 | 4.1977 | 0.0937 (80.48%) | 0.1323 (78.93%) |
| c40-p85 | 3.8418 | 3.9031 | 4.2161 | 0.0940 (80.68%) | 0.1325 (78.65%) |

The two colloquial arms lower `chat`, `web` and `web_span` more, while `plain` rises by about 0.05.

## Evaluation setup

- As in 2.3: script `scripts/ja/benchmarks/evaluate_ja_gen_stage2.py`, candidate index SHA256 prefix `b60c485d7ba2`, λ = 1.6 (the value frozen in 2.3). On kana and colloquial development the final list is AzooKey N-best 20 ∪ constrained beam-16 generation bucketed by key position (top 10); on Standard development the LM score reranks. No evaluation fell back (fallbacks = 0).
- **Statistics**: kana and Standard use author/dialogue-clustered sign-flip tests; the colloquial set is paired item by item.
- **Check**: every number was recomputed from the per-row result files, including the exact Top-1 paired differences.

## `ime-kana-ja-colloq-v1` development (colloquial, 600 items)

| Model | lenient Top-1 (primary) | lenient Top-3 | exact Top-1 | exact Top-3 | generation alone, lenient Top-1 | lenient zero / low / mid / high | exact zero / low / mid / high |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| kd-cont (start) | **89.17** | 92.83 | 74.50 | 85.83 | 72.67 | 90.73 / 86.57 / 90.07 / 89.08 | 78.15 / 70.90 / 69.50 / 78.16 |
| ctrl | 88.83 | 93.50 | 74.00 | 86.67 | 72.17 | 91.39 / 84.33 / 90.07 / 89.08 | 76.82 / 70.15 / 69.50 / 78.16 |
| c40 | 88.83 | **94.17** | **75.17** | 87.00 | **74.33** | 90.07 / 85.82 / 89.36 / 89.66 | 78.15 / 73.88 / 70.21 / 77.59 |
| c40-p85 | **89.17** | 94.00 | **75.17** | **87.33** | 74.00 | 90.73 / 84.33 / 89.36 / 91.38 | 78.81 / 72.39 / 70.21 / 78.16 |

## `ime-kana-ja-v1` development and Standard development (guards)

| Model | kana lenient Top-1 | kana exact Top-1 | kana lenient Top-3 | D5 items (/24) | generation alone, lenient Top-1 | Standard exact Top-1 (/728) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kd-cont (start) | **88.50** | 71.33 | 92.67 | 19 | 68.67 | 514 |
| ctrl | 87.83 | 70.67 | 92.33 | 17 | 69.33 | 513 |
| c40 | 88.17 | **73.33** | 92.67 | 17 | **70.00** | **518** |
| c40-p85 | 87.83 | 73.17 | 92.33 | 18 | 68.50 | 515 |

## Paired differences (items)

| Comparison | colloquial lenient Top-1 | colloquial exact Top-1 | kana lenient Top-1 | kana exact Top-1 | Standard exact Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| c40-p85 − c40 | +2 (+7 / −5, p 0.77) | | | | |
| c40 − ctrl | 0 (+7 / −7, p 1.00) | +7 (+15 / −8, p 0.21) | +2 (+7 / −5, p 0.78) | **+16 (+24 / −8, p 0.002)** | +5 (+22 / −17, p 0.51) |
| c40-p85 − ctrl | +2 (+8 / −6, p 0.79) | | | | |
| c40 − kd-cont | −2 (+3 / −5, p 0.73) | +4 (+12 / −8, p 0.50) | −2 (+6 / −8, p 0.77) | **+12 (+22 / −10, p 0.032)** | +4 (+19 / −15, p 0.55) |
| c40-p85 − kd-cont | 0 (+6 / −6, p 1.00) | +4 (+11 / −7, p 0.48) | −4 (+4 / −8, p 0.29) | +11 (+19 / −8, p 0.059) | +1 (+23 / −22, p 1.00) |
| ctrl − kd-cont | −2 (+4 / −6, p 0.75) | −3 (+5 / −8, p 0.58) | −4 (+3 / −7, p 0.13) | −4 (+10 / −14, p 0.49) | −1 (+17 / −18, p 1.00) |

## Decision

By the registered rule:

1. **p85 against p70**: c40-p85 is +2 against c40, p 0.77, not significant. The default colloquial arm c40 is kept.
2. **Guards**: no arm is significantly below kd-cont in kana development lenient Top-1 (lowest p 0.13). Standard is 513 / 518 / 515, all at least 501 and none significantly below kd-cont. Met.
3. **Replacement**: c40 against ctrl on the primary metric is 0, p 1.00, not significantly higher. No replacement.

**`20m-kd-cont` (`5a448396…`) is kept.**

## Observations

- **The colloquial data changes notation, not lenient hits**: c40 and ctrl differ only in the colloquial data. Their lenient Top-1 differs by 0–2 items on both development sets. Exact Top-1 is 16 items higher on kana development and 7 higher on colloquial development, and generation alone is 1–2 points higher. Most of the extra exact hits are items that lenient matching already counted as correct but whose notation differed from the answer, such as 下さい / ください, 分かる / わかる, オシャレ / おしゃれ. Changes go both ways; c40 matches the answer more often.
- **Continued training alone gains nothing**: ctrl only adds 30,000 updates and is level with or slightly below kd-cont everywhere. c40's exact Top-1 gain comes from the colloquial data, not from more updates.
- **p85 and p70 do not differ**: the two colloquial arms are within 0–2 items on every metric.
- **Single seed**: c40's p 0.032 against kd-cont is one of about ten metrics examined. 3.4 replicates c40 and ctrl with s2–s4, with kana development exact Top-1 as the primary metric; its rule was registered before the replica results.

## Next

- **3.4**: `colloq-ctrl-s{2,3,4}` and `colloq-c40-s{2,3,4}`; the configs are in the repo. If the rule passes, `colloq-c40` (s1) replaces kd-cont for stage 4; otherwise stage 4 confirms kd-cont.
