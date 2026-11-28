# ja-gen stage 3.4: colloquial post-training seed replicas

> [Chinese version](../../../zh-CN/ja/reports/ja-gen-stage3-seeds-20261128/report.md)

2026-11-28. Follows the 3.4 registration (2026-11-28, submitted before any replica result) in the [ja-gen plan](../../training/training-ja-gen-plan.md). Development sets only: neither the `ime-kana-ja-v1` test split nor the `ime-kana-ja-colloq-v1` test split was read. For 3.3, see the [3.3 report](../ja-gen-stage3-colloq-20261128/report.md).

**Result: pass.** `colloq-c40` (s1, `19b04ddf…`) replaces `kd-cont` and goes to the stage 4 confirmation.

- kana development exact Top-1 (primary): c40 four-seed mean 73.25, ctrl 71.21, kd-cont 71.33. c40 is above the same-seed ctrl at 4 of 4 seeds and above kd-cont at 4 of 4 seeds.
- Lenient Top-1 does not drop: c40 mean 88.54 on kana development (kd-cont 88.50) and 89.17 on colloquial development (kd-cont 89.17).
- Standard: c40 seeds 515–521, mean 517.75 (kd-cont 514).

## Training

`configs/ja/train-ja-kana-20m-colloq-{ctrl,c40}-s{2,3,4}.toml` differ from the two 3.3 arms only in seed (20261206 / 07 / 08). All start from `20m-kd-cont` (`5a448396…`) and ran 30,000 updates in one go; update time alone was 2,238–2,296 seconds per arm. The two 3.3 arms count as s1.

| Arm | final.pt SHA256 |
| --- | --- |
| `colloq-ctrl-s2` | `587fe79a2c86321963a7c160172a088ee11c4989ad48390a80a0d5303b671d96` |
| `colloq-ctrl-s3` | `d2d7eb2b63b2bb10b01eb76831012e5591cc75e612817ea54e4def95c35fc919` |
| `colloq-ctrl-s4` | `528a85b439bfafa42eff4e550c5ab5cde0376bd0814156152813434af5a98da5` |
| `colloq-c40-s2` | `d7e01806c3216edba9d0dcc110cd4e7e8b601296bafa94b5881dde84fa7e283c` |
| `colloq-c40-s3` | `b5e5f234cd3e6e015fdd27503f4ad3fc97d1c80c20501d1370351bca3f6ab8dc` |
| `colloq-c40-s4` | `ec9fb74470c2085e6a15f206680c14c6ee72b185e2cabaf29d484b9d24682927` |

Evaluation is as in 3.3 with λ = 1.6; no evaluation fell back (fallbacks = 0). Every number was recomputed from the per-row result files.

## Results (items, percentages in parentheses)

| Arm | kana exact Top-1 (primary) | kana lenient Top-1 | kana lenient Top-3 | colloquial lenient Top-1 | colloquial exact Top-1 | Standard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kd-cont | 428 (71.33) | 531 (88.50) | 556 | 535 (89.17) | 447 (74.50) | 514 |
| ctrl s1 / s2 / s3 / s4 | 424 / 428 / 432 / 425 | 527 / 529 / 529 / 528 | 554 / 555 / 553 / 554 | 533 / 531 / 534 / 533 | 444 / 446 / 444 / 441 | 513 / 521 / 521 / 518 |
| **ctrl mean** | 427.25 (71.21) | 528.25 (88.04) | 554.00 | 532.75 (88.79) | 443.75 (73.96) | 518.25 |
| c40 s1 / s2 / s3 / s4 | 440 / 434 / 443 / 441 | 529 / 532 / 534 / 530 | 556 / 555 / 557 / 555 | 533 / 535 / 535 / 537 | 451 / 455 / 454 / 453 | 518 / 517 / 515 / 521 |
| **c40 mean** | **439.50 (73.25)** | 531.25 (88.54) | 555.75 | 535.00 (89.17) | **453.25 (75.54)** | 517.75 |

Generation alone, mean lenient Top-1: c40 69.62 and ctrl 68.88 on kana development; c40 74.37 and ctrl 73.34 on colloquial development.

## Paired differences

| seed | c40 − ctrl: kana exact | c40 − kd-cont: kana exact | c40 − kd-cont: kana lenient | c40 − kd-cont: colloquial lenient | c40 − kd-cont: Standard |
| --- | ---: | ---: | ---: | ---: | ---: |
| s1 | +16 (p 0.002) | +12 (p 0.032) | −2 (p 0.77) | −2 (p 0.73) | +4 (p 0.55) |
| s2 | +6 (p 0.24) | +6 (p 0.31) | +1 (p 1.00) | 0 (p 1.00) | +3 (p 0.73) |
| s3 | +11 (p 0.16) | +15 (p 0.029) | +3 (p 0.55) | 0 (p 1.00) | +1 (p 1.00) |
| s4 | +16 (p 0.020) | +13 (p 0.045) | −1 (p 1.00) | +2 (p 0.75) | +7 (p 0.30) |

c40 − ctrl on colloquial exact Top-1 is +7 / +9 / +10 / +12 (p 0.21 / 0.076 / 0.041 / 0.022). Each ctrl seed against kd-cont on kana exact Top-1 is −4 / 0 / +4 / −3, none significant.

## Decision

By the registered rule:

1. **Primary**:
   - c40 mean 73.25 > ctrl mean 71.21, and 4 of 4 seeds are above the same-seed ctrl. Met.
   - c40 mean 73.25 > kd-cont 71.33, and 4 of 4 seeds are above 71.33. Met.
2. **Guards**:
   - c40 seeds against kd-cont in lenient Top-1: −2 / +1 / +3 / −1 on kana development, −2 / 0 / 0 / +2 on colloquial development, none significant. Met.
   - c40 seeds on Standard: 518 / 517 / 515 / 521, all at least 501 and all above kd-cont. Met.

**Pass**: `colloq-c40` (s1, `19b04ddfe0f98041ffe50511cab9d64d2f2a20b2918f5e78cc78c0768981d15f`) replaces `kd-cont` as the stage 4 candidate.

## Observations

- **The exact Top-1 gain replicates**: c40 is +6 to +16 items above ctrl at each of the 4 seeds, +12.25 on average, significant at 2 seeds individually. Colloquial exact Top-1 moves the same way (4 of 4, +9.5 on average).
- **The gain comes from the colloquial data**: ctrl only adds 30,000 updates, and its exact Top-1 mean is level with kd-cont (71.21 against 71.33).
- **Lenient Top-1 is about unchanged**: c40 is above ctrl on kana development at 4 of 4 seeds (+2 to +5), none significant individually, and level with kd-cont on average.
- **s1 is not the best seed**: the fixed s1 ranks third of four in kana exact Top-1 and fourth in colloquial lenient Top-1. As registered, no other seed is substituted.

## Next

- **Stage 4**: after preregistration, read the `ime-kana-ja-v1` test split (and the `ime-kana-ja-colloq-v1` test split) once to confirm `colloq-c40` against the current V4.2 deployment (AzooKey + V4.2 reranking).
