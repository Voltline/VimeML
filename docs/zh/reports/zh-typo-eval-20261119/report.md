# zh keyboard-typo arm: evaluation (2026-11-19)

The `typo` arm of the [keyboard-typo plan](../../training/training-zh-typo-plan.md) against its control r3-ctrl,
on development only. **Result: the arm does not pass.** Task-mode scoring learns to correct typos
(13.0 → 32.2 typo @2), but the LM rule that Vime already uses reaches 34.3 with any of the models,
v1 included, and the plan's primary metric requires the arm to be significantly above it.

## Arms

| Arm | Checkpoint sha256 | Recipe |
| --- | --- | --- |
| r3-ctrl (control) | — | kd-cont, 200M teacher, α 0.5, seed 20261117, 30,000 updates, 512 samples per update |
| typo | `a7f6e1e9…` | the same plus 48 `typo` samples per update (cross-entropy only; store manifest `3bc0164b…`), 560 per update, 0.19 passes over the typo store |

Validation loss by kind, at the end of training:

| Kind | typo arm | r3-ctrl |
| --- | --- | --- |
| typo | 0.8389 (1.8871 at step 0) | — |
| full | 0.2487 | 0.2457 |
| mixed | 0.8182 | 0.8020 |
| initials | 1.4125 | 1.4066 |
| error | 0.5533 | 0.5280 |
| plain | 4.2550 | 4.2395 |

## Guards

| Metric | typo arm | r3-ctrl | v1 | Result |
| --- | --- | --- | --- | --- |
| Public pinyin self-rescored Top-1 (λ 1.6) | 54.2 | 54.6 | 52.7 | not significant vs r3-ctrl: pass |
| — full / initials | 86.9 / 27.9 | 86.7 / 28.7 | — | |
| Standard full-pinyin Top-1 | 0.9354 | 0.9368 | 0.9361 | net −1 vs v1, p = 1.0: pass |
| Prediction A Top-3 (descriptive) | 0.2526 | 0.2582 | — | |

## `ime-typo-zh-v1-r1` development

578 sentences, each clean, with one typo and with two typos (1,734 items). Pool manifest
`5d990a7a…`, built once by v1. The reference is among the corrections for 79.6% of one-typo inputs,
0.3% of two-typo inputs (the pool holds single edits only) and 0.9% of clean inputs. So typo1 @2 is
capped at about 80 and typo2 at about 0.3.

| Run | Rule | Margin | typo @2 | typo1 @2 | typo2 @2 | Typo shown correct | Clean false display | Clean @2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | lm | 3.00 fixed | 34.1 | 67.1 | 1.0 | 31.7 | 12.6 | 88.6 |
| v1 | lm | 3.48 calibrated | 34.0 | 67.0 | 1.0 | 31.6 | 9.5 | 89.1 |
| r3-ctrl | lm | 3.00 fixed | **34.3** | 67.6 | 1.0 | 31.9 | 14.2 | 88.6 |
| r3-ctrl | lm | 3.72 calibrated | 34.0 | 67.0 | 1.0 | 31.6 | 9.2 | 88.8 |
| r3-ctrl | task | −1.86 calibrated | 13.0 | 25.4 | 0.5 | 10.7 | 9.7 | 87.9 |
| typo | lm | 3.00 fixed | 34.3 | 67.5 | 1.0 | 31.8 | 13.3 | 88.6 |
| typo | lm | 3.88 calibrated | 34.0 | 67.0 | 1.0 | 31.6 | 9.2 | 88.8 |
| typo | task | −1.64 calibrated | **32.2** | 63.7 | 0.7 | 29.9 | 9.7 | 88.1 |

typo @2 is the mean of typo1 and typo2. Calibrated margins cap clean false display at 10% and then
maximize typo @2. Paired typo @2 (sentence-cluster sign-flip, 20,000 permutations):

| Run | Against | Diff | +/− | p |
| --- | --- | ---: | --- | ---: |
| typo / task / calibrated | r3-ctrl / lm / fixed 3 (the primary) | −2.2 | +6/−31 | 0.0001 |
| typo / task / calibrated | r3-ctrl / lm / calibrated | −1.8 | +9/−30 | 0.0011 |
| r3-ctrl / task / calibrated | r3-ctrl / lm / fixed 3 | −21.4 | +3/−250 | 5e-05 |
| r3-ctrl / lm / calibrated | r3-ctrl / lm / fixed 3 | −0.3 | +0/−4 | 0.13 |
| v1 / lm / calibrated | r3-ctrl / lm / fixed 3 | −0.3 | +4/−8 | 0.39 |
| typo / lm / calibrated | r3-ctrl / lm / fixed 3 | −0.3 | +0/−4 | 0.13 |

## Decision

| Rule | Result |
| --- | --- |
| Primary: typo arm (task rule) significantly above the control's LM rule (margin 3) | 32.2 vs 34.3, significantly lower: **fail** |
| Guard 1: self-rescored Top-1 not significantly below the control | pass |
| Guard 2: clean false display not above the control's; Standard not significantly below v1 | 9.7 vs 14.2; pass |

What the numbers show:

- Typo training teaches task-mode scoring to correct (r3-ctrl task 13.0 → typo task 32.2), but task
  mode stays below the LM rule.
- The LM rule gives the same typo @2 for v1, r3-ctrl and the typo arm (34.0–34.3). On this pool,
  correction quality does not depend on the model.
- Calibration matters more than the model: margin 3 shows a correction on 12.6–14.2% of clean
  sentences; the calibrated margin (3.5–3.9) brings that to 9.2–9.5% for −0.3 typo @2 (p ≥ 0.13).

## Consequences

- The typo arm is not a v2-20m candidate; the candidate stays kd-cont (or round 3's arm if better).
- Vime keeps the LM rule. `calibration.json` carries the rescoring λ and the LM-rule margin calibrated
  for the shipped model on this benchmark; there is no taskScore rule, so no taskScore reference or
  fixture is made.
- `typo` data is not added to round 4 or to the 800M.
- The typo store and the benchmark stay: the benchmark calibrates the margin of every new model.

Not reported: valid-spelling typos on their own and the private benchmark (descriptive items of the
plan); they do not change the decision.
