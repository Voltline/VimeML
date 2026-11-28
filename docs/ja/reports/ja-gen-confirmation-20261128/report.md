# ja-gen stage 4: colloq-c40 test confirmation

> [Chinese version](../../../zh-CN/ja/reports/ja-gen-confirmation-20261128/report.md)

> Naming, 2026-11-28: the ja-gen batch is named vimeml-ja-v6 (20M and 200M; see the [v6 release scope](../../release-ja-v6-plan.md)); this report was first written with the name `vimeml-ja-v2-20m`.

2026-11-28. Run once (run-01) under the [ja-gen test confirmation plan](../../benchmarks/confirmation-ja-gen-plan.md) (commit `8eb7bc6`, registered before any test result). This run was the first time any model scored the `ime-kana-ja-v1` or `ime-kana-ja-colloq-v1` test split.

**Result: pass.** `colloq-c40` (`19b04ddf…`) is released as `vimeml-ja-v6-20m` and replaces V4.2.

- **H1**: kana test lenient Top-1, V4.2 reranking 510 (85.00%) → `colloq-c40` 528 (88.00%), +18 items (+30 / −12), clustered sign-flip p 0.0024. Passed.
- **Guards**: `colloq-c40` against `kd-cont` in lenient Top-1 is +2 on kana test (p 0.82) and +3 on colloquial test (p 0.65); no drop. Met.
- **H2**: kana test exact Top-1, `kd-cont` 427 → `colloq-c40` 430, +3 (+16 / −13), p 0.73. Not passed. As planned, this does not change the release, but the release notes do not claim that colloquial post-training raised exact Top-1 on test.

## Run

- **Preflight** (development): V4.2 reranking of kana development matched the stage 0 order on 600/600 items; `colloq-c40` regenerated kana and colloquial development and matched the judged 3.3 rows item by item in lenient and exact Top-1 hits (529 / 440 and 533 / 451, 0 flips). Only then were the test splits opened.
- **Test**: V4.2 reranked both test splits; `colloq-c40` and `kd-cont` ran on both. Settings as on development: constrained beam 16, top 10, merged with AzooKey N-best 20, λ = 1.6, candidate index `b60c485d…`. Neither generation nor V4.2 fell back.
- **Consumption record**: run-01 only, status finished.
- **Check**: every number was recomputed from the per-row result files and matches the `summary.json` the run wrote.

## kana test (`ime-kana-ja-v1`, 600 items, 59 author/dialogue clusters)

| Setup | lenient Top-1 | lenient Top-3 | exact Top-1 | exact Top-3 | generation alone, lenient / exact Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AzooKey order | 449 (74.83) | 520 | 356 (59.33) | 469 | |
| V4.2 reranking (deployed) | 510 (85.00) | 543 | 402 (67.00) | 499 | |
| `kd-cont` | 526 (87.67) | 555 | 427 (71.17) | 524 | 412 / 276 |
| **`colloq-c40`** | **528 (88.00)** | 555 | **430 (71.67)** | 523 | 416 / 281 |

Lenient Top-1 by stratum (V4.2 / kd-cont / c40):

| Stratum | Items | V4.2 | kd-cont | c40 |
| --- | ---: | ---: | ---: | ---: |
| 4–10 chars | 180 | 153 | 159 | 156 |
| 11–20 chars | 195 | 174 | 179 | 182 |
| 21–40 chars | 225 | 183 | 188 | 190 |
| left context | 350 | 309 | 317 | 319 |
| no left context | 250 | 201 | 209 | 209 |
| WRIME | 400 | 318 | 333 | 335 |
| JMultiWOZ | 200 | 192 | 193 | 193 |
| D5 items | 19 | 16 | 17 | 17 |

## Colloquial test (`ime-kana-ja-colloq-v1`, 600 items, paired by item)

| Setup | lenient Top-1 | lenient Top-3 | exact Top-1 | exact Top-3 | generation alone, lenient / exact Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AzooKey order | 435 (72.50) | 515 | 326 (54.33) | 443 | |
| V4.2 reranking | 509 (84.83) | 552 | 381 (63.50) | 487 | |
| `kd-cont` | 534 (89.00) | 564 | 437 (72.83) | 523 | 438 / 317 |
| **`colloq-c40`** | **537 (89.50)** | **570** | **444 (74.00)** | **528** | 441 / 318 |

Lenient / exact Top-1 by rule band:

| band | Items | V4.2 | kd-cont | c40 |
| --- | ---: | --- | --- | --- |
| zero | 142 | 111 / 89 | 126 / 103 | 126 / 104 |
| low | 136 | 114 / 81 | 120 / 101 | 120 / 103 |
| mid | 148 | 133 / 94 | 135 / 110 | 138 / 111 |
| high | 174 | 151 / 117 | 153 / 123 | 153 / 126 |

## Paired differences (items)

| Comparison | Set | Metric | Difference (+ / −) | p |
| --- | --- | --- | ---: | ---: |
| **H1**: V4.2 → c40 | kana | lenient Top-1 | **+18** (+30 / −12) | **0.0024** |
| **H2**: kd-cont → c40 | kana | exact Top-1 | +3 (+16 / −13) | 0.73 |
| Guard: kd-cont → c40 | kana | lenient Top-1 | +2 (+9 / −7) | 0.82 |
| Guard: kd-cont → c40 | colloquial | lenient Top-1 | +3 (+11 / −8) | 0.65 |
| Fallback test: V4.2 → kd-cont | kana | lenient Top-1 | +16 (+29 / −13) | 0.025 |
| V4.2 → c40 | kana | exact Top-1 | +28 (+55 / −27) | 0.0054 |
| V4.2 → kd-cont | kana | exact Top-1 | +25 (+57 / −32) | 0.016 |
| AzooKey → V4.2 | kana | lenient / exact Top-1 | +61 / +46 | < 0.001 |
| V4.2 → c40 | colloquial | lenient / exact Top-1 | +28 (+39 / −11) / +63 (+81 / −18) | < 0.001 |
| V4.2 → kd-cont | colloquial | lenient / exact Top-1 | +25 / +56 | < 0.001 |
| kd-cont → c40 | colloquial | exact Top-1 | +7 (+18 / −11) | 0.27 |
| AzooKey → V4.2 | colloquial | lenient Top-1 | +74 | < 0.001 |

## Observations

- **The gain over V4.2 matches development**: kana lenient Top-1 was +18 on development (529 vs 511) and is +18 on test (528 vs 510), with p going from 0.013 to 0.0024. The exact Top-1 gain is larger (+28). On the colloquial test split the gain over V4.2 is clearer still: +28 lenient, +63 exact.
- **c40's exact Top-1 gain over kd-cont does not replicate on test**: +12 on development (p 0.032; four-seed mean +12.25), only +3 on kana test (p 0.73) and +7 on colloquial test (p 0.27). Both test splits point the same way, but neither the size nor the significance supports the claim that colloquial post-training makes notation match the answer more often.
- **c40 is not below kd-cont anywhere**: lenient Top-1, Top-3 and the colloquial bands are level or slightly higher. Between the two 20M models, c40 is no worse than kd-cont; the preregistered rule decides that it ships.
- **The merged list beats generation alone**: generation-alone lenient Top-1 (kana 69.33%, colloquial 73.50%) is below the merged list; the final result depends on merging AzooKey N-best with generated candidates plus self rescoring.

## Next

- **Release `vimeml-ja-v6-20m`**: `colloq-c40` goes to Mac Core ML alignment and iOS integration; the handoff script `scripts/ja/deployment/prepare_ja_v6_20m.py` switches to `colloq-c40`.
- **Test status**: both test splits are used and are no longer described as unused.
