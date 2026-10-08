# Standard IME initial comparison — 2026-10-08

The preregistered comparison evaluates frozen V2.1 extension step 40,000 and V3 A step 25,146 on identical AzooKey pools. It uses release `ime-standard-ja-v1-ai-expert-r1`, AI-reviewed labels, and FP32 scoring. The originally blind split is consumed by this experiment and becomes a fixed reference thereafter.

## Results

| Track | Engine | V2.1 | V3 A |
| --- | ---: | ---: | ---: |
| Development WRIME / 483 | — | 298 | 310 |
| Development JMultiWOZ / 245 | — | 207 | 198 |
| Development micro / 728 | 453 | 505 | 508 |
| Development equal-source macro | — | 73.09% | 72.50% |
| Reference WRIME / 475 | — | 309 | 318 |
| Reference JMultiWOZ / 250 | — | 226 | 228 |
| Reference micro / 725 | 467 | 535 | 546 |
| Reference equal-source macro | — | 77.73% | 79.07% |
| Historical regression / 500 | 317 | 374 | 368 |

Candidate coverage is 630/728, 650/725, and 443/500. Development Top-5 is 618 for both models; MRR is 0.75934 / 0.76814 and MinCER 0.04255 / 0.03871. Reference Top-5 is 636 / 639, MRR 0.79875 / 0.81147, and MinCER 0.04061 / 0.03642. Source-level effects differ; a favorable aggregate does not establish uniform domain improvement.

## Paired changes: A relative to V2.1

| Track | Corrected | Regressed | Net | Exact McNemar two-sided p |
| --- | ---: | ---: | ---: | ---: |
| Development WRIME | 40 | 28 | +12 | 0.1818 |
| Development JMultiWOZ | 10 | 19 | -9 | 0.1360 |
| Development micro | 50 | 47 | +3 | 0.8392 |
| Reference WRIME | 36 | 27 | +9 | 0.3135 |
| Reference JMultiWOZ | 9 | 7 | +2 | 0.8036 |
| Reference micro | 45 | 34 | +11 | 0.2604 |
| Historical regression | 14 | 20 | -6 | 0.3915 |

Relative to engine order, development corrections/regressions are 100/48 for V2.1 and 104/49 for A; reference values are 107/39 and 120/41. These are full-denominator results, including recall misses.

## Interpretation and records

A shows a small reference improvement and historical regression, without a stable source-wide or statistically established replacement advantage. V2.1 remains deployed. AI-expert labels are not native-speaker gold; case-level statistics do not correct author/user dependence or multiple comparisons. Corpus overlap screening remains limited.

The plan, manifest, and consumption receipt preserve the initial evaluation boundary. New model comparisons treat the 725 cases as exposed reference, not fresh blind evidence. Original per-case outputs remain in `outputs/ime-eval/standard-v1-{v21,v3-a}-{development,blind}/`; historical caches reside in `outputs/benchmarks/ime-standard-ja-v1/legacy-{v21,v3-a}/`. [Protocol](standard-ime.md) and [B evaluation](../reports/v3-20261008/b-evaluation.md) describe later use.
