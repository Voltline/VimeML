# IME metrics and statistics v2

Registered 2026-10-21; the policy file is [`configs/ime-metrics-v2.json`](../../configs/ime-metrics-v2.json). Every later IME comparison, including the first scoring of any new blind set, is reported this way. Results frozen before this date keep their original primary metric and are not rewritten. Background: the [IME error decomposition](../reports/analysis-20261021/ime-errors.md) found that about half of the remaining Standard errors are pool misses and about four in ten are notation variants, so exact Top-1 alone and per-item McNemar tests are not enough to judge model differences.

## Metrics

| Level | Metric | Meaning |
| --- | --- | --- |
| Primary | Exact Top-1 | As in v1: frozen acceptable notations, full denominator (pool misses included) |
| Secondary | Notation-lenient Top-1 | Exact hits plus choices that are conservative notation variants of an in-pool reference (same UniDic lexeme sequence **and** kanji-set containment); implemented in `vimeml.benchmarks.notation` |
| Secondary | Pool-conditional Top-1 | Only items whose frozen candidate pool contains a reference |
| Diagnostic | Lenient upper bound, error classes | Analysis only, never used for decisions |

The candidate pool stays AzooKey n-best 20 (converter `d59a28e4`, typo correction and Zenzai off). n-best 50 is reported alongside on v2 blind and the homophone set; switching the primary pool needs a new metrics version.

## Statistics

Items from the same author, dialogue user or sentence template tend to flip together, so per-item McNemar overstates the evidence. v2 tests at the cluster level:

- **Primary test**: two-sided paired sign-flip permutation test over author clusters (WRIME authors, JMultiWOZ users, Legacy source documents, homophone groups for the homophone set; every other item is its own cluster). Exact enumeration with at most 20 discordant clusters, otherwise 20,000 draws with a fixed seed.
- **Interval**: 95% percentile interval of the net difference from a cluster bootstrap (10,000 draws, seed 0).
- **Sensitivity**: template clusters (character-trigram Jaccard ≥ 0.5 after NFKC and digit masking) and the union of author and template clusters. The union links Standard into one 185-item component, so it is sensitivity analysis only, never the primary clustering.
- Per-item McNemar is still reported for comparison with v1 reports.

**Decision rule**: claiming a Top-1 improvement requires a positive exact Top-1 net on the primary track with author-cluster p < 0.05, and no notation-lenient regression with p < 0.05 on any reported track. Significance by per-item McNemar alone is not enough.

Implemented in [`statistics.py`](../../src/vimeml/benchmarks/statistics.py); the report entry point [`report_ime_metrics.py`](../../scripts/benchmarks/report_ime_metrics.py) reads only completed FP32 scores.

## Track roles

| Track | Role |
| --- | --- |
| Standard v1 development /728 | Development and diagnosis; used many times |
| [Standard v2 blind](standard-ime-v2-blind.md) /729 | Primary confirmation track: commit the model plan first, then score once; coverage statistics open together with scoring |
| Legacy regression /500 | Notation regression alarm on old web text; references are drafts, not a capability metric |
| AJIMEE /200 | Public encyclopedic written-text check, semantic errors reported; contamination not excluded |
| Historical development /137 | Smoke test: flagged when exact Top-1 drops by more than 3, never used to compare |
| [Homophone challenge](homophone-ime.md) /619 | Written-text semantic challenge; candidates exported after freezing |
| [Segmented-input prototype](incremental-ime.md) | Protocol prototype on exposed development data; no capability conclusions |

## Retrospective description of earlier results

The completed non-blind scores were recomputed under v2; results are in `outputs/metrics-v2-20261021/summary.json`. All of these sets have been used repeatedly. The recomputation only shows how the statistical treatment changes the picture and changes no earlier conclusion.

Standard development /728: 62 author clusters, 630 items in pool.

| Model | Exact | Notation-lenient | Pool-conditional |
| --- | ---: | ---: | ---: |
| V2.1 | 505 | 578 | 0.802 |
| V4.2 epoch 20 | **523** | **594** | **0.830** |
| V4.3 epoch 30 | 519 | 588 | 0.824 |
| V5 epoch 30 | 514 | 583 | 0.816 |

| Contrast (Standard) | Exact gain/loss | Per-item p | Author-cluster p | Cluster bootstrap 95% | Design effect |
| --- | --- | ---: | ---: | --- | ---: |
| V2.1 → V4.2 | 46/28 (+18) | 0.047 | 0.081 | [−1, 37] | 1.22 |
| V2.1 → V4.2 (notation-lenient) | 28/12 (+16) | 0.017 | 0.079 | [0, 33] | 1.68 |
| V4.2 → V5 epoch 30 | 31/40 (−9) | 0.34 | 0.30 | [−24, 6] | 0.80 |
| V4.2 → V5 epoch 30 (notation-lenient) | 11/22 (−11) | 0.080 | 0.061 | [−21, −1] | 0.77 |

V2.1 → V4.2 is just significant per item but not by author cluster (p = 0.081), consistent with the earlier fixed-reference result of only +6, not significant. Under the v2 decision rule V4.2 is not a confirmed improvement. No other pairwise comparison is significant under any treatment. Legacy, AJIMEE and historical development have one item per cluster, so their cluster tests match the per-item tests.

## Limitations

The notation-lenient rule agrees with the earlier per-item AI labels about 90% of the time; it is not a native-speaker judgement. Author clusters only cover known source groupings; topic, wizard and web-repost relations are not modeled.
