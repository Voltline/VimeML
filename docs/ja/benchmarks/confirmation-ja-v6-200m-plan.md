# vimeml-ja-v6-200m test confirmation plan (preregistered)

> [Chinese version](../../zh-CN/ja/benchmarks/confirmation-ja-v6-200m-plan.md) (reference text)

Drafted 2026-11-28. The plan counts as registered only once it and the execution code are committed; until then the 200M is not run on the `ime-kana-ja-v1` or `ime-kana-ja-colloq-v1` test split. Release gate principles are in the [v6 release scope](../release-ja-v6-plan.md#release-gates); this plan gives the concrete tests.

The entry point is [confirm_ja_v6_200m.py](../../../scripts/ja/benchmarks/confirm_ja_v6_200m.py), and everything the plan fixes is a constant in [ja_v6_200m_confirm.py](../../../src/vimeml/benchmarks/ja_v6_200m_confirm.py). Search, merging and scoring are those of the [20M confirmation](confirmation-ja-gen-plan.md) (`kana_gen_ja.evaluate_arm`, constants from `ja_gen_confirm`).

## Questions

The 200M is the desktop model evaluated on development in 2.2 ([2.2 report](../reports/ja-gen-stage2-200m-20261128/report.md)) and is used on the desktop only. This plan asks on test:

- Is its kana conversion lenient Top-1 above the deployed V4.2 setup (AzooKey N-best 20 + V4.2 reranking)?
- Is its lenient Top-1 significantly below the phone model `vimeml-ja-v6-20m` on the kana or colloquial test?
- Is it significantly above `vimeml-ja-v6-20m`?

On development the 200M is only slightly ahead of `vimeml-ja-v6-20m` (see "Reading the outcome"), so the primary hypothesis is not "above the 20M". It is the 20M confirmation's test against V4.2, plus guards that the 200M is not below the 20M. The Chinese v2-200m's primary hypothesis was "above v2-20m"; this differs on purpose.

## Models

| Role | Model | Checkpoint SHA256 |
| --- | --- | --- |
| Candidate | `vimeml-ja-v6-200m`: 2.2 joint reading pretraining 200M (`configs/ja/train-ja-kana-200m-joint.toml`, 193,625,088 parameters, 845,000 updates) | `47721ddd357cb3a720d7108e4aeaa60f671d83167504fd9f649365b37d70a843` |
| Baseline (H1) | V4.2 reranking, the rows the 20M confirmation run-01 wrote on both test splits | (V4.2 `0e83ea74…`, not rerun) |
| Guards and H2 | `vimeml-ja-v6-20m` (`colloq-c40`), the rows the 20M confirmation run-01 wrote on both test splits | (`19b04ddfe0f98041ffe50511cab9d64d2f2a20b2918f5e78cc78c0768981d15f`, not rerun) |

The 200M uses exactly the 20M settings: tokenizer `ja-unigram-12288` (`61d71460…`), candidate index `b60c485d…`, constrained beam 16 bucketed by key position, top 10, merged with AzooKey N-best 20 and ordered by gen + λ·LM with λ = 1.6, using the 200M's own scores. FP32, TF32 off.

**Baselines from the run-01 rows**: V4.2 and c40 are not run on test again. The script checks that the 20M confirmation's consumption record has a single finished run-01 (shipping c40), that its `summary.json` SHA256 matches the record, and that the four per-row files (V4.2 and c40, two test splits each) reproduce the `summary.json` tables exactly. Their SHA256 go into this run's `preflight.json` and `summary.json`.

## Test sets

As in the 20M confirmation, with the same registered SHA256:

- **`ime-kana-ja-v1` test**: 600 items (WRIME 400 + JMultiWOZ 200, 59 author / dialogue clusters), pool manifest `b11df321…`.
- **`ime-kana-ja-colloq-v1` test**: 600 colloquial spans, pool manifest `667e12ce…`.

Both were first used by the 20M confirmation on 2026-11-28. This is their second use, for the 200M only, run once.

## Hypotheses and decision

Kana test: cluster sign-flip test by author / dialogue. Colloquial test: item-level pairing. Two-sided, α = 0.05.

**Primary hypothesis H1**: kana test, V4.2 reranking → 200M, lenient Top-1, difference > 0 and p < 0.05.

**Guards** (both must hold):

1. kana test, `vimeml-ja-v6-20m` → 200M, lenient Top-1: no difference < 0 with p < 0.05;
2. colloquial test, `vimeml-ja-v6-20m` → 200M, lenient Top-1: the same.

**H2** (fixed sequence, tested only after H1 passes, α = 0.05): kana test, `vimeml-ja-v6-20m` → 200M, lenient Top-1, difference > 0 and p < 0.05. It is the same pairing as guard 1, in the other direction.

H1 and H2 are tested in fixed sequence, without a further multiple-comparison correction. The Standard blind is not used.

**Reading the outcome**:

- On development, the 200M is +25 over V4.2 reranking in lenient Top-1 (536 vs 511, p 0.0002). Against `vimeml-ja-v6-20m`: kana lenient 536 vs 529 (+7), exact 432 vs 440 (−8); colloquial lenient 538 vs 533 (+5), exact 451 vs 451.
- On test, the 20M confirmation gave: kana lenient V4.2 510, c40 528; colloquial lenient V4.2 509, c40 537.
- **H1 passes and the guards hold**: the 200M is used on the desktop as `vimeml-ja-v6-200m`. H2 does not change the release. If H2 passes, the release notes say the 200M's test lenient Top-1 is above the 20M's; if not, they report the numbers and say the desktop and phone models do not differ significantly on test.
- **Otherwise**: the 200M is not released as `vimeml-ja-v6-200m`, the desktop uses `vimeml-ja-v6-20m`, and v6 releases only the 20M. The test splits are not reopened for another 200M checkpoint.
- Standard development reranking is 517, which meets release gate 3 (at or above 501); no test check of it is made.

## Descriptive report (no decision)

- Lenient / exact Top-1 and Top-3 of V4.2, c40 and the 200M on both test splits, by length band, left context and source; generation-only Top-1; D5 items; colloquial rule bands.
- The remaining pairings of the 200M against V4.2 on both splits (kana exact; colloquial lenient and exact), and the 200M against c40 in exact Top-1.
- Search fallback counts.

## Execution

One GPU, one process, in the checkout that ran the 20M confirmation (it needs `outputs/benchmarks/ja-gen-confirmation/run-01`).

1. Commit this plan and the execution code. At run time the script checks that they are committed and unmodified and that every registered file has its SHA256, then checks the run-01 baselines as above.
2. **Preflight** (development only): the 200M regenerates kana and colloquial development; its lenient and exact Top-1 hit counts must each be within 3 of the judged 2.2 results (kana 536 / 432, colloquial 538 / 451), with no search fallback. Otherwise test is not opened. The preflight can run alone (`--preflight-only`).
3. Write the consumption record `outputs/benchmarks/ja-v6-200m-confirmation/consumption.json`, run the 200M on both test splits, compute the decision and write `summary.json`.
4. Run once. If a technical failure stops the run before `summary.json` is written, it may be rerun under the same plan and recorded in the consumption record; it is never rerun to change a result.

Per-row results move only internally, and transit copies are deleted after transfer. Reports publish only aggregate numbers and item ids, not test texts or candidates.

## Limitations

- Both test splits were already used for the 20M confirmation, and this plan was drafted after the 20M test results were seen. The choice of primary hypothesis and guards rests on the development gap between the 200M and the 20M; no 200M test result was used.
- The preflight compares hit counts, not per-item hits: the 2.2 per-row files are not in this run's checkout.
- Colloquial readings are automatic, confirmed by an LLM, not human-reviewed.
