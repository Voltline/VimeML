# ja-gen 20M candidate colloq-c40: test confirmation plan (preregistered)

> [Chinese version](../../zh-CN/ja/benchmarks/confirmation-ja-gen-plan.md) (reference text)

> Run on 2026-11-28 (run-01): pass. H1 +18 (p 0.0024), guards met, H2 +3 (p 0.73) not passed; `colloq-c40` is released as `vimeml-ja-v2-20m`. See the [confirmation report](../reports/ja-gen-confirmation-20261128/report.md).

> Naming, 2026-11-28: the ja-gen batch is named vimeml-ja-v6 (20M and 200M; see the [v6 release scope](../release-ja-v6-plan.md)). `vimeml-ja-v2-20m` in this plan is `vimeml-ja-v6-20m`.

Drafted 2026-11-28. The plan counts as registered only once it and the execution code are committed; until then neither the `ime-kana-ja-v1` test split nor the `ime-kana-ja-colloq-v1` test split is opened. This is stage 4 of the [ja-gen plan](../training/training-ja-gen-plan.md).

The entry point is [confirm_ja_gen.py](../../../scripts/ja/benchmarks/confirm_ja_gen.py); every fixed setting is a constant in [ja_gen_confirm.py](../../../src/vimeml/benchmarks/ja_gen_confirm.py). Search, reranking and scoring are those of the 2.3 / 3.x development evaluations (`kana_gen_ja.evaluate_arm`, as used by [evaluate_ja_gen_stage2.py](../../../scripts/ja/benchmarks/evaluate_ja_gen_stage2.py)).

## Questions

`colloq-c40` was selected on development sets in stage 3 ([3.4 report](../reports/ja-gen-stage3-seeds-20261128/report.md)). This plan answers on test:

- Is its kana conversion lenient Top-1 above the deployed V4.2 setup (AzooKey N-best 20 + V4.2 reranking)?
- Is its exact Top-1 above `kd-cont` (does the 3.4 primary metric replicate on test)?
- Is its lenient Top-1 significantly below `kd-cont` on either the kana or the colloquial test split?

On a pass, `colloq-c40` becomes `vimeml-ja-v2-20m`, goes to Mac alignment and iOS integration, and replaces V4.2.

## Models

| Role | Model | checkpoint SHA256 |
| --- | --- | --- |
| Candidate | `colloq-c40` (s1; 30,000 updates from `kd-cont`, colloquial 208/512, seed 20261205) | `19b04ddfe0f98041ffe50511cab9d64d2f2a20b2918f5e78cc78c0768981d15f` |
| Baseline (H1) | V4.2 epoch 20 `best-lm.pt`, reranking AzooKey N-best 20 (contextual sum score) | `0e83ea7417b9816b37183044468343b151f4e610da0764c31f1f780c5fa5796d` |
| Baseline (H2 and guards) | `kd-cont` (2.3 distilled 20M) | `5a44839657569b34dea57fae4a45a2050506f892a441e38715756727fe46a92f` |

Settings for both 20M models: tokenizer `ja-unigram-12288` (`61d71460…`), candidate index `b60c485d…`, constrained beam 16 bucketed by key position, top 10 per model, merged with AzooKey N-best 20 and ordered by gen + λ·LM with λ = 1.6 (frozen in 2.3). V4.2 as in stage 0 (FP32, TF32 off, ties keep AzooKey order).

## Test sets

- **`ime-kana-ja-v1` test**: 600 items (WRIME 400 + JMultiWOZ 200, 59 authors / users), manifest SHA256 `8d51fc00…`. AzooKey candidates were exported on 2026-11-25 together with development (pool manifest `b11df321…`) and have not been scored by any model since.
- **`ime-kana-ja-colloq-v1` test**: 600 colloquial spans (rule band zero 142 / low 136 / mid 148 / high 174), manifest SHA256 `6538b128…`. AzooKey candidates were exported on 2026-11-28 with the development settings (converter `d59a28e4c7ca`, N-best 20, typo off, Zenzai off); the pool SHA256 is registered in the execution code. The export log was neither imported nor read.

## Hypotheses and decision

On the kana test split, tests are author/dialogue-clustered sign-flip tests; on the colloquial test split (at most one item per document) items are paired individually. Two-sided, α = 0.05.

**Primary hypothesis H1**: kana test, V4.2 reranking → `colloq-c40`, lenient Top-1, difference > 0 and p < 0.05.

**H2** (fixed sequence, tested only after H1 passes, also α = 0.05): kana test, `kd-cont` → `colloq-c40`, exact Top-1, difference > 0 and p < 0.05.

**Guards** (all must hold):

1. kana test, `kd-cont` → `colloq-c40`, lenient Top-1: not (difference < 0 and p < 0.05);
2. colloquial test, `kd-cont` → `colloq-c40`, lenient Top-1: the same.

H1 and H2 are tested in fixed sequence, with no further multiplicity correction. Standard blind is not used.

**Reading the result**:

- On development, `colloq-c40` is +18 items over V4.2 in lenient Top-1 (529 vs 511, p 0.013) and +12 over `kd-cont` in exact Top-1 (440 vs 428, p 0.032); against `kd-cont` in lenient Top-1 it is −2 on both the kana and the colloquial development sets (p 0.77 / 0.73).
- **H1 passes and the guards hold**: `colloq-c40` is released as `vimeml-ja-v2-20m` and replaces V4.2. H2 does not change the release: if H2 passes, the release notes state that colloquial post-training raised exact Top-1 on test; if it fails, the numbers are reported without that statement.
- **H1 passes, a guard fails**: `colloq-c40` is not released. If `kd-cont` is also above V4.2 in kana test lenient Top-1 with difference > 0 and p < 0.05, `kd-cont` is released as `vimeml-ja-v2-20m`; otherwise V4.2 stays.
- **H1 fails**: the V4.2 deployment stays. Neither test split is described as unused afterwards, and neither is reopened to confirm another model.

## Descriptive reporting (no decision)

- On both test splits: lenient / exact Top-1 and Top-3 of AzooKey order, V4.2, `kd-cont` and `colloq-c40`, by length band, left context, source and colloquial rule band; generation-only Top-1; D5 items.
- The remaining paired differences on both splits (V4.2 against AzooKey, each 20M against V4.2), and `colloq-c40` against `kd-cont` in exact Top-1 on the colloquial split.
- Generation fallback counts.

## Execution

One GPU, one process, sequential.

1. Commit this plan and the execution code. At run time the script checks that they are committed and unmodified and that every registered file has its SHA256.
2. **Preflight** (development only): V4.2 reranks `ime-kana-ja-v1` development again, with at least 99% of items in the same order as in stage 0; `colloq-c40` regenerates the kana and colloquial development sets, compared item by item with the judged 3.3 rows on lenient and exact Top-1 hits: at most 1% of items (6) may flip and the hit count may change by at most 0.5% (3 items). Otherwise the test splits are not opened. The preflight can run on its own (`--preflight-only`).
3. Write the consumption record `outputs/benchmarks/ja-gen-confirmation/consumption.json`, then V4.2-rerank both test splits, run both 20M models, compute the decision and write `summary.json`.
4. Run once. If a technical failure stops the run before `summary.json` is written, it may be rerun under the same plan and recorded in the consumption record; it is never rerun to change a result.

Per-item results move only through internal transfer, and transit copies are deleted afterwards. The report publishes only aggregate numbers and item ids, not test texts or candidates.

## Limitations

- `colloq-c40` was selected after several comparisons on the kana and colloquial development sets; this confirmation is an independent test of that choice.
- Colloquial readings are automatic readings confirmed by an LLM, not human-reviewed.
- V4.2 only reranks AzooKey candidates and does not generate; H1 compares two complete deployment setups, not two models scoring the same candidates.
