# vimeml-tiny-zh-v1 blind confirmation plan (preregistered)

> Run on 2026-11-11 (run-01) and passed; results in [blind confirmation and baselines](../reports/zh-v1-20261111/blind-and-baselines.md).

Drafted 2026-11-11. The plan counted as registered only once it and the run script were committed; blind stayed closed until then. Benchmark version `ime-standard-zh-v1-r1`. Entry point [confirm_zh_v1.py](../../scripts/benchmarks/confirm_zh_v1.py); every fixed item is in [zh_v1_confirm.py](../../src/vimeml/benchmarks/zh_v1_confirm.py). The registered text is the Chinese version, [docs/zh-CN/benchmarks/confirmation-zh-v1-plan.md](../zh-CN/benchmarks/confirmation-zh-v1-plan.md); this page is its translation.

## Question

The [eight-arm decision](../reports/zh-v1-20261110/ime-benchmark.md) used development only, so its conclusions are exploratory, and the [experiment plan](../training-zh-v1-plan.md) requires a preregistered test on the Chinese blind set before deployment. This plan asks one question: on the unopened blind set, is the chosen model's exact Top-1 really above Rime's first candidate? It also gives a blind accuracy estimate.

## Model

| Item | Value |
| --- | --- |
| Release name | `vimeml-tiny-zh-v1` |
| Arm | `12k-fresh-s1` (12K vocabulary, 9 layers, fresh data) |
| Checkpoint | `artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/final.pt`, step 844,730, SHA256 `44c423a1e7cb0e32161afc94d6167cabb439a3d25fd9de6d1fbb150a927542d4` |
| Tokenizer | `zh-unigram-12288/tokenizer.model`, SHA256 `68bd37c45bcb018b32b2eef5b241561ab5b6dc1b77c1c3aadcc1b41c4bf47e5a` |
| Control | The frozen candidate pool's original order, i.e. the first candidate of librime 1.17.0 + rime-ice |

**Why this model**: fresh data and 12K come from the experiment plan's default rules. The three seeds of that recipe score 1,363, 1,359 and 1,363 Top-1 on development, within noise, so the choice is not made by score. s1 is chosen because it is the arm that represents `12k-fresh` in the preregistered contrasts. The choice was made before blind was opened and does not use blind.

Only this one model is compared. The other seven arms do not see blind: their differences from this model on development all fell short of the effectiveness standard, and blind is not used to reselect.

## Hypothesis and decision

**Primary hypothesis H1**: full-pinyin track, all 1,445 blind items, Rime first candidate → `vimeml-tiny-zh-v1`, exact Top-1.

Passing requires all of:

1. **H1 holds**: net > 0 and exact McNemar two-sided p < 0.05.
2. **No source regresses**: Top-1 net > 0 in each of CrossWOZ, RiSAWOZ and KdConv.
3. **Top-3 does not drop**: no Top-3 drop with McNemar p < 0.05.

There is only one primary hypothesis, so no multiplicity correction. Each dialogue contributes at most one item, so a dialogue cluster is a single item and item-level and cluster-level tests are equivalent.

**How to read it**:

- The development net is +362/1,456, so this test is almost certain to pass and is not a hard hurdle. Its value is ruling out "works only on development" and providing an unselected accuracy estimate.
- Pass: `vimeml-tiny-zh-v1`'s Top-1 improvement over Rime is confirmed on blind. Report blind Top-1 with its Wilson 95% interval; that number replaces development's 93.6%.
- Fail: this checkpoint is not released, and blind is not reopened for another arm. The analysis goes into the report, and blind can no longer be called an unused independent set.
- Passing does not mean ready to ship. On-device performance and client keyword blocking are verified separately.

## Descriptive reports (no decision)

- Full pinyin: Top-3, Top-5, Top-10, pool-conditional Top-1, MRR; micro per source, macro over the three sources; strata by target length; McNemar and bootstrap intervals.
- The same metrics on the abbreviated track (964 items). Its pool covers only about four in ten answers, so it is not pooled with full pinyin.
- A sensitivity subset without the overlap-screen hits.
- Differences between development and blind are descriptive only. Blind's Rime Top-1 (72.3%) is already higher than development's (68.8%), so absolute scores should not be subtracted and interpreted directly.

## Scoring

As on development: `BOS + left context + candidate` tokenized jointly, FP32, full-vocabulary log-probability sum, no EOS, TF32 off, ties keep the pool's original order. Items whose pool lacks the answer or is empty stay in the denominator. An item with a roundtrip or context-window problem falls back to the pool's original order and is counted in `fallbacks` (as the [benchmark description](standard-ime-zh.md) specifies). There were none on development.

## Running

1. Commit this plan and the script. At run time `confirm_zh_v1.py` checks that the plan and code are committed and unmodified; that checkpoint, tokenizer, items and pools match the registered SHA256s; and that no completed consumption record exists.
2. **Preflight**: rescore development full and abbreviated pinyin; every item's rank must equal the result of the 2026-11-11 decision, or blind is not opened.
3. Write a consumption record to `outputs/benchmarks/zh-v1-confirmation/consumption.json`, then score blind full and abbreviated pinyin, check the Rime baselines (full 1,045, abbreviated 147) and write `summary.json`.
4. Run once only. If a technical fault interrupts the run before `summary.json` is written, it can be rerun under the same plan and recorded in the consumption record; never rerun to change the result.
5. Any change after the first run is reported as a deviation, and the original outputs are kept.

The report publishes only summary numbers and item ids, never blind item text or candidates.

## Limits

- Labels went through AI expert review and human spot checks, not formal native-speaker annotation.
- Blind and development come from the same sources, all dialogue data, not representative of real keyboard traffic.
- The result measures "reranking given the Rime pool", not the accuracy of a complete IME.
- The overlap checks find only verbatim overlap. The [leakage check](../reports/zh-v1-20261110/ime-benchmark.md#leakage-check) covered development only; for blind there are only the original verbatim screen results (44 contiguous-Han-run hits, 27 whole-message hits).
