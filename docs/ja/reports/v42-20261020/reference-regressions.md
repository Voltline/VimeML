# V4.2 fixed-reference evaluation and regression analysis

2026-10-20. The frozen V4.2 epoch 20 (`best-lm.pt`, step 1,081,942) received one preregistered evaluation on the 725-item reference set, and every Top-1 flip against V2.1 on the four development/regression sets and the reference was classified item by item. Conclusion: V4.2 scores **541/725** on the reference, 6 more than V2.1 (40 gains / 34 losses, p=0.56) and below V3 A's 546; the +18 development advantage does not reproduce at the same size on the reference. About 61% of the flips are notation changes of the same word (kana/kanji/okurigana), not semantic errors; the AJIMEE regression is mostly encyclopedic homophone errors.

## Reference evaluation

| Item | Record |
| --- | --- |
| Plan | `outputs/evaluation-v42-reference-20261020/reference-test-plan.json`, plan_id `81512f147d3749ef9a36e61470fce173`, SHA256 `76105753…d8de5` |
| Registered | Before scoring; model identity includes path, size, mtime, step and weight SHA256 |
| Role | `reference_test_after_consumption`: these 725 items were consumed in the first V2.1/V3 A comparison and are not a new blind test |
| Execution | Local CUDA FP32, TF32 off; run once, consumption record written to `blind-consumption.json` |
| Scoring | Frozen candidates and references; jointly tokenized suffix log-probability sum, stable ties, no fallbacks |
| Baselines | Existing V2.1, V3 A and V3 B reference scores reused, with ID, reading, context, reference, source, candidates and engine order checked item by item |

`evaluate_standard_ime.py` gained `--architecture v4`, which uses `JapaneseLMV4` and checks the step and weight SHA256 against the plan; the V1/V2 paths are unchanged.

| Model | WRIME /475 | JMultiWOZ /250 | Micro /725 | Equal-weight macro |
| --- | ---: | ---: | ---: | ---: |
| V2.1 | 309 | 226 | 535 (73.79%) | 77.73% |
| V3 A | **318** | **228** | **546** (75.31%) | **79.07%** |
| V3 B epoch 2 | 308 | 225 | 533 (73.52%) | 77.42% |
| V4.2 epoch 20 | 315 | 226 | 541 (74.62%) | 78.36% |

V4.2 Top-5 is 643/725 and MRR 0.80517 (V2.1: 636, 0.79875); against AzooKey it fixes 117 items and breaks 43; no fallbacks.

| Paired comparison | V4.2 gains | V4.2 losses | Net | Exact McNemar two-sided p |
| --- | ---: | ---: | ---: | ---: |
| vs V2.1 | 40 | 34 | +6 | 0.561 |
| vs V2.1, WRIME only | 32 | 26 | +6 | — |
| vs V2.1, JMultiWOZ only | 8 | 8 | 0 | — |
| vs V3 A | 34 | 39 | −5 | 0.640 |
| vs V3 B | 38 | 30 | +8 | 0.396 |

The result changes no checkpoint choice, label, candidate or scoring setting. V3 A still holds the highest recorded score on this set, but its development and regression results did not support replacing V2.1 earlier.

## Regression analysis method

Scope: all 210 items whose Top-1 correctness flips against V2.1: Standard development 74, Legacy 34, AJIMEE 21, Historical 7, reference 74. The two models' first choices, the reference set and the score gap were compared item by item and labeled in three classes:

- **A notation**: the same word with the same reading, differing only in kana/kanji/katakana or okurigana; both spellings are usually acceptable, and the reference set recorded only one.
- **S semantic**: homophones, wrong segmentation or conversions that change the meaning.
- **P proper noun**: spellings of personal, place and shop names.

Labels are AI judgements, not native-speaker annotation; borderline cases (such as subtle usage differences between near-synonym kanji) count as S. Per-item labels and scripts are in `outputs/analysis-v42-regressions-20261020/`. The analysis is for understanding differences only and changes no reference or benchmark set.

## Results

| Set | A gain/loss | S gain/loss | P gain/loss | S+P net | S+P exact p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard development | 28/23 | 15/5 | 3/0 | **+13** | 0.011 |
| Reference 725 | 27/24 | 11/7 | 2/3 | +3 | 0.678 |
| Legacy regression | 12/17 | 3/2 | 0/0 | +1 | 1.000 |
| AJIMEE | 1/2 | 7/11 | 0/0 | **−4** | 0.481 |
| Historical development | 3/1 | 1/2 | 0/0 | −1 | 1.000 |

**Notation is the majority.** 128 of the 210 flips (61%) are class A. V4.2's notation changes lean toward hiragana overall: in 82 its first choice has more hiragana than V2.1's, in 51 fewer. On the conversational sets (WRIME, JMultiWOZ) this drift's gains and losses roughly cancel; Legacy's old web references prefer formal kanji, and there hiragana-leaning flips give 7 gains and 14 losses. Legacy's −4 is essentially explained by notation (−5), with +1 at the semantic level.

**Many small-gap flips.** On Standard and the reference, about 30% of flips each have a log-probability gap under 1 nat, almost all class A; for example 「欲しい／ほしい」 and 「いい／良い」 flip together with the same gap across several templated JMultiWOZ sentences. Such changes look like drift in notation preference rather than capability differences.

**AJIMEE is a real semantic regression.** 11 of 13 losses are class S, concentrated in encyclopedic homophones (military, broadcasting, organization names and similar) and digit/counter confusions; 7 of 8 gains are class S too. Net −4, small sample, not significant. V4-B, which also uses the 8K tokenizer, scores 144 on AJIMEE, the same as V2.1, so the tokenizer alone cannot explain it. V4.1/V4.2 both mix in about 5% chat data, which is consistent with the AJIMEE drop, but this experiment does not isolate that factor.

**The development semantic gain does not reproduce.** The S+P net of +13 (18/5) on Standard development is the main source of V4.2's development advantage; on the reference the same net is only +3 (13/10). Standard has been used repeatedly by many V4 experiments, and the gap is what regression of development gains toward the mean would predict, but the cause cannot be determined from this.

## Judgement

- V4.2 is a usable plain-LM reranking model: it is not below V2.1 on the reference, and semantic errors do not increase systematically.
- The current evidence does not support replacing V2.1 with V4.2: the reference +6 is not significant and the main difference is notation style; AJIMEE shows a few real encyclopedic regressions.
- Exact-match Top-1 is highly sensitive to notation. Later comparisons should also report a notation-insensitive diagnostic (for example counting same-reading form changes separately), but frozen references are not expanded after seeing results.
- If the deployment cares about consistent notation style (for example users who prefer kanji), V4.2's hiragana drift should be assessed separately as a client experience risk.
- The historical 1,000-item blind set is still unscored. V4.2 should not consume it without a new decision need; if it is used, it should be preregistered once as "V2.1 INT8 vs V4.2 INT8" after the Mac INT8 results.
