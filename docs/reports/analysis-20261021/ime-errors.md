# IME Top-1 error decomposition: the bottleneck is not LM capacity

2026-10-21. V4.3 (more data) and V5 (larger model) both lower BPC significantly, but Top-1 does not improve (see [V5 scale-up](../../training-v5.md)). This analysis classifies every Top-1 error of six models on four non-blind sets, item by item, to find where the remaining errors come from. Conclusion: of V4.2's 205 errors on Standard, 48% have no reference in the candidate pool and 35–44% are notation differences; only 17 (8%) are clear semantic errors. Model scale can only affect that small share.

## Method

Only completed FP32 rerank scores (`scores.jsonl`) are read; no model is run and neither the fixed reference nor any blind set is read. Entry point [`analyze_ime_errors.py`](../../../scripts/benchmarks/analyze_ime_errors.py); results in `outputs/analysis-ime-errors-20261021/`.

Each error's Top-1 is compared with the closest reference and put in one of three classes:

| Class | Rule |
| --- | --- |
| Pool miss | No reference is in the frozen AzooKey pool (20 candidates); no reranking can get it right |
| Notation (A) | The same words, differing only in kana/kanji/katakana or okurigana |
| Semantic (S) | Everything else: homophones, segmentation errors, proper nouns |

Every pool candidate is converted from the same kana input and has the same reading, so A versus S is a question of whether the words are the same. A uses a conservative rule: the same UniDic lexeme sequence (`fugashi` + `unidic-lite`, versions in `requirements/evaluation.txt`) **and** one side's kanji set contains the other's. Against the earlier per-item labels of 210 V2.1→V4.2 flips, the rule never classed a semantic flip as A (72/72) and caught 118/138 notation flips. Cases meeting only one of the two conditions are counted separately as the "A upper bound".

## Results

### Standard development /728 (reference in pool: 630)

| Model | Top-1 | Pool miss | A | A upper-bound increment | S | Notation-lenient Top-1 (upper bound) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AzooKey original order | 453 | 98 | 69 | 49 | 59 | 522 (571) |
| V2.1 | 505 | 98 | 73 | 26 | 26 | 578 (604) |
| V4.2 epoch 20 | **523** | 98 | 71 | 19 | 17 | **594** (613) |
| V4.3 controlled epoch 10 | 514 | 98 | 71 | 27 | 18 | 585 (612) |
| V4.3 epoch 30 | 519 | 98 | 69 | 28 | 14 | 588 (616) |
| V5 controlled epoch 10 | 504 | 98 | 85 | 22 | 19 | 589 (611) |
| V5 epoch 30 | 514 | 98 | 69 | 30 | 17 | 583 (613) |

"Notation-lenient Top-1" counts class A under the conservative rule as correct. It is a diagnostic only and does not replace the frozen exact-match metric.

### Other sets

| Set | In pool | V4.2 Top-1 / A / S | V5 epoch 30 Top-1 / A / S | AzooKey Top-1 / S |
| --- | ---: | --- | --- | --- |
| Legacy /500 | 443 | 370 / 57 / 4 | 365 / 58 / 9 | 317 / 31 |
| AJIMEE /200 | 162 | 139 / 3 / 16 | 145 / 3 / 11 | 87 / 57 |
| Historical dev /137 | 136 | 123 / 8 / 1 | 124 / 9 / 1 | 111 / 9 |

### Scale-only paired comparisons (notation-lenient)

| Contrast | Standard | Legacy | AJIMEE | Historical dev |
| --- | --- | --- | --- | --- |
| V4.3-10ep → V5-10ep | +4 (22/18, p=0.64) | −3 (4/7, p=0.55) | +5 (9/4, p=0.27) | 0 |
| V4.3 epoch 30 → V5 epoch 30 | −5 (16/21, p=0.51) | +1 (6/5, p=1.00) | +2 (7/5, p=0.77) | +1 |
| V4.2 → V5 epoch 30 | −11 (11/22, p=0.080) | −4 (5/9, p=0.42) | +6 (12/6, p=0.24) | +2 |

## Findings

1. **Pool misses are the largest single class.** On Standard, Legacy and AJIMEE, 13%, 11% and 19% of items have no reference among the 20 candidates. Of Standard's 98 pool misses, in about 32 the model picked a notation variant of the reference (the reference spelling is not in the pool but an equivalent one is); in the other ~66 the pool may contain no correct conversion at all. Only candidate generation can fix these.
2. **In-pool errors are mostly notation.** Of V4.2's 107 in-pool errors on Standard, 71–90 are notation; of 61 in-pool errors on Legacy, 57 are notation and only 4 semantic. The references usually record one spelling, so most of these "errors" are differences of style, not capability.
3. **The LM has already fixed most semantic errors.** AzooKey's original order has 59 semantic errors on Standard; every LM brings that down to 14–26. On AJIMEE, from 57 to 10–16. Few semantic errors remain, so better LM quality has almost nothing left to improve.
4. **Scale shows a directional gain only on AJIMEE.** V5 has the fewest AJIMEE semantic errors (11 vs 16 for V4.2) and a notation-lenient net of +6, as expected if a stronger LM helps encyclopedic homophones, but the sample is small and the difference not significant. On the conversational sets the effect of scale drowns in notation differences; V5 has no advantage under the notation-lenient treatment either.
5. **No single-direction drift toward hiragana.** Among Standard's class-A errors, about 40% have the model choosing the more kana-heavy spelling (V4.2: 28/71); on Legacy 58% (33/57), where the old web references prefer kanji. Notation preference depends on register more than on any fixed direction.
6. **Most errors are not near misses.** Only about 25–35% of in-pool errors have a log-probability gap under 1 nat; most are the model confidently choosing another spelling.

## Implications

- **Candidate pool coverage** has the highest ceiling: about 66 Standard items and about 35 AJIMEE items have no correct answer in the pool. Check on the Mac how much coverage a larger AzooKey N-best (for example 20 → 50) adds. That requires regenerating candidates, which is a new benchmark version and must be frozen before scoring.
- **Notation** should be treated as a style question, not an LM capability question: as a client-side user preference or register prior, and in evaluation as a secondary notation-lenient Top-1 reported alongside after preregistration. The frozen references are not changed by this analysis.
- **Scoring changes (PMI, AzooKey prior fusion)** were run under this protocol and none passed held-out checks; see [training-free rescoring](rescoring.md).
- **Further LM pretraining gains little**: on the conversational sets only a dozen or so semantic errors remain. If semantic capability is the goal, encyclopedic domains such as AJIMEE are the only place with room left, and that is a domain-data question.

## Limitations

The classification rule agrees with the earlier per-item AI labels about 90% of the time and is not native-speaker annotation; class A is a conservative estimate (about 15% missed). Whether pool misses can be recovered with a larger N-best can only be settled by actually regenerating candidates. All numbers come from development/regression sets used many times and serve diagnosis only.
