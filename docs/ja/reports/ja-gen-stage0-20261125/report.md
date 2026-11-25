# Japanese generation line, stage 0: candidate-pool coverage and development baselines

> [Chinese version](../../../zh-CN/ja/reports/ja-gen-stage0-20261125/report.md)

2026-11-25. This is stage 0 (0.1–0.4) of the [ja-gen plan](../../training/training-ja-gen-plan.md); deviations D1–D3 are at the end of the plan page. Only development sets are used here. Candidates for the `ime-kana-ja-v1` test split were exported and imported, but they were neither loaded nor scored, and the CliTool log for test was not imported.

## What was done

- **New public set `ime-kana-ja-v1`**, frozen ([build script](../../../../scripts/ja/benchmarks/prepare_kana_ime_ja.py)):
  - development and test have 600 items each (WRIME 400 + JMultiWOZ 200) and share no author;
  - manifest SHA256 `8d51fc00…f9a9d`, recorded on the plan page;
  - all 8,000 documents used by earlier sets are excluded, and 45 items that overlap existing training text were removed before selection.
- **Mac candidate export** ([handoff script](../../../../scripts/ja/benchmarks/build_candidate_handoff_v3.py)):
  - the same converter, dictionary and emoji revisions as v1, with typo correction and Zenzai off;
  - Standard development at N-best 100, plus `ime-kana-ja-v1` development and test at N-best 20.
- **Evaluation** ([script](../../../../scripts/ja/benchmarks/evaluate_ja_gen_stage0.py)):
  - V4.2 epoch 20 in FP32 reranks the N-best 20 pools of both development sets;
  - on Standard development, all 728 rerank orders match the 2026-10-20 record exactly, giving exact Top-1 523;
  - notation-lenient judging follows deviation D3: a candidate counts if `notation.relation` returns `=` or `A` for some accepted answer;
  - statistics are clustered by author and dialogue (metrics v2).

## 0.1 Standard development pool coverage

| AzooKey N-best | Mean pool size | Pool ceiling (exact) | Pool ceiling (lenient) |
| --- | ---: | ---: | ---: |
| 20 (deployed) | 12.1 | 630 / 728 (86.5%) | 674 (92.6%) |
| 50 | 28.8 | 652 (89.6%) | 699 (96.0%) |
| 100 | 55.8 | 670 (92.0%) | 703 (96.6%) |

At N-best 20, 98 items miss the pool (exact). What a deeper pool recovers:

| Outcome | Items |
| --- | ---: |
| Recovered at N-best 50 | 22 |
| Recovered at N-best 100 (includes the row above) | 40 |
| **Still missing at N-best 100: generation only** | **58** |
| For reference: of the 98, items with a notation variant already in the N-best 20 pool | 44 |

Pool misses are concentrated in long spans:

| Span length / left context | Missing at N-best 20 | Generation only |
| --- | ---: | ---: |
| 4–10 chars, context | 3 | 2 |
| 4–10 chars, no context | 2 | 0 |
| 11–20 chars, context | 12 | 6 |
| 11–20 chars, no context | 9 | 4 |
| 21–40 chars, context | 48 | 29 |
| 21–40 chars, no context | 24 | 17 |

Of the 58 generation-only items, 46 are spans of 21–40 characters.

Deepening the AzooKey list to 100 recovers 40 items, but the mean pool grows to 56 candidates, roughly 4.6× the current on-device scoring cost. The generation line targets the remaining 58.

## 0.3 Development baselines

Top-1 / Top-3 refer to the first one / three entries of the final displayed list.

| Set | Order | Exact Top-1 | Exact Top-3 | Lenient Top-1 (primary) | Lenient Top-3 |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard development, 728 | AzooKey order | 62.2 | 79.0 | 75.7 | 88.2 |
| | V4.2 rerank | 71.8 | 82.3 | 86.1 | 90.4 |
| | Pool ceiling | 86.5 | | 92.6 | |
| `ime-kana-ja-v1` development, 600 | AzooKey order | 57.3 | 76.7 | 74.3 | 86.5 |
| | V4.2 rerank | 66.3 | 81.5 | 85.2 | 90.5 |
| | Pool ceiling | 85.7 | | 92.3 | |

V4.2 against AzooKey order, Top-1:

| Set | Judging | Gains / losses | Net | Cluster 95% interval | Cluster sign-flip p |
| --- | --- | --- | ---: | --- | ---: |
| Standard development (62 clusters) | exact | 110 / 40 | +70 | [+44, +96] | <0.0001 |
| | lenient | 97 / 21 | +76 | [+53, +101] | <0.0001 |
| `ime-kana-ja-v1` development (63 clusters) | exact | 91 / 37 | +54 | [+32, +78] | <0.0001 |
| | lenient | 77 / 12 | +65 | [+44, +87] | <0.0001 |

Lenient Top-1 on `ime-kana-ja-v1` development, by stratum:

| Stratum | AzooKey order | V4.2 rerank |
| --- | ---: | ---: |
| 4–10 chars | 81.2 | 85.8 |
| 11–20 chars | 75.3 | 89.3 |
| 21–40 chars | 67.5 | 80.4 |
| With context | 78.2 | 90.0 |
| No context | 69.5 | 79.2 |
| WRIME | 67.8 | 79.5 |
| JMultiWOZ | 87.5 | 96.5 |

## Conclusions

- **The new set is about as hard as Standard development.** Pool ceilings are 85.7% vs 86.5%, and V4.2 lenient Top-1 is 85.2 vs 86.1. It serves as the main development set of the generation line; Standard development stays as the reranking guard (0.4).
- **The baseline stage 2 must beat is V4.2 reranking:** lenient Top-1 85.2 and exact 66.3 on `ime-kana-ja-v1` development.
  - The direct gain from generated candidates is bounded by the 7.7% (lenient) outside the pool.
  - The rest of the gap is ranking within the pool, where V4.2 is about 7 points below the pool ceiling.
- **Generation mainly has to fill long spans.** Of the 58 Standard development items that N-best 100 still misses, 46 are spans of 21–40 characters. Stage 1's `sentence` samples and long `span` samples target exactly these.
- **Items without left context are harder.** On the new set, V4.2 scores 79.2 without context and 90.0 with context.

All stage 0 completion conditions are met:
- the coverage report exists;
- the set is frozen and its SHA256 is recorded;
- both baselines are measured;
- test is unread.
