# ja-gen stage 2.1: 20M reading post-training and development evaluation

> [Chinese version](../../../zh-CN/ja/reports/ja-gen-stage2-20m-20261127/report.md)

2026-11-27. Follows the 2.1 registration (2026-11-26) in the [ja-gen plan](../../training/training-ja-gen-plan.md). Development sets only: `ime-kana-ja-v1` development (600 items) and Standard development (728 items). `ime-kana-ja-v1` test was not read.

**Result**: the reading recipe is kept. The reading arm's four-seed mean lenient Top-1 on kana development is 87.17, above V4.2 reranking (85.2) and the LM control (85.79), and above both at every seed. Its Standard reranking mean is 504.5, above the 501 guard. Reading post-training costs about 11 items of Standard reranking against the LM control; see below.

## Training

- **Tokenizer and data**: 12K tokenizer `ja-unigram-12288`, token store `ja-kana-v1-12k` (records 1.4 and 1.5 in the plan).
- **Pretraining**: four seeds, 1,081,942 updates each (V4.2's budget), `plain` + `chat` only. Validation loss of final.pt:

  | seed | final.pt (first 12 of SHA256) | plain | chat |
  | --- | --- | ---: | ---: |
  | s1 | `b12d41ed1f6c` | 3.7063 | 4.0154 |
  | s2 | `6d4c161726ab` | 3.7187 | 4.0167 |
  | s3 | `2b6d86ad54e9` | 3.7306 | 4.0101 |
  | s4 | `2e31dde3835f` | 3.7017 | 4.0030 |

- **Post-training**: per seed one reading arm and one LM control, eight arms of 60,000 updates each, as registered; final.pt only.

  | seed | reading arm final.pt | LM control final.pt |
  | --- | --- | --- |
  | s1 | `35ac57685062` | `6a1eedfa2c87` |
  | s2 | `10b16c52206a` | `d26407b49756` |
  | s3 | `44cf41718cc6` | `4de370c2bdc2` |
  | s4 | `4345b8edd74c` | `797795858ab7` |

## Evaluation setup

- Script `scripts/ja/benchmarks/evaluate_ja_gen_stage2.py`; candidate index `kana-index.json`, SHA256 prefix `b60c485d7ba2`.
- **kana development**: the reading arm's final list is AzooKey N-best 20 ∪ beam-16 generation bucketed by key position (top 10), ordered by generation score + λ × LM score with λ = 1.6. The LM control cannot generate and reranks the AzooKey pool with its LM score.
- **Standard development**: both arms rerank the AzooKey pool with the LM score only.
- Lenient matching follows deviation D3; D5 items are reported separately.

## Results

`ime-kana-ja-v1` development (600 items):

| Metric | reading s1 / s2 / s3 / s4 | reading mean | LM control mean | V4.2 reranking |
| --- | --- | ---: | ---: | ---: |
| lenient Top-1 (primary) | 87.33 / 87.67 / 87.00 / 86.67 | **87.17** | 85.79 | 85.2 |
| exact Top-1 | 71.33 / 70.67 / 71.83 / 71.17 | 71.25 | 66.67 | 66.3 |
| lenient Top-3 | 92.00 / 92.50 / 91.17 / 91.83 | 91.88 | 90.63 | 90.5 |
| D5 items, lenient Top-1 | 75.00 / 66.67 / 66.67 / 70.83 | 69.79 | 70.83 | |

LM control lenient Top-1 by seed: 86.17 / 86.33 / 84.67 / 86.00.

Standard development (728 items, exact Top-1 count):

| seed | reading arm | LM control | difference |
| --- | ---: | ---: | ---: |
| s1 | 501 | 507 | −6 |
| s2 | 505 | 518 | −13 |
| s3 | 508 | 512 | −4 |
| s4 | 504 | 524 | −20 |
| mean | **504.50** (69.30%) | 515.25 (70.78%) | −10.75 |

Paired differences in kana development lenient Top-1 (items):

| Comparison | s1 | s2 | s3 | s4 |
| --- | ---: | ---: | ---: | ---: |
| reading − V4.2 reranking | +13 | +15 | +11 | +9 |
| reading − LM control | +7 | +8 | +14 | +4 |

For s3, the author/dialogue-clustered sign-flip p of reading against the LM control is 0.031.

Other results:

- **Generated candidates alone** (not merged with the AzooKey pool): mean lenient Top-1 68.13 (66.67–69.33).
- **Constrained search** never fell back (fallbacks = 0).
- **λ**: lenient Top-1 is flat across λ 1.2–2.5. 1.6 lies on that plateau and is kept.

## Decision

By the registered rule:

- (a) The reading arm's four-seed mean lenient Top-1 on kana development, 87.17, is above V4.2 reranking 85.2 (+1.97 points) and above the LM control 85.79 (+1.38 points), and higher at 4 of 4 seeds when paired. Met.
- (b) The reading arm's four-seed mean Standard reranking, 504.50, is at least 501, and no seed is below 501. Met.

`keep_reading_recipe = True`: the reading recipe is kept for 2.2 / 2.3.

## Observations

- **Exact Top-1 gains more**: the reading arm is 4.6 points above the LM control in exact Top-1 (71.25 against 66.67) and 1.4 points in lenient Top-1. Generation and reading conditioning mostly fix the written form.
- **Cost on Standard reranking**: reading post-training spends 320 of 512 samples per update on the reading task for 60,000 updates, and its LM score reranks Standard about 11 items worse than the LM control, in the same direction at all four seeds. V4.1 fell from 505 to 461. This time the mean is 504.5, inside the V4.2 seed band (501–523) but in its lower half. The 2.3 distillation from 200M is to be checked on this as well.
- **D5 items** (about 24; the set is frozen with dictionary readings): 69.79 for the reading arm against 70.83 for the LM control, less than one item apart on average.
- **Generation alone is not enough**: generated candidates alone reach 68.13, below AzooKey order (74.3). The gain comes from merging with the AzooKey pool and scoring the union.

## Next

- **2.2**: the 200M joint reading pretraining is running.
- **2.3**: after the 200M finishes, distill to 20M with the kd-cont recipe. `configs/ja/train-ja-kana-20m-kd-cont.toml` is ready; the teacher's SHA256 is filled in then.
