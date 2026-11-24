# Beam-width sweep for Vime's pinyin search (2026-11-15)

This report sets the default beam width for Vime's online pinyin search. It measures how much accuracy is lost, and how much compute is saved, when the beam drops from 16 to 12, 8 and 4. All results are on development; the public set's test split was not read. The Chinese version is at [docs/zh-CN/zh/reports/zh-pinyin-beam-20261115/report.md](../../../zh-CN/zh/reports/zh-pinyin-beam-20261115/report.md).

## Conclusions

- **The default beam stays 16.** Every narrower beam is significantly worse on both sets:

  | Beam | Private benchmark | Public set |
  | ---: | ---: | ---: |
  | 12 | −0.7 | −0.9 (p ≤ 0.001) |
  | 8 | −1.6 | −3.4 |
  | 4 | −4.4 | −8.4 |

- **Almost all of the loss is on abbreviated input.**
  - Full pinyin is unchanged at beam 12 (public 83.6 vs 83.6) and loses only 1.8 at beam 4.
  - All-initials and mixed input drop at every step. An initial matches the most characters, so with a narrow beam the correct hypothesis is pushed out at the first few key positions.
- **Compute scales with the beam.**
  - Predicted rows per generation are about 225 at beam 16 and 112 at beam 8.
  - The number of calls equals the number of key positions and does not depend on the beam.
  - If mobile latency ever gets tight, beam 12 for full pinyin only is an option. It is not needed now: a full search at beam 16 takes p50 79 ms on an M3 Mac (see the [batched KV v2 report](../mac-20261115/zh-v1-kvbatch-v2.md)).
- **In this mode, v1 rescoring after generation adds nothing.**
  - Results are equal or 0.1–0.2 lower at every width. Here generation is already scored by v1, so adding 1.6 × the same model's score hardly changes the order.
  - The +6.4/+11.8 in the [rescoring report](../zh-pinyin-rescore-20261115/report.md) were measured on the three-task model `r2-cont`, where generator and prior are different models. The two results do not conflict.

## Setup

- **Model**: `vimeml-tiny-zh-v1` in LM mode. It decodes under pinyin constraints right after the in-field context, without task markers, and finishes when the keys are used up.
- **Search**: [zh_search.py](../../../../src/vimeml/deployment/zh_search.py) keeps the beam per key position.
  - Only hypotheses that consumed the same number of keys compete.
  - At each position the bucket is deduplicated, cut to the beam and advanced in one batched KV call.
  - This is the Vime implementation; the backend is `zh_v1_batch.TorchBatchBackend` (FP32).
- **Context**: the text left of the cursor in the input field, as online.
- **Candidates**: the top 10 per generation. The "rescored" column reorders them by genScore + 1.6 × v1 score, with v1 also given the context.
- **Data**:
  - the private benchmark: 4,612 inputs, 774 sentences;
  - the `ime-pinyin-zh-v1` development split: 3,428 inputs.
  - Both are reported by input kind: full pinyin, mixed abbreviations, all initials.
- **Tests**: paired against beam 16, with a sentence-cluster sign-flip test and 20,000 permutations (smallest p 5e-5).

## Results

### Public set development (3,428 inputs)

| Beam | Kind | Top-1 | Top-3 | Top-10 | Rescored Top-1 | Rows/gen |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | all | **52.5** | 63.9 | 68.4 | 52.5 | 225.0 |
| 16 | full | 83.6 | 93.9 | 96.4 | 83.6 | 316.8 |
| 16 | mixed | 51.5 | 63.9 | 68.8 | 51.4 | 225.0 |
| 16 | initials | 25.6 | 33.9 | 38.9 | 25.4 | 133.1 |
| 12 | all | 51.6 | 62.4 | 66.2 | 51.5 | 168.8 |
| 12 | full | 83.6 | 94.1 | 96.0 | 83.6 | 237.6 |
| 12 | mixed | 50.4 | 61.8 | 66.0 | 50.2 | 168.8 |
| 12 | initials | 24.6 | 32.7 | 37.4 | 24.4 | 99.8 |
| 8 | all | 49.1 | 58.7 | 61.5 | 48.9 | 112.5 |
| 8 | full | 82.9 | 93.3 | 94.5 | 82.9 | 158.5 |
| 8 | mixed | 47.4 | 57.4 | 60.6 | 47.2 | 112.5 |
| 8 | initials | 22.1 | 29.2 | 32.2 | 21.8 | 66.5 |
| 4 | all | 44.1 | 50.9 | 51.5 | 44.0 | 56.3 |
| 4 | full | 81.8 | 88.9 | 89.1 | 81.8 | 79.2 |
| 4 | mixed | 41.4 | 48.9 | 49.5 | 41.3 | 56.3 |
| 4 | initials | 17.1 | 20.6 | 21.5 | 17.0 | 33.3 |

### Private benchmark (4,612 inputs)

| Beam | Kind | Top-1 | Top-3 | Top-10 | Rescored Top-1 | Rows/gen |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | all | **33.6** | 43.5 | 50.0 | 33.6 | 216.6 |
| 16 | full | 66.9 | 81.0 | 86.8 | 66.8 | 286.4 |
| 16 | mixed | 28.5 | 37.9 | 44.7 | 28.5 | 212.3 |
| 16 | initials | 10.7 | 16.5 | 21.1 | 10.7 | 104.4 |
| 12 | all | 32.9 | 42.3 | 47.7 | 32.9 | 162.5 |
| 12 | full | 66.7 | 80.9 | 86.0 | 66.5 | 214.9 |
| 12 | mixed | 27.6 | 36.4 | 42.1 | 27.6 | 159.3 |
| 12 | initials | 10.7 | 15.3 | 18.8 | 10.7 | 78.3 |
| 8 | all | 32.0 | 40.5 | 44.6 | 32.0 | 108.4 |
| 8 | full | 65.6 | 78.8 | 82.7 | 65.5 | 143.3 |
| 8 | mixed | 26.6 | 34.6 | 38.8 | 26.7 | 106.2 |
| 8 | initials | 10.7 | 15.0 | 17.6 | 10.7 | 52.2 |
| 4 | all | 29.2 | 35.9 | 37.0 | 29.2 | 54.2 |
| 4 | full | 62.1 | 73.4 | 74.3 | 62.0 | 71.7 |
| 4 | mixed | 23.9 | 29.8 | 31.0 | 23.9 | 53.1 |
| 4 | initials | 9.8 | 13.9 | 14.7 | 9.8 | 26.1 |

### Paired comparisons (all inputs, Top-1, against beam 16)

| Set | Beam | Diff | Gained/lost | Cluster sign-flip p |
| --- | ---: | ---: | --- | ---: |
| public | 12 | −0.9 | +10/−42 | 0.00095 |
| public | 8 | −3.4 | +13/−131 | 5e-5 |
| public | 4 | −8.4 | +26/−314 | 5e-5 |
| private | 12 | −0.7 | +11/−43 | 0.00045 |
| private | 8 | −1.6 | +34/−109 | 5e-5 |
| private | 4 | −4.4 | +62/−263 | 5e-5 |

With rescoring the paired results are the same: they differ by at most 0.1 point, and the p-values are of the same order.

## Limitations

- Only v1's LM mode was tested. The three-task models (`r2-cont`, the future v2-20m) use a different generator; their beam width should be swept again once the model is fixed.
- Beams 24 and 32 were not tested, so it is unknown whether 16 is saturated. The gain from 12 to 16 (+0.7/+0.9) is already smaller than from 8 to 12 but still significant, so a wider beam may add a little more.
- Compute is counted only as predicted rows; on-device time was not measured. iPhone latency is still unmeasured.
