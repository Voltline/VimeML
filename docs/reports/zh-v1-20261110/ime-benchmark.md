# vimeml-tiny-zh-v1 eight-arm IME scoring and decision (2026-11-11)

Chinese IME benchmark results and the decision for the eight arms of the [preregistered plan](../../training-zh-v1-plan.md). Training is described in the [eight-arm training record](training.md). All results come from the development split of `ime-standard-zh-v1-r1`; blind was not read.

## Conclusions

- **All eight arms are far above Rime.** Full-pinyin development has 1,456 items; Rime's first candidate hits 1,001 (68.8%), the eight arms 1,359–1,374 (93.3%–94.4%).
- **Fresh data and 12K are chosen, but only as defaults.** None of the five preregistered contrasts is effective; every Holm-corrected p is 1. By the plan, when none is effective the defaults are fresh data and 12K; this does not mean fresh data or 12K won.
- **No confirmable difference among the eight arms.** The best, `8k-ep5` (1,374), and the worst, `12k-fresh-s2` (1,359), differ by 15 items. The seed range of the same recipe is 4 items; two contrasts (7 items each) exceed it, but neither is significant after Holm correction.
- **No corpus leakage found.** A scan of the 1,456 development dialogues found no source dataset copied into the training corpus; see [Leakage check](#leakage-check). The high scores come mainly from the conditions of the task itself and are not the accuracy of a complete IME.
- **The release candidate is `12k-fresh-s1`.** The three seeds differ within noise, and it is the arm that represents `12k-fresh` in the preregistered contrasts, not chosen by score. The confirmation design is the [blind confirmation plan](../../benchmarks/confirmation-zh-v1-plan.md).
- These are exploratory development conclusions. The later blind confirmation passed (Top-1 93.8%) and was compared with Qwen and stronger Rime configurations, see [blind confirmation and baselines](../zh-v1-20261111/blind-and-baselines.md).

## Accuracy

Full-pinyin track, 1,456 items, exact Top-1 is the primary metric. Each arm's final checkpoint, not chosen by results.

| System | Top-1 hits | Top-1 | Top-3 | Pool-conditional Top-1 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rime first candidate (baseline) | 1,001 | 68.8% | 90.0% | 70.5% | 0.800 |
| `12k-fresh-s1` | 1,363 | 93.6% | 97.3% | 96.0% | 0.954 |
| `12k-fresh-s2` | 1,359 | 93.3% | 97.2% | 95.7% | 0.952 |
| `12k-fresh-s3` | 1,363 | 93.6% | 97.1% | 96.0% | 0.953 |
| `12k-ep5` | 1,373 | 94.3% | 97.2% | 96.7% | 0.956 |
| `12k-ep20` | 1,366 | 93.8% | 97.2% | 96.2% | 0.954 |
| `8k-fresh` | 1,370 | 94.1% | 97.1% | 96.5% | 0.956 |
| `8k-ep5` | 1,374 | 94.4% | 97.2% | 96.8% | 0.957 |
| `8k-ep20` | 1,363 | 93.6% | 97.1% | 96.0% | 0.954 |

The pool-conditional denominator is the items whose correct answer is in the Rime pool (1,420); for the other 36 the answer is not in the pool and reranking cannot hit it. The "Top-1 hits" denominator is always 1,456, including 3 items with an empty pool.

**Top-1 by source and target length**

| System | CrossWOZ (478) | RiSAWOZ (493) | KdConv (485) | 2–3 chars (523) | 4–6 chars (467) | 7–12 chars (466) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rime | 64.9% | 72.4% | 68.9% | 75.9% | 74.5% | 54.9% |
| `12k-fresh-s1` | 94.6% | 94.7% | 91.5% | 96.6% | 95.5% | 88.4% |
| `12k-fresh-s2` | 93.1% | 95.1% | 91.8% | 96.7% | 95.5% | 87.3% |
| `12k-fresh-s3` | 93.9% | 96.1% | 90.7% | 97.1% | 95.9% | 87.3% |
| `12k-ep5` | 95.4% | 95.5% | 92.0% | 98.1% | 95.5% | 88.8% |
| `12k-ep20` | 94.8% | 95.1% | 91.5% | 97.3% | 94.6% | 89.1% |
| `8k-fresh` | 96.4% | 94.3% | 91.5% | 96.6% | 95.9% | 89.5% |
| `8k-ep5` | 95.6% | 95.1% | 92.4% | 96.9% | 96.4% | 89.5% |
| `8k-ep20` | 94.8% | 94.1% | 92.0% | 96.9% | 95.5% | 88.0% |

**Overlap-free subset, item-level comparison with Rime, and the abbreviated track**

| System | Overlap-free Top-1 | Overlap-free | Wins vs Rime | Losses vs Rime | Abbrev. Top-1 | Abbrev. | Abbrev. Top-3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rime | 950 / 1,392 | 68.2% | — | — | 142 / 990 | 14.3% | 23.6% |
| `12k-fresh-s1` | 1,301 / 1,392 | 93.5% | +377 | −15 | 337 / 990 | 34.0% | 39.3% |
| `12k-fresh-s2` | 1,298 / 1,392 | 93.2% | +379 | −21 | 335 / 990 | 33.8% | 39.0% |
| `12k-fresh-s3` | 1,302 / 1,392 | 93.5% | +380 | −18 | 341 / 990 | 34.4% | 39.3% |
| `12k-ep5` | 1,311 / 1,392 | 94.2% | +386 | −14 | 342 / 990 | 34.5% | 39.2% |
| `12k-ep20` | 1,304 / 1,392 | 93.7% | +383 | −18 | 334 / 990 | 33.7% | 39.1% |
| `8k-fresh` | 1,308 / 1,392 | 94.0% | +383 | −14 | 341 / 990 | 34.4% | 39.4% |
| `8k-ep5` | 1,312 / 1,392 | 94.3% | +388 | −15 | 339 / 990 | 34.2% | 39.1% |
| `8k-ep20` | 1,301 / 1,392 | 93.5% | +381 | −19 | 337 / 990 | 34.0% | 39.1% |

- Overlap-free: the 64 items with verbatim overlap with the training corpus removed (screening method in the [benchmark description](../../benchmarks/standard-ime-zh.md)). Top-1 barely changes: `12k-fresh-s1` goes from 93.6% to 93.5%, so the high score is not due to verbatim overlap. This screen cannot find paraphrases or near-duplicates.
- Wins/losses vs Rime: items where the LM's first candidate hits and Rime's misses, and the reverse. Exact McNemar p is below 1e-85 for all eight arms.
- The abbreviated track has 990 items, only 403 (40.7%) with the answer in the pool, so its accuracy is limited mainly by the pool.

## Preregistered decision

Condition: the absolute Top-1 difference between two arms exceeds the range of the three `12k-fresh` seeds, and the Holm-corrected exact McNemar p < 0.05. The five contrasts form one test family, family α = 0.05, two-sided.

| Contrast | A | B | Top-1 A | Top-1 B | B − A | +/− | McNemar p | Holm p | Exceeds range | Effective |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| Fresh vs 20 epochs, 12K | `12k-fresh-s1` | `12k-ep20` | 1363 | 1366 | +3 | +20/−17 | 0.743 | 1 | No | No |
| Fresh vs 20 epochs, 8K | `8k-fresh` | `8k-ep20` | 1370 | 1363 | −7 | +14/−21 | 0.311 | 1 | Yes | No |
| Vocabulary, fresh data | `8k-fresh` | `12k-fresh-s1` | 1370 | 1363 | −7 | +16/−23 | 0.337 | 1 | Yes | No |
| Vocabulary, 5 epochs | `8k-ep5` | `12k-ep5` | 1374 | 1373 | −1 | +19/−20 | 1 | 1 | No | No |
| Vocabulary, 20 epochs | `8k-ep20` | `12k-ep20` | 1363 | 1366 | +3 | +20/−17 | 0.743 | 1 | No | No |

**Noise band**: the three `12k-fresh` arms score 1363, 1359, 1363: mean 1361.7, range 4, SD 2.31. The three seed pairs differ by +20/−24, +17/−17, +22/−18 items, McNemar p 0.652, 1, 0.636. Two runs of the same recipe already differ on about 40 items, with gains and losses roughly cancelling.

**Choice by the plan**: neither data contrast is effective, so fresh data by default. None of the three vocabulary contrasts is effective, so 12K by default.

## Descriptive observations (not used for the decision)

- **5 epochs**: both `ep5` arms are slightly above the matching fresh and 20-epoch arms, none significantly: at 12K, 10 above fresh (+21/−11, p = 0.11) and 7 above 20 epochs (p = 0.281); at 8K, 4 above fresh (p = 0.618) and 11 above 20 epochs (p = 0.0989).
- **BPC**: among the 12K recipes, lowest web validation BPC is `12k-ep5` (4.3469); the three `12k-fresh` seeds are 4.3533–4.3556 and `12k-ep20` rises to 4.3774. BPC and IME Top-1 orderings do not fully agree: `12k-ep20` has worse BPC than `12k-fresh-s1` but 3 more Top-1 hits, also within noise.
- **Memorization gap**: web validation BPC minus P20 memorization-sample BPC is about 0.014 for fresh data, 0.021 for 5 epochs, 0.049 for 20 epochs.
- **Vocabulary**: 12K's web BPC is 0.007–0.017 below 8K at the same recipe, with no matching difference in IME Top-1.

## Leakage check

The original overlap screen only looks for verbatim matches of contiguous Han runs or whole messages, so a stricter check was run against the actual training data, `export-r1/train` (60 shards, 45,925,410 documents, web and LCCC chat). Text is compared with every non-Han character removed, so differences in punctuation and spacing do not matter. Only development was read; blind was not.

| Check | Scope | Result |
| --- | --- | --- |
| Dialogue-level 16-character contiguous match | 4,605 messages of at least 16 characters in the 1,456 development dialogues, 49,600 16-character strings | 5 strings hit, from only 2 messages in 2 dialogues, both KdConv sentences paraphrasing encyclopedic text; no document contains two messages of the same dialogue |
| Last 16 characters of left context + target | 246 items | 0 found in the training corpus |
| Last 8 characters of left context + target | 578 items | 7 found, of which 6 hit (85.7%), below the 95.6% of items in this group that were not found |
| Code paths | Training, data building and formal tokenizers | None reads the benchmark. Only the exploratory vocabulary probe read the development candidate pools, to count token lengths and coverage; it produces no training data, but those statistics were one input to the 8K/12K choice |

**Scores with all suspected overlap removed** (`12k-fresh-s1`, Top-1):

| Subset | Items | LM | Rime |
| --- | ---: | ---: | ---: |
| All | 1,456 | 93.6% | 68.8% |
| Minus original overlap-screen hits | 1,392 | 93.5% | 68.2% |
| Also minus items whose "last 4+ characters of left context + target" occur, and 16-character hits | 1,204 | 92.5% | 66.5% |
| Also minus items whose "last 2 characters of left context + target" occur | 1,122 | 92.1% | 65.2% |

The strictest subset still scores 92.1%. Rime falls along with the LM, so these filters mainly remove common phrases; the 1.5-point drop cannot be read as the amount of leakage.

**Left-context controls** (`12k-fresh-s1`, same pools, same scorer; the real left context reproduces the 1,363 of the decision):

| Condition | All Top-1 | 1,080 items with left context | 376 items without |
| --- | ---: | ---: | ---: |
| Real left context | 1,363 (93.6%) | 1,031 (95.5%) | 332 (88.3%) |
| Left context removed | 1,193 (81.9%) | 861 (79.7%) | 332 (88.3%) |
| Left context from another item | 1,171 (80.4%) | 862 (79.8%) | 309 (82.2%) |

On items with left context, removing it costs 15.7 points, the same as swapping in an unrelated one: the model really uses context.

**Why the scores are so high**: Rime provides the candidates and 97.5% of answers are already in the pool, so the model only ranks; targets are only 2–12 characters, and accuracy falls with length (88.4% at 7–12); items are common conversational sentences, and problematic items were removed in review; Rime's first candidate (68.8%) ranks by frequency and dictionary only, uses little context, and is an easy baseline to beat.

Two limits: exact matching cannot find paraphrases or near-duplicates; messages shorter than 16 characters can only be checked through "left context + target", and short sentences are naturally common on the web, so they cannot be told apart from leakage. Scripts and results are in `outputs/benchmarks/zh13-20m/leak-check/` (not in git).

## Error analysis (`12k-fresh-s1`)

93 misses in all, 54 of them long targets of 7–12 characters.

- **Answer not in the candidate pool: 36.** Only an expanded Rime pool could fix these.
- **In the pool but misranked: 57**, 42 of them ranked second.
  - 13 differ only in 他/她/它 and 8 only in 那/哪. These 21 often cannot be decided from the left context alone and are common in this kind of dialogue data.
  - The other 36 are mostly homophone confusions.

## Limits

- This is development, not a confirmatory result. Blind was not read.
- The benchmark's three sources are all task-oriented or knowledge dialogues in fixed proportions, not representative of real keyboard traffic.
- Candidates come from Rime and the LM only reranks. The result measures "reranking given the Rime pool", not a complete IME.
- Labels went through AI expert review and human spot checks, not formal native-speaker annotation (the score output records `native_chinese_review: false`, `labels_formal_gold: false`).
- Left context is only the part of the current message before the target (at most 64 characters), with no earlier turns; 376 items have no left context. Latency and on-device performance were not measured.

## Reproduction

- Models: downloaded from private storage; all 72 files in `SHA256SUMS` checked; placed in `artifacts/models/tiny-zh13-*/`.
- Environment: local RTX 5060 Laptop, PyTorch 2.10.0+cu128, FP32, code at `1357e17` (`706a68f`, after scoring, changed docs only).
- Before producing a result, `judge_zh13.py` checked score-file SHA256s, that scores come from each arm's `final.pt`, training status complete, all 20 measurement points, shared tokenizer per vocabulary and the Rime baseline of 1,001; all passed.
- Outputs in `outputs/benchmarks/zh13-20m/` (not in git): `runs/<arm>/{full,abbrev}.jsonl` and `.meta.json`, the decision `judgement/judgement.json` (SHA256 `911f77b0c6ffd9f52a5d693d03c76ab59a674741c4583f84b0f982bf46c3784e`) and `judgement/report.md`.

```powershell
# Once per arm and track: 12K arms with zh-unigram-12288, 8K arms with zh-unigram-8192
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/score_zh13_lm.py <final.pt> <tokenizer.model> outputs/benchmarks/zh13-20m/runs/<arm>/full.jsonl --device cuda --track full
# Once all eight are done
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/judge_zh13.py outputs/benchmarks/zh13-20m/runs outputs/benchmarks/zh13-20m/judgement
```
