# V3 B joint retraining results — 2026-10-08

Joint retraining does not establish a stable IME replacement advantage. V2.1 remains deployed. The selected epoch-2 checkpoint scores 507/728 on Standard development, 533/725 on the exposed fixed reference, and 364/500 on historical web regression; V2.1 scores 505, 535, and 374 respectively.

The randomly initialized V2 architecture completes four epochs, 265,841 updates, and 4,001,730,586 effective tokens, including 5% chat. Input scope is old V2 text, the selected eight-shard JpnMix pool, and two dialogue corpora. The run takes 13,428.8 seconds (3 hours 44 minutes); mean formal-update throughput is 307,681 effective tokens/s, excluding validation, saving, and initialization intervals. [Training design](b-plan.md) records the frozen configuration.

## Epoch and checkpoint measurements

| Checkpoint | New-pool BPC ↓ | Chat BPC ↓ | Old-corpus BPC ↓ | AJIMEE / 200 | Historical dev / 137 | Standard dev / 728 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Epoch 1, step 66,459 | 3.413605 | 3.329645 | 3.329031 | 128 | 123 | 496 |
| Epoch 2, step 132,915 | 3.375466 | 3.255242 | 3.277100 | 134 | 120 | 507 |
| Epoch 3, step 199,377 | 3.303812 | 3.181846 | 3.211323 | 128 | 125 | 489 |
| Epoch 4, step 265,841 | 3.237611 | 3.130899 | 3.164418 | 139 | 126 | 498 |
| Subset-loss best, step 250,000 | 3.235060 | Not evaluated | Not evaluated | 135 | 126 | 502 |

BPC reuses original complete validation through the BF16 training path; epoch IME uses FP32 parameters. Existing same-step scores are reused. Step 250,000 receives one supplemental new-pool/full-task evaluation. BPC across different text pools is not directly comparable. Epoch-4 old BPC remains above frozen V2.1 3.080269. Decreasing validation loss does not correspond to a stable IME Top-1 trend.

## Selection and development comparison

Before reference evaluation, Standard development micro Top-1 selects epoch 2 from the four epoch checkpoints and step 250,000. This decision is frozen. Epoch 4 remains the best complete-epoch BPC checkpoint; step 250,000 remains the fixed-subset-loss best. Neither subsequent reference nor regression results change selection.

| Standard development | V2.1 | V3 A | V3 B epoch 2 |
| --- | ---: | ---: | ---: |
| WRIME / 483 | 298 | 310 | 297 |
| JMultiWOZ / 245 | 207 | 198 | 210 |
| Micro / 728 | 505 (69.37%) | 508 (69.78%) | 507 (69.64%) |
| Equal-source macro | 73.09% | 72.50% | 73.60% |

B relative to V2.1 corrects 43 and regresses 41 cases, net +2, exact case-level McNemar two-sided p=0.9132. Relative to A it corrects 43 and regresses 44, net -1, p=1.0000. WRIME and JMultiWOZ effects differ; neither aggregate alone establishes uniform improvement.

Comparison checks frozen IDs, references, readings, left context, source/provenance, candidate text, and engine order. References and candidate aliases are not added after observing model results. Case-level p-values are exploratory: author/user correlation, multiple comparisons, and development selection are not fully corrected. This is not an equal-data/equal-compute controlled architecture comparison.

## Post-selection reference and historical regression

Step 132,915 is registered under a reference-test plan and scored once with local CUDA/FP32. The 725-case split was already consumed by the A comparison; this evaluation is a fixed reference, not a new blind test. Historical 500-case labels remain drafts. The old 1,000 blind cases are not scored.

| Fixed track | V2.1 | V3 A | V3 B epoch 2 |
| --- | ---: | ---: | ---: |
| Reference WRIME / 475 | 309 | 318 | 308 |
| Reference JMultiWOZ / 250 | 226 | 228 | 225 |
| Reference micro / 725 | 535 (73.79%) | 546 (75.31%) | 533 (73.52%) |
| Reference equal-source macro | 77.73% | 79.07% | 77.42% |
| Historical regression / 500 | 374 (74.80%) | 368 (73.60%) | 364 (72.80%) |

| B relative to baseline | Corrected | Regressed | Net | Exact two-sided p |
| --- | ---: | ---: | ---: | ---: |
| Reference: V2.1 | 37 | 39 | -2 | 0.9088 |
| Reference: A | 32 | 45 | -13 | 0.1711 |
| Regression: V2.1 | 16 | 26 | -10 | 0.1641 |
| Regression: A | 19 | 23 | -4 | 0.6440 |

Reference Top-5 is 636/725, MRR 0.79740, and MinCER 0.03843, with 113 engine corrections and 47 regressions. Historical regression Top-5 is 443/500, MRR 0.79963, and MinCER 0.02989, with 72 corrections and 25 regressions. Both tracks have zero fallbacks. Reference coverage is 650/725; historical coverage is 443/500. New-source scores remain separate from historical web scores.

Legacy caches use an earlier external-benchmark draft wrapper; current reports use the later reviewed wrapper. The 500 historical references are unchanged. Metadata differences are limited to external-track review/naming/version wrappers, with per-case reference/context/source/provenance/candidate identity checked before baseline reuse. Original caches remain untouched. Development uses the FP32 compiled training evaluation; added reference/regression uses ordinary checkpoint FP32 inference with the same tokenizer and frozen scoring rules.

## Qualitative behavior and interpretation

Four fixed developer prefixes complete 16-token greedy inference. Repetition and semantic inconsistency remain, including `明日の朝は、雨が降って、雨が降った。`. These examples establish loadability and illustrative behavior, not conversational accuracy or UI latency.

Validation loss continues to decrease, so complete convergence is not established. Exposed development/reference comparisons do not supply independent evidence for additional training benefits. A incorporates V2.1 pretraining; B starts randomly and receives approximately four billion effective tokens, with different mixtures, schedules, and chat repetition. Results do not isolate a single source or architecture as the cause of task tradeoffs.

## Completed archival

Training, final validation, selected-model scoring, and archival are complete. Original checkpoints retain optimizer state under `artifacts/models/tiny-ja-v3-b-all8-chat05-d320-l6/`:

| File | Selection meaning | Step | Bytes |
| --- | --- | ---: | ---: |
| `epoch-2.pt` | Selected IME experiment checkpoint | 132,915 | 150,527,943 |
| `best.pt` | Fixed-subset-loss best | 250,000 | 150,526,803 |
| `best-epoch.pt` | Best full-epoch BPC | 265,841 | 150,528,507 |

Byte counts, checkpoint steps, and random-initialization records are checked after transfer. No incomplete model files remain. Original epoch JSON/scores reside in `outputs/training-v3-b/remote-reports/`; supplemental step-250,000 results reside in `best-step250000/`. Matching epoch-2 AJIMEE/historical/Standard development scores are reused directly.

`outputs/training-v3-b/` retains `selection.json`, `reference-test-plan.json`, `legacy500-selected/`, `reference725-selected/`, `paired-*.json`, `inference-selected.json`, `archive-manifest.json`, and `stages.json`. AI-expert labels are not native-speaker gold. The consumed reference comparison does not alter selection. No further training or deployment replacement follows from this completed record.
