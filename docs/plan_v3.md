# V3 data-mixture experiments

Status: adaptation A and joint retraining B completed on 2026-10-08. Neither experiment establishes a stable IME replacement advantage; V2.1 remains deployed. Architecture and the V2 16K tokenizer remain fixed. The primary variables are source distribution, optimization schedule, initialization, and training budget.

## Source selection and preprocessing

The new web pool is sampled from eight downloaded JpnMix `minhash_deduped` shards. Sixteen shards exist locally, but only the selected eight-shard cleaned sample participates. It contributes 349,384 selected documents and approximately 500 million prediction pairs before filtering. RealPersonaChat and the multi-relational multi-party corpus add 509,299 turns across 14,543 conversations.

Dialogue splits hold out complete conversations. Speaker-disjointness is not established. New-source training/validation/test initially contain 1,563,989 / 33,787 / 35,341 sequences and 505,598,379 / 13,466,346 / 14,206,605 prediction pairs. The chat-only split contains 496,266 / 4,986 / 5,097 sequences. These counts precede the subsequent train-only virtual-map filtering and repeated exposure.

49,205 obvious pricing/catalogue blocks are excluded from training. Validation/test retain the broad original source distribution. B additionally screens exact/anchor overlaps between old and new pools, excluding 1,062 old sequences, 16,854 new sequences, and 67 exact cross-pool web duplicates. This is limited screening rather than complete semantic decontamination.

## Effective mixture and loss scaling

Both runs target web/chat = 95/5 by effective post-crop tokens, using probability 0.30 and at least eight retained prediction targets for eligible first windows. Frozen virtual mappings reference existing stores without duplicating all encoded data. Repeated chat exposure is reported separately from unique data volume.

`epoch_mean_tokens` uses a fixed update-scale denominator. This prevents short chat batches from receiving the same aggregate weight as much longer web batches solely through per-batch mean normalization. Corpus composition, crop, mapping, and loss scaling are frozen for each run.

## Schedules and completed budgets

| Setting | V3 A | V3 B |
| --- | --- | --- |
| Initialization | V2.1 extension step 40,000 weights, new AdamW | Random initialization |
| Sources | Selected new web pool + chat | Old V2 train + selected new web pool + chat |
| Epochs / updates | 2 / 25,146 | 4 / 265,841 |
| Effective tokens | 952,851,047 | 4,001,730,586 |
| Chat effective tokens | 47,642,545 | 200,086,509 |
| Chat exposure | Approximately 9.08× | Approximately 38.53× |
| Batch / precision | 512 / BF16 | 512 / BF16 |
| Learning rate / warmup | 1e-4 → 1e-5 / 500 | 1e-3 → 1e-4 / 2,000 |
| Backbone | Compiled | Compiled |
| GPU | RTX 4090 D | Reported RTX 4080 SUPER, 32,760 MiB |
| Update throughput | 614,843 effective tokens/s | 307,681 effective tokens/s |
| Configuration | [train-v3-a.toml](../configs/train-v3-a.toml) | [train-v3-b.toml](../configs/train-v3-b.toml) |

Reported throughput excludes the initialization/validation/saving intervals defined in each report. A benefits from V2.1's earlier pretraining budget; B starts from random weights. These runs are not a controlled architecture comparison at equal data or compute.

## Evaluation and selection

Domain validation distinguishes new web, held-out chat, and the old corpus; BPC is comparable only on identical text and measurement protocol. Original AJIMEE and historical development remain fixed. Standard Japanese IME development uses 728 cases with frozen AI-reviewed labels. Its previously consumed 725-case split is a fixed reference; the old 500-case web set remains a separate draft-label regression track. The historical 1,000-case blind set is unscored.

A uses its preselected final checkpoint at step 25,146. B selection uses the highest Standard development micro Top-1 among four epoch checkpoints and the subset-loss best: epoch 2, step 132,915, 507/728. This selection is frozen before fixed-reference and regression scoring. B's subset-loss best and full-epoch BPC best remain separate archived checkpoints.

| Fixed task | V2.1 | V3 A | Selected V3 B |
| --- | ---: | ---: | ---: |
| Standard development / 728 | 505 | 508 | 507 |
| Exposed reference / 725 | 535 | 546 | 533 |
| Historical regression / 500 | 374 | 368 | 364 |

AI-reviewed labels are not native-speaker gold. Case-level paired tests are exploratory, with author/user dependence, multiple comparisons, and development selection limitations. No post-hoc label or candidate alias changes determine selection.

## Outcomes and records

A substantially reduces held-out chat BPC but worsens old-corpus BPC and the historical 2,000-case draft score. B continues to improve language-model validation loss without a stable corresponding IME Top-1 trend. Neither result isolates a single dataset or architecture as the cause.

All necessary checkpoints, evaluations, and inference examples are archived. [A evaluation](reports/v3-20261008/a-evaluation.md), [B training scope](reports/v3-20261008/b-plan.md), and [B final evaluation](reports/v3-20261008/b-evaluation.md) provide detailed results. Local stage/selection/reference records reside in `outputs/training-v3/` and `outputs/training-v3-b/`. No additional run is implied by this completed experiment record.
