# Documentation index

V2.1 `extend5` step 40,000 / INT8 block-32 is the current deployment release. V1 remains a baseline and fallback. V3 A and V3 B training, evaluation, and archival are complete; their results do not establish a stable replacement advantage. Primary documents are English. [Chinese documentation](zh-CN/index.md) retains the corresponding reference material.

## Model, data, and runtime

| Document | Scope |
| --- | --- |
| [Model card](../MODEL_CARD.md) | Architecture, intended use, releases, metrics, and limitations |
| [Data](data.md) | Frozen corpora, cleaning, splits, and provenance |
| [V1 training](training.md) | Original baseline and reproducibility |
| [V2 training](training-v2.md) / [V2 design](plan_v2.md) | Architecture, tokenization, cropping, and schedules |
| [V3 experiments](plan_v3.md) | Adaptation and joint retraining scope |
| [Evaluation](evaluation.md) | Fixed-pool scoring, statistics, and benchmark roles |
| [Standard IME](benchmarks/standard-ime.md) / [initial comparison](benchmarks/standard-ime-results-20261008.md) | Source splits, AI review, candidate export, and reference-set consumption |
| [Core ML](coreml.md) | Conversion, quantization, and client contract |
| [Artifact layout](artifacts.md) / [cross-platform exchange](reference/artifact-exchange.md) | Local assets, Git collaboration, and ZIP transfer |

## Experiment records

| Experiment | Records |
| --- | --- |
| V2 family | [Summary](reports/v2-summary.md) |
| V2 data/model preparation | [Tokenizer](reports/v2-20261007/tokenizer.md), [architecture](reports/v2-20261007/model.md), [token store](reports/v2-20261007/token-data.md), [prefix crop](reports/v2-20261007/prefix-crop.md) |
| V2.0 | [Environment](reports/v2-20261007/autodl.md), [evaluation](reports/v2-20261007/evaluation.md) |
| V2.1 restart1 | [Plan](reports/v2-20261007/v21-plan.md), [throughput](reports/v2-20261007/v21-performance.md), [interrupted predecessor](reports/v2-20261007/v21-recovery.md), [evaluation](reports/v2-20261007/v21-evaluation.md) |
| V2.1 extension | [Training and selection](reports/v2-20261007/v21-extend.md) |
| Historical expanded IME set | [Construction](reports/v2-20261007/ime-3000-handoff.md), [initial evaluation](reports/v2-20261007/expanded-ime-evaluation.md), [label/error audit](reports/v2-20261007/label-error-audit.md) |
| V2.1 Apple deployment | [Core ML](reports/mac-20261008/v21-coreml.md), [quantization review](reports/mac-20261008/v21-quantization-review.md), [iPhone measurements](reports/mac-20261008/v21-iphone.md) |
| V3 A | [Evaluation](reports/v3-20261008/a-evaluation.md) |
| V3 B | [Training plan](reports/v3-20261008/b-plan.md), [final evaluation](reports/v3-20261008/b-evaluation.md) |

## Historical V1 references

[FP32 results](reference/results.md), [Core ML measurements](reference/coreml-v1.md), [simulator memory](reports/mac-20261006/simulator-memory.md), [real keyboard extension](reports/mac-20261006/iphone-keyboard-memory.md), [artifact import](reports/mac-20261006/handoff.md), and [merge verification](reports/mac-20261006/merge-verification.md).

Methods: [corpus pipeline](reference/data-pipeline.md), [AJIMEE export](reference/ajimee-benchmark.md), [development generation](reference/development-generation.md), [hybrid ranking](reference/hybrid-ranking.md), and [phrase demo](reference/phrase-demo.md).

`history/` preserves pre-measurement [deployment design](history/deployment.md), [conversion commands](history/coreml-guide.md), and [workflow status](history/workflow.md). Historical expectations are distinct from subsequent measurements. Original JSON, candidate-level scores, and traces remain in Git-ignored `outputs/`.
