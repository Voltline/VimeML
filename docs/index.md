# Documentation index

V4.2 epoch 20 / INT8 block-32 is the current deployment release; it replaced V2.1 `extend5` step 40,000. V1 remains a baseline and fallback. V3 A and V3 B training, evaluation, and archival are complete; their results do not establish a stable replacement advantage. V4.2 leads V2.1 on Standard development and by +22 on the preregistered Standard v2 blind, but neither difference reaches author-cluster significance, and it is +6 (not significant) on the fixed reference. Primary documents are English. [Chinese documentation](zh-CN/index.md) retains the corresponding reference material.

## Model, data, and runtime

| Document | Scope |
| --- | --- |
| [Model card](../MODEL_CARD.md) | Architecture, intended use, releases, metrics, and limitations |
| [Data](data.md) | Frozen corpora, cleaning, splits, and provenance |
| [vimeml-tiny-zh-v1 corpus](zh-CN/data-zh-v1.md) (Chinese) | In progress: FineWeb2 cmn_Hani (21 shards) and LCCC; Traditional-to-Simplified, rule filter, spam-site removal, 955K-item human review queue |
| [vimeml-tiny-zh-v1 experiment plan](zh-CN/training-zh-v1-plan.md) (Chinese) | Preregistration: eight arms at the same 22.4B characters, fresh data vs 5 vs 20 epochs crossed with 8K vs 12K vocabularies, three seeds of 12K fresh for the noise band |
| [vimeml-tiny-zh-v1 eight-arm training record](zh-CN/reports/zh-v1-20261110/training.md) (Chinese) | 2026-11-10: all eight arms completed with no preemption or failure; final-checkpoint BPC and the memorization probe |
| [vimeml-tiny-zh-v1 eight-arm IME benchmark and judgement](zh-CN/reports/zh-v1-20261110/ime-benchmark.md) (Chinese) | Pinyin development Top-1 93.3%-94.4% for the eight arms vs 68.8% for Rime; none of the five contrasts was effective, so the preregistered defaults apply (fresh data, 12K) |
| [vimeml-tiny-zh-v1 blind confirmation plan](zh-CN/benchmarks/confirmation-zh-v1-plan.md) (Chinese) | Preregistration: `12k-fresh-s1` against Rime on the blind pinyin track, run once; run on 2026-11-11 and passed |
| [vimeml-tiny-zh-v1 baseline plan](zh-CN/benchmarks/baselines-zh-v1-plan.md) (Chinese) | Preregistration: rime-ice, rime-ice + Wanxiang grammar model, Luna Pinyin, Qwen2.5-0.5B, Qwen3-0.6B-Base on development and blind (as a reference test) |
| [vimeml-tiny-zh-v1 blind confirmation and baselines](zh-CN/reports/zh-v1-20261111/blind-and-baselines.md) (Chinese) | Blind pinyin Top-1 93.8% vs Rime 72.3%: confirmed; ahead of Qwen2.5-0.5B (91.8%), Qwen3-0.6B (89.2%), rime-ice + grammar (82.2%), Luna Pinyin (68.9%) |
| [vimeml-tiny-zh-v1 Mac results check and iOS release](zh-CN/reports/zh-v1-20261111/ios-release.md) (Chinese) | Handoff verified; INT8 b32 vs FP32 +2/−1 pinyin, +0/−2 abbreviated; about 1.3× V4.2 on the Mac; iPhone not measured; Hugging Face release |
| [Chinese prediction benchmark and desktop model roadmap](zh-CN/benchmarks/predict-zh-plan.md) (Chinese) | Preregistration: the `ime-predict-zh-v1` benchmark (next-phrase hit, keystroke savings, prefix completion, confidence), a no-training scale probe with public models, then gated 200M and 800M desktop models |
| [Chinese prediction benchmark ime-predict-zh-v1](zh-CN/benchmarks/predict-ime-zh.md) (Chinese) | Frozen 2026-11-11: 1,200 dialogues each from CrossWOZ, RiSAWOZ, KdConv and CPED; per split 7,200 next-phrase items, about 6,000 keystroke messages, 2,400 completion items; blind closed |
| [Chinese tokenizer vocabulary probe](zh-CN/reports/zh-v1-20261106/tokenizer-vocab.md) (Chinese) | Tokenizer-only comparison of 8K–32K Unigram vocabularies and coverage 0.9995 / 0.9999: the 20M model will compare only 8K and 12K at 0.9995; 16K–32K are deferred to a 200M model |
| [V1 training](training.md) | Original baseline and reproducibility |
| [V2 training](training-v2.md) / [V2 design](plan_v2.md) | Architecture, tokenization, cropping, and schedules |
| [V3 experiments](plan_v3.md) | Adaptation and joint retraining scope |
| [V4 training](training-v4.md) | Fixed 8K architecture, completed A/B schedules, results, and recovery |
| [V4 post-training](posttraining-v4.md) / [conversion data](data-v4-c1.md) | Completed C1/C1r2 experiments, frozen data, conditional scoring, and first-run settings |
| [V4.1 joint pretraining](training-v41.md) | Completed sixteen-epoch LM/chat/conversion experiment and local evaluation |
| [V4.2 LM/chat pretraining](training-v42.md) | Completed twenty-epoch LM/chat experiment without conversion, and local evaluation |
| [V5 scale-up pretraining](training-v5.md) | d512/L8 (28.4M) size-only comparison against V4.3: LM BPC −0.10, IME Top-1 unchanged; not deployed |
| [WSD schedule runs](zh-CN/training-wsd.md) (Chinese) | V4.2 and V5 for 50 epochs with warmup-stable-decay at peak lr 3e-4: best BPC at each size (3.0328, 2.9317), confounded with a larger budget; IME Top-1 unchanged; not deployed |
| [V4.4 ablation plan](zh-CN/training-v44-plan.md) (Chinese) | Preregistered six-arm single-factor ablation of the V4.2 recipe: three replicate seeds for the first run-to-run IME noise band, chat fraction 0 / 0.10, prefix crop 0.60; epoch 20 only, author-cluster tests against V4.2 |
| [V4.4 ablation results](zh-CN/training-v44.md) (Chinese) | No treatment arm effective (removing chat hurts); Standard Top-1 seed band 501–523 (SD 9.5) with V4.2 at the top, so every earlier single-seed Standard difference is within noise; exploratory 4-seed log-probability ensemble reaches 535 |
| [V4.5 distillation plan](zh-CN/training-v45-plan.md) (Chinese) | Preregistered ensemble distillation: eight 15M students (geometric vs arithmetic pool of the four V4.2-recipe seeds, each student the exact twin of one CE seed), 0.5 CE + 0.5 soft target; paired student-minus-twin test on Standard development |
| [IME error decomposition (Chinese)](zh-CN/reports/analysis-20261021/ime-errors.md) | Remaining Top-1 errors: candidate-pool misses, notation variants, few semantic errors |
| [Training-free rescoring (Chinese)](zh-CN/reports/analysis-20261021/rescoring.md) | PMI, length and engine-score fusion; none passes held-out tracks, deployment keeps plain LM scoring |
| [Evaluation](evaluation.md) | Fixed-pool scoring, statistics, and benchmark roles |
| [Standard IME](benchmarks/standard-ime.md) / [initial comparison](benchmarks/standard-ime-results-20261008.md) | Source splits, AI review, candidate export, and reference-set consumption |
| [IME metrics v2 (Chinese)](zh-CN/benchmarks/metrics-v2.md) | Notation-lenient and pool-conditional Top-1, author-cluster tests, decision rule, and track roles |
| [Confirmation plan v2](zh-CN/benchmarks/confirmation-v2-plan.md) and [results](zh-CN/benchmarks/confirmation-v2-results-20261021.md) (Chinese) | Preregistered scoring of Standard v2 blind and the homophone challenge. V2.1 -> V4.2 +22 exact on blind, author-cluster p = 0.0508: not confirmed; V4.2 is already deployed. V4.3 -> V5 +18 on homophones (Holm p = 0.025) |
| [Seed-ensemble replication](zh-CN/benchmarks/ensemble-replication-v2-results-20261105.md) (Chinese) | Preregistered second use of v2 blind and homophones: the four-seed ensemble beats V4.2 on Standard blind 619 vs 598 (+21, author-cluster p = 0.0012), replicated; homophones +7, not significant |
| [Standard v2 blind](zh-CN/benchmarks/standard-ime-v2-blind.md), [homophone challenge](zh-CN/benchmarks/homophone-ime.md), [incremental prototype](zh-CN/benchmarks/incremental-ime.md), [candidate export v2](zh-CN/benchmarks/candidates-v2-handoff.md) (Chinese) | New frozen confirmation, written-text homophone, and segmented-input tracks; the first two are now consumed |
| [Core ML](coreml.md) | Conversion, quantization, client contract, and V2.1/V4.2 measurements |
| [Artifact layout](artifacts.md) / [cross-platform exchange](reference/artifact-exchange.md) | Local assets, Git collaboration, and ZIP transfer |

## Experiment records

| Experiment | Records |
| --- | --- |
| V2 family | [Summary](reports/v2-summary.md) |
| V2 data/model preparation | [Tokenizer](reports/v2-20261007/tokenizer.md), [architecture](reports/v2-20261007/model.md), [token store](reports/v2-20261007/token-data.md), [prefix crop](reports/v2-20261007/prefix-crop.md) |
| V2.0 | [Environment](reports/v2-20261007/environment.md), [evaluation](reports/v2-20261007/evaluation.md) |
| V2.1 restart1 | [Plan](reports/v2-20261007/v21-plan.md), [throughput](reports/v2-20261007/v21-performance.md), [interrupted predecessor](reports/v2-20261007/v21-recovery.md), [evaluation](reports/v2-20261007/v21-evaluation.md) |
| V2.1 extension | [Training and selection](reports/v2-20261007/v21-extend.md) |
| Historical expanded IME set | [Construction](reports/v2-20261007/ime-3000-handoff.md), [initial evaluation](reports/v2-20261007/expanded-ime-evaluation.md), [label/error audit](reports/v2-20261007/label-error-audit.md) |
| V2.1 Apple deployment | [Core ML](reports/mac-20261008/v21-coreml.md), [quantization review](reports/mac-20261008/v21-quantization-review.md), [iPhone measurements](reports/mac-20261008/v21-iphone.md) |
| V3 A | [Evaluation](reports/v3-20261008/a-evaluation.md) |
| V3 B | [Training plan](reports/v3-20261008/b-plan.md), [final evaluation](reports/v3-20261008/b-evaluation.md) |
| V4-a data preparation | [Tokenizer and vocabulary comparison](reports/v4-20261014/tokenizer.md), [8K token data and windows](reports/v4-20261014/token-data.md) |
| V4-a model preparation | [Fixed architecture and local acceptance](reports/v4-20261014/model.md) |
| V4 base LM | [A continuation experiment](reports/v4-20261015/evaluation.md), [B twelve-epoch cosine comparison](reports/v4-20261015/b12cosine-evaluation.md) |
| V4 conversion post-training | [C1r2 data revision and diagnostics (Chinese)](zh-CN/reports/v4-20261016/c1r2.md) |
| V4.1 joint pretraining | [Results summary](training-v41.md), [complete evaluation (Chinese)](zh-CN/reports/v41-20261016/evaluation.md) |
| V4.2 LM/chat pretraining | [Results summary](training-v42.md), [complete evaluation (Chinese)](zh-CN/reports/v42-20261020/evaluation.md), [fixed reference and regression analysis (Chinese)](zh-CN/reports/v42-20261020/reference-regressions.md) |
| V4.2 Apple deployment | [Core ML, INT8 and Mac measurements](reports/mac-20261020/v42-coreml.md) |
| vimeml-tiny-zh-v1 Apple deployment | [Core ML, INT8 and quantization survey](reports/mac-20261111/zh-v1-coreml.md) |
| External Japanese LMs | [TinySwallow](zh-CN/reports/v4-20261016/external-1p5b.md), [RakutenAI](zh-CN/reports/v4-20261016/external-rakuten-1p5b.md), [Sarashina](zh-CN/reports/v4-20261016/external-sarashina-1p4b.md) |

## Historical V1 references

[FP32 results](reference/results.md), [Core ML measurements](reference/coreml-v1.md), [simulator memory](reports/mac-20261006/simulator-memory.md), [real keyboard extension](reports/mac-20261006/iphone-keyboard-memory.md), [artifact import](reports/mac-20261006/handoff.md), and [merge verification](reports/mac-20261006/merge-verification.md).

Methods: [corpus pipeline](reference/data-pipeline.md), [AJIMEE export](reference/ajimee-benchmark.md), [development generation](reference/development-generation.md), [hybrid ranking](reference/hybrid-ranking.md), and [phrase demo](reference/phrase-demo.md).

`history/` preserves pre-measurement [deployment design](history/deployment.md), [conversion commands](history/coreml-guide.md), and [workflow status](history/workflow.md). Historical expectations are distinct from subsequent measurements. Original JSON, candidate-level scores, and traces remain in Git-ignored `outputs/`.
