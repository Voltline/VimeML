# VimeML model card

## Model identity and availability

VimeML is a family of compact Japanese decoder-only language models for input-method candidate reranking. Public releases are [V1](https://huggingface.co/Voltline/vimeml-tiny-ja-v1), [V2.1](https://huggingface.co/Voltline/vimeml-tiny-ja-v2.1) and the research release [V4.2](https://huggingface.co/Voltline/vimeml-tiny-ja-v4.2). V2.0, V3 A, V3 B, V4-B, V4.1, V4.3 and V5 are experiment checkpoints. V4.2 is the deployed model; V2.1 is the previous deployment.

| Architecture | V1 | V2 / V2.1 / V3 experiments | V4 family (V4-B / V4.1 / V4.2) |
| --- | --- | --- | --- |
| Parameters | 7,386,624 | 12,537,920 | 15,070,848 |
| Vocabulary / context | 16,384 / 128 tokens | 16,384 / 128 tokens | 8,192 / 128 tokens |
| Hidden width / layers / attention heads | 256 / 4 / 4 | 320 / 6 / 5 | 384 / 7 / 6 |
| Feed-forward network | GELU, width 1,024 | SwiGLU, width 832 | SwiGLU, width 960 |
| Normalization | Pre-LayerNorm | Pre-RMSNorm, FP32 variance, epsilon 1e-5 | Pre-RMSNorm, epsilon 1e-5 |
| Attention / position representation | Causal SDPA / learned absolute embeddings | Causal SDPA / learned absolute embeddings | Causal SDPA / learned absolute embeddings |
| Output weights | Tied token embedding and LM head | Tied token embedding and LM head | Tied token embedding and LM head |
| Dropout | 0 | 0 | 0 |

V2 and V4 use bias-free linear layers. All families use SentencePiece Unigram with byte fallback and PAD/UNK/BOS/EOS IDs 0/1/2/3. V2 disables `add_dummy_prefix`; token IDs differ from V1. The V4 8K tokenizer reserves IDs 4–6 for conversion markers, which V4.2 does not use and generation excludes. Matching model, tokenizer, and manifests form a single artifact identity. The deployed client runtime is stateless; KV-cache Core ML graphs exist for V2.1 and V4.2 as validated research exports.

## Intended use

The primary task is reranking real AzooKey kana-to-kanji candidates using supplied left context. Scoring jointly tokenizes context and each candidate, then sums full-vocabulary log probabilities after the candidate pool's shared token prefix. This token-suffix likelihood is a scoring surrogate rather than an exact conditional probability over strings. Dictionary retrieval, tokenization, fallback, stable ordering, and search remain application-side responsibilities.

Short continuation and next-word suggestions are experimental. Instruction following, factual question answering, long-form generation, and general conversational assistance are outside the validated use cases. Generated continuations can repeat text or become semantically inconsistent.

## Training data and procedure

V1/V2 use a frozen sentence corpus from nine FineWeb2-Edu Japanese shards and Tatoeba: 25,713,003 unique sentences, with document-grouped train/validation/test splits of 25,185,368 / 260,337 / 267,298. Exact duplicate sentences share a split. Comprehensive semantic near-duplicate and benchmark-contamination audits were not completed.

V1 trains from random initialization for one epoch. V2.0 trains from random initialization for four epochs. V2.1 `restart1` continues from V2.0 step 375,000 for two epochs with a new AdamW optimizer. The subsequent `extend5` experiment carries optimizer state, runs 3.273 additional epochs, and selects step 40,000 as the deployed checkpoint. A deterministic 30% eligible-first-window prefix crop is used for V2 training, not validation.

V3 A adapts V2.1 to a JpnMix sample and two real-dialogue corpora. V3 B trains the V2 architecture from scratch on old V2 text, the selected eight-shard JpnMix pool, and the same dialogue sources. Dialogue contributes 5% of effective post-crop tokens; repeated exposure is not additional unique text. V3 does not replace V2.1. [Data](docs/data.md) and [training](docs/training-v2.md) provide artifact scope and configuration references.

The V4 family uses a new 8K tokenizer and a fixed 15.07M architecture. V4-B trains the original V2 text from random initialization; V4.1 adds dialogue and a reading-conditioned conversion objective; V4.2 removes the conversion objective and trains twenty epochs on the original text plus about 5% dialogue targets (12.1 billion supervised targets). Epoch 20 was selected on complete LM validation BPC before any IME scoring. [V4.2 training](docs/training-v42.md) records data, objective and settings.

V4.3 extends the V4.2 recipe with nine more web shards; V5 trains the V4.3 recipe at d512/L8 (28,385,792 parameters, separate `tiny_gpt_v5` identity). V5 lowers validation BPC to 2.942180 but its IME Top-1 (Standard 514, Legacy 365, AJIMEE 145, historical development 124) is not significantly different from V4.2 or V4.3, so neither is released or converted. [V5 training](docs/training-v5.md).

## Evaluation

The [README table](README.md#evaluation-summary) reports historical fixed-pool results. The V2.1 FP32 release has validation BPC 3.0802686, AJIMEE Top-1 144/200, historical development Top-1 122/137, and draft development Top-1 1,487/2,000. INT8 has 144/200, 122/137, and 1,481/2,000, respectively.

Standard Japanese IME development results are V2.1 505/728, V3 A 508/728, and selected V3 B 507/728. The originally held-out 725-case split has already been consumed and is now a fixed reference set. Scores are 535/725, 546/725, and 533/725. The corresponding historical 500-case regression scores are 374, 368, and 364. These small differences do not establish a stable V3 deployment advantage.

V4.2 scores 523/728 on Standard development, 541/725 on the fixed reference (registered before scoring, run once), 370/500 on legacy regression, 139/200 on AJIMEE and 123/137 on historical development. Against V2.1 the reference difference is +6 (40 improvements, 34 regressions, p=0.56). Most Top-1 flips are notation variants of the same word; AJIMEE losses are mainly encyclopedic homophone errors. V4.2 is not a demonstrated accuracy improvement over V2.1: on the preregistered Standard v2 blind it is +22/729 (author-cluster p = 0.0508, not confirmed) and on the written-text homophone challenge −12/619 (not significant). It is nonetheless the deployed model. [Reference and regression analysis (Chinese)](docs/zh-CN/reports/v42-20261020/reference-regressions.md).

Standard labels were reviewed by AI before initial candidate/model scoring; they are not native-speaker gold. Historical draft labels and the exposed reference set have separate roles. Case-level paired statistics are exploratory and do not account fully for author/user correlation, multiple comparisons, or development-based checkpoint selection. [Evaluation methodology](docs/evaluation.md) describes denominators and fallback behavior.

## Core ML deployment and numerical limitations

The V2.1 package uses symmetric INT8 block-32 weight compression with FP32 computation and `CPU_ONLY`, targeting iOS 18 or later. Input is `int32[1,T]`, 1 ≤ T ≤ 128; output is raw `float32[1,T,16384]` logits. The package is 14.33 MB, versus 50.36 MB uncompressed.

Strict logits alignment fails after quantization: maximum absolute error 1.00949144, with mean absolute error 0.108517 at T=128. Uncompressed conversion passes the recorded tolerance. Unchanged small-set accuracy does not imply unchanged distributions or rankings. The selected package remains an accepted deployment tradeoff supported by fixed-pool and limited-device measurements. Quantization failure is not established as an unavoidable consequence of model size or architecture. [Quantization review](docs/reports/mac-20261008/v21-quantization-review.md) retains the failure evidence.

The deployed V4.2 package uses the same INT8 recipe: 17.16 MB, output `float32[1,T,8192]`. Uncompressed conversion passes (maximum error 5.2e-5); INT8 fails strict alignment with maximum error 0.477 and mean 0.0495 at T=128, while Top-1 changes by at most one case per track. On an iPhone 16 Pro Max XCTest host, V4.2 INT8 runs 0.16–0.35 ms slower per `predict` call, has lower rerank p95 (120.9 vs 136.8 ms) and a 5.6 MB lower footprint peak than V2.1 INT8. Keyboard-extension memory and typing latency were not measured. [V4.2 Core ML report](docs/reports/mac-20261020/v42-coreml.md).

## Limitations and licensing

Candidate recall bounds reranking accuracy. Short context and sentence-oriented pretraining limit broader discourse handling. Web text can contain noise, biases, and personal information; cleaning is incomplete. Device measurements describe specific recorded sessions, not universal latency, memory, power, or long-term stability guarantees.

Project code and published weights use GPL-3.0-or-later with an additional permission for application distribution services such as the App Store ([LICENSE-EXCEPTION](LICENSE-EXCEPTION)); releases before 2026-10-22 used GPL-2.0-only. Libraries and datasets retain independent terms; [NOTICE.md](NOTICE.md) records attribution and redistribution boundaries. Model bundles contain aggregate benchmark results rather than restricted raw benchmark text. Published inference weights omit optimizer state and cannot resume training directly.
