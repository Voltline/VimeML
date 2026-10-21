# V4 conversion post-training

V4-C1 adds reading-conditioned conversion and ordinary-LM replay to the completed V4-B language model. The first run stopped after epoch two under its LM retention guard on 2026-10-16. The [C1r2 revision](zh-CN/reports/v4-20261016/c1r2.md) completed eleven conversion passes before its conversion plateau stop; best conditional Standard development was 473/728, below V4-B's ordinary-LM 498/728. The sections below retain the initial C1 design; subsequent experiments are recorded separately. [Chinese document](zh-CN/posttraining-v4.md).

C1 tests whether reading supervision improves the existing AzooKey candidate pools while retaining ordinary language modeling. C2, a separate candidate-ranking experiment, depends on positive C1 results. These stages are distinct from the A/B base-LM schedule experiments in the [V4-B evaluation](reports/v4-20261015/b12cosine-evaluation.md).

## Initialization and frozen data

| Component | Identity |
| --- | --- |
| Source | `artifacts/models/tiny-ja-v4-b12cosine-e8k-d384-l7/best.pt`, step 584,000 |
| Source SHA256 | `18c49f1d4de5b2852f98c9bc30bf48ca7c84077cfb09a25e3c9002cedf5d1106` |
| Architecture | 15,070,848 parameters; d384, 7 layers, 6 heads, SwiGLU 960, context 128 |
| Tokenizer | Existing 8K Unigram, identity normalization and byte fallback |
| Marker IDs | `<CTX>` = 4, `<READ>` = 5, `<OUT>` = 6; PAD/UNK/BOS/EOS = 0/1/2/3 |
| Conversion data | 2,841,003 training and 18,956 validation examples |
| LM replay | Original frozen V4 train token store and context-128 windows |
| LM validation | Original complete 260,337 sentences |

Initialization loads all source weights with fresh AdamW state and a new post-training step counter. Source step and token exposure remain recorded separately. The tokenizer and model dimensions are unchanged.

The [conversion dataset](data-v4-c1.md) derives one span per parent sentence from original train/validation text. Dual-dictionary agreement, original-surface alignment, orthographic kana rules, and conservative ambiguity filters produce automatic silver labels. Parent-sentence and exact task-case isolation are checked; document-level and semantic independence are not established. Benchmark text does not enter training, and blind data is excluded from preparation and evaluation. The conversion validation parent set previously informed base-LM selection, so it is not an independent test of the entire research process.

## Input and objective

```text
[BOS]<CTX>昨日友達と<READ>はなした<OUT>話した[EOS]
```

Text fields are encoded separately and markers inserted by ID. Reserved marker strings in raw fields are rejected. Conversion labels supervise only OUT and EOS, including the first surface token predicted from `<OUT>`. Labels are shifted once; CTX, READ, markers, and padding are masked as targets. Conditional input embeddings still receive gradients.

At most 129 serialized tokens produce 128 input/target positions. Preparation crops only the left context at text boundaries and preserves complete reading/output fields. Examples whose reading/output alone exceed the limit are excluded. A fixed 10% of training examples have an empty-READ view; validation retains its reading. Ordinary LM retains its unmarked format and 30% eligible prefix crop, with at least eight remaining targets. No `<LM>` marker is added.

```text
L = 0.7 * mean_token_loss(OUT + EOS)
  + 0.3 * mean_token_loss(ordinary LM)
```

Each optimizer update consumes 256 conversion and 256 LM examples. Losses are normalized separately by each task's actual supervised-token count, weighted, and accumulated before one clipping/optimizer step. Loss weights are not also sampling probabilities. The final partial conversion batch is retained; LM tail batches continue across epochs to supply 256 examples per update. Smaller microbatches preserve the same task-wide normalization.

## First-run configuration

The [C1 configuration](../configs/posttrain-v4-c1.toml) freezes these initial experiment settings. Measured outcomes and the revised configuration are recorded in the C1r2 report rather than applied retroactively to this first-run design.

| Setting | Value |
| --- | --- |
| Environment | Linux, RTX 4090 |
| Precision and runtime | BF16, fused AdamW, compiled backbone, eager valid-token head |
| Batch / microbatch | 256 conversion + 256 LM; one microbatch per task |
| AdamW | Betas (0.9, 0.95), weight decay 0.01, gradient clipping 1.0 |
| Learning rate | Single cosine from 1e-4 to 1e-5; 1,332 warmup updates |
| Maximum budget | 12 conversion-pool passes; 133,176 updates |
| Conversion exposure | 137,842,152 OUT/EOS targets; LM exposure counted separately |
| Data loading | Independent length buckets; four workers per training stream, zero validation workers |
| Subset validation / saving | Every 2,000 updates; fixed 4,096-example conversion subset |
| Full validation | Complete conversion NLL, original LM BPC, and fixed Standard development every epoch |
| Conversion stopping | Cumulative relative improvement above 0.2% from the stopping reference; patience three full epochs |
| LM retention guard | Stop after two full epochs more than 1% above initial LM BPC |

Complete LM and IME validation use the primitive FP32 path without cloning a second model. The initial LM reference is V4-B full FP32 BPC 3.0277770232964114; initial conversion NLL is evaluated before updates. Subset loss remains separate from complete-validation metrics. Report supervised tokens, conditional input tokens, padding, and throughput for both streams. Twelve conversion passes are distinct from twelve passes over the original 25-million-sentence LM corpus. Sustained C1 throughput and elapsed training time have not been measured.

## Scoring and model selection

The primary conditional score is the full-vocabulary log-probability sum of OUT tokens, excluding EOS. Training includes EOS for termination. Reading comes only from the original query, normalized by NFKC and katakana-to-hiragana conversion; it is never reconstructed from gold or candidates. Candidate pools, references, stable ties, and the full denominator remain fixed. Evaluation does not crop context. If any candidate exceeds the structured-input limit, the entire case falls back to the original AzooKey order, with fallback counts reported.

Compare V2.1 ordinary LM, V4-B ordinary LM, C1 ordinary LM, C1 conditional scoring, and same-checkpoint empty-READ/empty-CTX diagnostics. The latter measure conditional dependence and are not separately trained controls with equal compute.

Selection requires full LM BPC at most 3.058054794, the 1% retention cap. Eligible checkpoints are ordered by Standard contextual Top-1, then lower complete conversion NLL, then earlier step. LM, conversion-loss, and reranking selections remain separate.

| Evaluation criterion | First-run target |
| --- | --- |
| Standard development | At least 505/728, matching V2.1; V4-B starts at 498/728 |
| Engineering target | 513/728; not a statistical significance threshold |
| Source regression | WRIME and JMultiWOZ each within one percentage point of V2.1 |
| Legacy regression | At least 369/500, compared with V2.1's 374/500 |
| Ordinary LM | Full FP32 BPC at most 3.058054794 |
| Suggestions | Existing 14 examples remain qualitative diagnostics |

These are experimental acceptance criteria, not measured outcomes. Report corrected/damaged cases, net changes, per-source results, and paired uncertainty. Reused development data provides exploratory evidence; a replacement claim requires independent evaluation.

## Implementation and recovery

| Module | Responsibility |
| --- | --- |
| `src/vimeml/posttraining/readings.py` | Dual-dictionary span alignment and reading rules |
| `src/vimeml/posttraining/data.py` | Frozen storage, codec, and OUT/EOS masks |
| `src/vimeml/posttraining/streams.py` | Length buckets, microbatches, and committed task cursors |
| `src/vimeml/posttraining/runtime.py` | Source identity, task normalization, stopping, and resume checks |
| `src/vimeml/posttraining/train.py` | Dedicated training, validation, checkpointing, and selection |
| `src/vimeml/posttraining/infer.py` | Conditional encoding and frozen-pool scoring |
| `scripts/posttraining/` | `prepare_conversion.py` (preparation), `check_conversion.py` (full audit), `refine_conversion.py` (C1 ambiguity quarantine), `prepare_revision.py` (C1r2 views), `train.py`, `evaluate.py`, `prepare_handoff.py` (remote payload list), `launch_v4_c1.sh` |

Cross-stage initialization restores weights only. Same-stage resume requires matching configuration, data, code, precision, and task/scoring identities, and restores AdamW, RNG, both committed cursors, and stopping counters. Cursors advance only after successful updates. A safe checkpoint precedes complete validation; pending validation resumes without repeating an optimizer update. Existing V4 pretraining and V2 initialization protections remain intact.

Output directory: `artifacts/models/tiny-ja-v4-c1-e8k-d384-l7/`. `best-rerank.pt` and its `best.pt` alias require LM qualification. `best-conversion.pt`, `best-lm.pt`, `epoch-N.pt`, and `last.pt` preserve other selections and recovery state. If no reranking checkpoint qualifies, that role remains absent.

Fifteen CPU tests cover encoding/masks, source-weight equality and fresh AdamW state, microbatch gradient equivalence, prefetched-stream recovery, optimizer/RNG continuation, stopping, reading provenance, and scoring/fallback contracts. A constructed-data training-loop test recovers from failure during complete validation to bitwise-identical final weights. These verify program behavior; Linux compilation, sustained GPU throughput, and task quality await formal execution.

The handoff archive and SHA256 inventory under `outputs/handoff-v4-c1/` contain source/configuration and an explicit payload list. Required payloads are frozen conversion binaries, manifests/acceptance, LM tokens/windows, tokenizer, source checkpoint, and allowlisted nonblind snapshots. Existing remote files can be reused after hash matching. Raw corpus text and annotation dictionaries are not GPU training dependencies.

## Local capacity measurement

A BF16 eager forward/backward check on RTX 5060 Laptop 8 GB used microbatch 32, length 128, both task masks, and two reserved FP32 AdamW moment buffers. Peak allocated/reserved memory was 1,150.78/1,340 MiB, with zero optimizer updates. This supports further local feasibility work but does not measure full-trainer or compiled memory and speed. Raw record: `outputs/posttraining-v4-c1/local-memory.json`. The prepared formal configuration targets RTX 4090.

## Conditional C2 ranking stage

After positive C1 results, a separate experiment may generate real AzooKey lists from training context/reading with frozen engine/dictionary versions. Pool misses contribute no ranking term. Ambiguous valid spellings require multiple positives or quarantine. The listwise objective uses the same inference OUT-sum scores:

```text
Lrank = logsumexp(all scores) - logsumexp(positive scores)
```

LM and conversion objectives remain active. Ranking weight and budget require a separate frozen configuration after candidate counts and C1 results are available. If C1 fails, diagnose reading quality, span distribution, overflow, LM forgetting, and scoring before choosing another experiment.
