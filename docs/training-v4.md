# V4 base language-model training

V4 uses an 8K vocabulary to reallocate embedding capacity to the Transformer backbone. The fixed 15,070,848-parameter architecture and tokenizer preparation are documented in [model acceptance](reports/v4-20261014/model.md) and [tokenization](reports/v4-20261014/tokenizer.md). Two base-LM experiments completed on RTX 4090 on 2026-10-15. [Chinese document](zh-CN/training-v4.md).

## Data and shared settings

Both experiments use the same original FineWeb/Tatoeba train/validation text as V2.1, with the new 8K tokenizer. The token store contains 25,208,300 training windows. At batch 512, each epoch has 49,235 updates including the final partial batch. Original validation contains 260,337 sentences and 11,110,489 Unicode characters; all 6,627,568 V4 prediction targets, including EOS, contribute to complete BPC. Different tokenizers are compared by BPC.

| Setting | Value |
| --- | --- |
| Environment | Linux, RTX 4090 24 GiB; pinned CUDA 12.8 / PyTorch 2.7 environment |
| Initialization | Fresh random weights, seed 42 |
| Vocabulary / context | 8,192 / 128 |
| Architecture | d384, 7 layers, 6 heads, SwiGLU 960; 15,070,848 parameters |
| Precision | BF16; FP32 parameters and optimizer state |
| Batch / accumulation | 512 / 1; at most 65,536 padded positions per update |
| AdamW | Fused CUDA; betas (0.9, 0.95), weight decay 0.1, gradient clipping 1.0 |
| Prefix crop | Probability 0.30 on eligible first windows; at least eight remaining targets |
| Data loading | Four workers, four CPU threads, length buckets, pinned memory, persistent workers, prefetch 2 |
| Compilation | Dynamic hidden stack; valid-token vocabulary head and loss remain eager |
| Validation | Fixed 4,096 sampled windows expanded to complete sentences every 2,000 updates; full each epoch and completion |
| Saving | Every 5,000 updates and at epoch/completion/interruption boundaries |
| Tracking | TensorBoard with optional W&B synchronization |

Effective target-token counts depend on sentence lengths and prefix cropping; padded positions are not effective batch tokens. The base-LM configurations omit in-training IME benchmarks. Frozen candidate-pool evaluations were performed separately after training.

## Completed schedules

| Experiment | Schedule | Actual completion | Configuration |
| --- | --- | --- | --- |
| V4-A | Initial four-epoch cosine, extended to eight and then twelve epochs | Plateau stop after epoch 9, step 443,115 | [Initial](../configs/train-v4-b.toml), [eight-epoch extension](../configs/train-v4-b-extended.toml), [twelve-epoch extension](../configs/train-v4-b-extended12.toml) |
| V4-B | Full twelve-epoch cosine selected at initialization | Epoch 12, step 590,820; no early stop | [Twelve-epoch comparison](../configs/train-v4-b12cosine.toml) |

Both schedules begin at 6e-4 with 1,000 warmup updates and a 6e-5 minimum. A's extensions start from the committed optimizer rates at steps 144,039 and 368,826 and smoothly decay over the remaining budget. They preserve weights, AdamW, RNG, data cursors, and existing stopping counters. Original checkpoints/configurations remain in `extension-from-four-epochs/` and `extension-from-eight-epochs/` within A's model directory. Checkpoint-bound receipts record permitted configuration/code transitions and source SHA256 identities; ordinary resume remains strict.

Plateau stopping requires a cumulative improvement greater than 0.01 full BPC from its stopping reference; two complete epochs below that improvement threshold stop training. Actual best checkpoints are retained even for smaller gains. A's resumed schedule is distinct from B's initial twelve-epoch design. B consumed 33.33% more effective targets, so their final comparison includes both schedule and realized training budget.

## Results and selection

| Experiment | Effective training targets | Full local FP32 BPC, LM best | Standard dev, LM best | Development selection |
| --- | ---: | ---: | ---: | --- |
| V4-A | 5,174,759,074 | 3.051781125, step 372,000 | 492/728 | Epoch 8, step 393,880; 503/728 |
| V4-B | 6,899,499,874 | 3.027777023, step 584,000 | 498/728 | LM best, step 584,000; 498/728 |
| V2.1 baseline | — | 3.080190570 | 505/728 | Deployed extend5 step 40,000 |

Training `best.pt` uses fixed-subset loss; `best-epoch.pt` uses full epoch BPC. Subsequent development selection compares these recorded candidates separately. Remote BF16 metrics remain distinct from local FP32 results. Both V4 runs improve LM BPC without establishing a stable reranking replacement advantage over V2.1. Complete methods, source results, paired statistics, and limitations are in the [A evaluation](reports/v4-20261015/evaluation.md) and [B comparison](reports/v4-20261015/b12cosine-evaluation.md).

## Runtime and artifacts

The runtime reuses V2.1/V3 BF16 SDPA, fused AdamW, nonblocking pinned-memory transfers, and length buckets. Only valid target positions enter vocabulary projection and cross entropy. `torch.compile(fullgraph=True, dynamic=True)` preserves parameter names and checkpoint format. Prefix crops and epoch ordering support deterministic recovery.

A recorded 441,043.7 effective tokens/s across 443,095 timed updates and 11,732.5 update seconds; B recorded 439,458.1 effective tokens/s. These measurements exclude validation, saving, and initial warmup timing. The [V2.1 RTX 4090 D results](reports/v2-20261007/v21-performance.md) used a different model/tokenizer and do not form a controlled cross-model throughput comparison. The pinned Linux dependencies remain in [requirements-cuda.txt](../requirements-cuda.txt); launcher wrappers retain persistent TorchInductor/Triton caches.

| Path | Role |
| --- | --- |
| `artifacts/tokenizers/v4-a/ja-unigram-8192/` | Frozen tokenizer |
| `artifacts/token-data/corpus-v4-8k/` | Full train/validation token store |
| `artifacts/training-data/corpus-v4-c128/` | Context-128 window indices |
| `artifacts/models/tiny-ja-v4-b-e8k-d384-l7/` | A checkpoints, progress, and validation records |
| `artifacts/models/tiny-ja-v4-b12cosine-e8k-d384-l7/` | Independent B checkpoints and records |
| `runs/training/` / `artifacts/tracking/` | TensorBoard / W&B records |

Binary artifacts are excluded from Git. Source, tokenizer, token/window manifests, precision, and configuration identities are checked for recovery. Training consumes train/validation only; blind data remains excluded.

## Conversion post-training

[V4-C1](posttraining-v4.md) uses B's step-584,000 weights with reading-conditioned conversion and ordinary-LM replay. C1 and the [C1r2 revision](zh-CN/reports/v4-20261016/c1r2.md) are complete. Their conversion-pool epochs are separate from the base-LM schedules above.

[V4.1 joint pretraining](training-v41.md) completed sixteen LM mixture epochs from new random weights using the base LM pool, V3 natural chat, and C1r2 conversion data. It is a separate experiment and preserves the A/B and C1 artifacts.
