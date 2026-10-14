# V4 base language-model training design

V4 targets AutoDL Linux with an RTX 4090 D (24 GiB). The tokenizer, full train/validation token data, context128 indices, and fixed 15,070,848-parameter model are prepared. Completed local checks are recorded in [model acceptance](reports/v4-20261014/model.md). [Chinese document](zh-CN/training-v4.md).

Vocabulary8192 moves parameter capacity from embedding to the Transformer backbone. The experiment is full base-LM training from fresh random weights. [train-v4-b.toml](../configs/train-v4-b.toml) describes the configuration.

## Training settings

| Setting | Value |
| --- | --- |
| Target environment | AutoDL Linux, RTX 4090 D 24 GiB; existing pinned CUDA12.8/PyTorch2.7 environment |
| Initialization | Fresh random weights |
| Data | Frozen full train; independent validation |
| Vocabulary / context | 8192 / 128 |
| Model | d384, 7 layers, 6 heads, SwiGLU960 |
| Precision | CUDA BF16; FP32 parameter/optimizer state |
| Batch / accumulation | 512 / 1; at most65,536 padded input positions per update |
| Budget | 4 epochs, 49,235 updates per epoch, 196,940 total |
| AdamW | CUDA fused; betas(0.9,0.95), weight decay0.1 |
| Learning rate | 6e-4 → 6e-5, cosine |
| LR warmup | First1000 training updates |
| Gradient clipping | 1.0 |
| Prefix crop | Baseline probability0.30, minimum8 remaining tokens |
| DataLoader | 4 workers, 4 CPU threads, length buckets, pinned memory, persistent workers, prefetch2 |
| Validation | Fixed4096 sampled windows, expanded to complete sentences, every2000 updates; full each epoch and completion |
| Checkpoints | Every5000 updates and epoch/completion/interruption |
| Compilation | Dynamic-shape hidden stack; valid-token output head/loss remain eager |
| Logs | TensorBoard plus existing W&B synchronization |

The epoch budget follows 25,208,300 training windows and includes the final partial batch. Actual target-token counts depend on sentence lengths and prefix cropping; 65,536 is a padding ceiling, not an effective-token batch size. Validation BPC includes EOS and uses original Unicode character counts. Different tokenizer versions are compared primarily by BPC. The benchmark table is empty. Hyperparameters are initial experimental settings, not measured optima. The existing epoch-level degradation guard retains patience2 and minimum BPC delta0.01, so repeated deterioration can end training before the four-epoch ceiling; the best checkpoint is retained.

## Runtime efficiency

The implementation reuses the V2.1/V3 runtime: BF16 SDPA attention, fused AdamW, nonblocking transfers from pinned memory, and length buckets that reduce padding. Only valid target positions enter the vocabulary projection and cross entropy. The hidden stack uses `torch.compile(fullgraph=True, dynamic=True)`; the variable-size valid-token head remains outside compilation. Compilation preserves parameter names and checkpoint format. Prefix crops and epoch ordering retain their deterministic resume behavior.

Existing [V2.1 RTX4090 D measurements](reports/v2-20261007/v21-performance.md) recorded 215,837 effective tokens/s for eager batch512 and 278,818 for compiled batch512, with compiled peak allocated memory10.26 GiB. Those are V2.1 measurements. V4 has a larger backbone, a smaller vocabulary, and different token lengths; its throughput and peak memory on AutoDL have not been measured. Batch512 and four workers reuse the established configuration without claiming optimal GPU utilization.

The AutoDL entry follows previous launcher conventions, including persistent TorchInductor/Triton caches and the existing tracking environment. [requirements-autodl.txt](../requirements-autodl.txt) records the prior Linux runtime. Compilation occurs within normal training; no separate preliminary training phase is part of this configuration.

## Artifacts and identity

| Path | Role |
| --- | --- |
| `artifacts/tokenizers/v4-a/ja-unigram-8192/` | Frozen tokenizer |
| `artifacts/token-data/corpus-v4-8k/` | Full train/validation tokens |
| `artifacts/training-data/corpus-v4-c128/` | Window indices |
| `artifacts/models/tiny-ja-v4-b-e8k-d384-l7/` | Weights, optimizer/RNG checkpoints, progress and validation results |
| `runs/training/tiny-ja-v4-b-e8k-d384-l7/` | TensorBoard events |
| `artifacts/tracking/` | W&B synchronization records |

Prepared binary artifacts are local files excluded from Git. The training configuration depends on the tokenizer, token store, and indices together; rebuilding them on AutoDL is not part of training. Checkpoints record model/data/code signatures and configuration, and continuation requires the same identity and restores optimizer/RNG state and committed data cursors. V2 token IDs/checkpoints and synthetic fixtures are not training initializers. Training consumes only train and validation and does not access blind data. No formal V4 corpus training has started during preparation.
