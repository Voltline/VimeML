# V4 base language-model training

Tokenizer, full train/validation token data, context128 indices, and the fixed 15,070,848-parameter model are prepared. The completed basic checks are recorded in [model acceptance](reports/v4-20261014/model.md). [Chinese guide](zh-CN/training-v4.md).

V4 uses vocabulary8192 to move capacity from embedding to the Transformer backbone. There is one training route: start the full base-LM run directly. No preliminary training run or Apple test is required by this local workflow.

## Start and resume

Start on 2026-10-15 morning from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 -u scripts/training/train_v4.py
```

The default is [train-v4-b.toml](../configs/train-v4-b.toml). An explicit equivalent:

```powershell
.venv/Scripts/python.exe -X utf8 -u scripts/training/train_v4.py --config configs/train-v4-b.toml
```

Resume an interrupted run with its unchanged configuration:

```powershell
.venv/Scripts/python.exe -X utf8 -u scripts/training/train_v4.py --resume
```

No formal corpus training was started during preparation. These are manual launch commands, not a scheduled job.

## Training settings

| Setting | Value |
| --- | --- |
| Initialization | Fresh random weights |
| Data | Frozen full train; independent validation |
| Vocabulary / context | 8192 / 128 |
| Model | d384, 7 layers, 6 heads, SwiGLU960 |
| Precision | CUDA BF16 |
| Batch / accumulation | 128 / 1 |
| Budget | 2 epochs, 393,880 updates |
| AdamW | betas(0.9,0.95), weight decay0.1 |
| Learning rate | 6e-4 → 6e-5, cosine |
| LR warmup | First1000 training updates |
| Gradient clipping | 1.0 |
| Prefix crop | Baseline probability0.30, minimum8 remaining tokens |
| DataLoader | 4 workers, length buckets |
| Validation | Subset every2000 updates; full at each epoch and completion |
| Checkpoints | Every5000 updates and epoch/completion/interruption |
| Compilation | Disabled |
| Logs | Local TensorBoard |

Actual target-token counts depend on sentence lengths and prefix cropping. Validation BPC includes EOS and uses original Unicode character counts. Different tokenizer versions are compared primarily by BPC. The benchmark table is empty; this run does not consume blind/reference sets or perform automatic IME model selection. Training hyperparameters are initial experimental settings, not measured optima.

## Files

| Path | Role |
| --- | --- |
| `artifacts/tokenizers/v4-a/ja-unigram-8192/` | Frozen tokenizer |
| `artifacts/token-data/corpus-v4-8k/` | Full train/validation tokens |
| `artifacts/training-data/corpus-v4-c128/` | Window indices |
| `artifacts/models/tiny-ja-v4-b-e8k-d384-l7/` | Weights, optimizer/RNG checkpoints, progress and validation results |
| `runs/training/tiny-ja-v4-b-e8k-d384-l7/` | TensorBoard events |

Model/data signatures and configuration are recorded in checkpoints. V2 token IDs/checkpoints and synthetic test fixtures are not training initializers. Apple export, quantization and device deployment will be handled in a later phase.
