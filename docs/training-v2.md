# V2-family training

V2, V2.1, and the V3 experiments use the 12,537,920-parameter architecture in the [model card](../MODEL_CARD.md). V1 code and frozen artifacts remain separate. `model_factory.py` selects the architecture; legacy configurations without an architecture field retain V1 behavior.

## Tokenization and input windows

V2 uses a 16K Unigram tokenizer with `add_dummy_prefix=false`, trained on two million train sentences. V1 token IDs are incompatible. Full context/candidate strings are jointly encoded at inference because boundary retokenization still occurs.

The token store is `artifacts/token-data/corpus-v2-16k/`; the context-128 window index is `artifacts/training-data/corpus-v2-c128/`. Training has 25,196,843 windows and 556,462,657 uncropped prediction pairs per epoch. Labels exclude padding but include EOS. A deterministic 30% crop affects eligible first windows only, retaining at least eight targets; validation and test remain uncropped.

## Completed schedules

| Experiment | Initialization | Epoch budget | Batch | Learning rate | Warmup | Configuration |
| --- | --- | ---: | ---: | --- | ---: | --- |
| V2.0 | Random | 4 | 256 | 1e-3 → 1e-4 | 1,000 | [train-v2.toml](../configs/train-v2.toml) |
| V2.1 restart1 | V2.0 step 375,000 weights; new AdamW | 2 | 512 | 3e-4 → 3e-5 | 500 | [train-v21-restart.toml](../configs/train-v21-restart.toml) |
| V2.1 extend5 | restart1 step 98,426, including AdamW state | Up to 5; 3.273 completed | 512 | 3e-5 plateau, then cosine → 1e-5 | 0 | [train-v21-extend.toml](../configs/train-v21-extend.toml) |

Runs use BF16, AdamW betas 0.9/0.95, weight decay 0.1, and gradient clipping 1. V2.1 compiles the hidden stack while the vocabulary head/loss remain eager. The final release is extension step 40,000, not the last update. [Extension results](reports/v2-20261007/v21-extend.md) distinguish subset-selected and full-epoch checkpoints.

## Runtime and measurement

Linux training uses the recorded CUDA environment in `requirements-autodl.txt`, with large assets under `/root/autodl-tmp/vimeml`. Detached `screen` sessions preserve jobs across SSH disconnects. Launch wrappers reside in `scripts/training/`.

Progress and completion state are recorded in `progress.json` and `summary.json`. Fixed-subset validation runs every 5,000 updates; full BPC and IME evaluations run at epoch boundaries. Checkpoint state preserves committed batch position. The recorded degradation rule stops after two full epochs more than 0.01 BPC above the historical best; the extension was interrupted before its maximum budget.

W&B online tracking is optional through `[tracking]`; external environment credentials remain outside source/transfer packages. Run URLs and account records remain in ignored local tracking outputs. Throughput excludes validation, saving, and compilation according to each report; short benchmark measurements are not full-run measurements.

The V3 [adaptation](plan_v3.md) and [joint retraining](reports/v3-20261008/b-plan.md) use separate frozen mixtures and fixed `epoch_mean_tokens` loss scaling.
