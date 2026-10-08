# V2.1 restart design — 2026-10-07

The completed `restart1` experiment initializes model weights from frozen V2.0 best step 375,000 with a new AdamW optimizer. Data, tokenizer, architecture, and prefix crop remain fixed. Two epochs use data offset 4, batch 512, BF16, learning rate 3e-4 → 3e-5, warmup 500, and 98,426 updates.

The runtime compiles the hidden stack while preserving eager vocabulary/loss operations. Larger batch and compilation target observed underutilization; changed update frequency and gradient statistics remain optimization variables rather than neutral runtime changes.

Configuration: [train-v21-restart.toml](../../../configs/train-v21-restart.toml). [Real-data performance](v21-performance.md) records the throughput comparison. [Interrupted predecessor](v21-recovery.md) contains step-zero archives that are not restart1 progress.

The completed restart uses detached screen and optional online tracking; private run metadata remains local. Model/evaluation directories are separate from V1/V2.0. [Final evaluation](v21-evaluation.md) determines quality, not synthetic timing or training loss alone.
