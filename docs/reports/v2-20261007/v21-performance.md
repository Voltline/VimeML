# V2.1 real-data throughput comparison — 2026-10-07

Recorded on the original RTX 4090 D using the same frozen V2.0 initialization, real token store, epoch-4 shuffle/crop, AdamW, and BF16. Each condition has 32 warmup and 128 measured updates; measured time includes loader, transfer, update, and synchronization.

| Execution | Batch | Effective tokens/s | Step p50 / p95, ms | Loader wait | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 256 | 112,738 | 44.96 / 56.34 | 2.2% | 6.15 |
| Compiled | 256 | 152,335 | 33.72 / 39.69 | 3.0% | 5.22 |
| Eager | 512 | 215,837 | 43.68 / 64.70 | 3.1% | 12.12 |
| Compiled | 512 | 278,818 | 33.86 / 51.13 | 4.2% | 10.26 |

Measured intervals last only approximately 4–6 seconds, include recompilation, and reuse existing disk caches. They do not establish whole-epoch thermal behavior. Snapshot GPU utilization cannot be attributed as a condition-level average. Four loader workers have low wait fractions; no worker-count increase is adopted.

The formal batch-512 compiled run completes 98,426 updates and 998,854,924 effective tokens in 3,739.8 seconds. Update-only throughput is 273,368 tokens/s over 3,650.2 seconds, excluding validation, saving, and the initial 100 updates. This is approximately 2.26× V2.0 throughput under the reported definitions.

The 64-batch fixed validation subset changes size with batch size, so initial subset losses are not directly comparable to the old subset; full-corpus BPC is unchanged in definition. Tooling is `scripts/training/benchmark_v21.py`; original evidence resides in `outputs/model-checks/v21-real-data-performance-restart1/`. [Restart evaluation](v21-evaluation.md) records the completed quality comparison.
