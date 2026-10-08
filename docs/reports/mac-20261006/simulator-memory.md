# V1 simulator and historical memory audit — 2026-10-06

This record covers bounded simulator stress workloads and offline historical-trace analysis. The separate [real extension record](iphone-keyboard-memory.md) is not replaced by simulator measurements.

## Method and bounded scenarios

An optimized Release CPU-only arm64 iOS 27.0 simulator uses a new host process per scenario, 10-ms `task_vm_info.phys_footprint` sampling, and the kernel process-lifetime footprint ledger peak. Values are MiB and include UIKit/XCTest/framework overhead rather than pure model memory.

Each scenario runs warmup plus 80 input/selection/commit rounds and 100 rapid input/backspace/cancel operations. The worker remains alive at the post-round measurement.

| Scenario | Initial MiB | After 80 rounds | Kernel peak | Round-20/40/60/80 range |
| --- | ---: | ---: | ---: | ---: |
| Engine ranking, suggestions off | 36.21 | 46.52 | 48.24 | 0.47 |
| LM ranking, suggestions off | 37.63 | 49.24 | 54.02 | 0.25 |
| LM ranking + suggestions, successful retry | 37.25 | 48.89 | 52.52 | 0.16 |

Different process baselines/framework caches prevent interpreting scenario subtraction as precise model cost. Session/worker release followed by one-second waiting yields approximately 43.19/44.00/43.66 MiB. Suggestions appear within the 100-ms observation window in 69/81 rounds; other rounds are not automatically counted as successful generation.

The first LM-plus-suggestions run times out at round 24 under the unchanged five-second limit, with kernel peak 62.77 MiB. Diagnostic overhead can contribute to the failed-run peak. A retry passes, but the timeout root cause remains unresolved; failure evidence is retained.

## Isolated model workloads

Loading moves footprint 33.91 → 37.74 MiB, with process peak 47.71. One hundred predictions at T=8/64/128 finish at 39.97/42.10/43.58 MiB. Candidate scoring increases only 0.016 MiB from iteration 20 to 100; next-word calls increase 0.047 MiB. Twenty reload/predict/release cycles end at 36.58 MiB initially and 37.25 MiB finally, with process peak 53.35.

These bounded workloads show no observed linear physical-memory accumulation after warmup. They do not exclude all leaks or certify iPhone extension budgets. Heap-object/Leaks certification is not performed.

## Historical trace and cache semantics

An older approximately 150-second extension trace has 146 samples, footprint 24.00 → 29.95 MiB, and sampled peak 30.36. It lacks complete switch/load/build/model evidence, so it is not identified as a fully documented LM-plus-suggestions measurement.

Ranking and suggestions switches are independent. Engine ranking can coexist with suggestions that load the model. The worker retains a cached model after first load; UI switch changes do not establish immediate memory reclamation. Character-based left-context limits do not guarantee token bounds; explicit token-length fallback remains necessary.

Original scenario JSON, failed/retry records, traces, and summaries reside in `outputs/deployment/memory-audit-v2/`. `scripts/deployment/summarize_memory.py` reconstructs summaries from existing reports without rerunning stress workloads. Real-device identity, cold/continuous use, request latency, cancellation, and system termination require separate recorded evidence.
