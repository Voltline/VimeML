# V1 Core ML measurements

V1 deployment uses symmetric INT8 block-32 weights, FP32 computation, `CPU_ONLY`, and minimum iOS 18. The 7,386,624-parameter model retains a 128-token context. The conservative uncompressed package is 16.57 MB; INT8 is 8.08 MB.

## Numerical and task behavior

Strict INT8 logits alignment fails, with maximum absolute error approximately 3.66. Distribution diagnostics average positions within each example, then examples; they are not corpus-token-weighted. Core ML INT8 KL is approximately 0.0059, similar to simulated INT8 0.0060. Aggregate similarity does not establish a correct implementation at every position.

| Compression diagnostic | KL | NLL | Top-1 token agreement |
| --- | ---: | ---: | ---: |
| FP16 | 0.0003 | 5.568 | 98.7% |
| INT8 | 0.0059 | 5.573 | 94.4% |
| Tested 4-bit palette | 1.316 | 6.944 | Approximately 33% |

The tested palette package is 3.94 MB and performs poorly; this result does not generalize to all 4-bit methods. Inputs use BOS plus at most 127 content tokens for these diagnostics.

INT8 historical development is 122/137 with Top-5 134; AJIMEE is 124/200 with Top-5 151. Full rankings change in 119 development cases relative to Windows FP32 and 163 AJIMEE cases. Two AJIMEE first choices change between accepted spellings. Maximum suffix-sum differences are 1.985330 and 2.692832. Unchanged hit counts do not imply unchanged rankings or numerical equivalence. Sixteen of twenty completion first choices match the conservative Core ML package, a separate comparison.

## Runtime and memory

Recorded Mac CPU reranking median/p95 is 18.2/21.9 ms; beam generation is 53.8/57.3 ms. Warm-cache loading is approximately 95 ms. Recorded iPhone host measurements include loading 80 ms, reloading 16 ms, inference T=16/64/128 at 0.67/1.8/3 ms, reranking 10.4/12.7 ms, next-word 5.8 ms, and beam 44.5 ms. Host loading adds approximately 24.2 MB with a 74.8 MB peak; these are not keyboard-extension measurements.

The separate real-extension trace covers approximately 152.7 seconds, with sampled footprint peak 21.25 MiB and private RSS peak 51.922 MiB. Installed build/model identity and approximately one-second sampling limit the conclusion. No long-term plateau is established.

The CPU plan places all 119 operations on CPU. GPU/ALL paths triggered quantized-gather assertions. Recorded CPU/ANE prototypes using length buckets and alternate outputs run approximately 3–4 times slower than CPU. These observations are workload-specific; power is unmeasured. [Simulator audit](../reports/mac-20261006/simulator-memory.md) and [extension trace](../reports/mac-20261006/iphone-keyboard-memory.md) preserve separate evidence.
