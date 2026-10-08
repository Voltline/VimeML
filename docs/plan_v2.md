# V2 experiment design

Status: completed, 2026-10-08. V2 changes model capacity, tokenizer boundaries, and training duration while retaining the frozen V1/V2 sentence corpus. Subsequent V2.1 runs improve runtime throughput and extend optimization. The deployed result is V2.1 `extend5` step 40,000 / INT8 block-32.

## Experimental variables

| Component | V1 | V2-family change |
| --- | --- | --- |
| Parameters | 7.39M | 12.54M |
| Backbone | 4 × 256, LayerNorm/GELU | 6 × 320, RMSNorm/SwiGLU, five heads |
| Tokenizer | 16K Unigram V1 | 16K Unigram V2, dummy prefix disabled |
| Context | 128 | 128 |
| Data | Frozen sentence corpus | Same frozen corpus |
| Training crop | None | 30% eligible-first-window prefix crop |
| Optimization | One epoch | Four initial epochs, two restart epochs, extension experiment |

Joint tokenization remains necessary despite the dummy-prefix change. Context expansion, reading augmentation, dictionary replacement, and training on evaluation labels are outside this experiment.

## Completed stages

1. Tokenizer preparation and full-split round-trip measurement.
2. Architecture implementation and training-entry integration.
3. V2 token-store encoding and window indexing.
4. Deterministic prefix-crop integration.
5. AutoDL training and same-pool FP32 evaluation.
6. Batch/compilation throughput comparison and V2.1 continuation.
7. Core ML conversion, quantization analysis, and device measurements.

Configurations reside in `configs/`; [dated reports](index.md#experiment-records) retain measurements and artifact paths. Experiments reuse frozen inputs and completed score caches rather than repeatedly rebuilding or evaluating identical artifacts.

## Outcome

Validation BPC improves from V1 3.4736559 to V2.1 3.0802686. AJIMEE Top-1 improves from 124/200 to 144/200; historical development remains 122/137. The 2,000-case draft score improves from 1,453 to 1,487 before quantization and is 1,481 after quantization. These datasets have different label quality and exposure; they do not constitute a single independent generalization test.

The [V2 summary](reports/v2-summary.md) records the deployment decision and accepted numerical/device limitations.
