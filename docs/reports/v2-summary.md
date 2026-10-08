# V2-family experiment summary

Completed on 2026-10-08. The deployment release is V2.1 `extend5` step 40,000, INT8 block-32, FP32 computation, and CPU-only Core ML. V1 remains a baseline/fallback. The release has 12.54M parameters, a 16K tokenizer, context 128, and a 14.33 MB Core ML package.

## Fixed evaluation results

| Model | Old validation BPC ↓ | AJIMEE / 200 | Historical dev / 137 | Draft dev / 2,000 |
| --- | ---: | ---: | ---: | ---: |
| V1 FP32 | 3.4736559 | 124 | 122 | 1,453 |
| V2.0 FP32 | 3.2598221 | 125 | 124 | 1,463 |
| V2.1 restart1 FP32 | 3.0811788 | 137 | 121 | 1,476 |
| V2.1 extension release FP32 | 3.0802686 | 144 | 122 | 1,487 |
| V2.1 extension INT8 | Not measured | 144 | 122 | 1,481 |

The release reduces old validation BPC by approximately 11.32% relative to V1. Draft development gains 34 cases over V1, with exploratory paired p=0.00648; labels and repeated development inspection limit the inference. The final extension step is not the selected release checkpoint.

## Runtime and deployment outcome

Batch 512 plus compiled hidden-stack execution improves full-run training throughput from V2.0 approximately 121k to restart1 approximately 273k effective tokens/s on the recorded RTX 4090 D. Different batch size, update count, and optimizer state also change optimization, so throughput improvements are not isolated task-quality evidence.

V2.1 uncompressed conversion passes strict numerical alignment; INT8 fails it. Small-set hit counts remain unchanged while the expanded draft loses six cases. The quantized release is an accepted measured tradeoff, not numerically equivalent to FP32. The failure is not established as an intrinsic consequence of small size or architecture.

Recorded real-keyboard measurements show reranking mean 32.93 ms/p95 53.92 ms, a 28.00–28.30 MiB sampled footprint peak, and 34.72 MiB kernel lifetime peak. An approximately 1.93-second publication maximum and increasing late-session footprint remain unresolved. Limited-session input experience is acceptable; long-term memory/power guarantees are not established.

## Evidence and limitations

[Training](../training-v2.md), [extension selection](v2-20261007/v21-extend.md), [quantization analysis](mac-20261008/v21-quantization-review.md), and [iPhone report](mac-20261008/v21-iphone.md) retain specific conditions and failures. Original score/trace assets remain local and immutable. Historical development labels are AI-reviewed or drafts, not native-speaker gold. The old 1,000-case blind set remains unscored.

Later [V3 experiments](../plan_v3.md) do not establish a stable replacement advantage. Public release links and architecture details appear in the [model card](../../MODEL_CARD.md).
