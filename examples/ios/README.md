# iOS integration references

`CoreMLProbe.swift` is a manual model-loading, prediction, and measurement probe. `VimeSentencePiece/` provides SentencePiece 0.2.1 with a C bridge, preserving Apache-2.0 and third-party notices, with minimum iOS 18. `integration/` contains the historical first INT8 candidate-reranking Swift implementation and fixtures.

The historical integration predates later sentence-context, next-word, settings, cancellation, and memory-audit changes. Current V2.1 client implementation is maintained in the separate Vime repository. Local client/deployment snapshots preserve their original base and do not replace current client source.

The selected INT8 model uses FP32 CPU-only computation. GPU/ALL paths previously triggered quantized-gather assertions. Matching model/tokenizer/manifest assets are required. SentencePiece, joint-string scoring, stable ordering, and bounded search remain application-side; the Core ML model emits raw logits only.

[Core ML documentation](../../docs/coreml.md) records the runtime contract and measurement limitations. [Upstream attribution](VimeSentencePiece/UPSTREAM.md) preserves vendored provenance.
