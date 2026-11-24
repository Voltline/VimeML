# Historical V1 deployment design

Scope: pre-measurement design, 2026-10-06. This document preserves initial assumptions rather than current measured deployment behavior. Current results appear in [Core ML deployment](../coreml.md) and [V1 measurements](../reference/coreml-v1.md).

The design retains AzooKey dictionary retrieval and adds a small Japanese LM for reranking using within-sentence left context. Tokenization, stable ranking, bounded generation, and overflow fallback remain application-side. A Core ML wrapper outputs raw logits for context lengths up to 128.

Initial compression options include FP16, linear quantization, and 4-bit palettization. The theoretical 4-bit weight payload for 7.39M parameters is approximately 3.69 MB; this is not a package-size or runtime-memory measurement. Subsequent measurements select INT8 block-32 weight storage with FP32 CPU computation. The tested 4-bit palette performs poorly and GPU/ALL execution encounters assertions.

Conversion correctness, probability distribution changes, same-pool IME results, and actual device memory are separate criteria. Host-app memory and simulator tests are not substitutes for real keyboard-extension measurements. Historical planned experiments do not imply unsupported production guarantees.
