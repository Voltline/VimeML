# Core ML deployment

The current deployment asset is V2.1 `extend5` step 40,000, symmetric INT8 block-32 weights, FP32 computation, and `CPU_ONLY`, with minimum iOS 18. Model package size is 14.33 MB. V1 remains a fallback and comparison release.

## Runtime contract

| Property | Contract |
| --- | --- |
| Input | `int32[1,T]` token IDs, 1 ≤ T ≤ 128 |
| Output | Raw `float32[1,T,16384]` logits |
| Computation | FP32; compressed weight storage does not imply integer execution |
| Tokenization | Matching SentencePiece model, application-side |
| Candidate ranking | Joint context/candidate encoding, suffix logP sum, stable ties |
| Generation | Application-side bounded search; no KV cache |
| Overflow | Explicit fallback; no silent context truncation |

The V2 tokenizer is incompatible with V1 token IDs. Model, tokenizer, manifest, and client fixtures are updated as a unit. Dictionary retrieval remains necessary. The separate Vime repository maintains current client implementation; [iOS examples](../examples/ios/README.md) are historical references.

## Conversion and packaging

macOS is required for Core ML runtime conversion/evaluation. The recorded environment uses Python 3.11, PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5, and scikit-learn 1.5.1. Platform pins reside in `requirements.txt`.

`scripts/deployment/coreml_v2.py` provides V2 conversion/evaluation; `package_v2_ios.py` packages iOS assets. V1 tooling remains separate. `prepare_mac.py` creates inference-input transfers, and `package_mac_results.py` creates result archives. Existing release artifacts retain original manifests; later tool edits do not rewrite historical source identities. [Cross-platform exchange](reference/artifact-exchange.md) describes Git and artifact boundaries.

## Measurements and limitations

Uncompressed V2.1 conversion passes the original tolerance, with maximum logits error 0.0000457764. INT8 fails strict alignment, with maximum error 1.00949144. AJIMEE and historical-development hit counts remain unchanged; expanded draft Top-1 decreases from 1,487 to 1,481. These measurements support an accepted deployment tradeoff, not numerical equivalence or an intrinsic architecture explanation.

The recorded iPhone 16 Pro Max session reports LM reranking mean 32.93 ms / p95 53.92 ms and next-word inference mean 29.05 ms / p95 38.11 ms. Extension physical-footprint sample peaks range from 28.00 to 28.30 MiB depending on sampling source; kernel lifetime peak is 34.72 MiB. UI publication includes an unexplained approximately 1.93-second maximum. Short-session experience was acceptable, but long-term memory plateau and the delay mechanism remain unestablished.

GPU/ALL execution previously triggered quantized-gather assertions; CPU/ANE experiments were not faster than CPU under the recorded V1 workloads. Power was not measured. [V2.1 conversion](reports/mac-20261008/v21-coreml.md), [quantization review](reports/mac-20261008/v21-quantization-review.md), and [device report](reports/mac-20261008/v21-iphone.md) preserve detailed boundaries and failures.
