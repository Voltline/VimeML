# Core ML deployment

The current deployment asset is V4.2 epoch 20, symmetric INT8 block-32 weights, FP32 computation, and `CPU_ONLY`, with minimum iOS 18. Model package size is 17.16 MB. V4.2 has validated FP32/INT8 stateless, single-token KV and chunked KV packages. The previous deployment was V2.1 `extend5` step 40,000 with the same recipe (14.33 MB). V1 remains a fallback and comparison release.

## Runtime contract

| Property | Contract |
| --- | --- |
| Input | `int32[1,T]` token IDs, 1 ≤ T ≤ 128 |
| Output | Raw `float32[1,T,16384]` logits (V2.1); `float32[1,T,8192]` (V4.2) |
| Computation | FP32; compressed weight storage does not imply integer execution |
| Tokenization | Matching SentencePiece model, application-side |
| Candidate ranking | Joint context/candidate encoding, suffix logP sum, stable ties |
| Generation | Application-side bounded search; the client uses the stateless model |
| KV cache (research) | Caller-owned FP32 snapshots `(layers,1,heads,128,64)`; V2.1 and V4.2 chunk graphs accept `int32[1,T]` at a given position |
| Overflow | Explicit fallback; no silent context truncation |

The V2 tokenizer is incompatible with V1 token IDs, and the V4 8K tokenizer is incompatible with both; V4 IDs 4–6 are reserved markers that clients must reject in input and exclude from generation. Model, tokenizer, manifest, and client fixtures are updated as a unit. Dictionary retrieval remains necessary. The separate Vime repository maintains current client implementation; [iOS examples](../examples/ios/README.md) are historical references.

## Conversion and packaging

macOS is required for Core ML runtime conversion/evaluation. The recorded environment uses Python 3.11, PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5, and scikit-learn 1.5.1. Platform pins reside in `requirements.txt`.

`scripts/deployment/coreml_v2.py` provides V2 conversion/evaluation; `coreml_v4.py` provides the separate V4 stateless (`float32[1,T,8192]`), single-token KV and chunked KV (`vimeml.deployment.v4_chunk`) paths, `measure_v42.py` same-harness Mac measurements, and `package_v4_ios.py` V4 client resources ([V4.2 report](reports/mac-20261020/v42-coreml.md)); `package_v2_ios.py` packages iOS assets. V1 tooling remains separate. `prepare_mac.py` creates inference-input transfers, and `package_mac_results.py` creates result archives. Existing release artifacts retain original manifests; later tool edits do not rewrite historical source identities. [Cross-platform exchange](reference/artifact-exchange.md) describes Git and artifact boundaries.

## Measurements and limitations

Uncompressed V2.1 conversion passes the original tolerance, with maximum logits error 0.0000457764. INT8 fails strict alignment, with maximum error 1.00949144. AJIMEE and historical-development hit counts remain unchanged; expanded draft Top-1 decreases from 1,487 to 1,481. These measurements support an accepted deployment tradeoff, not numerical equivalence or an intrinsic architecture explanation.

The recorded iPhone 16 Pro Max session reports LM reranking mean 32.93 ms / p95 53.92 ms and next-word inference mean 29.05 ms / p95 38.11 ms. Extension physical-footprint sample peaks range from 28.00 to 28.30 MiB depending on sampling source; kernel lifetime peak is 34.72 MiB. UI publication includes an unexplained approximately 1.93-second maximum. Short-session experience was acceptable, but long-term memory plateau and the delay mechanism remain unestablished.

### V4.2

Same recipe and harness as V2.1. FP32 stateless, single-token KV and chunked KV conversions match PyTorch within 5.2e-5 and reproduce all four fixed-pool Top-1 counts and orders. INT8 fails strict alignment (maximum 0.477, mean 0.0495 at T=128, about half the V2.1 INT8 error) and changes Top-1 by at most one case per track.

| Measurement | V2.1 INT8 | V4.2 INT8 |
| --- | ---: | ---: |
| Package | 14.33 MB | 17.16 MB |
| Top-1, Standard / Legacy / AJIMEE / Historical (Mac Core ML) | 506 / 375 / 144 / 122 | 522 / 371 / 139 / 123 |
| Mac warm stateless call, T=1 / T=128 p50 | 0.99 / 5.68 ms | 1.30 / 6.07 ms |
| Mac KV prefill P=119, one chunk call | 6.1 ms | 6.6 ms |
| iPhone 16 Pro Max test host, `predict` T=128 p50 | 4.55 ms | 4.90 ms |
| iPhone test host, rerank p50 / p95 | 21.3 / 136.8 ms | 25.1 / 120.9 ms |
| iPhone test host, footprint peak | 49.95 MB | 44.35 MB |

The single-token V4 KV graph needs one call per token (170 ms prefill at P=119); the chunked graph removes this. Device runs used a warm, charging phone (thermal state `fair`) in the XCTest host, not the keyboard extension. [V4.2 Core ML report](reports/mac-20261020/v42-coreml.md).

### Earlier observations

GPU/ALL execution previously triggered quantized-gather assertions; CPU/ANE experiments were not faster than CPU under the recorded V1 workloads. Power was not measured. [V2.1 conversion](reports/mac-20261008/v21-coreml.md), [quantization review](reports/mac-20261008/v21-quantization-review.md), and [device report](reports/mac-20261008/v21-iphone.md) preserve detailed boundaries and failures.
