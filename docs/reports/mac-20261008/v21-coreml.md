# V2.1 Core ML conversion and packaging — 2026-10-08

The deployment source is V2.1 extension best step 40,000, 12,537,920 parameters, V2 16K tokenizer, and context 128. The inference transfer retains matching FP32 tensors and omits optimizer state. Conversion uses Python 3.11, PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5, and scikit-learn 1.5.1 on macOS.

## Selected package

The package compresses 32 learned weight matrices with symmetric INT8 block size 32, uses FP32 computation and CPU-only execution, and targets iOS 18 or later. Input is `int32[1,T]` for T=1–128; output is raw `float32[1,T,16384]` logits. There is no model-side softmax, tokenization, candidate search, or KV cache.

Uncompressed size is 50.36 MB; INT8 is 14.33 MB, a 71.5% reduction. Matching tokenizer, manifest, and client fixtures accompany the package. The two source repositories merge code independently; ignored deployment artifacts travel in the result ZIP.

## Numerical and task evidence

Uncompressed maximum logits error is 0.0000457764 and passes the recorded tolerance. INT8 maximum error is 1.00949144 and fails strict alignment. At T=128, mean absolute error is 0.108517 and RMSE 0.139438. Padding/causal-prefix diagnostic deltas are zero.

| Fixed task | FP32 | INT8 |
| --- | ---: | ---: |
| AJIMEE / 200 | 144 | 144 |
| Historical development / 137 | 122 | 122 |
| Draft development / 2,000 | 1,487 | 1,481 |

Quantization corrects three and regresses three AJIMEE cases, while the expanded draft has eight corrections and fourteen regressions, paired p=0.286279. Identical small-set counts do not imply identical case decisions or full rankings. Fixed next-token fixtures match in 12/12 cases; 32-token greedy generations match in 8/12.

Strict failure evidence remains preserved. The selected compression is an accepted deployment tradeoff under measured conditions, not FP32 equivalence. [Quantization review](v21-quantization-review.md) and [device measurements](v21-iphone.md) retain remaining limitations. Original conversion reports and returned assets remain in ignored artifact/output directories.
