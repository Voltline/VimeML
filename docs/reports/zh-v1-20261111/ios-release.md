# vimeml-tiny-zh-v1 Mac handoff check and iOS release (2026-11-11)

The Mac side completed Core ML conversion, INT8 quantization, a survey of quantization schemes and Mac measurements; the process and all numbers are in the [Mac report](../mac-20261111/zh-v1-coreml.md). This page does three things: checks the handoff package independently on Windows, analyzes the results, and records what was released to Hugging Face. This version is deployed to iOS.

Handoff package `vimeml-zh-v1-mac-results-20261111.zip`, SHA256 `0a97da02650bbbd9a4e32f4ec8f389047c4026b29666470649f9cc7af191b3bf`, 171 files. Only development was used; blind was not read again.

## Conclusions

- **iOS uses INT8 block32** (weight-only quantization, FP32 compute, CPU_ONLY, iOS 18 and later), in a stateless and a chunked-KV package. Against FP32, development full-pinyin Top-1 is +2/−1 and abbreviated +0/−2; Top-3 is unchanged on both tracks; the differences are within noise.
- **The handoff check passed**: file sizes, bundle hashes and weights match the training checkpoint tensor by tensor, and the Mac per-item results match the Windows per-item ranks (next section).
- **iPhone measurements added (see [iPhone and simulator](#iphone-and-simulator-measurements-2026-11-11))**: in the iPhone 16 Pro Max test host, stateless INT8 `predict` at T=128 has p50 6.82 ms against V4.2's 4.90 ms, 1.39×, matching the Mac ratio (the earlier proportional estimate was about 6.5 ms). Memory growth after loading and 50 predictions is 14.1 MiB, 2.8 MiB more than V4.2.
- **Keyboard-extension integration measured (see [keyboard extension](#keyboard-extension-integration-2026-11-11))**: in the extension the model reranks Rime's first 8 candidates. With the model on, keypress to tappable candidates averages 26.1 ms with p95 45.0 ms; with it off, 20.3 ms and 26.1 ms. 221 of 222 scorings finished within the 60 ms window, and the extension's memory peak is 29.9 MiB. Chinese prediction is not implemented yet, so it was not measured.
- **Training-data license**: LCCC's repository and Hugging Face data card say MIT, but the repository README's disclaimer says the dataset and pretrained dialogue models are "for research use only". The project owner decided to release publicly, and the model card and NOTICE state both terms as they are. See [License](#license).

## Independent check on Windows

| Check | Result |
| --- | --- |
| Byte sizes of the 171 handoff files against `handoff.json` | All match |
| SHA256 of the three inference bundle files against the bundle manifest | All match; manifest SHA256 `9d9dad19…`, the same as recorded in both Core ML packages and the iOS resource manifest |
| Bundle weights against the training checkpoint (`final.pt`, `44c423a1…`) | All 66 tensors equal; `lm_head` is tied to the embedding and not stored separately; 20,036,736 parameters |
| Top-1 / Top-3 recomputed from per-item results | Same as the Mac report for all 8 packages and both tracks |
| Mac per-item ranks against Windows CUDA FP32 (the decision's scoring) | FP32 ranks identical on every item of both tracks (1,456 + 990); INT8 differs on only 3 + 3 items, with ids matching the Mac report one by one |
| Environment information in the package | Only local Mac information, nothing about training infrastructure; 3 record files contain a Mac user directory path and are not in the release package |

## Analysis

### Effect of quantization on ranking

Reference is Core ML FP32 (Top-1 1,363 / 337, the same as Mac PyTorch and Windows CUDA).

| Package | Size | Full Top-1 (1,456) | Abbrev. Top-1 (990) | More right / more wrong vs FP32 | Verdict |
| --- | ---: | ---: | ---: | --- | --- |
| FP32 | 80.4 MB | 1,363 | 337 | — | Reference |
| **INT8 b32 (deployed)** | **22.8 MB** | 1,364 | 335 | Full +2/−1, abbrev. +0/−2 | No significant difference, Top-3 unchanged |
| INT8 b32 + FP16 embedding | 26.9 MB | 1,365 | 337 | +2/−0, +0/−0 | No significant difference, 18% larger |
| 6-bit palette | 15.7 MB | 1,364 | 339 | +7/−6, +4/−2 | Top-1 level, but many ranking changes |
| INT4 b32 | 12.8 MB | 1,349 | 332 | +10/−24, +8/−13 | Full pinyin significantly worse (p = 0.024) |
| 4-bit palette | 10.4 MB | 1,320 | 316 | +16/−59, +17/−38 | Significantly worse (p = 6e-7) |

- As usual INT8 fails the strict logits gate (max error 0.392; V4.2 0.477): expected drift from lossy quantization, not numerical equivalence. Chunked-KV INT8 and stateless INT8 differ by at most 4.2e-5 across 21 checks with identical rankings, so the two INT8 packages are interchangeable.
- The 6-bit palette matches FP32 on Top-1 but changes candidate order on 1,424 full and 983 abbreviated items, with the first candidate changing on 15 + 62 items, far more than INT8's 4 + 17. Top-1 alone understates its error; using it on iOS needs separate evaluation.
- As planned, the survey only records numbers and does not switch schemes. If the 18% size matters later, FP16 embedding is an option; for a smaller package, the 6-bit palette is worth checking on the prediction benchmark first.

### Mac performance (same machine as V4.2, interleaved)

| Metric | zh-v1 INT8 | V4.2 INT8 | Ratio |
| --- | ---: | ---: | ---: |
| Package size | 22.8 MB | 17.2 MB | 1.33× |
| Stateless `predict` p50, T=1 / T=128 | 1.65 / 8.49 ms | 1.29 / 6.43 ms | 1.28× / 1.32× |
| Chunked-KV prefill 119 tokens / single-step decode p50 | 9.33 / 2.71 ms | 7.18 / 2.05 ms | 1.30× / 1.32× |
| Model's own peak memory growth | 18.4 MiB | 13.4 MiB | 1.37× |

The ratios match the larger vocabulary (12,288 vs 8,192) and depth (9 vs 7 layers). INT8 only shrinks size, not CPU latency, since compute is still FP32.

### iPhone and simulator measurements (2026-11-11)

The Vime maintainer measured with `benchmark_lm_variants.py --variants zhV1,v42` in the Vime test host: iPhone 16 Pro Max (iOS 27.3) and the iOS simulator (iOS 27.2), 4 rounds each, alternating the two models within each round, each run a separate test process. All 16 runs passed, thermal state stayed nominal, Low Power Mode off. The table gives medians of the 4 iPhone rounds; `predict` is the stateless INT8 package, CPU_ONLY, p50 over 10 calls per length.

| Metric | zh-v1 INT8 | V4.2 INT8 | Ratio |
| --- | ---: | ---: | ---: |
| `predict` p50, T=1 | 1.70 ms | 1.25 ms | 1.35× |
| `predict` p50, T=8 / 16 / 32 | 2.25 / 2.37 / 3.31 ms | 1.68 / 1.78 / 2.45 ms | 1.34× / 1.33× / 1.35× |
| `predict` p50, T=64 / 128 | 4.56 / 6.82 ms | 3.30 / 4.90 ms | 1.38× / 1.39× |
| `predict` p95, T=128 | 6.97 ms | 5.04 ms | 1.38× |
| Load / reload | 195 / 7.4 ms | 155 / 5.9 ms | 1.26× / 1.26× |
| First `predict` (excluding round-0 cold start) | 6.2 ms | 4.5 ms | 1.39× |
| Memory growth after loading and 50 predictions | 14.1 MiB | 11.3 MiB | 1.25× |
| Memory growth, test peak | 20.0 MiB | 18.8 MiB | — |

- The latency ratio roughly matches the Mac's 1.28–1.32×, slightly higher, for the same reasons: vocabulary 12,288 vs 8,192, 9 vs 7 layers. T=128 p95 is only 2% above p50, and variation across the 4 rounds is small.
- zh-v1's round-0 load of 244 ms and first `predict` of 13.9 ms are cold start; the next 3 rounds settle at 174–201 ms and 6.2 ms. V4.2's 4 loads are 144–167 ms with first `predict` at 4.4–4.6 ms, with no clear first-round difference.
- Peak memory does not compare directly: V4.2's runs also include Japanese candidate reranking and prediction (40.9 MiB after reranking), which zh-v1's runs skip. For the model alone, zh-v1 uses 2.8 MiB more after loading and 50 predictions. The Mac report's 5.0 MiB is peak model growth, a different measure.
- Simulator numbers only show that things work, not device performance: T=128 p50 is 69.4 ms for zh-v1 and 44.7 ms for V4.2, over 9 times slower than the device, with large variation between rounds. The earlier simulator functional checks (resource identity, logits shape, native SentencePiece tokenization, joint scoring) passed 4/4.

### Keyboard-extension integration (2026-11-11)

Results come from the Vime maintainer's second handoff package `zh-v1-vime-integration-20261111.zip`. Device iPhone 16 Pro Max (iOS 27.3), optimized build (Debug configuration with `-O`), CPU_ONLY. Thermal state stayed nominal and Low Power Mode was off. The model is the stateless INT8 package.

The integration code is on Vime's `feat/chinese-language-model` branch, not yet merged. How it works:

- Rime produces candidates. The model reranks only the first 8 full conversions, scoring the Simplified text of each candidate appended to the sentence before the cursor.
- Rime's original order is a prior: each position down costs 1.0 log-probability.
- The candidate bar waits at most 60 ms for scoring, then shows Rime's order. While the model is still loading, the current list is shown immediately and the model is used from later input.
- The model never moves user-learned words, partial conversions or excluded words.
- Only one of the Chinese and Japanese models is loaded in the extension at a time.

**Inside the extension process**: XCUITest taps on the installed keyboard, 5 minutes per round, about 1.2 taps per second. The model-on round had 361 taps and committed 36 words or phrases; the model-off round used Rime's order and timed 309 calls in the extension. All timings below were taken in the extension process. p95 is the maximum of the per-window p95s, each window the latest 256 calls.

| Metric | Model on | Model off |
| --- | ---: | ---: |
| Keypress → tappable candidates (`inputToCandidateInteractive`) mean / p95 / max | 26.1 / 45.0 / 49.7 ms | 20.3 / 26.1 / 27.4 ms |
| Waiting for scoring (`chineseRankingWait`) mean / p95 / max | 13.9 / 36.4 / 40.9 ms | — |
| Model scoring (`lmCandidateScoring`) mean / p95 / max | 13.2 / 36.3 / 40.8 ms | — |
| Touch-up to type (`touchUpToType`) mean / p95 | 0.40 / 0.93 ms | 0.41 / 0.83 ms |
| Physical memory median / max; process peak | 20.0 / 29.5; 29.9 MiB | 24.3 / 25.3; 25.7 MiB |
| Model load in the extension | 253 ms (once) | — |
| Scorable / scored in time / order changed | 222 / 221 / 173 | — |

- With the model on, keypress-to-tappable p95 rises by about 19 ms, with a maximum of 49.7 ms, still within the 60 ms wait limit. Only 1 of 222 did not use the scoring result, either from a timeout or because the model had not finished loading; that time Rime's order was shown.
- 173/221 means the order of the first 8 candidates changed, not that the first choice changed.
- Scoring in the extension is several times slower than in the test host: 2.7 ms mean in the host, 13.2 ms in the extension. The cause is not confirmed; it may be the longer left context from real documents (up to 64 characters, cut at a full stop, exclamation mark, question mark or newline), or different CPU scheduling for extensions.
- The extension's memory peak is 29.9 MiB, 4.2 MiB above model-off. The median physical memory is actually lower with the model on; that is a difference between the two runs and cannot be taken as the model's memory cost.

**End-to-end reranking in the test host**: 24 "pinyin + left context" cases typed key by key, each keypress going through Rime, the first 8 candidates and model scoring. 3 passes, the 2nd and 3rd reported: 428 keypresses, 300 scorings, 7.1 candidates per scoring on average. Rime per key p50 0.09 ms, p95 0.31 ms. Model scoring p50 1.9–2.0 ms, p95 9.2–9.8 ms, max 14.8 ms. Rime plus scoring p95 9.0 ms. No scoring exceeded 60 ms and none failed. The per-case results show the model adjusting order by left context; for example, after 数学 (mathematics) with `gongshi`, 公式 (formula) moves ahead of 公示 (public notice).

**Chunked-KV package (device)**: first load 236–280 ms, reload 1.3 ms. 37.6 MiB after loading, about 48.5 MiB after prediction. Single-call p50 by length:

| Tokens per call | p50 |
| --- | ---: |
| 1 | 2.4–2.9 ms |
| 32 | 4.0 ms |
| 64 | 5.2–5.3 ms |
| 128 | 7.5–7.6 ms |

Reranking 3–6 short candidates, the KV path has p50 13.5–14.0 ms against about 9.0 ms for stateless joint scoring: each call moves two caches of about 1.8 MB, which costs more than the compute it saves. So the keyboard uses the stateless package. The test process's peak of about 119 MiB comes from the test framework not releasing objects between calls, not from the model's resident footprint.

The simulator (iPhone 17,2, Debug build) was used only to confirm the flow runs; its numbers are not used.

### Measurements still needed before launch

- Chinese prediction: not implemented in the keyboard yet, no data.
- Scoring 4–5 times slower in the extension than in the host: needs a cause. One approach is to fix the left-context length and measure once in each.
- Behavior under memory pressure, cold start and power use.

## Release contents

Hugging Face repository `Voltline/vimeml-tiny-zh-v1`, in the same format as the Japanese V4.2 release:

| Path | Contents |
| --- | --- |
| `inference/` | FP32 weights (no optimizer), config, tokenizer, bundle manifest |
| `coreml/ios18-int8-block32/` | Stateless INT8 Core ML package and manifest |
| `coreml/ios18-int8-block32-kvchunk/` | Chunked-KV INT8 Core ML package and manifest |
| `source/` | Repository source snapshot at the release commit (tracked files only) |
| `infer.py` | FP32 candidate scoring and prediction example |
| `evaluation/summary.json` | Summary numbers only, no item text, candidates or item ids |
| `RELEASE.json`, `SHA256SUMS.txt`, `verify_release.py` | Manifest and verification |
| `LICENSE`, `LICENSE-EXCEPTION`, `NOTICE.md` | GPL-3.0-or-later, App Store additional permission, training-data attribution |

Neither the model card nor the release package contains any information about training infrastructure; they say only "single-GPU training".

On 2026-11-11 the model card's Status and Performance sections were updated with the iPhone test host, chunked-KV package and keyboard extension measurements, and the hashes in `RELEASE.json` and `SHA256SUMS.txt` were updated to match (Hugging Face commit `8cc5ef3`). Weights and Core ML packages did not change, and the raw device records are not published with the release.

## License

- Weights and code keep the project's current GPL-3.0-or-later plus App Store additional permission, the same as the current Japanese V4.2 release.
- FineWeb2 `cmn_Hani`: ODC-By 1.0, also subject to Common Crawl's terms of use.
- **LCCC-base** (thu-coai, CDial-GPT repository): the repository LICENSE and the Hugging Face data card are MIT, but the README disclaimer says the LCCC dataset and pretrained dialogue models provided by the project are for research use only. The two statements disagree, and the README one is about use. LCCC is only 5% of supervised training tokens, but the model was trained on it. The project owner decided to release publicly; the model card and NOTICE state both terms as they are.
- Evaluation sources (CrossWOZ, RiSAWOZ, KdConv, CPED) are not in the weights; the release package contains only summary numbers.
