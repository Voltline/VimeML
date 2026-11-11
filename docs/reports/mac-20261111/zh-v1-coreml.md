# vimeml-tiny-zh-v1 Core ML conversion, INT8 quantization and quantization survey — 2026-11-11

The source is `artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/deployment.pt`, step 844,730 (source SHA256 `44c423a1…`, deployment tensors `7959fea4…`), `TinyGPTZh13`: 12,288 vocabulary, d384, 9 layers, 6 heads, SwiGLU 960, context 128, tied head, 20,036,736 parameters. Tokenizer `zh-unigram-12288` (`68bd37c4…`). The repository was later recreated; work continued in a fresh clone of the recreated `main` (`9d3a687`), with ignored artifacts and outputs copied over and the code checked to be identical. It follows the [handoff brief](../../reference/mac-zh-v1-coreml.md). Only the development split was read. No retraining, checkpoint selection, candidate-pool or reference change, or scoring change. V4.2, V4.3 and V2.1 packages and records were not touched.

Environment: Apple M3, 16 GiB, macOS 27.3 (26C5035r), Xcode 27.2 (27B48), Python 3.11.17 (uv venv), PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5, scikit-learn 1.5.1, sentencepiece 0.2.1. Core ML targets: mlprogram, iOS 18, FP32 compute, `CPU_ONLY`.

Development denominators: full track 1,456 items, abbreviated track 990 items. Item references below are ids only; no benchmark text is quoted.

## Summary

| Step | Result |
| --- | --- |
| 1. Mac PyTorch CPU FP32 baseline | Passed: fixtures max 3.7e-5, `forward_cached` steps max 1.5e-5; Top-1 1363/337 equals the Windows reference; 0 order changes on both tracks |
| 2. Generalized Core ML code | New `vimeml.deployment.zh_v1` (12,288 / 9 layers, cache `(9,1,6,128,64)`); V4 modules and formats unchanged |
| 3. FP32 Core ML | Stateless max 3.9e-5, chunk-KV max 4.6e-5 (22 checks); both 1363/337, +0/−0 and 0 order changes vs Mac PyTorch |
| 4. INT8 block32 | Strict logits gate failed as expected (max 0.392; V4.2 was 0.477). Top-1 full 1364 (+2/−1), abbreviated 335 (+0/−2) vs FP32. Chunk-KV INT8 equals stateless INT8 within 4.2e-5 and on every ranking |
| 5. Quantization survey | Size/accuracy only, no recipe selection: INT4 b32, palette 6-bit and 4-bit (grouped channel 32), INT8 b32 + FP16 embedding |
| 6. Mac measurements vs V4.2 | Done with the supplied V4.2 packages (not rebuilt). INT8 is 1.33× the size of V4.2; stateless `predict` p50 is 1.28–1.37× (T=128: 8.49 ms vs 6.43 ms); the model's peak memory delta is 18.4 vs 13.4 MiB |
| 7. Vime iPhone test host | Resources prepared. The Vime-side `.zhV1` variant is left to Vime's own maintainers; a one-off local check on the simulator passed (4/4) and was reverted. Benchmark and iPhone runs **not done** |

## 1. Mac PyTorch baseline

`outputs/deployment/zh-v1-mac-pytorch-baseline-20261111`. `TinyGPTZh13` loads `deployment.pt` on CPU FP32 (4 threads). All five fixture cases (`bos`, `short`, `long`, `right_pad`, `changed_future` valid prefixes) are within atol=rtol=3e-4 of the Windows logits; maximum error 3.72e-5. Single-token `forward_cached` over `long` matches `long_kv_step_logits` with maximum error 1.5e-5.

| Track | Windows CUDA FP32 | Mac PyTorch | Improved / regressed | Order-changed items | Max candidate-score delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full / 1,456 | 1363 | 1363 | +0/−0 | 0 | 5.7e-5 |
| Abbreviated / 990 | 337 | 337 | +0/−0 | 0 | 5.3e-5 |

Score files hash to `cde45ed7…` (full) and `746b9697…` (abbreviated). Scoring is `zh13_lm_scores.score_item`, unchanged: BOS + joint context/candidate tokenization, common-prefix suffix log-probability sum, no EOS; `score_standard_ime_zh.py score` and `compare` produce the results and comparisons.

## 2. Generalizing the Core ML code

The V4 code (`v4_coreml`, `v4_chunk`) hard-codes 8,192 / 7 layers in its contract constants and formats. Rather than changing it, `src/vimeml/deployment/zh_v1.py` is a thin module beside it that reuses the shared pieces:

- Contract: `VOCAB=12288`, `LAYERS=9`, `CACHE_SHAPE=(9,1,6,128,64)`, the same tolerance (3e-4) and weight threshold (2048).
- New formats: `vimeml_inference_bundle_zh_v1`, `vimeml_coreml_zh_v1`, `vimeml_coreml_zh_v1_kv_chunk`, `vimeml_alignment_zh_v1`, `vimeml_kv_chunk_alignment_zh_v1`.
- Stateless conversion reuses `graph.trace_graph`; chunk-KV conversion reuses `v4_chunk.KVChunkGraphV4` (it is shape-generic).
- Runtime adapters for the unchanged scorer: `BundleLMZh`, `CoreMLLMZh`, and `CachedChunkLMZh`, which serves `score_item` with one chunk call for the common prefix plus one call per candidate suffix. `zh13_lm_scores.run` gained an optional `loaded=(model, processor, meta)` argument; its default path is unchanged.
- `scripts/deployment/coreml_zh_v1.py` exposes the stages (`baseline`, `export`, `convert`, `convert-chunk`, `align`, `align-chunk`, `compress`, `evaluate`), each into a fresh output directory.
- `tests/test_deployment_zh_v1.py` (6 tests): contract constants, chunk-graph trace vs full forward, cached scoring vs stateless `score_item`, stable-tie rank differences, changed-item ids, bundle identity. The full unit suite run with `python -m unittest` passes: 39 tests OK, 5 skipped (22 OK / 5 skipped before the change).

## 3. FP32 conversion and alignment

| Package | Interface | Logical bytes |
| --- | --- | ---: |
| `tiny-zh-v1-fp32-v1` | `input_ids int32[1,T]`, 1≤T≤128 → `logits float32[1,T,12288]` | 80,367,198 |
| `tiny-zh-v1-kvchunk-fp32-v1` | `input_ids int32[1,T]`, `position int32[1]`, `key_cache`/`value_cache float32(9,1,6,128,64)` → `logits`, new caches | 80,404,135 |

| Check | Max abs error | Result |
| --- | ---: | --- |
| PyTorch inference bundle, five fixtures (`zh-v1-align-pytorch-bundle`) | 3.72e-5 | Passed |
| Stateless Core ML, five fixtures (`zh-v1-align-fp32`) | 3.91e-5 | Passed; padding / future-token invariants exact |
| Chunk-KV Core ML vs fixtures: one-shot, split chunks, single-token steps, shared-prefix branches, rollback, snapshot immutability — 22 checks (`zh-v1-align-kvchunk-fp32`) | 4.63e-5 | Passed |

| Track | Mac PyTorch | Core ML FP32 stateless | Core ML FP32 chunk-KV |
| --- | ---: | ---: | ---: |
| Full | 1363 | 1363 (+0/−0, 0 order changes) | 1363 (+0/−0, 0 order changes vs stateless, max delta 3.4e-5) |
| Abbreviated | 337 | 337 (+0/−0, 0 order changes) | 337 (+0/−0, 0 order changes vs stateless, max delta 3.1e-5) |

## 4. INT8 block32

Recipe as for V4.2: `linear_symmetric`, INT8, `per_block` block 32, weight threshold 2048, parameter selection via `v2.parameter_selection`. 47 matrices (20,029,440 elements) are quantized: token embedding `[12288,384]` (shared by the gather and the tied head), position embedding `[128,384]`, and per layer × 9 the fused QKV `[1152,384]`, attention projection `[384,384]`, FFN gate and up `[960,384]` and FFN down `[384,960]`. Left unchanged: `causal_mask` `[128,128]` and the zero head bias `linear_45_bias_0` `[12288]`; norms are not matrices. FP32 computation, no activation quantization.

| Package | Logical bytes |
| --- | ---: |
| `tiny-zh-v1-int8-b32-v1` | 22,792,958 |
| `tiny-zh-v1-kvchunk-int8-b32-v1` | 22,830,199 |

Strict logits gate (`zh-v1-align-int8-b32`) **failed as expected**; it is preserved as evidence, not tuned around:

| Case | Max | Mean | RMSE | p50 | p90 | p99 | p99.9 | Outside tolerance | Argmax equal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bos | 0.128 | 0.0267 | 0.0327 | 0.0238 | 0.0522 | 0.0805 | 0.1089 | 95.2% | 1/1 |
| short | 0.310 | 0.0408 | 0.0525 | 0.0330 | 0.0859 | 0.1468 | 0.2015 | 95.2% | 15/16 |
| long | 0.392 | 0.0463 | 0.0598 | 0.0374 | 0.0972 | 0.1700 | 0.2349 | 95.5% | 127/128 |
| right_pad, changed_future | same as short | | | | | | | | |

Padding and future-token invariants are exactly 0.

Chunk-KV INT8 against the FP32 fixtures fails (max 1.294, from the shared-prefix branch check; one-shot `long` 0.392) — the expected INT8 error compounded through cached prefixes. Against the **same-weight stateless INT8** package it passes: 21 checks, max 4.15e-5 (`zh-v1-align-kvchunk-int8-b32-vs-stateless-int8`). That is the gate used for the iOS resources.

Development ranking, reference = Core ML FP32 stateless:

| Track | FP32 | INT8 stateless | Improved / regressed | McNemar p | Order-changed items | Top-1-changed items | Max score delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 1363 | 1364 | +2/−1 | 1 | 878 | 4 | 0.661 |
| Abbreviated | 337 | 335 | +0/−2 | 0.5 | 619 | 17 | 1.465 |

"Order-changed items" counts items whose full candidate order changed with any score movement (stable ties); "Top-1-changed" counts items whose top candidate changed, correct or not. Every item whose correctness changed:

- Full improved: `zh-std-crosswoz-ea1daad81c`, `zh-std-risawoz-e8c74bc9fe`
- Full regressed: `zh-std-crosswoz-0c2c5afa15`
- Abbreviated improved: none
- Abbreviated regressed: `zh-std-kdconv-422abcb556`, `zh-std-risawoz-f9b102f070`

Chunk-KV INT8 (`zh-v1-eval-kvchunk-int8-b32`) gives identical Top-1, identical changed ids, 0 order changes against stateless INT8 on both tracks (max score delta 3.1e-5 full, 2.3e-5 abbreviated).

## 5. Quantization survey (size and accuracy only)

Stateless packages only, same parameter selection, FP32 computation. The survey records numbers; it does not select a recipe, and the deployed recipe remains INT8 b32.

| Recipe | Package | Bytes | Strict max (long mean) | long argmax | Full Top-1 (vs FP32) | Abbrev Top-1 (vs FP32) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | `tiny-zh-v1-fp32-v1` | 80,367,198 | 3.9e-5 | 128/128 | 1363 | 337 |
| INT8 b32 (deployed) | `tiny-zh-v1-int8-b32-v1` | 22,792,958 | 0.392 (0.046) | 127/128 | 1364 (+2/−1) | 335 (+0/−2) |
| INT8 b32 + FP16 embedding | `tiny-zh-v1-survey-int8-b32-embfp16-v2` | 26,921,756 | 0.200 (0.032) | 128/128 | 1365 (+2/−0) | 337 (+0/−0) |
| Palette 6-bit, grouped channel 32 | `tiny-zh-v1-survey-palette6-gc32-v1` | 15,658,010 | 2.04 (0.240) | 117/128 | 1364 (+7/−6) | 339 (+4/−2) |
| INT4 b32 | `tiny-zh-v1-survey-int4-b32-v1` | 12,778,238 | 6.99 (0.822) | 95/128 | 1349 (+10/−24) | 332 (+8/−13) |
| Palette 4-bit, grouped channel 32 | `tiny-zh-v1-survey-palette4-gc32-v1` | 10,368,794 | 9.49 (1.328) | 76/128 | 1320 (+16/−59) | 316 (+17/−38) |

Order-changed / Top-1-changed items vs FP32 (full; abbreviated): FP16-embedding 634/3; 462/10. Palette6 1424/15; 983/62. INT4 1451/50; 990/190. Palette4 1450/104; 990/289. Changed ids for every survey package are in each `zh-v1-eval-survey-*/judgement.json` (`improved_ids`, `regressed_ids`).

Notes:

- FP16 embedding: the token-embedding const is replaced with an FP16 const plus `constexpr_cast` to FP32 after INT8 compression; FP16 rounding of the embedding is at most 1.22e-4. The first build (`…-embfp16-v1`) lost the stateless input range (upper bound 2 instead of 128) when re-converting from the MIL program; it was deleted and rebuilt as `-v2` with inputs, outputs and metadata restated and the input spec checked against the source. Only v2 is evaluated.
- Palette and INT4 recipes change many more candidate orders than INT8; the 6-bit palette Top-1 is within noise of FP32 on these tracks despite a 2.0 strict max, so Top-1 alone understates its error.

## 6. Mac measurements vs V4.2

The four V4.2 epoch 20 packages accepted on 2026-10-20 (stateless FP32/INT8, chunk-KV FP32/INT8) were supplied as a ZIP (SHA256 `4728fc85…`), unpacked unchanged into `artifacts/deployment/`, and checked against their SHA256 list; package sizes equal the V4.2 report. They were not rebuilt.

Harness: `measure_v42.py run --set zh-v1` → `outputs/deployment/zh-v1-mac-measure-v1`. Each package is compiled with `coremlcompiler`, then measured in a fresh Python worker per model per round, eight models interleaved over five rounds, `CPU_ONLY`, 5 warm-up and 50 timed calls. Latency is the median over the five processes of each process's p50 / p95. Conditions: AC power (charging 35→39%), Low Power Mode off, and `pmset -g therm` recorded no thermal or performance warning at any of the six checkpoints. The 1-minute load average was 14.3 at the start, left over from setting up the fresh checkout, and fell to 3.9 by the end. Both families saw the same conditions because they were interleaved, but absolute numbers may read slightly high. These are Mac host-process numbers. They do not substitute for iPhone or keyboard-extension measurements.

Stateless:

| Metric | zh-v1 FP32 | zh-v1 INT8 | V4.2 FP32 | V4.2 INT8 | zh-v1 / V4.2 INT8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Package bytes | 80,367,198 | 22,792,958 | 60,467,999 | 17,164,271 | 1.33× |
| Load, first process (ms) | 261.4 | 197.5 | 196.2 | 137.3 | 1.44× |
| Load, p50 over processes (ms) | 30.6 | 28.4 | 27.6 | 26.2 | 1.08× |
| First `predict` in a process, T=32, p50 over processes (ms) | 107.2 | 102.4 | 89.3 | 13.7 | — |
| T=1 p50 / p95 (ms) | 1.66 / 2.06 | 1.65 / 2.11 | 1.34 / 1.52 | 1.29 / 1.71 | 1.28× |
| T=8 p50 / p95 | 2.45 / 3.06 | 2.43 / 2.88 | 1.76 / 2.19 | 1.78 / 2.35 | 1.37× |
| T=32 p50 / p95 | 3.98 / 4.53 | 3.85 / 4.28 | 2.98 / 3.32 | 2.95 / 3.35 | 1.31× |
| T=64 p50 / p95 | 5.46 / 5.88 | 5.52 / 5.88 | 4.21 / 4.85 | 4.12 / 4.55 | 1.34× |
| T=128 p50 / p95 | 8.40 / 9.28 | 8.49 / 8.88 | 6.31 / 6.84 | 6.43 / 6.71 | 1.32× |
| Footprint after load (MiB) | 309.7 | 310.1 | 309.8 | 310.2 | — |
| Load delta / peak delta (MiB) | 1.1 / 18.5 | 1.1 / 18.4 | 1.0 / 12.3 | 1.0 / 13.4 | 1.37× peak |
| Lifetime peak footprint (MiB) | 327.1 | 327.5 | 320.8 | 322.1 | +5.4 MiB |

The ratio column compares the INT8 p50 values. The first `predict` in each process is the first call after load, at T=32, and includes Core ML's first-shape work. It varies widely between processes (V4.2 INT8 13.7 ms against about 90–107 ms for the others), so it should be read as an indicator only. The first-process cold calls at the other lengths are 1.8–9.4 ms for zh-v1 and 1.5–7.0 ms for V4.2. The footprint baseline of about 309 MiB is the Python, PyTorch and coremltools host process; the load delta and peak delta are the model's own share.

Chunk-KV (prefix P prefilled in one call, then single-token decode):

| Metric (ms) | zh-v1 FP32 | zh-v1 INT8 | V4.2 FP32 | V4.2 INT8 |
| --- | ---: | ---: | ---: | ---: |
| Package bytes | 80,404,135 | 22,830,199 | 60,496,928 | 17,193,424 |
| Load, first process / p50 | 255.8 / 27.2 | 202.2 / 26.0 | 200.3 / 28.7 | 167.1 / 28.2 |
| First prefill T=128, p50 over processes | 10.2 | 10.0 | 8.0 | 7.9 |
| Prefill P=7 / 31 / 63 / 119, p50 | 3.60 / 5.25 / 6.67 / 9.37 | 3.60 / 5.14 / 6.74 / 9.33 | 2.90 / 3.98 / 4.99 / 7.13 | 2.77 / 3.78 / 4.90 / 7.18 |
| Prefill P=119 p95 | 9.69 | 9.57 | 7.33 | 7.29 |
| Decode after P=7 / 119, p50 | 2.61 / 2.70 | 2.63 / 2.71 | 2.10 / 2.05 | 2.21 / 2.05 |
| Decode after P=119, p95 | 3.05 | 3.05 | 2.54 | 2.49 |
| Peak delta / T=128 chunk peak delta (MiB) | 31.3 / 18.4 | 29.8 / 18.4 | 20.8 / 13.0 | 20.8 / 12.4 |

On the Mac, zh-v1 is about 1.3× V4.2 in size, stateless latency, chunk prefill and decode, and about 1.4× in the model's peak memory delta. That is in line with the brief's expectation from the 12,288 vocabulary and nine layers. As with V4.2, INT8 reduces size but not CPU latency, because computation stays FP32.

## 7. iOS resources and Vime test host

`scripts/deployment/package_zh_v1_ios.py` → `artifacts/deployment/tiny-zh-v1-ios-resources-v1`:

- `Resources/TinyChineseV1INT8.mlmodelc`, `TinyChineseV1KVChunkINT8.mlmodelc` (about 22 MB each), `VimeChineseTokenizerV1.model`, `VimeLMManifestZhV1.json` (format `vime_ios_lm_zh_v1`, compiled-file SHA256s, `strict_fp32_logits_alignment_passed: false`).
- `VimeLMFixturesZhV1.json`: tokenization, scoring and chunk-KV step fixtures on texts written for this purpose, none of them from the benchmark. Chunk-KV suffix sums match stateless sums within 6.2e-6.

Packaging requires the same-weight chunk/stateless INT8 gate to have passed.

Changes to Vime source are made by Vime's own maintainers, so this session made no lasting change there. To check that the resources work in the client runtime, a temporary local Vime worktree was used, then removed:

- It had an opt-in `.zhV1` case outside `runtimePreference`.
- It loaded resources from the UIKit test bundle only, so the app and keyboard bundles were unchanged.
- It had four functional tests.

Results:

- The simulator (iPhone 18 Pro Max, iOS 27.2, Debug) passed the four zh-v1 tests:
  - Not a runtime choice and not in the app bundle.
  - Identity and logits shape `[1,T,12288]`.
  - Native SentencePiece matches all tokenization fixtures.
  - Joint suffix scores match Mac Core ML INT8 within 0.002, with and without the conversion cache.
- The existing V4.2 tests (6) and model tests (24) also passed in the same run, 34/34 in total.
- The worktree, branch, installed resources and build products were then deleted; nothing was committed or pushed to Vime.
- The diff is kept as a reference for the Vime side in `outputs/deployment/zh-v1-vime-test-host-reference-v1/` (in the results ZIP, not in Git).

**Not done:** the simulator and iPhone 16 Pro Max `VimeLanguageModelBenchmarkTests` runs (four rounds alternating with V4.2, one variant per process). These need the Vime-side variant and can run once it lands with the existing `Scripts/benchmark_lm_variants.py --variants zhV1,v42`. Only stateless INT8 was checked in the client: Vime has no KV runtime for V4.2 either, so the chunk-KV resource is covered only by the Mac fixture gate.

## Tests

The targeted unit tests for this work (zh-v1, V4 and V2 deployment, KV cache, Core ML compression, deployment tools, zh-v1 confirmation, zh13 tokenizer) pass: 44 OK, 5 skipped, both before and after moving to the fresh clone. The full `unittest discover` run has errors on this Mac. They come from Japanese assets that are not in this checkout (V4 tokenizer, V4 token data, posttraining data, a V4 checkpoint) and from the `fugashi` and `opencc` packages, which are not installed. The same errors occur without this change; no dependencies were installed for them.

## Outputs

Packages under `artifacts/deployment/tiny-zh-v1-*`; records under `outputs/deployment/zh-v1-*` (alignment, evaluation scores, results, `judgement.json`, `model.json`, Mac measurements `zh-v1-mac-measure-v1`). Large assets are returned through `scripts/deployment/package_zh_v1_mac_results.py`.
