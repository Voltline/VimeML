# V4.2 Core ML conversion, INT8 quantization and Mac measurements — 2026-10-20

The source is V4.2 epoch 20 `best-lm.pt`, step 1,081,942 (source SHA256 `0e83ea74…`, deployment tensors `818a8f57…`), 15,070,848 parameters, V4-a 8K tokenizer, context 128. Work started from VimeML `b0f1fe6` on `claude/mac-v42-coreml`, following the [handoff brief](../../reference/mac-v42-coreml.md). No retraining, checkpoint selection, candidate-pool or reference change, scoring change or blind/test read. V2.1 remains the deployed model; its release assets were not modified.

Environment: Apple M3, 16 GiB, macOS 27.2 (26B5112a), Xcode 27.1, Python 3.11.17, PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5, scikit-learn 1.5.1, sentencepiece 0.2.1. coremltools warns that PyTorch 2.7.1 is untested (2.7.0 is the last tested); all gates below ran under this pairing.

## Summary

| Step | Result |
| --- | --- |
| 1. Mac PyTorch CPU FP32 baseline | Passed: fixtures max error 6.1e-5; Top-1 523/370/139/123 equals CUDA; 0 per-case order differences in any LM method |
| 2. FP32 Core ML conversion | Stateless `int32[1,T]→float32[1,T,8192]` and single-token KV graph; iOS 18, FP32, CPU_ONLY |
| 3. FP32 gates | Passed: stateless max 5.2e-5, KV max 5.0e-5, all invariants exact; stateless and KV Top-1 and full orders equal Mac PyTorch on all four tracks |
| 4. INT8 block32 | Strict logits gate failed as expected (max 0.477); task-level Top-1 522/371/139/123 vs FP32 523/370/139/123; KV INT8 equals stateless INT8 within 4.0e-5 and on all Top-1 (one diagnostic-method tie swap) |
| 5. Measurements | V4.2 and V2.1 FP32/INT8, stateless and KV, same Mac harness; INT8 changes size, not CPU latency; V4.2 single-token KV prefill is ~30× V2.1 chunk prefill at 119 tokens |
| 6. iOS resources | v1 (single-token KV) and v2 (chunk KV) prepared |
| 7. Chunked KV graph | FP32 gate passed (max 5.2e-5), INT8 equals stateless INT8 within 4.0e-5; Top-1 and primary orders equal stateless on all tracks; P=119 prefill 6.6 ms vs 170 ms single-token |
| 8. iOS | Vime opt-in V4.2 variant; simulator and iPhone 16 Pro Max functional tests pass; device `predict` +0.16–0.35 ms vs V2.1, rerank p95 lower, peak footprint 5.6 MB lower (thermal `fair`) |

## 1. Mac PyTorch baseline

`JapaneseLMV4` loads `deployment.pt` on CPU FP32 (4 threads). All five fixture cases (`bos`, `short`, `long`, `right_pad`, `changed_future` valid prefixes) are within atol=rtol=3e-4 of the Windows PyTorch logits; maximum error is 6.1e-5. Single-token `forward_cached` over `long` matches `long_kv_step_logits` within the same tolerance.

| Track | CUDA FP32 | Mac PyTorch | Order differences | Max suffix-score delta |
| --- | ---: | ---: | ---: | ---: |
| Standard development / 728 | 523 | 523 | 0 | 1.23e-4 |
| Legacy regression / 500 | 370 | 370 | 0 | 1.25e-4 |
| AJIMEE / 200 | 139 | 139 | 0 | 1.33e-4 |
| Historical development / 137 | 123 | 123 | 0 | 6.2e-5 |

Order differences are counted per case for `lm_context_sum`, `lm_context_mean_secondary` and `lm_no_context_sum`; all are zero.

## 2. Conversion

`vimeml.deployment.v4_coreml` adds a V4 inference bundle (unique FP32 tensors, tied head stored once, tokenizer and token manifest), a stateless converter reusing the V2 explicit-attention `ConversionGraph`, and a KV converter from `vimeml.deployment.v4.KVStepGraphV4`. Formats are new (`vimeml_inference_bundle_v4`, `vimeml_coreml_v4`, `vimeml_coreml_v4_kv_v1`); V1/V2 modules and formats are untouched. `scripts/deployment/coreml_v4.py` exposes each stage with fresh output directories.

| Package | Interface | Logical bytes |
| --- | --- | ---: |
| `tiny-ja-v4.2-epoch20-fp32-v1` | `input_ids int32[1,T]`, 1≤T≤128 → `logits float32[1,T,8192]` | 60,467,999 |
| `tiny-ja-v4.2-epoch20-kv-fp32-v1` | `input_ids int32[1,1]`, `position int32[1]`, `key_cache`/`value_cache float32(7,1,6,128,64)` → `logits [1,1,8192]`, `new_key_cache`, `new_value_cache` | 60,421,926 |

The KV graph writes slot `position` with a select against a static slot index and masks slots greater than `position`; it has no scatter or dynamic slicing. One K/V snapshot is 2,752,512 bytes. Callers own snapshots: inputs are not modified and outputs are fresh arrays.

## 3. FP32 alignment

Tolerance is unchanged: atol=rtol=3e-4.

| Check | Max abs error | Result |
| --- | ---: | --- |
| Stateless, five fixture valid prefixes | 5.2e-5 | Passed; last-position and all-position argmax equal |
| Stateless `right_pad` / `changed_future` prefix vs `short` | 0 | Passed |
| KV one-shot (uninterrupted single-token prefill) vs full logits, five cases | 5.0e-5 | Passed |
| KV one-token vs `long_kv_step_logits` | 2.9e-5 | Passed |
| KV split-chunk prefill from copied snapshots ([3,·] and half/half) | 5.0e-5; identical to one-shot (difference 0) | Passed |
| KV shared-prefix branches [9], [47,81,102], [9] vs full logits | 2.8e-5 | Passed |
| A/B/A rollback, dirty unused cache tail (999), input snapshot immutability | 0 / 0 / unchanged | Passed |

| Track | Mac PyTorch | Core ML FP32 stateless | Core ML FP32 KV |
| --- | ---: | ---: | ---: |
| Standard development | 523 | 523 | 523 |
| Legacy regression | 370 | 370 | 370 |
| AJIMEE | 139 | 139 | 139 |
| Historical development | 123 | 123 | 123 |

Both Core ML FP32 runs have 0 per-case order differences from Mac PyTorch in every LM method. Maximum suffix-score deltas are 5.3e-5 (stateless) and 7.5e-5 (KV). The KV evaluation shares the exact joint-tokenized common prefix within each scoring call and scores each candidate suffix from that immutable snapshot, one token per call.

## 4. INT8 quantization

Recipe identical to V2.1: `linear_symmetric` int8, `per_block` block 32, `weight_threshold=2048`, FP32 compute, CPU_ONLY. The same V2 parameter selection compresses 37 finite matrices (token and position embeddings plus 35 block linears; 15,065,088 elements). RMSNorm weights, the causal mask (stateless only) and the converter-inserted lm_head bias constant `linear_35_bias_0` remain FP32. Packages: stateless 17,164,271 bytes (−71.6%), KV 17,118,210 bytes.

Strict logits alignment fails, as expected, and the gate is not loosened:

| Fixture | Max | Mean | RMSE | p50 | p90 | p99 | p99.9 | Outside tolerance | Argmax equal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bos` (1 position) | 0.127 | 0.0195 | 0.0252 | 0.0151 | 0.0405 | 0.0731 | 0.104 | 94.7% | 1/1 |
| `short` (16) | 0.453 | 0.0467 | 0.0648 | 0.0345 | 0.100 | 0.232 | 0.325 | 96.8% | 14/16 |
| `long` (128) | 0.477 | 0.0495 | 0.0651 | 0.0391 | 0.105 | 0.195 | 0.284 | 97.2% | 124/128 |
| `right_pad`, `changed_future` (16 valid) | 0.453 | 0.0467 | 0.0648 | 0.0345 | 0.100 | 0.232 | 0.325 | 96.8% | 14/16 |

V2.1 INT8 reached 1.009 maximum and 0.1085 mean at T=128 under the same recipe. Padding and future-token invariants remain exact for INT8. INT8 KV against the FP32 fixtures fails with maximum 0.708 (branch diagnostic); against the same-weight stateless INT8 package it passes every check within 4.0e-5, with exact split/rollback/tail invariants.

| Track | Core ML FP32 | Core ML INT8 | Improved / regressed | Exact McNemar p | Full-order changes | Max score delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard development | 523 | 522 | 2 / 3 | 1.0 | 254 | 0.751 |
| Legacy regression | 370 | 371 | 1 / 0 | 1.0 | 46 | 0.812 |
| AJIMEE | 139 | 139 | 0 / 0 | 1.0 | 106 | 0.985 |
| Historical development | 123 | 123 | 0 / 0 | 1.0 | 88 | 0.724 |

Every regressed case ID (IDs only; benchmark text stays in ignored outputs):

- `ajimee:ime-standard-v1-development-wrime-0216`
- `ajimee:ime-standard-v1-development-jmultiwoz-0023`
- `ajimee:ime-standard-v1-development-jmultiwoz-0136`

Improved: `ajimee:ime-standard-v1-development-wrime-0389`, `ajimee:ime-standard-v1-development-jmultiwoz-0098`, `ajimee:ime-v21-development-1717`. Thirteen first choices change in total (11 Standard, 2 Legacy); seven of them leave correctness unchanged. All thirteen had an FP32 score margin of at most 0.139 between the two candidates. Nonsignificance is not an equivalence result.

INT8 KV reproduces stateless INT8 on all four tracks: Top-1 522/371/139/123, 0 per-case order differences in the primary `lm_context_sum` and in `lm_context_mean_secondary`, maximum suffix-score delta 5.4e-5. The only difference is one AJIMEE case (`ajimee:1763`) in the diagnostic `lm_no_context_sum`. Its candidates at ranks 11 and 12 are tied at −67.4044114 in stateless INT8 and separated by about 2e-5 in KV, so the stable-tie order swaps; Top-1 is unchanged. The acceptance check "KV INT8 orders equal stateless INT8 in all methods" is therefore recorded as false. Apart from this tie, cache reuse adds no error beyond the shared weight quantization.

### V2.1 under the same Mac harness

The deployed V2.1 packages were scored with the same frozen pools, scorer and Mac Core ML CPU_ONLY harness (`coreml_v4.py evaluate-v21`). V2.1 Core ML FP32 reproduces its recorded PyTorch FP32 run exactly: 505/374/144/122 with 0 per-case order differences. Earlier records had V2.1 INT8 only for AJIMEE (144) and Historical (122); Standard and Legacy are new here.

| Track | V2.1 FP32 | V2.1 INT8 | V2.1 INT8 vs FP32, improved / regressed | V4.2 INT8 | V2.1 INT8 → V4.2 INT8, improved / regressed | Net | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard development | 505 | 506 | 4 / 3 | 522 | 44 / 28 | +16 | 0.076 |
| Legacy regression | 374 | 375 | 3 / 2 | 371 | 13 / 17 | −4 | 0.585 |
| AJIMEE | 144 | 144 | 3 / 3 | 139 | 9 / 14 | −5 | 0.405 |
| Historical development | 122 | 122 | 0 / 0 | 123 | 4 / 3 | +1 | 1.0 |

For reference, the FP32 pairing V2.1 → V4.2 under this harness is 46/28 (p=0.047), 15/19, 8/13 and 4/3, identical to the recorded CUDA comparison for Standard. Quantization leaves the direction of the V2.1/V4.2 differences unchanged: V4.2 is ahead on Standard and behind on Legacy and AJIMEE, and none of the INT8 differences reaches p<0.05. These are exploratory development comparisons, not a replacement decision; the preregistered fixed-reference evaluation (FP32, +6 against V2.1, p=0.56) is recorded separately in the [reference and regression analysis](../../zh-CN/reports/v42-20261020/reference-regressions.md).

## 5. Measurements

`scripts/deployment/measure_v42.py` compiles each package with `xcrun coremlcompiler`, then loads the `.mlmodelc` through `coremltools.models.CompiledMLModel` with CPU_ONLY in a fresh Python process per model per round. Eight models are interleaved over five rounds, with 5 warm-up and 50 timed calls per shape and seeded random token IDs. Memory is the macOS `phys_footprint` from `proc_pid_rusage`; deltas are measured from the point after `coremltools` import (about 282 MiB of Python/coremltools/PyTorch runtime).

Conditions: AC power (charging 55→58%), Low Power Mode off, and `pmset -g therm` recorded no thermal or performance warning at any of six checkpoints. The quality evaluations had finished before measurement and nothing else was running; the 1-minute load average was decaying from 10.7 to 4.3 over the run. These are Mac host-process numbers and do not substitute for iPhone, simulator or keyboard-extension measurements.

The V2.1 KV packages are not in the payload. `measure_v42.py prepare-v21-kv` rebuilt them from the deployed V2.1 bundle with the unchanged `kv_cache` converter and the same INT8 recipe, under measurement-only names `tiny-ja-v2.1-extend5-kv-{fp32,int8-b32}-mac-v42-measure-v1`. Their logical sizes equal the accepted V2.1 KV packages (50,382,727 / 14,355,753 bytes), and they match the deployed V2.1 stateless FP32/INT8 packages within 4.4e-5 for one-shot prefill and for prefill-31-then-single-token decode. The V2.1 KV graph accepts multi-token chunks; the V4.2 graph is single-token by contract.

| Model | Package MB | Compiled MB | Load, first process (ms) | Load p50, 5 processes (ms) | Footprint Δ after load (MiB) | Peak footprint Δ (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4.2 FP32 | 60.47 | 60.47 | 599.9 | 27.8 | 1.0 | 12.1 |
| V4.2 INT8 | 17.16 | 17.17 | 242.3 | 26.6 | 1.0 | 12.1 |
| V2.1 FP32 | 50.36 | 50.36 | 597.0 | 27.0 | 0.9 | 24.2 |
| V2.1 INT8 | 14.33 | 14.34 | 151.8 | 27.1 | 0.9 | 24.1 |
| V4.2 KV FP32 | 60.42 | 60.43 | 211.7 | 22.1 | 1.0 | 14.0 |
| V4.2 KV INT8 | 17.12 | 17.13 | 163.5 | 23.6 | 0.9 | 16.6 |
| V2.1 KV FP32 | 50.38 | 50.39 | 171.9 | 26.4 | 1.1 | 28.7 |
| V2.1 KV INT8 | 14.36 | 14.36 | 149.7 | 24.8 | 1.0 | 28.8 |

The first process after compilation is the closest Mac equivalent of a cold load. Later loads reuse system Core ML caches. File-backed weights hardly change `phys_footprint` at load. Resident size grows by about 73–80 MiB after the workload for every model, mostly shared runtime pages, so it does not separate the models. Peak footprint deltas for V4.2 are about half those of V2.1, consistent with its 8,192-entry rather than 16,384-entry logits; that cause was not isolated.

Stateless warm per-call latency, median over processes of the per-process p50 / p95 (ms):

| Model | T=1 | T=8 | T=32 | T=64 | T=128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V4.2 FP32 | 1.36 / 1.53 | 1.81 / 2.11 | 2.70 / 3.29 | 3.83 / 4.07 | 6.14 / 6.28 |
| V4.2 INT8 | 1.30 / 1.51 | 1.79 / 2.07 | 2.72 / 3.02 | 3.94 / 4.08 | 6.07 / 6.27 |
| V2.1 FP32 | 0.97 / 1.15 | 1.61 / 1.85 | 2.37 / 2.55 | 3.57 / 3.81 | 5.75 / 5.89 |
| V2.1 INT8 | 0.99 / 1.20 | 1.64 / 1.84 | 2.51 / 2.71 | 3.56 / 3.79 | 5.68 / 5.89 |

KV warm latency: single-token decode after a prefix of P tokens, p50 / p95 (ms), and prefill of P tokens, p50. V4.2 prefill uses P sequential single-token calls; V2.1 prefill uses one chunk call.

| Model | Decode P=7 | P=31 | P=63 | P=119 | Prefill 7 / 31 / 63 / 119 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V4.2 KV FP32 | 1.49 / 1.73 | 1.47 / 1.62 | 1.45 / 1.57 | 1.49 / 1.67 | 10.7 / 46.3 / 96.3 / 180.8 |
| V4.2 KV INT8 | 1.51 / 1.73 | 1.45 / 1.61 | 1.51 / 1.71 | 1.47 / 1.66 | 10.5 / 45.1 / 96.0 / 180.7 |
| V2.1 KV FP32 | 1.62 / 1.75 | 1.61 / 1.78 | 1.59 / 1.68 | 1.61 / 1.75 | 2.3 / 3.1 / 4.2 / 6.0 |
| V2.1 KV INT8 | 1.62 / 1.83 | 1.55 / 1.71 | 1.59 / 1.76 | 1.61 / 1.78 | 2.3 / 3.1 / 4.4 / 6.4 |

Cold first call after load (stateless T=32; KV decode at position 0), first process / median / range over five processes (ms):

| Model | Cold first call |
| --- | --- |
| V4.2 FP32 / INT8 | 11.5 / 44.0 / 5.6–118.7 · 5.8 / 5.8 / 5.3–108.8 |
| V2.1 FP32 / INT8 | 5.5 / 39.4 / 5.5–87.4 · 5.2 / 54.7 / 5.0–103.3 |
| V4.2 KV FP32 / INT8 | 5.0 / 6.5 / 5.0–87.8 · 5.3 / 74.7 / 4.9–110.7 |
| V2.1 KV FP32 / INT8 | 4.7 / 15.0 / 3.7–64.4 · 3.7 / 13.2 / 3.7–65.0 |

Cold first calls are bimodal, at roughly 4–12 ms or 40–120 ms, and every model is slow in the last two of five rounds, with occasional slow calls earlier; this is a system-level spread, not a model difference, and its cause was not isolated. Under this harness INT8 storage changes size, not CPU latency, because computation stays FP32. Warm stateless calls for V4.2 run about 0.1–0.4 ms slower than V2.1. V4.2 single-token decode runs about 0.1 ms faster. Prefill on the V4.2 single-token graph grows by about 1.5 ms per token, reaching 30× the V2.1 chunk prefill at 119 tokens, so long-context client scoring through this KV graph would be dominated by prefill. A chunked V4 KV graph would be a separate, unbuilt experiment.

## 6. iOS client resources

`scripts/deployment/package_v4_ios.py` writes `artifacts/deployment/tiny-ja-v4.2-epoch20-client-resources-v1`: `TinyJapaneseV42INT8.mlmodelc`, `TinyJapaneseV42KVINT8.mlmodelc` (compiled with `coremlcompiler`), `VimeJapaneseTokenizerV4.model`, `VimeLMManifestV42.json` (format `vime_ios_lm_v4`, forbidden generation IDs 0/1/2/4/5/6, compiled-file hashes) and `VimeLMFixturesV42.json`. The fixtures reuse the Vime V2.1 template inputs read-only: 6,755 tokenization texts, 4 scoring sets with INT8 sums, 20 beam phrase results, and 12 single-token KV step sequences whose suffix sums equal the stateless sums within 1e-5. Resource names are new, so V1/V2 client resources are unaffected. Device runs in the Vime repository remain a later step.

## 7. Chunked KV graph

The single-token graph makes prefill cost one call per token (§5). `vimeml.deployment.v4_chunk.KVChunkGraphV4` exports the same weights for T new tokens per call; `KVStepGraphV4`, its packages and records are unchanged. Interface: `input_ids int32[1,T]` (1≤T≤128, dynamic), `position int32[1]` (current cache length; caller guarantees position+T≤128), `key_cache`/`value_cache float32(7,1,6,128,64)` → `logits float32[1,T,8192]`, `new_key_cache`, `new_value_cache`. Format `vimeml_coreml_v4_kv_chunk_v1`; iOS 18, FP32 compute, CPU_ONLY.

Cache writes use the `index_put` pattern of the V2.1 `kv_cache.CacheGraph` (14 `scatter_nd` ops after conversion). Conversion and every gate passed with it, so the one-hot matmul fallback was not needed. A static 128×128 causal mask gathered by absolute query position lets query p+i see only slots ≤ p+i, which also hides unused tail slots. Before export, the traced graph matches `model(ids)` and `forward_cached` (logits and both caches, 3e-4) for the schedules [1,3,124], [128], [3,1,124], [124,3,1] and [127,1], covering start 0, nonzero starts and the last slot.

| Package | Logical bytes |
| --- | ---: |
| `tiny-ja-v4.2-epoch20-kvchunk-fp32-v1` | 60,496,928 |
| `tiny-ja-v4.2-epoch20-kvchunk-int8-b32-v1` | 17,193,424 |

INT8 uses the same recipe and selection: 37 matrices; the causal mask and lm_head bias constant stay FP32.

| Check (atol=rtol=3e-4) | FP32 vs PyTorch fixtures | INT8 vs stateless INT8 | INT8 vs FP32 fixtures |
| --- | --- | --- | --- |
| One-shot prefill, five cases | max 5.2e-5 | 0 (bit-identical) | 0.477 (expected fail) |
| Chunks [1]×T, [3,T−3], half/half | max 5.2e-5; ≤3.8e-5 from one-shot | max 4.0e-5; ≤4.0e-5 from one-shot | 0.477 |
| [1]×T vs `long_kv_step_logits` | 2.9e-5 | — | 0.477 |
| T=1 decode vs single-token KV package | 0 (bit-identical) | 0 (bit-identical) | — |
| Shared-prefix branches [9], [47,81,102], [9] | max 2.3e-5 | max 2.9e-5 | 0.708 |
| `right_pad`/`changed_future` prefix, A/B/A rollback, dirty tail (999, T=1 and T=3), snapshot immutability | exact / unchanged | exact / unchanged | exact / unchanged |
| Gate | Passed (27 checks) | Passed (26 checks) | Failed (22/22), as for the other INT8 graphs |

Different chunkings of the same tokens are equal within tolerance, not bit-identical (up to 4.0e-5), because chunk size changes the matmul shapes.

Four-track scoring makes one chunk call for the exact joint-tokenized common prefix, then one chunk call per candidate suffix from that immutable snapshot.

| Track | Stateless FP32 / chunk FP32 | Stateless INT8 / chunk INT8 | Order differences FP32 / INT8 (all LM methods) | Max suffix-score delta FP32 / INT8 |
| --- | ---: | ---: | ---: | ---: |
| Standard development | 523 / 523 | 522 / 522 | 0 / 0 | 4.8e-5 / 4.1e-5 |
| Legacy regression | 370 / 370 | 371 / 371 | 0 / 0 | 3.9e-5 / 2.9e-5 |
| AJIMEE | 139 / 139 | 139 / 139 | 0 / 1 | 4.3e-5 / 4.2e-5 |
| Historical development | 123 / 123 | 123 / 123 | 0 / 0 | 2.3e-5 / 3.1e-5 |

The single INT8 difference is the same diagnostic `lm_no_context_sum` tie in `ajimee:1763` seen for the single-token INT8 graph (ranks 11/12, tied in stateless INT8, separated by about 2e-5); Top-1 is unchanged. The corresponding acceptance check is recorded as false.

Measurements rerun all ten packages with the unchanged harness (`measure_v42.py run`, now including the two chunk packages) into `v42-mac-measurements-v2`: five interleaved rounds, a fresh process per model per round, 5 warm-up and 50 timed calls (prefill: 10). Conditions: AC power (98–99%, charging), Low Power Mode off, no thermal or performance warning at any of six `pmset -g therm` checkpoints, 1-minute load average 3.8–4.8 (decaying after the evaluations; nothing else running). For chunk graphs the worker also makes one T=128 prefill right after load, before any other workload, to isolate its footprint. In this rerun, stateless calls and single-token decode are within about 0.35 ms of §5, and single-token prefill is about 5% lower (171 vs 181 ms at P=119).

| Model (warm p50 / p95, ms) | Prefill P=7 | P=31 | P=63 | P=119 | Decode after P=7 | P=31 | P=63 | P=119 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4.2 KV single-token FP32 (P calls) | 10.2 / 10.3 | 44.2 / 45.7 | 90.2 / 92.0 | 171.5 / 183.1 | 1.43 / 1.57 | 1.41 / 1.66 | 1.40 / 1.55 | 1.42 / 1.60 |
| V4.2 KV single-token INT8 (P calls) | 10.1 / 10.5 | 45.1 / 47.2 | 89.6 / 91.5 | 169.5 / 171.3 | 1.43 / 1.56 | 1.42 / 1.60 | 1.42 / 1.55 | 1.41 / 1.56 |
| V4.2 KV chunk FP32 (1 call) | 2.47 / 2.64 | 3.40 / 3.57 | 4.54 / 4.65 | 6.63 / 6.73 | 1.94 / 2.06 | 1.92 / 2.07 | 1.91 / 2.05 | 1.92 / 2.06 |
| V4.2 KV chunk INT8 (1 call) | 2.49 / 2.59 | 3.52 / 3.79 | 4.55 / 4.80 | 6.56 / 6.66 | 1.94 / 2.13 | 1.93 / 2.10 | 1.94 / 2.16 | 1.93 / 2.04 |
| V2.1 KV FP32 (1 call) | 2.20 / 2.28 | 3.01 / 3.10 | 4.14 / 4.21 | 6.02 / 6.13 | 1.56 / 1.75 | 1.53 / 1.70 | 1.52 / 1.70 | 1.51 / 1.62 |
| V2.1 KV INT8 (1 call) | 2.19 / 2.33 | 3.03 / 3.17 | 4.13 / 4.30 | 6.11 / 6.23 | 1.53 / 1.67 | 1.52 / 1.68 | 1.51 / 1.65 | 1.51 / 1.59 |

| Model | Package / compiled MB | Load, first process / p50 (ms) | Cold first T=128 prefill (ms) | Footprint Δ after T=128 chunk (MiB) | Peak footprint Δ (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| V4.2 KV chunk FP32 | 60.50 / 60.50 | 218.0 / 26.3 | 7.4 | 13.2 | 21.6 |
| V4.2 KV chunk INT8 | 17.19 / 17.20 | 224.0 / 25.8 | 7.4 | 13.2 | 21.4 |
| V4.2 KV single-token FP32 / INT8 | 60.42 / 17.12 | 206.1 / 21.5 · 145.3 / 21.3 | — | — | 16.4 / 17.9 |
| V2.1 KV FP32 | 50.38 / 50.39 | 239.5 / 25.5 | 7.4 | 19.7 | 29.2 |
| V2.1 KV INT8 | 14.36 / 14.36 | 165.2 / 26.3 | 7.4 | 19.7 | 29.6 |

One chunk call cuts V4.2 prefill at P=119 from about 170 ms to 6.6 ms (26×), within 1.1× of V2.1 chunk prefill. A T=1 call through the chunk graph costs about 1.9 ms, roughly 0.5 ms more than the single-token graph (1.4 ms) and 0.4 ms more than V2.1, although its outputs are bit-identical to the single-token package. A client can therefore prefill with the chunk graph and decode with either graph; switching graphs mid-request reuses caches that agree within 4e-5, not bit-exactly. The T=128 chunk raises footprint by 13.2 MiB over the post-import baseline, against 19.7 MiB for V2.1 with its 16K logits. Cold first decode calls show the same bimodal spread as §5 (first process 3.6–5.0 ms; medians 37–81 ms). Mac host-process numbers only.

Client resources v2 (`tiny-ja-v4.2-epoch20-client-resources-v2`; v1 unchanged) contain `TinyJapaneseV42INT8.mlmodelc`, `TinyJapaneseV42KVChunkINT8.mlmodelc`, the V4 tokenizer and manifest (`kv_graph: chunk`). The fixtures add 12 `kv_chunk_steps`: one chunk call for the common prefix, then single-token decode of each candidate suffix. Their suffix sums equal the stateless INT8 sums within 9.7e-6, inside the required 1e-5 but with little margin.

## 8. iOS simulator and iPhone measurements

Client integration lives on Vime branch `claude/v42-lm-measure` (separate worktree from `origin/main` `7e8bccc`; local commits only). `VimeLanguageModel` gains an explicit `Variant` (`.v21` default, `.v42` opt-in) that selects resource names, manifest format (`vime_ios_lm_v4`), vocabulary 8,192, and the V4 reserved conversion markers 4–6. Those markers are rejected in scoring/beam input and excluded from next-word and beam output, matching the Python V4 contract. Every production call site still constructs the default V2.1 model; only tests and the benchmark pass `.v42`. The client loads only the INT8 stateless model; the KV graphs are not integrated.

The first V4.2 simulator run failed in all five V4.2 tests: the native SentencePiece bridge hard-coded a 16,384-piece check, so `vime_sp_load` rejected the 8,192-piece V4 tokenizer. Both tokenizers otherwise share the same normalizer settings (identity, byte fallback). The bridge now takes the expected piece count per variant, and the check stays exact. `Scripts/install_v42_resources.py` installs the hash-verified V4.2 resources from `client-resources-v1`; `validate_model_resources.py` additionally verifies them when their manifest is present.

Functional checks (iPhone 18 Pro Max simulator, iOS 27.2, Debug): 5/5 new V4.2 tests pass. Native tokenization matches all 6,755 fixtures, joint suffix scores match Mac Core ML INT8 within 0.002 (with and without the conversion cache), and the first beam result matches all 20 Mac prefixes in token IDs and text (sums within 0.005). Literal `<CTX>` input is rejected and reserved IDs never appear in output. The 24 existing V2.1 model tests pass unchanged. The full `VimeKeyboardTests` suite on the branch executes 381 tests: 13 skipped (opt-in benchmark/memory scenarios) and 4 failed with 9 assertion failures (an intelligent-candidate profile assertion, a personal-recall latency budget, a settings-layout width and one segmented-conversion orthography expectation). The same four tests fail with the same nine assertions on a clean `origin/main` `7e8bccc` worktree, so they predate this change; they were not modified.

Simulator benchmark: `VimeLanguageModelBenchmarkTests` (unchanged workload: fixed-length `predict`, 20 dictionary rerank cases × 5, 10 prompts × 3 beam and next-word requests), optimized `-O` build, `CPU_ONLY`, one variant per test process, variants alternating over four rounds (`Scripts/benchmark_lm_variants.py`). The simulator reported thermal state `nominal` and Low Power Mode off at the start and end of every run; the Mac was on AC power with no thermal warning; the 1-minute host load average fell from 8.8 (just after the build) to 5.9. Medians over the four processes of each process's p50 / p95:

| Metric (ms unless noted) | V2.1 INT8 | V4.2 INT8 |
| --- | ---: | ---: |
| First load in process / reload | 243.6 / 73.3 | 267.8 / 33.0 |
| First predict | 11.7 | 12.7 |
| `predict` T=1 / 8 / 32 / 128 | 1.91 / 6.21 / 12.66 / 35.83 | 2.33 / 6.73 / 14.24 / 40.76 |
| `predict` T=128 p95 | 36.3 | 41.0 |
| Rerank scoring (`scores`, 90 calls) p50 / p95 / max | 54.1 / 438.7 / 454.7 | 61.5 / 478.2 / 489.9 |
| Beam suggestions p50 / p95 | 427.6 / 469.6 | 452.4 / 495.0 |
| Next words p50 / p95 | 56.8 / 78.7 | 59.8 / 77.1 |
| Log-softmax row | 0.10 | 0.06 |
| Footprint before / after load / peak (MB) | 29.8 / 31.9 / 35.5 | 30.0 / 31.4 / 35.7 |

The rerank workload includes pools of up to 93 candidates; 10 of 90 direct `scores` calls exceed 250 ms for both variants. The keyboard's `rerank` path applies its own 250 ms soft budget, which this direct call does not. In this simulator workload V4.2 runs about 5–14% slower than V2.1 for T≥8 and end-to-end requests (22% at T=1), while its smaller vocabulary halves the log-softmax row cost; footprint is within 0.5 MB. An earlier three-round simulator run (`sim-benchmark`, before the thermal-state fields were added) gave the same picture. Simulator numbers run on the Mac CPU and are not iPhone measurements.

iPhone 16 Pro Max (iPhone17,2, iOS 27.2 build 24B115, wired USB, connected to the Mac and charging), the same optimized signed build: the 29 model functional tests pass on the device, including all five V4.2 tests (6,755 tokenization fixtures, Mac INT8 scores within 0.002, 20 beam prefixes). The benchmark ran in four alternating rounds with one variant per test-host process. The device reported thermal state `fair` (not `nominal`) at the start and end of every run, and Low Power Mode off. Numbers therefore reflect a warm, charging phone. Medians over four processes:

| Metric (ms unless noted) | V2.1 INT8 | V4.2 INT8 |
| --- | ---: | ---: |
| First load in process / reload | 192.5 / 65.9 | 175.0 / 28.0 |
| First predict | 3.70 | 4.36 |
| `predict` T=1 / 8 / 32 / 64 / 128, p50 | 1.03 / 1.50 / 2.04 / 2.95 / 4.55 | 1.28 / 1.69 / 2.39 / 3.25 / 4.90 |
| `predict` T=128 p95 | 4.68 | 4.99 |
| Rerank scoring (`scores`, 90 calls) p50 / p95 / max | 21.3 / 136.8 / 139.8 | 25.1 / 120.9 / 122.6 |
| Calls over 250 ms | 0 of 90 | 0 of 90 |
| Beam suggestions p50 / p95 | 101.8 / 106.8 | 108.9 / 113.2 |
| Next words p50 / p95 | 13.2 / 18.3 | 16.1 / 20.1 |
| Log-softmax row | 0.09 | 0.04 |
| Footprint before / after load / after 50 predicts (MB) | 25.5 / 36.0 / 40.0 | 25.4 / 34.7 / 36.7 |
| Footprint peak (ledger, MB) | 49.95 | 44.35 |

On the device, V4.2 costs 0.16–0.35 ms more per `predict` call than V2.1 (8–24% depending on length); beam and next-word requests are 7% and 21% slower at p50. Rerank scoring is slower at the median (25.1 vs 21.3 ms) but faster at the tail (p95 120.9 vs 136.8 ms), where the 16K-vocabulary normalizer dominates long candidate pools. V4.2 reloads 2.4× faster and peaks 5.6 MB lower in this test host. Spread between processes is small (T=128 p50 within 0.2 ms for each variant). This is the XCTest host process, not the keyboard extension; extension memory limits, typing latency in third-party hosts, cold device boot and battery use were not measured, and the earlier V2.1 keyboard-session report is a different workload.

## Records

Reports are under ignored `outputs/deployment/v42-*` (baseline, alignments, four-track scores with per-case comparisons, measurements, and `v42-mac-acceptance-v1/status.json` and the superseding `v42-mac-acceptance-v2/status.json` with every pass/fail check, including the expected INT8 strict failures and the INT8 KV ties) and logs under `outputs/deployment/logs/`. Packages are under `artifacts/deployment/tiny-ja-v4.2-epoch20-*-v1`. The results ZIP carries these with `handoff.json`; source changes stay on the branch.
