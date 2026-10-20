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
| 6. iOS resources | Prepared; no device run |

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

## Records

Reports are under ignored `outputs/deployment/v42-*` (baseline, alignments, four-track scores with per-case comparisons, measurements, and `v42-mac-acceptance-v1/status.json` with every pass/fail check, including the expected INT8 strict failure and the KV INT8 tie) and logs under `outputs/deployment/logs/`. Packages are under `artifacts/deployment/tiny-ja-v4.2-epoch20-*-v1`. The results ZIP carries these with `handoff.json`; source changes stay on the branch.
