# vimeml-tiny-zh-v1 batched single-step KV decoding on Core ML — 2026-11-15

This follows the [handoff brief](../../reference/mac-zh-v1-kvbatch.md). The checkout was created with `setup_mac.py` from `vimeml-zh-kvbatch-mac-20261115.zip` at `main` `a8440c5` (recorded commit verified), with every payload file checked against `handoff-manifest.json`. Source model `vimeml-tiny-zh-v1` (checkpoint `44c423a1…`, tokenizer `68bd37c4…`, bundle manifest `9d9dad19…`). Not changed: checkpoint, tokenizer, INT8 recipe (INT8 block32, weight threshold 2048, same parameter selection), search rules. `RELEASE.json` and `SHA256SUMS.txt` are untouched (SHA256 still `899ca51a…` / `58096d45…`). No Vime-side code was written or committed.

Environment: Apple M3, 16 GiB, macOS 27.3 (26C5035r), Xcode 27.3 (27C5020n), Python 3.11.17 (uv venv), PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5, scikit-learn 1.5.1, sentencepiece 0.2.1, psutil. coremltools warns that PyTorch 2.7.1 is untested (2.7.0 is); no effect was seen. Core ML targets: mlprogram, iOS 18, FP32 compute, `CPU_ONLY` unless stated. AC power (battery 100%, charged), `pmset -g therm` recorded no thermal or performance warning, load average 2.9–4.1.

All outputs are under `outputs/deployment/zh-v1-kvbatch/`; stage logs are in `logs/`, exploratory helper scripts in `tools/`. No output was overwritten. One shell mistake (an unsplit `$P` variable in zsh) made the first step-4 loop run nothing; its logs are kept in `logs/shell-typo/` and the loop was rerun.

**`src/vimeml/deployment/zh_v1_batch.py` was not changed.** Conversion of the dynamic slot count and the one-hot write worked as written, so the "fix and rerun `check`" path was never needed.

## Summary

| Step | Result |
| --- | --- |
| 1. PyTorch check | Passed: eager and traced, 4 cases each, max 5.01e-5 (Windows 3.96e-5) |
| 2. FP32 flex | Converted; vs PyTorch FP32 max 5.05e-5, 507 rows, all argmax equal — passed |
| 3. INT8 flex | Vime gate vs stateless INT8 **passed**: max 4.82e-5 (limit 3e-4) |
| 4. Fixed B=16 / B=8 | FP32 5.05e-5 / 4.58e-5; INT8 vs stateless 4.82e-5 / 3.58e-5 — all passed, including padding rows |
| 5. FP16 cache | Strict gate **failed as expected** (FP32 max 7.0e-3, INT8 max 6.2e-3), argmax 507/507; `compress` refused; INT8 built by a separate evidence-only tool. No speed gain on CPU, no ANE gain |
| 6. Merged package | Both functions load and match the separate packages bit for bit; weights stored once (22.96 MB vs 45.59 MB); Mac footprint saving about 10 MiB, not a full weight copy |
| 7. Mac latency | INT8 flex B=16: 3.35 ms (S=16), 3.76 ms (S=32); B=8: 2.69 / 2.92 ms. Fixed 16 is no faster at B=16 and pays ~3.3 ms for B=1 |
| 8. Vime fixture | 16 rows, positions 1–9, one decode call: top-5 identical across package, stateless INT8 and PyTorch FP32; max 2.48e-5 vs stateless |
| 9. Search check | FP32 (FP32 prefill): 125/125 exact order, max 3.8e-5. INT8: identical to a stateless-INT8-only search (125/125); vs the FP32 fixture top-1 125/125, same text set 86/125 — see §9 |
| 10. Client resource | Built from **INT8 flex** (`ios-kvbatch/`) |
| 11. iPhone | **Not done** (no batched test host in Vime; not built from this session) |
| 12. Release copy | `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch/` = INT8 flex, identical bytes |

## Interface and the two deliberate differences

`vimeml.deployment.zh_v1_batch.KVBatchGraphZh`, one call advances B independent hypotheses by one token:

| Name | Type | Meaning |
| --- | --- | --- |
| `input_ids` | int32 `[B,1]` | one new token per row |
| `position` | int32 `[B]` | row cache length before this token; `position[b] < S` |
| `key_cache`, `value_cache` | float32 (float16 in the exploratory variant) `(9,B,6,S,64)` | row b's slots `>= position[b]` are ignored; junk there is masked (checked with ±30 fill) |
| → `logits` | float32 `[B,12288]` | the new token only; no softmax |
| → `new_key`, `new_value` | `(9,B,6,64)` | the new token's key/value; the caller writes them into slot `position[b]` of row b |

Rows never attend across each other; padding rows (token 0 at position 0) do not change real rows (checked in the fixed-16 alignment: 15 and 11 padding rows, and in fixed 8: 7 and 3).

1. **`S` is flexible (1..128) instead of a fixed 128.** The caller passes `max(position)+1` slots. A fixed 128-slot cache for B=16 is 28 MiB per cache, so about 56 MiB of input per call, while real rows are a short prompt plus about 20 tokens. Measured cost of S on Mac (INT8 flex, B=16): 3.35 ms at S=16, 3.76 at S=32, 5.45 at S=64, 8.73 at S=128 — the fixed-128 contract would roughly double the step.
2. **Only the new slot is returned, not full caches.** Returning two full caches would write as much as the inputs every call, more than the step itself. With caller-owned buffers only the `(9,B,6,64)` slot is copied; Vime can still back the inputs with its own buffers (`outputBackings` for the small outputs).

The prompt stays on the shipped batch-1 chunk package (or `prefill` of the merged package): run once, copy its cache rows into the batch buffer. MLState is not used (a bucket gathers rows whose parents sit in different earlier buckets).

## 1. PyTorch check

`check.json`. 16 rows of 3–25 tokens starting at different steps (positions differ within every call); each step's logits vs the row's own full forward.

| Case | Calls | Rows | Max abs (eager = traced) |
| --- | ---: | ---: | ---: |
| B=16, S=max(position)+1 | 30 | 221 | 4.10e-5 |
| B=1 | 3 | 3 | 2.19e-5 |
| B=5, S=128 (junk-free full buffer) | 30 | 62 | 3.91e-5 |
| B=16, seed 2026 | 30 | 221 | 5.01e-5 |

## 2–4. Conversion, compression and alignment

INT8 recipe: 47 matrices quantized (token embedding shared by gather and tied head, position embedding, 9 × QKV / projection / gate / up / down); 1 structural constant excluded (the zero head bias; this graph has no causal-mask constant). FP32 compute, no activation quantization.

| Package | Batch | Logical bytes |
| --- | --- | ---: |
| `fp32-flex` | 1..16 (`RangeDim`) | 80,337,051 |
| `fp32-b16` / `fp32-b8` | fixed | 80,337,661 |
| `int8-flex` | 1..16 | 22,762,761 |
| `int8-b16` / `int8-b8` | fixed | 22,763,413 |

Max abs error per alignment case (tolerance atol = rtol = 3e-4; every case 0 rows outside, every row argmax-equal):

| Package | Reference | B=16 | B=1 | B=5, S=128 | B=16 seed 2026 | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `fp32-flex` | PyTorch FP32 | 5.05e-5 | 2.57e-5 | 4.20e-5 | 4.72e-5 | passed |
| `fp32-b16` | PyTorch FP32 | 5.05e-5 | 2.96e-5 (+15 pad) | 4.58e-5 (+11 pad) | 4.72e-5 | passed |
| `fp32-b8` | PyTorch FP32 | — | 2.96e-5 (+7 pad) | 4.58e-5 (+3 pad) | — | passed |
| `int8-flex` | stateless INT8 | 4.24e-5 | 7.2e-6 | 3.43e-5 | 4.82e-5 | **passed** |
| `int8-b16` | stateless INT8 | 4.24e-5 | 1.72e-5 (+15 pad) | 3.58e-5 (+11 pad) | 4.82e-5 | **passed** |
| `int8-b8` | stateless INT8 | — | 1.72e-5 (+7 pad) | 3.58e-5 (+3 pad) | — | **passed** |

The B=8 alignment skips the 16-row cases, as the brief says. The chunk-KV INT8 package was within 4.2e-5 of stateless INT8; the batch packages are within 4.8e-5.

## 5. FP16 cache (exploratory, evidence only)

`fp32-flex-fp16cache` converted with `--cache-dtype float16`. Its strict alignment against PyTorch FP32 **failed** (`fp32-flex-fp16cache-align.json`, kept). `compress` then refused it ("First pass alignment for this exact FP32 batch package"; `logs/05-compress-int8-flex-fp16cache.log`). To still measure an INT8 FP16-cache package, `tools/compress_evidence_only.py` applies the identical recipe but records the failed gate instead of enforcing it; the result `int8-flex-fp16cache` carries `"evidence_only": true` in its manifest. It was not used for the resource or the release (the resource script also requires a passed stateless gate). Its alignment vs stateless INT8 also failed (`int8-flex-fp16cache-align.json`).

Error distribution (`tools/error_distribution.py`; same rows as the alignment, every logit):

| Package / reference | Case | Max | Mean | RMSE | p50 | p90 | p99 | p99.9 | Logits > 3e-4 | Rows outside | Argmax | Top-5 set |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 FP16-cache / PyTorch | B=16 | 4.38e-3 | 4.5e-4 | 6.5e-4 | 3.2e-4 | 1.04e-3 | 2.20e-3 | 3.19e-3 | 52% | 186/221 | 221/221 | 220/221 |
| | B=1 | 6.7e-4 | 1.2e-4 | 1.7e-4 | 8.9e-5 | 3.0e-4 | 4.4e-4 | 5.5e-4 | 10% | 2/3 | 3/3 | 3/3 |
| | B=5, S=128 | 5.49e-3 | 5.5e-4 | 8.5e-4 | 3.7e-4 | 1.23e-3 | 3.44e-3 | 4.49e-3 | 56% | 53/62 | 62/62 | 62/62 |
| | B=16 seed 2026 | 7.03e-3 | 5.3e-4 | 7.3e-4 | 4.0e-4 | 1.16e-3 | 2.29e-3 | 3.75e-3 | 60% | 199/221 | 221/221 | 221/221 |
| INT8 FP16-cache / stateless INT8 | B=16 | 6.17e-3 | 5.3e-4 | 7.7e-4 | 3.5e-4 | 1.27e-3 | 2.59e-3 | 3.77e-3 | 55% | 195/221 | 221/221 | 221/221 |
| | B=1 | 9.2e-4 | 1.7e-4 | 2.4e-4 | 1.5e-4 | 4.1e-4 | 5.7e-4 | 6.9e-4 | 25% | 2/3 | 3/3 | 3/3 |
| | B=5, S=128 | 3.05e-3 | 4.2e-4 | 5.7e-4 | 3.2e-4 | 9.8e-4 | 1.60e-3 | 2.02e-3 | 52% | 52/62 | 62/62 | 62/62 |
| | B=16 seed 2026 | 5.71e-3 | 5.3e-4 | 7.6e-4 | 3.9e-4 | 1.19e-3 | 2.63e-3 | 3.99e-3 | 59% | 197/221 | 221/221 | 221/221 |

The error is small relative to INT8's own (~0.4) but fails 3e-4 everywhere. Speed (INT8 FP16-cache, p50 ms, single run of the full grid, stateless loaded with the same compute units):

| B, S | INT8 flex FP32 cache, CPU_ONLY | FP16 cache CPU_ONLY | FP16 cache CPU_AND_NE | FP16 cache ALL |
| --- | ---: | ---: | ---: | ---: |
| 1, 16 | 1.90 | 1.67 | 1.75 | 2.56 |
| 4, 32 | 3.08 | 3.12 | 3.05 | 5.43 |
| 8, 32 | 3.51 | 3.97 | 3.78 | 4.21 |
| 16, 16 | 3.88 | 4.14 | 4.03 | 3.76 |
| 16, 32 | 4.22 | 5.60 | 5.24 | 5.01 |
| 16, 64 | 5.45 | 8.35 | 8.12 | 7.71 |
| 16, 128 | 8.73 | 14.37 | 13.69 | 12.86 |

FP16 caches do not help on CPU: the inputs are half the bytes, but the FP32-compute graph casts them up, and from B=16, S=32 it is slower. `CPU_AND_NE` is within noise of `CPU_ONLY` (an FP32-compute program does not run on the ANE). `ALL` is noisy and no better; loading under `ALL` also raised process RSS by ~450 MiB more than `CPU_ONLY`. This matches what Vime saw with the chunk package. Recommendation: keep FP32 caches and `CPU_ONLY`.

## 6. Merged package

`merge int8-flex int8-merged`, chunk = shipped `ios18-int8-block32-kvchunk`. One `weight.bin` (22,694,592 bytes) is shared by `prefill` and `decode`.

| | Bytes |
| --- | ---: |
| `int8-merged` package | 22,962,601 |
| shipped chunk package + `int8-flex` | 22,830,199 + 22,762,761 = 45,592,960 |

`tools/merged_memory.py` loads, in a fresh process each (3 runs per configuration): the stateless INT8 package (run once), then either the two separate packages or both functions of the merged package (`function_name="prefill"` / `"decode"`), and runs one prefill and one B=16 decode. Both functions load; their outputs equal the separate packages exactly (max abs 0 on prefill logits/keys and decode logits/keys).

| Mac, Python, CPU_ONLY (median of 3) | Separate | Merged |
| --- | ---: | ---: |
| Stateless INT8 load + run, phys_footprint delta | 15.7 MiB | 15.2 MiB |
| + load both + prefill, phys_footprint delta | 63.9 MiB | 54.4 MiB |
| + B=16 decode | 2.4 MiB | 2.4 MiB |
| Total added over stateless, phys_footprint | **66.2 MiB** | **56.8 MiB** |
| Total added over stateless, RSS | 222.2 MiB | 230.7 MiB |

The bytes on disk halve, but on the Mac the footprint saving is ~9.5 MiB, less than one weight copy (21.6 MiB); most of the load cost is something other than the weight file (per-function compiled program state and buffers, plus the Python/coremltools compile path). RSS, which includes the mapped weight file, is slightly higher merged. So "one copy instead of two" holds for the package, but only partly for memory on the Mac; it has to be confirmed on the device (`phys_footprint` of the keyboard extension), which was not done (§11).

## 7. Latency on the Mac

`measure`, 200 repeats after 20 warm-ups, stateless INT8 loaded and run alongside, `CPU_ONLY`. Times are Python `predict()` wall time including input conversion. The first full grid (`latency/*-latency.json`) showed `int8-merged` 10–20% faster than `int8-flex`, though its `decode` function is the same graph, so single runs carry that much noise. The packages were therefore measured again three times interleaved at the S values a real bucket uses (`latency/*-rerun{1,2,3}-latency.json`); the table gives the median of the three p50 / p95.

| B, S | INT8 flex | INT8 fixed 16 | INT8 fixed 8 | merged `decode` |
| --- | ---: | ---: | ---: | ---: |
| 1, 16 | 1.64 / 1.98 | 3.31 / 3.73 | 2.65 / 3.04 | 1.67 / 1.95 |
| 1, 32 | 1.80 / 2.12 | 3.86 / 4.18 | 2.94 / 3.28 | 1.77 / 2.13 |
| 4, 16 | 2.30 / 2.94 | 3.27 / 3.75 | 2.43 / 2.94 | 2.48 / 3.08 |
| 4, 32 | 2.47 / 3.11 | 3.84 / 4.17 | 2.83 / 3.35 | 2.66 / 3.21 |
| 8, 16 | 2.69 / 3.28 | 3.19 / 4.24 | 2.65 / 3.04 | 2.87 / 3.30 |
| 8, 32 | 2.92 / 3.54 | 3.63 / 4.68 | 2.89 / 3.33 | 2.79 / 3.44 |
| 16, 16 | 3.35 / 3.72 | 3.34 / 3.77 | — | 3.25 / 3.73 |
| 16, 32 | 3.76 / 4.25 | 3.67 / 4.18 | — | 3.89 / 4.24 |

Full grid, single run (p50 ms), for larger S:

| B, S | INT8 flex | fixed 16 | fixed 8 | merged |
| --- | ---: | ---: | ---: | ---: |
| 16, 64 | 5.45 | 4.24 | — | 4.63 |
| 16, 128 | 8.73 | 6.66 | — | 7.05 |
| 8, 64 | 3.94 | 5.44 | 3.13 | 3.41 |
| 8, 128 | 5.15 | 8.59 | 4.22 | 4.77 |

- Padding cost of the fixed packages: a B=1 row in a fixed-16 call costs the full 16-row step (3.3–3.9 ms at S=16/32, vs 1.6–1.8 ms in flex); in fixed 8, B=1 costs 2.7–2.9 ms.
- At B=16 and real S, flex and fixed 16 are equal within noise. Fixed 16 may be faster at S≥64 (single run), which real buckets do not reach.
- Mac targets: B=16 ≤ 8 ms and B=8 ≤ 5 ms hold with a wide margin. These are Mac numbers; the iPhone 13 Pro was 3.8 ms per batch-1 chunk predict in Vime, against ~1.7 ms here per B=1 step, so device times are expected to be roughly twice the Mac's, still inside the targets, but this is not measured.
- Script note: `measure` also runs `rows` larger than a fixed package's batch (e.g. "rows 16" on fixed 8), timed as an 8-row call and mislabelled. Those rows are excluded above; they remain in the JSON.

## 8. Vime fixture

`fixture int8-flex fixture` → `fixture/batch16.jsonl`, `batch16_logits.npz`, `meta.json`. Sixteen hand-written sentence prefixes (not benchmark items) at positions 1–9 (8 distinct positions, `slots_needed` 10); each is prefilled with the shipped chunk package and all are advanced in one decode call.

- Package vs stateless INT8 (full forward): max 2.48e-5.
- PyTorch batch graph vs PyTorch full forward: max 2.48e-5.
- Per-row top-5 ids: package = stateless INT8 = PyTorch FP32 in order, for all 16 rows.

## 9. Search check

`search` with the unchanged per-key-position beam search (`zh_search.search`, beam 16, top 10, lm mode) on the 125-case `zh-generation-v1-lm-r2` fixture. Per generation: mean 14.48 decode calls and 231.68 rows (16.0 rows per call). An unbatched runtime makes 231.68 batch-1 predicts per generation; batching turns them into 14.48 calls.

| Run | Prefill | Exact order | Same text set | Max score diff (stage, by rank) | Mac p50 / p95 per generation |
| --- | --- | ---: | ---: | ---: | ---: |
| `search-r2` (INT8 flex) | shipped INT8 chunk | 46 | 86 | 3.48 | 87.5 / 214.6 ms |
| `search-r2-fp32-flex-int8chunk` (as the brief's command runs it) | shipped INT8 chunk | 90 | 110 | 2.18 | 85.8 / 207.1 ms |
| `search-r2-fp32-flex-fp32chunk` | FP32 chunk `tiny-zh-v1-kvchunk-fp32-v1` | **125** | **125** | **3.8e-5** | 85.4 / 191.0 ms |

- The `search` stage defaults `--chunk` to the INT8 chunk package, so the "FP32 flex" run of the brief still prefills the prompt in INT8 and cannot be exact. The fully FP32 run used `--chunk` pointing at the FP32 chunk package from the 2026-11-11 zh-v1 work (`~/Documents/Sources/VimeML-zh-v1-coreml/artifacts/deployment/tiny-zh-v1-kvchunk-fp32-v1`, same bundle manifest `9d9dad19…`, read in place, not copied). It reproduces the fixture exactly: order exact, scores within 3.8e-5 (limit 1e-3). Batched graph, Core ML conversion and search are therefore correct.
- The stage's "max score diff" compares scores by rank. Once a text drops out of the list, the ranks shift, so this overstates the score error.

**INT8 attribution** (`tools/search_int8_attribution.py` → `search-r2-int8-attribution/`). The same search was run with a backend that never uses the batch or chunk packages: every row is one full forward of the shipped stateless INT8 package.

| Comparison | Exact order | Same text set | Top-1 equal | Max score diff on common texts (median per case) | Cases ≤ 0.5 | Dropped texts (reference rank) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| batch INT8 vs stateless-INT8 search | 125 | 125 | 125 | 2.2e-5 | 125 | none |
| stateless INT8 vs FP32 fixture | 46 | 86 | 125 | 0.565 (0.128) | 124 | 48: rank 10 ×24, 9 ×7, 8 ×5, 7 ×2, 6 ×4, 4 ×3, 3 ×2, 2 ×1 |
| batch INT8 vs FP32 fixture | 46 | 86 | 125 | 0.565 (0.128) | 124 | same 48 |

The batch INT8 package gives exactly the search result of the shipped stateless INT8 package. The difference from the FP32 fixture is INT8 itself. Top-1 never changes, and scores of shared texts stay within ~0.5 (one case at 0.565). The text set changes in 39 of 125 cases, mostly at ranks 9–10. Seven drops at ranks 2–4 come from the beam keeping a different hypothesis earlier, not from a swap. `meta.json` expects "the same text set" for on-device INT8. That holds for 86/125 cases with this INT8 recipe, for any INT8 runtime, batched or not. Vime should compare INT8 to INT8 (`search-r2-int8-attribution` or the fixture's INT8 rows), or accept set differences at the tail.

## 10. Client resource

Chosen: **INT8 flex**. The brief's rule is flex unless clearly slower than fixed 16. At B=16 and real S, flex and fixed 16 are equal within noise. At smaller buckets flex is up to 2× faster, while fixed 16 always pays the full 16-row step. One package covers every B from 1 to 16.

`package_zh_v1_kvbatch_ios.py --batch int8-flex --alignment int8-flex-align.json --search search-r2 --fixture fixture --output ios-kvbatch`:

- `ios-kvbatch/Resources/TinyChineseV1KVBatchINT8.mlmodelc` (compiled with Xcode 27.3 `coremlcompiler`, 22 MB)
- `ios-kvbatch/Resources/VimeLMManifestZhV1KVBatch.json`: `kv_batch` section with resource, compiled file SHA256s, interface, cache shape/slots/dtype, `batch_range` [1,16], alignment tolerance/reference/checks, search check, policy
- `ios-kvbatch/KVBatchFixture-*` (the step-8 fixture)

Existing zh-v1 resources (`TinyChineseV1INT8`, `TinyChineseV1KVChunkINT8`, `VimeLMManifestZhV1.json`) are not touched. The manifest's `search_check` carries the stage summary, including the rank-aligned 3.48. The text-aligned numbers are in §9.

## 11. iPhone

**Not done.** Both test phones (iPhone 13 Pro, iPhone 16 Pro Max) were paired and available, but Vime has no batched-KV test host. A per-predict and footprint harness would mean new client code, signed and installed from this session, which the brief leaves to Vime. Still open for the device: B = 1/4/8/16 at S = 16/32 on both phones, the keyboard-extension footprint added on top of today's 43 MB with the stateless package loaded, and whether the merged package's memory saving is larger on device than the Mac's ~9.5 MiB.

## 12. Release copy

`artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch/` = `outputs/deployment/zh-v1-kvbatch/int8-flex/` (`diff -r` identical). `manifest.json` SHA256 `472d148a…`, `weight.bin` `7fa1b5c8…`, `model.mlmodel` `0e8c9e70…`. `RELEASE.json` / `SHA256SUMS.txt` are left for the Windows side.

**Windows follow-up (2026-11-15).** The results ZIP was checked against its SHA256SUMS (159 files) and the release copy against `int8-flex` and its package manifest. The release inventory now lists the four `ios18-int8-block32-kvbatch/` files and `coreml_manifest_sha256.kvbatch_int8` = `472d148a…`; the README describes the batched package. `verify_release.py --hashes` passes on 370 files. The Hugging Face release was updated in place at revision `ce35c931e0102248e484fd93c36100f430067ead` (parent `8cc5ef3a…`); a fresh download of that revision passes `verify_release.py --hashes`. Two script issues from this report were fixed in `coreml_zh_v1_batch.py`: `measure` no longer runs more rows than a fixed package's batch, and `search` also reports Top-1 agreement, the score difference on texts present in both lists, and the dropped texts. `--chunk` is documented as needing an FP32 chunk package for an exact FP32 run.

## Memory and footprint

- Stateless INT8 alone (Mac, Python): +15.6 MiB phys_footprint.
- Batch package on top: one B=16 decode adds 2.4 MiB beyond loading. With the shipped chunk package, chunk + batch add 66 MiB (separate) or 57 MiB (merged) phys_footprint on the Mac. These are Python host-process numbers, inflated by coremltools' load path. They are not the keyboard-extension footprint, which remains to be measured (§11).
- Per call, the caller buffers at B=16 are 2 × 9·16·6·S·64·4 bytes = 0.84 MiB at S=16 and 1.69 MiB at S=32 (input), plus 48 KiB logits and 2 × 216 KiB new slots (output).

## Allowed-token logits and MLState

Not built, as the brief says. Log-softmax needs all 12,288 logits, so returning only allowed tokens cannot skip the head matmul. The `[B,12288]` output is 768 KiB at B=16. Whether its copy and the client softmax matter should be judged from the device numbers; nothing on the Mac suggests it dominates. MLState stays unused.

## Files

| Path | Contents |
| --- | --- |
| `check.json` | step 1 |
| `fp32-{flex,b16,b8}`, `*-align.json` | FP32 packages and their alignment vs PyTorch |
| `int8-{flex,b16,b8}`, `*-align.json` | INT8 packages and their gates vs stateless INT8 |
| `fp32-flex-fp16cache`, `int8-flex-fp16cache` (evidence only), `*-align.json`, `*-errors.json` | step 5 |
| `int8-merged`, `merged-memory/` | step 6 |
| `latency/` | step 7 (full grids and three interleaved reruns) |
| `fixture/` | step 8 |
| `search-r2*`, `search-r2-int8-attribution/` | step 9 |
| `ios-kvbatch/` | step 10 |
| `tools/` | `compress_evidence_only.py`, `error_distribution.py`, `merged_memory.py`, `search_int8_attribution.py` |
| `logs/` | every stage's console output; `shell-typo/` holds the empty first step-4 run |
