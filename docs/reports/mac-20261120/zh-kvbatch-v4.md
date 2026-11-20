# vimeml-tiny-zh batched KV v4 (`prefix` function, FP16 KV) on Core ML — 2026-11-20

This follows the [v4 brief](../../reference/mac-zh-kvbatch-v4.md) on a fresh checkout of `Voltline/VimeML` at `main` `b3d83d4` with `kvbatch4-handoff.zip` (SHA256 `6b454c8ed84c1e2289e4cf88e176da6f417cd5cd763dd35bc1009377600748a0`, all 16 files verified against `SHA256SUMS`).

**Unchanged**

- The checkpoint, tokenizer, INT8 recipe (`linear_symmetric`, per-block 32) and scoring rule.
- `src/vimeml/deployment/zh_v1_batch4.py` and `scripts/deployment/coreml_zh_kvbatch4.py`: no code changes were needed (no patch).
- Nothing was committed or pushed on the Mac.

**Environment**

- OS & Toolchain: macOS 27.3 (`26C5047e`, arm64), Xcode 27.3 (`27C5035i`).
- Python: CPython 3.11.17 in a `uv` venv, with PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5, and sentencepiece 0.2.1. coremltools warns that torch 2.7.1 is untested; no stage was affected.
- Core ML: mlprogram, iOS 18 target, `CPU_ONLY`.

All outputs are under `outputs/deployment/zh-kvbatch-v4/` (`$O`), with build logs at `outputs/deployment/zh-kvbatch-v4-build-<variant>.log`. No output was overwritten.

## Summary

- **Unit tests:** `python -m unittest tests.test_deployment_zh_kvbatch4 tests.test_deployment_zh_v1_batch3` passed (10/10 in 11.57 s).
- **Stage 1 (`check`):** passed (`eager_fp32kv` max `4.10e-5`, `traced_fp32kv` max `4.10e-5`, `eager_fp16kv` max `3.70e-3`).
- **Stage 2 (`build`):**
  - `fp32kv`: passed source (`4.20e-5`) and INT8 stateless alignment (`4.15e-5`, 0 rows outside tolerance). Merged INT8 package: **23,276,592 bytes** (23.27 MB, vs 89.70 MB separate).
  - `fp16kv`: passed source (`3.63e-3`) and INT8 stateless alignment (`3.23e-3`, score sum `1.06e-3`, 0 rows outside tolerance). Merged INT8 package: **23,280,397 bytes** (23.28 MB, vs 89.70 MB separate).
  - `fp16kv-fp16c`: converted cleanly, but **failed source alignment** (`logits` max `0.324` > `0.1`, 25 rows outside tolerance; `score` sum max `0.259` > `0.25`, 1 row outside tolerance; `prefix` max `0.149`, 2 rows outside tolerance). Stopped per decision rules; INT8 not built; alignment JSON and build log kept as evidence.
- **Stage 3 (`loadorder`, Mac reference):**
  - Mac does **not** reproduce the iPhone `+27.3 MB` steady jump on `score>scored` (all load-only deltas on Mac are `+0.9–1.5 MiB`). Final decision for `prefix` depends on the iPhone load-order run.
  - Under the 10-round workload on Mac, `fp16kv` retains `+36.2 MiB` (`peak 49.7 MiB`) when `scored` runs, compared with `+3.4 MiB` (`peak 16.9 MiB`) for `fp32kv`.
- **Stage 4 (`measure`, Mac latency):**
  - The `prefix` function speeds up `prompt_prefix` (48 tokens) by **19.1%** (`5.15 → 4.16 ms`) and `extend_prefix_1` by **19.4%** (`2.07 → 1.67 ms`) for `fp32kv` vs `v3`.
  - `fp16kv` is slower on `scored_step_b16` (`5.24 ms` vs `3.99 ms` for `fp32kv`).
- **Stage 5 (`search`, 125 cases):**
  - `fp32kv` vs `v3`: **125/125** exact order, **125/125** top-1, max common score diff `0.0`, p50 `64.51 ms`.
  - `fp16kv` vs `fp32kv`: **125/125** exact order, **125/125** top-1, max common score diff `3.19e-3`, p50 `66.55 ms`.
- **Stage 6 (`fixture`):**
  - `fp32kv`: step vs stateless max `2.86e-5`, score sums max `1.70e-5`, same order **8/8**.
  - `fp16kv`: step vs stateless max `9.09e-4`, score sums max `8.06e-4`, same order **8/8**.
- **Stage 7 (`ios`):** packaged `ios-fp32kv` (`TinyChineseKVBatchV4FP32KVINT8.mlmodelc`) and `ios-fp16kv` (`TinyChineseKVBatchV4FP16KVINT8.mlmodelc`) for on-device iPhone testing.

## 1. PyTorch check (`torch-check.json`)

| Suite | Max abs error | Rows outside tolerance |
| --- | ---: | ---: |
| `eager_fp32kv` | `4.10e-5` | 0 |
| `traced_fp32kv` | `4.10e-5` | 0 |
| `eager_fp16kv` | `3.70e-3` | 0 |

## 2. Build, package sizes, and alignment

### Package sizes (bytes, directory totals)

| Variant | Stage | Merged | Sum of 4 separate packages | `logits` | `scored` | `score` | `prefix` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp32kv` | source (FP32) | 80,827,621 | 316,408,345 | 80,390,590 | 80,392,454 | 80,351,729 | 75,273,572 |
| `fp32kv` | **INT8** | **23,276,592** | 89,800,551 | 22,838,463 | 22,840,931 | 22,799,609 | 21,321,548 |
| `fp16kv` | source (FP32) | 80,831,444 | 316,412,159 | 80,391,862 | 80,393,726 | 80,352,102 | 75,274,469 |
| `fp16kv` | **INT8** | **23,280,397** | 89,804,347 | 22,839,729 | 22,842,197 | 22,799,980 | 21,322,441 |
| `fp16kv-fp16c` | source (FP16) | 41,056,006 | 158,861,028 | 40,381,534 | 40,383,464 | 40,325,246 | 37,770,784 |

All four functions in `int8-merged` share one INT8 weight copy (~23.27–23.28 MB, comparable to v3's 23.14 MB).

### Source alignment (vs PyTorch FP32 reference)

| Variant | Tolerance (val / sum) | `logits` max | `scored` max | `score` sum max | `prefix` max | Rows outside | Result |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `fp32kv` | `3e-4` / `1e-3` | `4.20e-5` | `≤ 2.68e-5` | `≤ 3.70e-5` | `≤ 3.00e-5` | 0 | **passed** |
| `fp16kv` | `1e-2` / `1e-2` | `3.63e-3` | `≤ 2.41e-3` | `≤ 1.75e-3` | `≤ 1.71e-3` | 0 | **passed** |
| `fp16kv-fp16c` | `1e-1` / `2.5e-1` | **`0.324`** | `0.163` | **`0.259`** | **`0.149`** | **29** | **FAILED** |

### INT8 alignment (vs stateless `ios18-int8-block32`)

| Variant | Tolerance (val / sum) | `logits` max | `scored` max | `score` sum max (`scored` / `prefix`) | `prefix` max (`one vs ext`) | Rows outside | Result |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `fp32kv` | `3e-4` / `1e-3` | `4.15e-5` | `2.18e-5` | `3.70e-5` / `3.70e-5` | `2.91e-5` (`1.22e-5`) | 0 | **passed** |
| `fp16kv` | `1e-2` / `1e-2` | `3.23e-3` | `1.89e-3` | `1.06e-3` / `1.06e-3` | `1.29e-3` (`7.81e-3`) | 0 | **passed** |

## 3. Load-order memory matrix (`loadorder.json`, Mac reference)

Medians over fresh Python processes, MiB relative to `start`, `.mlmodelc` precompiled outside the process, 10 rounds (`prompt=48`, 8 extensions, 16 `B=16` beam steps when `scored` is loaded, 1 `B=16 T=16` score call):

| Package | Order | After load 1 | After load 2 | After load 3 | `load_peak` | `after_first_round` | `after_rounds` | `peak` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v3` | `score>scored` | 0.9 | 1.2 | — | 1.2 | 3.0 | 3.4 | 17.3 |
| `v3` | `scored>score` | 1.0 | 1.2 | — | 1.2 | 3.1 | 3.3 | 16.8 |
| `v3` | `score>logits` | 0.9 | 1.3 | — | 1.3 | 2.3 | 2.5 | 16.0 |
| `v3` | `logits>score` | 1.0 | 1.2 | — | 1.2 | 2.2 | 2.5 | 16.0 |
| `fp32kv` | `score>scored` | 0.9 | 1.2 | — | 1.2 | 3.1 | 3.4 | 16.9 |
| `fp32kv` | `scored>score` | 1.0 | 1.2 | — | 1.2 | 3.1 | 3.3 | 16.8 |
| `fp32kv` | `score>logits` | 1.0 | 1.3 | — | 1.3 | 2.4 | 2.6 | 16.1 |
| `fp32kv` | `logits>score` | 1.0 | 1.2 | — | 1.2 | 2.2 | 2.4 | 15.9 |
| `fp32kv` | `score>prefix` | 0.9 | 1.2 | — | 1.2 | 2.1 | 2.3 | 15.9 |
| `fp32kv` | `prefix>score` | 1.0 | 1.3 | — | 1.3 | 2.1 | 2.3 | 15.8 |
| `fp32kv` | `score>prefix>scored` | 0.9 | 1.2 | 1.4 | 1.4 | 3.4 | 3.6 | 17.1 |
| `fp16kv` | `score>scored` | 1.0 | 1.2 | — | 1.3 | 35.9 | 36.2 | 49.7 |
| `fp16kv` | `scored>score` | 1.1 | 1.2 | — | 1.2 | 35.8 | 36.2 | 49.7 |
| `fp16kv` | `score>logits` | 1.0 | 1.2 | — | 1.3 | 4.2 | 5.1 | 18.6 |
| `fp16kv` | `logits>score` | 1.1 | 1.2 | — | 1.2 | 4.2 | 5.1 | 18.7 |
| `fp16kv` | `score>prefix` | 0.9 | 1.2 | — | 1.3 | 5.3 | 6.1 | 19.7 |
| `fp16kv` | `prefix>score` | 1.0 | 1.2 | — | 1.2 | 5.3 | 6.2 | 19.7 |
| `fp16kv` | `score>prefix>scored` | 1.0 | 1.2 | 1.5 | 1.5 | 37.3 | 37.5 | 51.0 |

## 4. Latency (`latency.json`, Mac `predict()` p50 / p95 ms)

| Package | `prefix_via` | `prompt_prefix` (48) | `extend_prefix_1` | `scored_step_b16` | `score_b16` (T=16) |
| --- | --- | :---: | :---: | :---: | :---: |
| `v3` | `scored` | 5.15 / 5.74 | 2.07 / 2.52 | 3.89 / 4.39 | 18.04 / 18.62 |
| `fp32kv` | `prefix` | **4.16 / 4.80** | **1.67 / 2.07** | 3.99 / 4.50 | 18.04 / 18.63 |
| `fp16kv` | `prefix` | 4.08 / 4.47 | 1.94 / 2.39 | 5.24 / 5.75 | 18.23 / 18.95 |

## 5–7. Search, fixtures, and iOS resources

| Variant | Search vs reference (125 cases) | Search vs FP32 fixture (125 cases) | Search p50 | Fixture step max | Fixture score sum max | Fixture order |
| --- | --- | --- | ---: | ---: | ---: | :---: |
| `fp32kv` | **125/125** exact, **125/125** top-1 (`diff 0.0`, vs `v3`) | 46 exact, 125 top-1 (`diff 0.565`) | 64.51 ms | `2.86e-5` | `1.70e-5` | **8/8** |
| `fp16kv` | **125/125** exact, **125/125** top-1 (`diff 3.19e-3`, vs `fp32kv`) | 46 exact, 125 top-1 (`diff 0.568`) | 66.55 ms | `9.09e-4` | `8.06e-4` | **8/8** |

Both `ios-fp32kv` (`TinyChineseKVBatchV4FP32KVINT8.mlmodelc`, 23.32 MB) and `ios-fp16kv` (`TinyChineseKVBatchV4FP16KVINT8.mlmodelc`, 23.32 MB) were generated and verified.

## What the Mac numbers say for each decision rule

- **`prefix` function:** Pending iPhone measurement. Mac load-order deltas do not reproduce the iPhone `+27.3 MB` steady footprint jump; on latency alone, `prefix` cuts 48-token prompt and 1-token extend time by ~19%.
- **`fp16kv`:** Passes all accuracy gates on Mac (INT8 alignment `3.23e-3`, search `125/125` exact & top-1 vs `fp32kv`, fixture `8/8` same order). However, on Mac it uses **more** memory in the beam workload (`49.7 vs 16.9 MiB` peak) and is slower on `scored_step_b16` (`5.24 vs 3.99 ms`). Adoption depends on whether iPhone shows a lower peak in the beam workload.
- **`fp16kv-fp16c`:** **Rejected.** Failed source alignment (`max_abs = 0.324`, 29 rows outside tolerance). Stopped per decision rules.

## 8. On-device iPhone 16 Pro Max load-order verification (`iphone16pm-loadorder.jsonl`)

Fresh-process runs on an iPhone 16 Pro Max (iOS 18+ target, `-O`, `CPU_ONLY`, 2 ms `phys_footprint` sampling + kernel `ledger_phys_footprint_peak`), matching the Mac 10-round workload (`prompt=48`, 8 extensions, 16 `B=16` beam steps when `scored` is loaded, 1 `B=16 T=16` score call; Float16 KV `MLMultiArray` for `fp16kv`):

| Package | Order | Start | Load1 steady(Δ) / pk(Δ) | Load2 steady(Δ) / pk(Δ) | Load3 steady(Δ) / pk(Δ) | LoadPk (Δ) | AfterRnd (Δ) | TotalPk (Δ) | `prompt / ext / step / score` (p50 ms) |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| `v3` | `score>scored` | 30.9 | 33.9(+3.0) / +21.6 | 61.5(+30.6) / +30.6 | — | 61.5(+30.6) | 62.0(+31.2) | 75.5(+44.7) | 4.61 / 1.88 / 3.19 / 13.59 |
| `v3` | `scored>score` | 30.8 | 34.1(+3.3) / +49.0 | 34.2(+3.5) / +3.5 | — | 79.8(+49.0) | 34.8(+4.0) | 79.8(+49.0) | 4.63 / 1.88 / 3.19 / 13.57 |
| `v3` | `score>logits` | 30.7 | 33.7(+3.0) / +21.6 | 61.3(+30.6) / +30.6 | — | 61.3(+30.6) | 61.4(+30.7) | 74.9(+44.2) | 4.58 / 1.89 / — / 13.58 |
| `v3` | `logits>score` | 30.8 | 34.2(+3.4) / +49.0 | 34.3(+3.5) / +3.5 | — | 79.9(+49.0) | 34.4(+3.6) | 79.9(+49.0) | 4.59 / 1.84 / — / 13.55 |
| `fp32kv` | `score>scored` | 30.8 | 38.0(+7.3) / +26.0 | 65.6(+34.8) / +34.9 | — | 65.6(+34.9) | 65.7(+34.9) | 79.6(+48.8) | 4.60 / 1.93 / 3.25 / 13.70 |
| `fp32kv` | `scored>score` | 30.7 | 38.3(+7.6) / +53.4 | 38.3(+7.7) / +7.7 | — | 84.1(+53.4) | 38.9(+8.2) | 84.1(+53.4) | 4.61 / 1.89 / 3.21 / 13.64 |
| `fp32kv` | `score>logits` | 30.8 | 38.0(+7.3) / +25.9 | 65.5(+34.8) / +34.8 | — | 65.6(+34.8) | 65.7(+34.9) | 79.2(+48.4) | 4.60 / 1.84 / — / 13.54 |
| `fp32kv` | `logits>score` | 30.8 | 38.3(+7.5) / +53.2 | 38.4(+7.6) / +7.6 | — | 84.0(+53.2) | 38.5(+7.7) | 84.0(+53.2) | 4.59 / 1.90 / — / 13.63 |
| **`fp32kv`** | **`score>prefix`** | 30.7 | 37.9(+7.2) / +25.9 | **38.2(+7.5) / +7.6** | — | **56.6(+25.9)** | **38.3(+7.6)** | **56.6(+25.9)** | **3.61 / 1.26 / — / 13.45** |
| **`fp32kv`** | **`prefix>score`** | 30.8 | 38.1(+7.3) / +26.0 | **38.3(+7.6) / +7.6** | — | **56.7(+26.0)** | **38.4(+7.6)** | **56.7(+26.0)** | **3.59 / 1.24 / — / 13.48** |
| `fp32kv` | `score>prefix>scored` | 30.8 | 38.1(+7.3) / +25.9 | 38.5(+7.6) / +7.7 | 65.9(+35.0) / +35.1 | 65.9(+35.1) | 66.4(+35.6) | 79.9(+49.1) | 3.58 / 1.26 / 3.10 / 13.50 |
| `fp16kv` | `score>scored` | 28.7 | 36.0(+7.3) / +26.0 | 63.6(+34.8) / +34.9 | — | 63.6(+34.9) | 63.9(+35.2) | 77.4(+48.7) | 4.38 / 1.84 / 3.36 / 13.54 |
| `fp16kv` | `scored>score` | 28.7 | 36.4(+7.6) / +53.4 | 36.5(+7.7) / +7.8 | — | 82.1(+53.4) | 36.8(+8.0) | 82.1(+53.4) | 4.42 / 1.84 / 3.37 / 13.52 |
| `fp16kv` | `score>logits` | 28.7 | 36.1(+7.4) / +26.0 | 63.7(+34.9) / +35.0 | — | 63.7(+35.0) | 63.8(+35.1) | 77.3(+48.6) | 4.34 / 1.81 / — / 13.66 |
| `fp16kv` | `logits>score` | 28.7 | 36.3(+7.6) / +53.3 | 36.4(+7.7) / +7.7 | — | 82.0(+53.3) | 36.5(+7.8) | 82.0(+53.3) | 4.39 / 1.82 / — / 13.61 |
| `fp16kv` | `score>prefix` | 28.7 | 36.1(+7.4) / +26.0 | 36.5(+7.8) / +7.8 | — | 54.8(+26.0) | 36.6(+7.8) | 54.8(+26.0) | 3.56 / 1.28 / — / 13.65 |
| `fp16kv` | `prefix>score` | 28.7 | 36.1(+7.5) / +26.1 | 36.4(+7.7) / +7.7 | — | 54.8(+26.1) | 36.5(+7.8) | 54.8(+26.1) | 3.56 / 1.29 / — / 13.61 |
| `fp16kv` | `score>prefix>scored` | 28.7 | 36.0(+7.4) / +26.0 | 36.4(+7.7) / +7.7 | 63.8(+35.1) / +35.2 | 63.9(+35.2) | 64.2(+35.5) | 77.7(+49.0) | 3.54 / 1.31 / 3.38 / 13.52 |

### Final decisions (against preset rules)

- **`prefix` function: KEEP.** Both `score>prefix` (`+7.5 MB` steady, `+25.9 MB` peak) and `prefix>score` (`+7.6 MB` steady, `+26.0 MB` peak) cut steady footprint by **27.3 MB** vs `score>scored` (`+34.8 MB`) — well past the 10 MB threshold — and cut total peak by **23.0 MB** (`56.6 vs 79.6 MB`). `score>logits` does not save memory (`+34.8 MB` steady), confirming the output head (`lm_head`) is the cause. `prefix` also cuts 48-token prompt latency by 21.7% (`4.60 → 3.60 ms`) and 1-token extension by 35.2% (`1.93 → 1.25 ms`).
- **`fp16kv`: REJECT.** Passes all accuracy gates, but on iPhone 16 Pro Max relative peak memory in the beam workload is unchanged (`+48.7 vs +48.8 MB` for `score>scored`, `+53.4 vs +53.4 MB` for `scored>score`, `+49.0 vs +49.1 MB` for `score>prefix>scored`) and `scored_step_b16` is slightly slower (`3.36–3.38 ms` vs `3.10–3.25 ms`).
- **`fp16kv-fp16c`: REJECT.** Failed source alignment (`max_abs = 0.324`, 29 rows outside tolerance).
- **Selected variant for release & HF upload:** **`fp32kv`** (`outputs/deployment/zh-kvbatch-v4/fp32kv/int8-merged` and `ios-fp32kv/`).


## Review (2026-11-20)

Return ZIP sha256 `39481e7a39ed498cfd8a3657ff7fe9cb4a76611f14c427f77ce39b5b590eb7d2`, all 84 files match `SHA256SUMS`. The decisions above follow the preset rules and stand: **`fp32kv` with `prefix`**, `fp16kv` and `fp16kv-fp16c` rejected. Against the FP32 fixture `fp32kv` gives 46 exact and 125 top 1, the same as v2 and v3 (the INT8 effect).

What the iPhone matrix shows about the extra memory:

- It is a fixed cost of the second **head** function loaded after `score`: `scored` and `logits` both add about +27 MB steady, `prefix` (no head) adds 0.3 MB.
- Loading `scored` first moves the same cost into a transient: steady +7.7 MB, but a load peak of +53.4 MB.
- **The saving holds only while `scored` is not loaded.** `score>prefix>scored` is back at +35.0 MB steady and +49.1 MB peak. Pinyin generation needs `scored` for its beam steps, so a session that generates pays the cost whichever function builds the prefix.

For the client:

- Build and extend the shared prefix with `prefix`: 21.7% faster for 48 tokens and 35.2% for one token, with identical search results.
- Load `scored` lazily, on the first generation request, not at start-up. Sessions that only score or rerank then keep the 27 MB saving.
- Not measured: `scored>score>prefix` and `scored>prefix>score`. Vime adds these two orders to the next on-device run (both iPhones) before the client fixes its load order.

The same variant is rebuilt with the vimeml-zh-v2-20m weights once v2-20m is chosen.
