# vimeml-tiny-zh-v1 batched KV decoding v2 on Core ML — 2026-11-15

This follows the [v2 brief](../../reference/mac-zh-v1-kvbatch-v2.md). It continues in the first task's checkout (`~/Documents/Sources/VimeML-zh-kvbatch`), pulled fast-forward to `main` `65c42fd`.

- Source model: `vimeml-tiny-zh-v1` (bundle manifest `9d9dad19…`, tokenizer `68bd37c4…`).
- Not changed: the checkpoint, the tokenizer, the INT8 recipe (block32, threshold 2048, 47 matrices) and the search rules.
- `RELEASE.json` and `SHA256SUMS.txt` are untouched (still `899ca51a…` / `58096d45…`, recorded in `logs/07-release-meta-before.sha`).
- Nothing was committed on the Mac.
- The first task's outputs under `outputs/deployment/zh-v1-kvbatch/` were used read-only as the baseline.
  - The first report had not been committed when this task started; that local copy had collided with the upstream file on pull.
  - Before pulling, it was moved to `outputs/deployment/mac-20261115-local-report-backup/`. The committed upstream version differs only by 2 added lines.

**Environment**

- Hardware and OS: Apple M3, 16 GiB, macOS 27.3 (26C5035r), Xcode 27.3 (27C5020n).
- Python: 3.11.17 in a uv venv, with PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5 and sentencepiece 0.2.1.
- Core ML: mlprogram, iOS 18 target, FP32 compute, `CPU_ONLY`.
- Power and load: AC power, and `pmset -g therm` recorded no warning. The machine was not idle (load average 7–24 during the session), so single latency numbers carry noise. The v1↔v2 comparison was therefore interleaved three times.

**Outputs and change**

- All outputs are under `outputs/deployment/zh-v1-kvbatch-v2/` (`$O`): stage logs in `logs/`, helper scripts in `tools/`. No output was overwritten. Failed steps are kept as logs.
- **`src/vimeml/deployment/zh_v1_batch2.py` was changed** (shape specification only; the graph is unchanged). See [Change to zh_v1_batch2.py](#change-to-zh_v1_batch2py). `check` was rerun afterwards and passed.

## Summary

- **Gates:** all six variants and three fixed-shape fallbacks convert, align against PyTorch FP32 and pass Vime's INT8 gate against the stateless package (max ≤ 6.6e-5, tolerance 3e-4).
  - `dec-enum` and `dec-b16` first failed to convert. Both failures came from how the enumerated shapes were declared, not from the graph.
  - After the fix in `shape_spec`, both convert and pass.
- **Memory:** read-only caches alone do **not** reduce memory on the Mac. The T=1 variants are much worse:
  - **T=1 packages** (`dec-flex`, `dec-enum`, `scored`) add ~200 MiB at load. They grow by tens of MiB with every new (B,S) shape, to ~370 MiB of Core ML memory.
  - **Flexible-T packages** (`unified`, `unified-scored`) add ~39 MiB at load. They stay flat at ~40–41 MiB through every shape, the same as v1 (38–40 MiB). They also remove the chunk package (13 MiB).
- **Latency:** `unified` and `unified-scored` stay well inside Vime's targets: B=16 ≤ 4.4 ms and B=8 ≤ 3.6 ms on the Mac, at every S up to 128.
  - The T=1 variants are 2–6× slower than v1 at B=16.
  - A full prompt prefill is one call of 2–4 ms, instead of one call per token.
- **Search:** every INT8 variant gives the same results as the v1 INT8 batch search, 125/125 in exact order, with scores within 5.3e-5.
  - FP32 `dec-flex` and FP32 `unified` match the FP32 fixture 125/125 in exact order, with scores within 4.5e-5.
- **Choice: `unified-scored`.**
  - It ties with `unified` for the lowest footprint, needs no chunk package, and meets the latency targets.
  - Its end-to-end search is the fastest (p50 79 ms vs 106 ms for `unified`).
  - `unified` is the fallback if the client cannot use the scored head; its memory and latency are the same.
- **Outputs:** `ios-kvbatch-v2/` and the release copy are built from it.

## 1. PyTorch check

| Run | Decode max | Prefill + multi-token max | Scored head vs log-softmax | Result |
| --- | ---: | ---: | ---: | --- |
| `check.json` (before the shape fix) | 5.01e-5 | ≤ 2.7e-5 | 1.91e-6 | pass |
| `check-after-shape-fix.json` | 5.01e-5 | ≤ 2.7e-5 | 1.91e-6 | pass |

These agree with the Windows checks (3.96e-5 / 2.9e-5 / 1.9e-6).

After the change, `python -m unittest tests.test_deployment_zh_v1_batch` also passed: 12 tests, OK. pytest is not installed in the venv.

## 2. Conversion, compression and gates

Max abs error in both columns. FP32 align is against PyTorch FP32; the INT8 gate is against the stateless INT8 package. Tolerance is atol = rtol = 3e-4.

| Variant | Shapes | FP32 align | INT8 gate | Result |
| --- | --- | ---: | ---: | --- |
| `dec-flex` | B 1..16, T=1, S 1..128 (RangeDim) | 8.34e-5 | 5.96e-5 | pass |
| `dec-enum` (first try) | B {1,4,8,16} × S {16,32,64,128} | — | — | **convert failed** (`logs/02-convert-fp32-dec-enum.log`) |
| `dec-enum` (after fix) | 16 enumerated (B,S) pairs | 7.77e-5 | 6.58e-5 | pass |
| `dec-b16` (first try) | B=16, S {32,64,128} | — | — | **convert failed** (`logs/02-convert-fp32-dec-b16.log`) |
| `dec-b16` (after fix) | B=16 fixed, S enumerated | 5.02e-5 | 4.39e-5 | pass |
| `dec-b16-s32` / `-s64` / `-s128` (fallback) | B=16, S fixed | ≤ 4.49e-5 | ≤ 4.72e-5 | pass |
| `scored` | as `dec-flex`, scored head | 3.24e-5 | 4.10e-5 | pass |
| `unified` | B 1..16, T 1..128, S 1..128 | 5.29e-5 | 5.25e-5 | pass |
| `unified-scored` | as `unified`, scored head | 2.67e-5 | 2.96e-5 | pass |

The two conversion failures were raised by coremltools 9.0 for the iOS 18 target:

- **`dec-enum`:**
  - Error: *"Enumerated shape input mismatch. All enumerated shape inputs must have the same number of shapes for deployment targets iOS 18 or above."*
  - Cause: the caches listed 16 (B,S) shapes, while `input_ids` and `position` listed 4.
- **`dec-b16`:**
  - Error: *"EnumeratedShapes should be take a list or tuple with len >= 2, got 1 instead".*
  - Cause: with a single batch size, `input_ids` and `position` had one enumerated shape.

As the brief asked, the three fixed-S B=16 fallbacks were also built and kept before the fix. They are measured below.

The four variants built before the fix (`dec-flex`, `scored`, `unified`, `unified-scored`) record the earlier `conversion_source_sha256` in their manifests. The fix touches only the enumerated and fixed branches, which these RangeDim packages never use.

## 3. Memory

### `memory` stage (3 fresh processes per configuration, medians)

Values are `phys_footprint` added over the process after the stateless INT8 package was loaded and run, in MiB. The stateless package itself adds ~15.5 MiB. Source: `memory.json`, `logs/03-memory.log`.

| Config | Chunk | Load | Prefill 24 | First B16 S32 | Sweep to S=64 | B16 S128 | Peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 `int8-flex` | — | 37.4 | — | 60.0 | 98.8 | 179.9 | 189.6 |
| v1 + chunk | 13.0 | 47.2 | — | 68.7 | 102.9 | 184.0 | 193.7 |
| `dec-flex` | — | 209.3 | — | 251.8 | 345.7 | 547.0 | 547.0 |
| `dec-flex` + chunk | 13.2 | 214.1 | — | 256.6 | 349.9 | 551.1 | 551.1 |
| `dec-enum` | — | 199.6 | — | 249.6 | 349.9 | 551.1 | 551.1 |
| `dec-b16` | — | 51.9 | — | 61.0 | 101.6 | 182.7 | 183.0 |
| `dec-b16-s32` | — | 22.3 | — | 38.3 | 51.9 | — | 52.1 |
| `dec-b16-s64` | — | 22.4 | — | 65.3 | 92.4 | — | 92.7 |
| `dec-b16-s128` | — | 22.3 | — | 132.8 | 186.2 | 186.2 | 187.1 |
| `scored` | — | 209.4 | — | 250.7 | 343.5 | 544.8 | 544.8 |
| `unified` | — | 38.9 | 40.3 | 61.6 | 99.9 | 181.0 | 182.3 |
| `unified-scored` | — | 39.0 | 40.4 | 60.5 | 99.0 | 180.1 | 181.6 |

`proc_pid_rusage` V4 `phys_footprint` equals the V2 value used in the first report, within 1 MiB, in every process checked: `v2_equals_v4_phys_footprint: true` for all 7 labels in `memory-attribution/attribution.json`.

### Where the memory goes (`memory-attribution/`, `tools/memory_attribution.py`)

The `memory` stage creates each call's NumPy inputs inside the measured window. Those inputs are large: one B=16, S=128 key + value pair is 56 MiB. Much of the growth after load in the table above is therefore Python's, not Core ML's.

The attribution tool separates the two. In one fresh process per package, it:

1. allocates every input the run will use before loading the package;
2. runs the same call sequence, then repeats it once;
3. captures `footprint -p` and `vmmap --summary` on the live process.

Added over "stateless loaded + all inputs allocated", in MiB:

| Package | Load | Prefill 24 | B16 S32 | Sweep to S=64 | B16 S128 | Whole sequence repeated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 `int8-flex` | 38.1 | — | 39.7 | 40.0 | 40.0 | 40.0 |
| `dec-flex` | 209.1 | — | 224.2 | 296.3 | 395.5 | 396.5 |
| `dec-b16` (S enumerated) | 51.3 | — | 53.7 | 53.8 | 53.8 | 53.8 |
| `dec-b16-s32` | 22.3 | — | 24.8 | 24.8 | — | 24.8 |
| `dec-b16-s128` | 21.8 | — | 24.3 | 24.3 | 24.3 | 24.3 |
| `unified` | 39.3 | 40.4 | 41.0 | 41.0 | 41.0 | 41.0 |
| `unified-scored` | 39.5 | 40.5 | 40.3 | 40.3 | 40.3 | 40.3 |

What this shows:

- **Core ML's own cost for v1 and the flexible-T v2 packages is ~38–41 MiB, and it is flat.**
  - Nothing grows with B or S once the package is loaded.
  - On the Mac, read-only caches therefore save nothing over v1: v1's cache-sized temporaries are not held between calls.
  - The device figure Vime reported (~27 MiB of Core ML) has the same shape. The Mac number is larger and is not a device prediction.
- **The T=1 packages behave differently.**
  - They load at ~209 MiB and then add ~15–100 MiB per new (B,S) shape. Growth stops only after every shape has been seen once (the repeated sequence adds 1 MiB).
  - In `footprint`/`vmmap`, `dec-flex` has about 350 MiB more `Malloc Large` + `Malloc Small` than `unified` (762 vs 406 MB of malloc in the whole process).
  - Only 77 MB of the weight file is mapped, against 154 MB for `unified` and v1.
  - The pattern fits Core ML materializing dequantized weights onto the heap and caching a plan per shape for these graphs. That is an inference from the counters, not something the tools state.
- **The graphs are nearly identical.** MIL op counts (`logs/03c-op-histogram.log`):
  - `dec-flex` 997 ops, `unified` 1026 ops.
  - The same 47 `constexpr_blockwise_shift_scale`, 46 `linear` and 36 `matmul`.
  - The only difference is a static T=1 versus a flexible T dimension, so the static 1 is what triggers the other runtime path.
- **Fixed shapes** load lightest (~22 MiB). Their footprint stays flat whatever the fixed S, so the large S growth in the `memory` table for fixed packages is input allocation.
  - The cost of fixed shapes is that every call is padded to B=16 and the fixed S, which shows up in latency (below).
  - `dec-b16` with enumerated S costs 52–54 MiB.

## 4. Latency on the Mac

Settings: `measure` with the stateless package loaded, 200 repeats, `CPU_ONLY`. Values are p50 in ms (p95 in the JSON files). Fixed and enumerated packages pad rows up to their shape.

### Decode, p50 ms (B × S)

| Variant | B1 S16 | B1 S128 | B8 S16 | B8 S32 | B8 S64 | B8 S128 | B16 S16 | B16 S32 | B16 S64 | B16 S128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dec-flex` | 2.32 | 3.55 | 4.59 | 5.44 | 9.12 | 13.65 | 6.56 | 8.07 | 15.74 | 24.16 |
| `dec-enum` | 1.89 | 3.09 | 4.10 | 4.89 | 8.42 | 12.65 | 5.85 | 7.56 | 14.63 | 51.29 |
| `dec-b16` (pads to B16) | 6.04 | 9.06 | 5.72 | 6.30 | 6.22 | 8.89 | 6.95 | 5.07 | 5.66 | 8.22 |
| `dec-b16-s32` | 4.81 | — | 5.63 | 5.07 | — | — | 4.80 | 4.96 | — | — |
| `dec-b16-s64` | 5.91 | — | 5.45 | 4.45 | 4.43 | — | 4.12 | 4.16 | 3.71 | — |
| `dec-b16-s128` | 7.89 | 6.45 | 7.80 | 6.95 | 8.43 | 6.31 | 7.97 | 6.96 | 7.71 | 8.46 |
| `scored` K=149 | 2.98 | 3.54 | 4.89 | 5.68 | 9.15 | 13.56 | 6.00 | 8.38 | 16.32 | 24.16 |
| `unified` | 2.01 | 2.07 | 2.56 | 3.12 | 3.27 | 3.55 | 3.12 | 3.66 | 4.15 | 4.26 |
| `unified-scored` K=149 | 2.01 | 2.30 | 2.99 | 3.09 | 3.03 | 3.46 | 3.19 | 3.73 | 3.81 | 4.37 |
| `unified-scored` K=450 | 2.02 | 2.43 | 3.02 | 2.71 | 3.19 | 3.10 | 3.54 | 3.35 | 4.09 | 4.32 |
| v1 `int8-flex` (first report) | — | — | — | — | — | — | 3.35 | 3.76 | — | — |

For the fixed-S packages, a column at a smaller S than the package's own S is the same padded call.

`scored` at K=450 is within noise of K=149 (e.g. B16 S32 8.09 vs 8.38). Full grids are in `int8-<name>-latency.json`; an extract is in `latency-table.txt`.

### Prompt prefill, p50 ms

| Variant | 8 tokens | 16 tokens | 32 tokens | How |
| --- | ---: | ---: | ---: | --- |
| `unified` | 2.15 | 2.34 | 4.08 | one call |
| `unified-scored` | 2.31 | 2.69 | 3.76 | one call |
| `dec-flex` | 47.0 | 92.3 | 187.1 | one token per call, B=1 |
| `dec-b16-s32` | 43.1 | 97.6 | 163.1 | one token per call, padded to B=16 |

### Interleaved `int8-dec-flex` vs v1 `int8-flex` (3 rounds, p50 ms; `latency-interleaved/`)

| Rows | S | `dec-flex` runs | median | v1 runs | median | v2 / v1 |
| ---: | ---: | --- | ---: | --- | ---: | ---: |
| 1 | 16 | 2.00 / 2.43 / 2.13 | 2.13 | 1.88 / 1.66 / 1.89 | 1.88 | 1.13 |
| 1 | 32 | 2.00 / 2.55 / 2.01 | 2.01 | 1.89 / 1.70 / 1.92 | 1.89 | 1.06 |
| 4 | 16 | 2.90 / 3.30 / 2.83 | 2.90 | 2.51 / 2.65 / 2.36 | 2.51 | 1.16 |
| 4 | 32 | 3.48 / 3.65 / 3.58 | 3.58 | 2.48 / 2.86 / 2.52 | 2.52 | 1.42 |
| 8 | 16 | 4.52 / 4.97 / 4.69 | 4.69 | 2.72 / 2.92 / 2.94 | 2.92 | 1.61 |
| 8 | 32 | 4.97 / 5.86 / 4.95 | 4.97 | 3.23 / 2.90 / 3.26 | 3.23 | 1.54 |
| 16 | 16 | 6.27 / 6.66 / 6.58 | 6.58 | 3.43 / 3.16 / 3.34 | 3.34 | 1.97 |
| 16 | 32 | 7.64 / 8.20 / 7.83 | 7.83 | 3.66 / 4.00 / 3.61 | 3.66 | 2.14 |

The slowdown is consistent across rounds, so it is not noise. The T=1 v2 graph is slower than v1 on this runtime path. The flexible-T `unified` (B16 S32 3.66 ms) is as fast as v1.

## 5. Search check (r2 fixture, 125 cases, beam 16, top 10)

- "vs v1 batch" compares against the same search run with the v1 INT8 batch package and chunk prefill, which equals the stateless-INT8 search (first report).
- "vs FP32 fixture" compares against the shipped FP32 fixture.
- Prefill calls are the mean per case.

| Package | Prompt | vs v1 batch (exact / same set / max score diff) | vs FP32 fixture (exact / same set / top-1 / max diff) | p50 / p95 ms | Prefill calls |
| --- | --- | --- | --- | --- | ---: |
| `int8-unified-scored` | package, one call | 125 / 125 / 3.3e-5 | 46 / 86 / 125 / 0.565 | 79 / 212 | 1 |
| `int8-unified` | package, one call | 125 / 125 / 2.3e-5 | 46 / 86 / 125 / 0.565 | 106 / 234 | 1 |
| `int8-dec-flex` | package, per token | 125 / 125 / 2.7e-5 | 46 / 86 / 125 / 0.565 | 197 / 391 | 4.44 |
| `int8-dec-flex` | chunk | 125 / 125 / 2.1e-5 | same | 209 / 455 | 1 |
| `int8-dec-enum` | package, per token | 125 / 125 / 2.6e-5 | same | 325 / 877 | 4.44 |
| `int8-dec-enum` | chunk | 125 / 125 / 2.2e-5 | same | 206 / 481 | 1 |
| `int8-dec-b16` | package, per token | 125 / 125 / 2.1e-5 | same | 192 / 411 | 4.44 |
| `int8-dec-b16` | chunk | 125 / 125 / 2.0e-5 | same | 140 / 330 | 1 |
| `int8-dec-b16-s128` | package, per token | 125 / 125 / 2.2e-5 | same | 196 / 486 | 4.44 |
| `int8-dec-b16-s64` | package, per token | 125 / 125 / 2.2e-5 | same | 120 / 258 | 4.44 |
| `int8-dec-b16-s32` | package, per token | **failed** (`logs/05b-search-dec-b16-s32.log`) | — | — | — |
| `int8-scored` | package, per token | 125 / 125 / 5.3e-5 | same | 265 / 586 | 4.44 |
| `int8-scored` | chunk | **refused** (`logs/05-search-scored-chunk.log`) | — | — | — |
| `fp32-dec-flex` | package, per token | — | **125 / 125 / 125 / 4.4e-5** | 260 / 532 | 4.44 |
| `fp32-unified` | package, one call | — | **125 / 125 / 125 / 4.5e-5** | 138 / 298 | 1 |

Notes on the table:

- The INT8 vs FP32-fixture numbers are identical for every INT8 variant, and identical to the first report's v1 batch and stateless-INT8 runs (46/86/125). That difference is INT8 itself, not the batch packages.
- `scored --prompt chunk` is refused by design. The chunk package returns full logits, while the backend needs either a flexible-T scored package or token-by-token prefill with the scored head (*"Use a flexible-T package or token-by-token prefill with the scored head."*). The error is kept as evidence.
- `dec-b16-s64` holds every r2 case.
- `dec-b16-s32` fails: some cases need more than 32 slots.
  - The backend then finds no fitting S and fails on an empty `min()` (*"min() arg is an empty sequence"*) instead of a clear message.
  - The log is kept. A fixed S=32 package is therefore not usable on its own.

## 6. Choice: `int8-unified-scored`

Applying the brief's order of preference:

1. **Lowest added footprint, with the stateless package loaded.**
   - Among full-range packages: `unified-scored` 40.3 MiB and `unified` 41.0 MiB after the whole sequence (attribution), against v1 40.0 MiB plus 13 MiB for its chunk package.
   - The fixed-S packages are lighter (~24 MiB), but each has a catch:
     - `dec-b16-s32` cannot hold the r2 contexts.
     - `dec-b16-s64` holds them, but caps context plus keys at 64 tokens, needs the chunk package or per-token prefill (~100 ms for 16 tokens), and pads every call to B=16 (3.7–6.5 ms even at B=1).
     - `dec-b16-s128` pays 6.5–8.5 ms on every call on the Mac, which is ~13–17 ms on the 13 Pro, over the target.
   - `dec-b16-s64` is the only option that saves memory (~16 MiB less than `unified-scored`). It is not chosen, because it fails preference 2 and the 128-token range.
   - `dec-b16` with enumerated S (54 MiB) is heavier than `unified`.
   - All T=1 flexible or enumerated packages are ruled out (≥ 200 MiB).
2. **No chunk package.** Both `unified` variants prefill the prompt in one call (2–4 ms). With them the keyboard needs only the stateless package and this one.
3. **Latency.**
   - `unified-scored` on the Mac: B=16 ≤ 4.4 ms and B=8 ≤ 3.5 ms at every S up to 128, at K=149 and K=450.
   - At the Mac's roughly 2× speed, that is about 9 ms at B=16 S128 on the 13 Pro and about 7–8 ms at S ≤ 64, against the 8 ms target. B=8 stays within 5 ms after doubling only up to S≈32–64 (3.1–3.5 ms on the Mac). This needs confirming on device.

Between the two `unified` variants, footprint and decode latency tie within noise. `unified-scored` was chosen because:

- its end-to-end search is faster (p50 79 vs 106 ms; the client no longer copies and softmaxes 16×12,288 logits per call);
- its output per call is 16×(K+1) floats instead of 16×12,288.

The trade-off:

- The client must send each bucket's allowed-id list (K ≤ 1024; the r2 maximum is 536).
- The client gets log-probabilities only for those ids.

`unified` (`$O/int8-unified`, search 125/125) is the drop-in alternative with the full-logits interface and the same memory.

The fixture stage reported package vs stateless max 1.24e-5 (`logs/06-fixture.log`).

## 7. Client resource and release copy

- `$O/ios-kvbatch-v2/`:
  - `Resources/TinyChineseV1KVBatchV2INT8.mlmodelc`;
  - `Resources/VimeLMManifestZhV1KVBatchV2.json` (format `vime_ios_lm_zh_v1_kv_batch_v2`, section `kv_batch_v2`, which records head `scored` and flexible tokens);
  - `KVBatchFixture-*` and `manifest.json`.

  v1 resources are not touched.
- Release copy: `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch-v2/`, a byte-identical copy of `$O/int8-unified-scored` (`diff -r` clean, 22 MiB). `RELEASE.json` and `SHA256SUMS.txt` are unchanged.

**Windows follow-up (2026-11-15).** The results ZIP was checked against its SHA256SUMS (251 files). The release copy is byte-identical to `int8-unified-scored` and passes `verify_package` with the v2 format. The release inventory now lists the four `ios18-int8-block32-kvbatch-v2/` files and `coreml_manifest_sha256.kvbatch_v2_int8` = `33c90e2a…`; the README describes the v2 package. `verify_release.py --hashes` passes on 374 files. The Hugging Face release was updated in place at revision `8dd4f19729e921a6bd6844a79743355343569784` (parent `ce35c931…`); a fresh download of that revision passes `verify_release.py --hashes`. The Mac's `shape_spec`/`shapes_of` change was taken into `zh_v1_batch2.py` as it is. The backend now raises a clear error when no shape of a fixed or enumerated package fits a call, instead of the empty `min()` seen with `dec-b16-s32`.

The brief's reasoning for v2 was that v1's per-layer cache copies made Core ML allocate cache-sized temporaries. The Mac numbers do not support it: read-only caches change nothing for flexible-T graphs (~40 MiB, as v1), and T=1 graphs get much worse. What v2 saves on device is the chunk package, not Core ML's own memory. Only fixed-shape packages are lighter, at the costs listed in §6.

## 8. iPhone

Not done in this session, as the brief says; Vime measures it with its own test host. On the device, check whether:

- T=1 graphs show the same heap growth;
- `unified-scored`'s ~40 MiB Mac figure maps to Vime's ~27 MiB Core ML figure for v1.

## Change to zh_v1_batch2.py

The full diff is in `$O/zh_v1_batch2.py.diff`; the changed file is included in the ZIP. It has two parts.

1. **`shape_spec`, enumerated branch.**
   - Build the list of (B,S) pairs.
   - Give the caches one enumerated shape per pair.
   - Give `input_ids` and `position` the same number of shapes, the i-th repeating the i-th pair's B, as iOS 18 requires.
   - With a single batch size, `input_ids` and `position` become plain fixed shapes, since `EnumeratedShapes` needs at least two.
   - Raise a clear error if fewer than two pairs exist.
2. **`shapes_of`.** Recognize `"S fixed <n>"` manifests from the fixed-S fallbacks, so the Core ML backend and the `measure`/`search` stages feed exactly that S instead of treating S as flexible.

`check` was rerun after the change and passed (`check-after-shape-fix.json`, `logs/01b-check-after-shape-fix.log`), and the unit tests pass.

## Files

| Path | What |
| --- | --- |
| `check.json`, `check-after-shape-fix.json` | PyTorch checks |
| `fp32-<name>/`, `int8-<name>/`, `*-align.json` | packages and gates (6 variants + 3 fallbacks) |
| `memory.json` | `memory` stage (12 configurations × 3 processes) |
| `memory-attribution/` | preallocated-input runs, `footprint` and `vmmap` text per package, V2/V4 check |
| `int8-<name>-latency.json`, `latency-interleaved/`, `latency-table.txt` | latency |
| `search-<name>[-chunk]/`, `search-fp32-*/`, `search-table.txt` | search checks |
| `fixture/`, `ios-kvbatch-v2/` | Vime fixture and client resource |
| `tools/` | `memory_attribution.py`, `run_latency.sh`, `run_search.sh` |
| `logs/` | every stage, including the two failed conversions and the refused scored+chunk search |
| `zh_v1_batch2.py.diff` | the source change |
