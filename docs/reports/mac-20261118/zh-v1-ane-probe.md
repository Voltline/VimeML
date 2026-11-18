# vimeml-tiny-zh-v1 Neural Engine probe (step 0) on the Mac — 2026-11-18

This follows the [ANE probe brief](../../reference/mac-zh-v1-ane-probe.md), steps 1–7. It was run in `~/Documents/Sources/VimeML-zh-kvbatch`, pulled fast-forward to `main` `519784d`.

**Before the pull.** The untracked Mac copy of the v3 report (`docs/reports/mac-20261118/zh-v1-kvbatch-v3.md`) collided with the upstream file. It was moved to `outputs/deployment/mac-20261118-kvbatch-v3-local-backup/` (16 diff lines of editing vs upstream). `git status` was clean after the pull.

**Unchanged:**

- no tracked file was edited;
- nothing under `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/` was touched (`RELEASE.json` / `SHA256SUMS.txt` included);
- nothing was committed or pushed.

**Environment** (`logs/00-environment.log`):

- Hardware and OS: Apple M3 (16 GiB), macOS 27.3 (26C5047e), Xcode 27.3 (27C5035i).
- Python: 3.11.17 (uv venv), coremltools 9.0, PyTorch 2.7.1, NumPy 2.3.5.
- Power and load: AC power, load average about 2–3.

**Outputs.** Everything is under `outputs/deployment/zh-v1-ane-probe/` (`$OUT`): stage logs in `logs/`, helper scripts in `tools/`. No output was overwritten, and failed runs are kept.

**Ranking cases.** The ranking cases are `outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl`, the same file as the v3 search stage (133 cases).

## Verdict: **no-go** under the brief's rules, with a very large score-speed win

| Rule (on `CPU_AND_NE`) | Best package `bt-fp16-iofp32` | Result |
| --- | --- | --- |
| Score ≥ 1.5× faster than v3 INT8 CPU, warm p50, B16×T16 | 2.18 ms vs 16.11 ms, **7.4×** | pass |
| Score ≥ 1.5× faster than v3 INT8 CPU, warm p50, B16×T32 | 4.10 ms vs 28.33 ms, **6.9×** | pass |
| Peak memory ≤ v3 INT8 CPU, same workload | **149.7 MiB** vs 15.8 MiB | **fail** |
| Top-1 = PyTorch FP32 on every ranking case | **132/133** | **fail** |
| Flipped pairs close to v3 INT8 CPU | 41 vs 92 (fewer) | pass |

The other packages:

- Every package that places its work on the ANE fails the same two rules. They all miss top-1 on the same case, and `bt-int8c-iofp32` shows the same ~150 MiB peak.
- The only package with 133/133 top-1 is `ane-int8b32`. It runs on the CPU, not the ANE (section 3), and is slower than v3 on CPU_AND_NE.

So no package meets all three rules, and the line ends here as the brief defines it. Two failures are narrow, and they are worth knowing before the line is closed for good:

- **Top-1.** The miss is a single case, `zh-pin-crosswoz-94d91a6a27/initials` (10 candidates). The same case fails in every FP16 package, and also in `bt-fp16-iofp32` on `CPU_ONLY`. So it is FP16 arithmetic, not the ANE itself. PyTorch FP16 on Windows had 133/133. On every other measure the FP16 packages rank closer to FP32 than v3 INT8 does: 41–47 flipped pairs vs 92, and exact order 95–100 vs 71.
- **Memory.** The 150 MiB is a transient peak during model load, even with the ANE program already cached by a warm-up process. After the 20-round workload, 17 MiB stays (v3: 2.3 MiB). ANE buffers held by system daemons are not counted, so the true cost is higher.

The step functions are not gated:

- `bt-fp16-iofp32` decode is faster than v3 on the ANE (1.59 vs 3.11 ms at P32 S16), and slightly faster on `CPU_ONLY` as well.
- The `ane` layout's step is not faster than v3.

## Deviations from the brief

1. **Unit tests.** `python -m unittest tests.test_deployment_zh_ane` ran as written (3 tests OK). The v3 task's `PYTHONPATH` problem does not occur here.
2. **Step 3, `plan`.** The brief's stage plans only the package's default function. coremltools 9.0's `MLComputePlan.load_from_path` has no `function_name`, so for the other five functions every op returns no usage. `plan-<pkg>.json` therefore holds counts for `score_b16_t8_p32` only and empty entries for the rest. These files are kept as run.
   - `tools/plan_per_function.py` (new, outside the repository code) handles each function separately: it copies the package, sets that function as `defaultFunctionName`, compiles it and plans it.
   - The results are in `plan-perfn-*.json`. Section 3 uses these.
3. **`bt-fp16` and `bt-int8c` with FP16 I/O.** Their score functions load and run on `CPU_AND_NE`. Their two step functions do not:
   - the per-function plan fails with "doesn't support the multi-function description syntax";
   - loading fails with "`.functionName` must be nil unless the model type is ML Program" (`logs/03b`, `03c`, `03d`, `04-accuracy-bt-*.log`, `04b-load-probe-bt.log`);
   - so the brief's `accuracy` crashed at the step check, and no `accuracy-bt-fp16.json` / `accuracy-bt-int8c.json` was written. The logs are kept.

   Fix, as the brief allows:
   - Both were reconverted with `--io fp32` as `bt-fp16-iofp32` and `bt-int8c-iofp32`. Every function then loads and runs on `CPU_ONLY`, `CPU_AND_NE` and `ALL` (`tools/load_probe.py`).
   - Accuracy, latency and memory for the `bt` FP16 / per-channel INT8 packages use these `-iofp32` packages.
   - The `ane` layout and `bt-int8b32` kept FP16 I/O, which works for them.
4. **Step 5.** No package was dropped: the six-package run took 394 s.
   - `bt-fp16-iofp32` and `bt-int8c-iofp32` replace the FP16-I/O versions, which would crash `measure` at the step load.
   - The best two packages, `bt-fp16-iofp32` and `bt-int8c-iofp32`, were also run on `CPU_AND_GPU` and `ALL` (`latency-gpu-all.json`).
5. **Step 6.** The brief's example uses `ane-fp16`. I used the best package instead, `bt-fp16-iofp32`, on `CPU_AND_NE` and `CPU_ONLY`, and added `bt-int8c-iofp32` on `CPU_AND_NE`.
6. **Step 7, power: skipped.** `sudo -n true` reports that a password is required. There is no passwordless sudo on this Mac, and I do not enter passwords. So no power was measured, and the `loop` runs that go with the power sampling were not run either.

## 1. Check (`check.json`, `logs/01-check.log`, `logs/01-unittest.log`)

- Both layouts equal v3 in FP32 on every grid function: max **6.0e-5** (Windows: 4.1e-5).
- PyTorch FP16 on the Mac CPU, 133 ranking cases:

  | Layout | top-1 | exact order | flipped pairs | max sum diff |
  | --- | ---: | ---: | ---: | ---: |
  | `bt` | 133 | 122 | 11 | 0.040 |
  | `ane` | 133 | 112 | 25 | 0.354 |

- Step (16 beam rows):
  - `bt`: top-1 16/16, top-5 order 16/16;
  - `ane`: top-1 16/16, top-5 order 15/16.
- Unit tests: 3/3 OK.

## 2. Convert (`logs/02-*.log`)

All six packages converted on the first try, plus the two `--io fp32` rebuilds.

| Package | Bytes | vs v3 INT8 (23,139,825) |
| --- | ---: | ---: |
| `bt-fp16` | 40,975,536 | 1.77× |
| `bt-fp16-iofp32` | 40,985,026 | 1.77× |
| `bt-int8c` | 21,088,408 | 0.91× |
| `bt-int8c-iofp32` | 21,097,882 | 0.91× |
| `bt-int8b32` | 22,246,296 | 0.96× |
| `ane-fp16` | 52,996,963 | 2.29× |
| `ane-int8c` | 28,419,139 | 1.23× |
| `ane-int8b32` | 29,847,363 | 1.29× |

The `ane` layout is ~12 MB (fp16) / ~7 MB (int8) bigger than `bt` for the same weights. Its conv-layout projections and per-head split appear not to share the tied embedding/head constant. Not investigated further.

## 3. Plan: ANE placement on `CPU_AND_NE` (`plan-perfn-*.json`)

Ops on the ANE / all ops, and the ANE share of estimated cost (c):

| Package | score t8 p32 | score t16 p32 | score t32 p32 | score t16 p64 | step p32 s16 | step p64 s32 |
| --- | --- | --- | --- | --- | --- | --- |
| `bt-fp16` | 391/406 c.84 | 391/406 c.90 | 391/406 c.94 | 391/406 c.91 | plan error | plan error |
| `bt-fp16-iofp32` | 394/409 c.84 | 394/409 c.90 | 394/409 c.94 | 394/409 c.91 | 420/450 c.86 | 420/450 c.89 |
| `bt-int8c` | 391/410 c.63 | 391/410 c.76 | 391/410 c.85 | 391/410 c.77 | plan error | plan error |
| `bt-int8c-iofp32` | 394/413 c.63 | 394/413 c.76 | 394/413 c.85 | 394/413 c.77 | 420/458 c.68 | 420/458 c.74 |
| `bt-int8b32` | 11/410 c.02 | 191/410 c.04 | 191/410 c.05 | 191/410 c.05 | 0/450 | 0/450 |
| `ane-fp16` | 1186/1203 c.83 | 1186/1203 c.88 | 1186/1203 c.92 | 1186/1203 c.89 | 1524/1551 c.74 | 1524/1551 c.84 |
| `ane-int8c` | 1186/1207 c.58 | 1186/1207 c.71 | 1186/1207 c.82 | 1186/1207 c.72 | 1524/1559 c.49 | 1524/1559 c.63 |
| `ane-int8b32` | 14/1207 c.03 | 14/1207 c.03 | 968/1207 c.12 | 14/1207 c.03 | 0/1559 | 0/1559 |

Notes:

- **Off the ANE in the FP16 / per-channel packages.** About 15–21 ops per function stay on the CPU: input `cast`, the mask (`greater_equal`, `select`, `tile`), the target `gather` and a few `add`/`expand_dims`. In the INT8 per-channel packages, the dequantization adds a few more. The remaining cost share in those packages is not attributed to a device.
- **Per-block-32 INT8.** It moves the graph off the ANE: `linear` with block-32 weights is "ANE unsupported" (46 ops), and the rest follows it to the CPU. Under `ALL`, the block-32 packages go to the GPU (`plan-*.json`, ALL rows).
- **`bt-fp16` / `bt-int8c` steps with FP16 I/O.** On `CPU_AND_NE` they produce no plan (deviation 3). Under `ALL` the plan puts them entirely on the GPU (`plan-perfn-v2-bt-*.json`).

## 4. Accuracy (`accuracy-*.json`, `logs/04*-accuracy-*.log`)

All rows are against PyTorch FP32 over 133 ranking cases. The v3 INT8 package on `CPU_ONLY`, scored by the same stage on the same cases, is the comparison row. The stage writes probe-vs-FP32 and v3-vs-FP32, not probe-vs-v3 directly.

| Package / unit | top-1 | exact order | flipped pairs | max sum diff | median max diff | missed top-1 case |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **v3 INT8 / CPU_ONLY** | **133** | 71 | 92 | 0.309 | 0.119 | – |
| `bt-fp16-iofp32` / NE | 132 | **100** | **41** | 0.287 | 0.068 | crosswoz-94d91a6a27/initials |
| `bt-fp16-iofp32` / CPU | 132 | 97 | 46 | 0.148 | 0.052 | crosswoz-94d91a6a27/initials |
| `bt-int8c-iofp32` / NE | 132 | 40 | 176 | 0.543 | 0.247 | crosswoz-94d91a6a27/initials |
| `bt-int8b32` / NE (runs on CPU) | 132 | 70 | 103 | 0.307 | 0.127 | risawoz-88ae3e71d0/full |
| `ane-fp16` / NE | 132 | 95 | 47 | 0.319 | 0.075 | crosswoz-94d91a6a27/initials |
| `ane-int8c` / NE | 132 | 43 | 172 | 0.622 | 0.248 | crosswoz-94d91a6a27/initials |
| `ane-int8b32` / NE (runs on CPU) | **133** | 63 | 104 | 0.533 | 0.150 | – |

Step check (`step_b16_p32_s16`, 16 beam rows, vs PyTorch FP32):

| Package / unit | top-1 | top-5 order | max log-prob diff |
| --- | ---: | ---: | ---: |
| `bt-fp16-iofp32` / NE | 16/16 | 14/16 | 0.077 |
| `bt-fp16-iofp32` / CPU | 16/16 | 14/16 | 0.195 |
| `bt-int8c-iofp32` / NE | 16/16 | 13/16 | 0.468 |
| `bt-int8b32` / NE | 16/16 | 13/16 | 0.234 |
| `ane-fp16` / NE | 16/16 | 14/16 | 0.075 |
| `ane-int8c` / NE | 16/16 | 13/16 | 0.463 |
| `ane-int8b32` / NE | 16/16 | 12/16 | 0.283 |

What this shows:

- FP16 weights on the ANE rank closer to FP32 than v3 INT8 does: half the flipped pairs and better exact order. They miss top-1 on exactly one case.
- Per-channel INT8 on top of FP16 compute is clearly worse than v3's block-32 INT8: about 2× v3's flipped pairs. Block-32 does not run on the ANE.

## 5. Latency (`latency.json`, `latency-gpu-all.json`, `logs/05*-latency*.log`)

Setup:

- Mac Python `predict()` wall time, including input conversion.
- Median of 3 interleaved passes × 100 calls, after 20 warm-up calls.
- v3 = shipped INT8 package (`ios18-int8-block32-kvbatch-v3`) on `CPU_ONLY`, same B/T/P/S.

### Warm p50 / p95, ms, `CPU_AND_NE` (and v3)

| Function | v3 CPU | `bt-fp16-iofp32` | `bt-int8c-iofp32` | `bt-int8b32` | `ane-fp16` | `ane-int8c` | `ane-int8b32` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| score B16 T8 P32 | 9.29 / 9.50 | **1.59** / 1.61 | 1.34 / 1.37 | 4.11 / 4.25 | 5.27 / 5.33 | 5.27 / 5.34 | 11.62 / 12.01 |
| score B16 T16 P32 | 16.11 / 16.32 | **2.18** / 2.27 | 2.11 / 2.20 | 7.03 / 7.15 | 4.71 / 4.83 | 4.71 / 4.80 | 11.19 / 12.02 |
| score B16 T32 P32 | 28.33 / 28.66 | **4.10** / 4.21 | 4.10 / 4.19 | 11.44 / 11.69 | 5.30 / 5.39 | 5.27 / 5.34 | 11.90 / 12.07 |
| score B16 T16 P64 | 16.82 / 16.95 | **2.36** / 2.47 | 2.31 / 2.40 | 7.09 / 7.14 | 4.83 / 4.90 | 4.82 / 4.89 | 11.91 / 12.53 |
| step B16 P32 S16 | 3.11 / 3.15 | **1.59** / 1.62 | 1.30 / 1.34 | 2.19 / 2.23 | 3.01 / 3.10 | 2.88 / 2.95 | 3.41 / 3.49 |
| step B16 P64 S32 | 3.33 / 3.39 | **2.42** / 2.45 | 2.12 / 2.14 | 3.03 / 3.06 | 3.64 / 3.68 | 3.48 / 3.52 | 4.91 / 4.98 |

Speed-up of score over v3 on `CPU_AND_NE`:

| Package | T8 | T16 | T32 | T16 P64 |
| --- | ---: | ---: | ---: | ---: |
| `bt-fp16-iofp32` | 5.9× | **7.4×** | **6.9×** | 7.1× |
| `bt-int8c-iofp32` | 6.9× | 7.7× | 6.9× | 7.3× |
| `ane-fp16` | 1.8× | 3.4× | 5.4× | 3.5× |

### Same packages on `CPU_ONLY` (warm p50, ms)

| Function | v3 | `bt-fp16-iofp32` | `bt-int8c-iofp32` | `bt-int8b32` | `ane-*` (all three) |
| --- | ---: | ---: | ---: | ---: | ---: |
| score T8 P32 | 9.29 | 3.71 | 3.74 | 3.78 | 9.90 |
| score T16 P32 | 16.11 | 6.17 | 6.17 | 6.22 | 9.77 |
| score T32 P32 | 28.33 | 10.15 | 10.16 | 10.22 | 11.45 |
| score T16 P64 | 16.82 | 6.55 | 6.54 | 6.64 | 10.55 |
| step P32 S16 | 3.11 | 1.78 | 1.78 | 2.19 | 3.00–3.29 |
| step P64 S32 | 3.33 | 2.33 | 2.32 | 3.02 | 4.51 |

FP16 compute alone, in v3's `bt` math, makes the CPU path 2.5–2.8× faster than v3's FP32-compute INT8 package. This is a side result: it needs no ANE. It costs FP16 accuracy (the same single top-1 miss, section 4) and 124 MiB resident on `CPU_ONLY` (section 6).

### `CPU_AND_GPU` and `ALL`, best two (warm p50, ms; `latency-gpu-all.json`, v3 re-measured: T16 16.22, T32 29.11)

| Function | `bt-fp16-iofp32` GPU | `bt-fp16-iofp32` ALL | `bt-int8c-iofp32` GPU | `bt-int8c-iofp32` ALL |
| --- | ---: | ---: | ---: | ---: |
| score T8 P32 | 4.03 | 1.59 | 6.20 | 1.35 |
| score T16 P32 | 6.04 | 2.15 | 6.36 | 2.08 |
| score T32 P32 | 10.01 | 4.09 | 10.74 | 4.09 |
| score T16 P64 | 6.46 | 2.36 | 6.82 | 2.30 |
| step P32 S16 | 3.29 | 3.41 | 4.27 | 4.26 |
| step P64 S32 | 4.19 | 4.19 | 5.11 | 5.11 |

- With `ALL`, scoring matches `CPU_AND_NE` (it goes to the ANE), but the step functions go to the GPU and are slower than v3.
- The GPU alone is slower than the ANE for scoring.

### Cold load and first call (`loads` in the latency JSONs)

`load_ms` is the time to load all six functions of a package in one process; the first-call column is the first `predict` per function after load. On the ANE units these are the first loads of each package in the session, so they include ANE compilation.

| Package / unit | load (6 functions) | first call, score | first call, step |
| --- | ---: | ---: | ---: |
| `bt-fp16-iofp32` / CPU_ONLY | 3.4 s | 10–26 ms | 16–28 ms |
| `bt-fp16-iofp32` / CPU_AND_NE | **17.7 s** | 6–9 ms | 13–24 ms |
| `bt-int8c-iofp32` / CPU_AND_NE | **21.4 s** | 5–8 ms | 12–22 ms |
| `bt-int8b32` / CPU_AND_NE | 13.8 s | 12–21 ms | 14–27 ms |
| `ane-fp16` / CPU_AND_NE | **39.6 s** | 10–12 ms | 17–29 ms |
| `ane-int8c` / CPU_AND_NE | **44.0 s** | 9–11 ms | 15–27 ms |
| `ane-fp16` / CPU_ONLY | 12.2 s | 13–17 ms | 17–31 ms |
| `bt-fp16-iofp32` / CPU_AND_GPU | 3.7 s | 44–953 ms | 60–74 ms |
| `bt-fp16-iofp32` / ALL | 12.8 s | 5–8 ms | 50–60 ms |

- On the ANE, the first load of a package costs 14–44 s for all six functions. This is ANE compilation, and the brief flags it as a known cost.
- Once loaded, the first call is already warm.
- The memory stage loads each configuration after a warm-up process, which the stage uses so that the ANE compiles and caches its program first. Its `after_load` is ≤ 2 MiB, but load time is not recorded there. Whether and for how long iOS keeps that cache is not known from this Mac run.

## 6. Memory (`memory.json`, `logs/06-memory.log`)

Setup:

- Per round: 16 × `step_b16_p32_s16` and 1 × `score_b16_t16_p32`; for v3, `scored` / `score` on the same shapes.
- 20 rounds, 5 fresh processes per configuration after one unmeasured warm-up process, medians.
- `phys_footprint` relative to `start` (coremltools imported, inputs built); models loaded precompiled (`.mlmodelc`).

| Configuration | after load | after first round | after 20 rounds | peak | min peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| **v3 INT8 / CPU_ONLY** | 1.2 | 2.0 | 2.3 | **15.8** | 15.7 |
| `bt-fp16-iofp32` / CPU_AND_NE | 0.8 | 17.0 | 17.1 | **149.7** | 149.2 |
| `bt-int8c-iofp32` / CPU_AND_NE | 1.9 | 18.1 | 18.1 | **149.8** | 149.3 |
| `bt-fp16-iofp32` / CPU_ONLY | 124.1 | 124.9 | 125.3 | **132.0** | 132.0 |

- **On the ANE units, the peak comes during load.** The `after_load` snapshot already carries the lifetime peak of ~150 MiB, while its current footprint is under 2 MiB. Per-snapshot peaks are in the JSON runs.
  - This happens with the ANE program already cached by the warm-up process.
  - The INT8 per-channel package has the same peak, so it is not the weight size.
  - About 17 MiB stays after the work, against v3's 2.3 MiB.
- **ANE buffers are undercounted.** The ANE program and its buffers live partly in system daemons (`aned`), which this process's `phys_footprint` does not count. The ANE numbers above are a lower bound.
- **FP16 weights on `CPU_ONLY` stay resident.** That package keeps 124 MiB resident from load on, about 3× its 41 MB of weights. v3 maps its INT8 weights from the file instead.

## 7. Power

Skipped: no passwordless `sudo` on this Mac (deviation 6). No `power-*.txt` and no `loop` runs.

## Files

| Path | What |
| --- | --- |
| `check.json` | step 1 |
| `bt-*/`, `ane-*/` (not in the ZIP) | packages, incl. `bt-fp16-iofp32/`, `bt-int8c-iofp32/` |
| `plan-*.json` | brief's `plan` stage as run (default function only) |
| `plan-perfn-*.json`, `plan-perfn-retry-*.json`, `plan-perfn-v2-*.json` | per-function plans (`tools/plan_per_function.py`); retry/v2 are the `bt` FP16-I/O reruns |
| `accuracy-*.json` | step 4 (`-cpu` = `CPU_ONLY` run of the best package) |
| `latency.json`, `latency-gpu-all.json` | step 5 |
| `memory.json` | step 6 |
| `tools/` | `plan_per_function.py`, `load_probe.py`, `run_accuracy.sh`, `run_measure.sh`, `run_rest.sh` |
| `logs/` | every stage, including the failed runs |

## Windows follow-up (2026-11-18)

**The two top-1 misses are near ties in FP32.** PyTorch FP32 sums for the top two candidates:

| Case | FP32 gap, top-1 vs top-2 (nats) |
| --- | ---: |
| `zh-pin-crosswoz-94d91a6a27/initials` (missed by every FP16 package) | 0.033 |
| `zh-pin-risawoz-88ae3e71d0/full` (missed by `bt-int8b32`) | 0.044 |

Both gaps are below the per-sum error of every package here, including v3 INT8 on CPU (median 0.119, max 0.309). v3 keeps these two cases by chance. On ranking quality, FP16 on the ANE is closer to FP32 than v3 (41 vs 92 flipped pairs). The accuracy rule fails on ties, not on ranking quality.

**The blocker is memory at load.**
- On `CPU_AND_NE` the process peak reaches ~150 MiB during model load, with the ANE program already cached. Per-channel INT8 weights give the same peak.
- A keyboard extension is limited by its peak footprint, so a transient peak at that size rules the path out as measured.
- FP16 weights on `CPU_ONLY` keep 124 MiB resident, which rules out the FP16 CPU path for the same reason.

**Verdict.** No-go, as the brief defines it. The ANE line ends for zh-v1. These results stand for later work:
- scoring on the ANE: 6.9–7.4× faster than v3 on the Mac, using the plain `bt` layout. The conv/per-head `ane` layout was slower and larger;
- per-block INT8 does not run on the ANE;
- first-time ANE compilation takes 14–44 s per package.

**Tooling gaps found by this run** (not fixed; the probe is closed):
- `plan` covers only the default function. coremltools 9.0 plans one function per compiled model; the Mac's `plan_per_function.py` sets `defaultFunctionName` for each function.
- Multifunction FP16-I/O step functions in the `bt` layout fail to load. Packages built with `--io fp32` load.
