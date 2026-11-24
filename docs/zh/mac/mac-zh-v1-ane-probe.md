# vimeml-tiny-zh-v1 Neural Engine probe (step 0): Mac brief

This task decides whether the Apple Neural Engine (ANE) is worth a deployment path for the 20M zh models. It uses the zh-v1 weights because the architecture equals v2-20m. The zh-v1 release and its packages stay unchanged.

## Why

The ANE was set aside during ja-v1. That test used a stateless full-sequence package (FP16, enumerated lengths 16/32/64/128, per-channel INT8, full `[1,T,V]` logits). It ran 3-4× slower than CPU, and the cause was never isolated. Every zh-v1 package since then is FP32 and runs `CPU_ONLY`, so none of them can use the ANE.

The ANE needs FP16 compute and fixed shapes. It also prefers channels-first tensors and 1×1 convolutions over Linear layers, with attention split per head. Two workloads matter:

- **Sentence scoring** (`score`, B16×T16 over a shared prefix): about 10 GFLOP per call. This is the likely ANE win.
- **Beam decode** (`scored`, B16, T=1): small matrices. CPU may stay faster here, which would make a hybrid the outcome.

## What is measured: `vimeml.deployment.zh_ane`

Two layouts of the same math. Both take the v3 inputs (a shared prefix plus per-row caches) with fixed shapes.

| Layout | Description |
| --- | --- |
| `bt` | v3's math and cache layouts. The additive mask is -1e4, built with `where` (v3's -1e9 becomes -inf in FP16, and 0·inf gives NaN). |
| `ane` | Hidden state `[B,384,1,T]`, 1×1 `conv2d` for every projection, attention per head. Keys are (9,B,S,384) and values (9,B,384,S). RMSNorm inputs are scaled by 1/16 so x² sums stay within FP16 range (the real model's hidden values reach ~53). |

Function grid, one multifunction package per layout and weight format:

| Function | Shape |
| --- | --- |
| `score_b16_t8_p32`, `score_b16_t16_p32`, `score_b16_t32_p32`, `score_b16_t16_p64` | B, T continuation tokens, P prefix slots; output `target_logprobs` [B,T] |
| `step_b16_p32_s16`, `step_b16_p64_s32` | T = 1, S row-cache slots, 256 `allowed` ids; outputs `logsumexp`, `allowed_logits`, `new_key`, `new_value` |

Weights: `fp16`, `int8c` (per-channel INT8) and `int8b32` (per-block-32 INT8). Compute precision is always FP16. Inputs and outputs are FP16 by default (`--io fp32` is available).

## Windows checks

| Check | Result |
| --- | --- |
| Both layouts equal v3 in FP32 on the real weights, on every grid function (`check`) | max 4.1e-5 |
| Unit tests (`tests/test_deployment_zh_ane.py`): FP32 equality incl. new keys/values, traced = eager, layout conversion | 3/3 |
| PyTorch FP16 on CPU, `bt`: 133 ranking cases | top-1 133/133, exact order 122/133, 11 flipped pairs, max sum diff 0.034 |
| PyTorch FP16 on CPU, `ane`: 133 ranking cases | top-1 133/133, exact order 110/133, 26 flipped pairs, max sum diff 0.33 |
| PyTorch FP16, 16 beam rows (step) | `bt` and `ane` both top-1 16/16, top-5 order 15/16, max log-prob diff 0.030 / 0.037 |

The 133 ranking cases are the 125 search fixtures (context plus the distinct top candidates) and the 8 sentence-scoring fixtures. PyTorch half precision on CPU is not the ANE's arithmetic, so the Core ML run on the Mac is the measurement.

`coremltools` was not run on Windows.

## Mac steps

Run from the repository root with the venv used for the v3 task. Use `OUT=outputs/deployment/zh-v1-ane-probe`. Every stage refuses to overwrite an existing output. `check` and `accuracy` read `outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl`, the same file the v3 search stage used.

1. **Check.**
   ```
   python scripts/zh/deployment/ane_probe_zh_v1.py check $OUT/check.json
   python -m unittest tests.test_deployment_zh_ane
   ```
2. **Convert.** Six packages:
   ```
   for L in bt ane; do for W in fp16 int8c int8b32; do
     python scripts/zh/deployment/ane_probe_zh_v1.py convert $OUT/$L-$W --layout $L --weights $W
   done; done
   ```
   If a conversion fails, record the error and continue with the others. If FP16 I/O fails, try `--io fp32` for that package and say so.
3. **Plan.** For every package:
   ```
   python scripts/zh/deployment/ane_probe_zh_v1.py plan $OUT/<pkg> $OUT/plan-<pkg>.json --units CPU_AND_NE ALL
   ```
   This reports, per function, how many ops and how much estimated cost go to the CPU, GPU and ANE, and which op types stay off the ANE. Put the op counts for `CPU_AND_NE` in the report.
4. **Accuracy** on `CPU_AND_NE`, for every package that converted:
   ```
   python scripts/zh/deployment/ane_probe_zh_v1.py accuracy $OUT/<pkg> $OUT/accuracy-<pkg>.json --unit CPU_AND_NE
   ```
   The v3 INT8 package on `CPU_ONLY` is scored on the same cases (`v3_int8_cpu`). Both are compared against PyTorch FP32. For the best package, also run `--unit CPU_ONLY`.
5. **Latency.** All packages on `CPU_ONLY` and `CPU_AND_NE`, interleaved with v3 INT8 on `CPU_ONLY` with the same B/T/P/S:
   ```
   python scripts/zh/deployment/ane_probe_zh_v1.py measure $OUT/latency.json \
     bt-fp16=$OUT/bt-fp16 bt-int8c=$OUT/bt-int8c bt-int8b32=$OUT/bt-int8b32 \
     ane-fp16=$OUT/ane-fp16 ane-int8c=$OUT/ane-int8c ane-int8b32=$OUT/ane-int8b32
   ```
   If it takes too long, drop the packages that placed little work on the ANE in step 3 and say which ones. Then run the best one or two packages with `--units CPU_AND_GPU ALL` as well.
6. **Memory.** v3 INT8 on `CPU_ONLY` is always included. Add the best package on `CPU_AND_NE` and on `CPU_ONLY`:
   ```
   python scripts/zh/deployment/ane_probe_zh_v1.py memory $OUT/memory.json \
     ane-fp16=$OUT/ane-fp16:CPU_AND_NE ane-fp16=$OUT/ane-fp16:CPU_ONLY
   ```
   The workload per round is 16 step calls and one score call. Each configuration runs in fresh processes after one unmeasured warm-up process. ANE buffers partly live in system daemons, so `phys_footprint` may undercount them. Note this next to the numbers.
7. **Power**, if `sudo powermetrics` is available. In one terminal:
   ```
   sudo powermetrics --samplers cpu_power,ane_power -i 1000 -n 25 > $OUT/power-<label>.txt
   ```
   In another, at the same time:
   ```
   python scripts/zh/deployment/ane_probe_zh_v1.py loop <pkg or v3> score_b16_t16_p32 --unit CPU_AND_NE --seconds 30
   ```
   Do this for v3 (`loop v3 score_b16_t16_p32`, CPU), for the best package on `CPU_AND_NE`, and for `step_b16_p32_s16` of both. Report the mean CPU and ANE power and the calls per second. Energy per call is power divided by calls per second.

## Go / no-go

**Go** to an ANE deployment design for v2-20m if all of these hold for some package and function on `CPU_AND_NE`:

- the score functions are at least 1.5× faster than v3 INT8 on `CPU_ONLY`, at warm p50, on B16×T16 and B16×T32;
- peak memory is not above v3 INT8 on `CPU_ONLY` for the same workload;
- top-1 equals PyTorch FP32 on every ranking case, and flipped pairs are close to v3 INT8 on CPU.

Step functions are reported but not gated. If only scoring wins, the outcome is a hybrid: decode on CPU and scoring on the ANE.

**No-go** otherwise: report the numbers and the line ends. Report the cold-load and first-call times in both cases, because ANE compilation on first load is a known cost.

## Report

Write `docs/zh/reports/mac-<date>/zh-v1-ane-probe.md` with:

- the environment (chip, macOS, coremltools, Python);
- a table per stage: package size, ANE op share, accuracy (top-1, exact order, flipped pairs, max sum diff vs PyTorch FP32 and vs v3), warm p50/p95 per function and unit next to v3, load and first-call times, memory and power;
- a go / no-go verdict against the rules above;
- every deviation from these steps.

Return the report and the JSON outputs. Do not return the `.mlpackage` directories.
