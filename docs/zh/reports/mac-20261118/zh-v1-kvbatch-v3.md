# vimeml-tiny-zh-v1 batched KV v3 on Core ML (shared prefix, merged functions) — 2026-11-18

This follows the [v3 brief](../../mac/mac-zh-v1-kvbatch-v3.md). It continues in the checkout from the earlier tasks (`~/Documents/Sources/VimeML-zh-kvbatch`), pulled fast-forward to `main` `586a04e`.

**Before the pull**

- The untracked Mac copy of the kvsequence report (`docs/zh/reports/mac-20261116/zh-v1-kvsequence.md`) collided with the upstream file.
- It was moved to `outputs/deployment/mac-20261116-kvsequence-local-backup/` before the pull. It differs from the upstream copy by 3 diff lines of editing, so nothing was lost.
- After the pull, `git status` was clean.

**Unchanged**

- The checkpoint, the tokenizer, the INT8 recipe (linear_symmetric, per_block 32, 47 matrices) and the scoring rule.
- `RELEASE.json` and `SHA256SUMS.txt`: still `899ca51a…` / `58096d45…` before and after the release copy (`logs/12-release-copy.log`).
- `src/vimeml/deployment/zh_v1_batch3.py` and `scripts/zh/deployment/coreml_zh_v1_batch3.py`: no conversion or alignment problem needed a fix.
- Nothing was committed or pushed on the Mac.

**Environment**

- Hardware and OS: Apple M3, 16 GiB, macOS 27.3 (26C5035r), Xcode 27.3 (27C5020n).
- Python: 3.11.17 in a uv venv, with PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5 and sentencepiece 0.2.1. coremltools warns that torch 2.7.1 is untested; no stage was affected.
- Core ML: mlprogram, iOS 18 target, FP32 compute, `CPU_ONLY`.
- Power and load: AC power, and `pmset -g therm` recorded no warning. Load average was about 3.4.

All outputs are under `outputs/deployment/zh-v1-kvbatch-v3/` (`$O`): stage logs in `logs/`, helper scripts in `tools/`. No output was overwritten. Superseded memory attempts are kept as evidence (section 7).

## Summary

- **Gates:** every stage passed on the first try.
  - FP32 alignment against PyTorch has max 4.2e-5.
  - Vime's INT8 gate against the stateless package has max 4.15e-5 for both `int8-merged` and `int8-merged-c4`, with zero rows outside tolerance (3e-4 per value, 1e-3 per sum).
- **Size:**
  - `int8-merged` holds `logits`, `scored` and `score` in one 23.1 MB package with one shared weight copy. The three separate packages would take 68.4 MB.
  - The shipped v2 pair (kvbatch-v2 + kvsequence) is 45.5 MB, or 68.3 MB with stateless.
- **Latency (Mac p50, B=16):**
  - v3 decode is equal to or faster than kvbatch-v2: 2.9–3.3 ms vs 3.0–3.9 ms.
  - v3 `score` is 4–12% faster than kvsequence: T16 16.1–17.1 ms vs 17.2–17.8 ms.
- **Memory, the key question:** v3 uses clearly less memory. These are clean runs: precompiled `.mlmodelc`, 5 fresh processes, medians, MiB over start.

  | T | v2-shipped peak | v3 peak | v3-alone peak |
  | --- | ---: | ---: | ---: |
  | 16 | 40.5 | 18.5 | 17.8 |
  | 32 | 70.2 | 31.4 | 31.2 |
  | 64 | 150.0 | 56.8 | 56.2 |

  - After the workload, v2-shipped retains 16–57 MiB, while v3 retains about 4.5 MiB at every T.
  - Load cost is about 2 MiB for every configuration, because weights are file-mapped. The mapped weight files are 68.3 MB for v2-shipped, 45.9 MB for v3 and 23.1 MB for v3-alone.
- **Search:**
  - Against the v2 package, 125/125 cases match exactly.
  - Against the FP32 fixture: 46 exact, 86 same set and 125 top-1, identical to v2.
- **Fixture:** against stateless INT8, max 3.2e-5 per step value and 1.7e-5 per score sum.
- **Decisions** (see the end):
  - ship v3 (`int8-merged`);
  - do **not** choose -c4;
  - the stateless package can be dropped, after one on-device check by Vime.

## 0. Unit tests (`logs/00-*.log`)

- The brief's command, `python -m unittest tests.test_deployment_zh_v1_batch3`, fails with `ModuleNotFoundError: test_deployment_zh_v1_batch`. The v3 test module imports the v1 test module by bare name, and `tests/` is not on `sys.path`.
- With `PYTHONPATH=tests python -m unittest tests.test_deployment_zh_v1_batch3`, all 5 tests pass.
- This is an invocation issue, not a code change. Suggested fix upstream: make `tests` a package import, or put the `PYTHONPATH` in the brief.

## 1. PyTorch check (`torch-check.json`, `logs/01-check.log`)

- Passed. Eager vs reference: max 4.1e-5 over the `logits`, `scored` and `score` cases (batch 1/5/16, prefix 11).
- The 4-chunk `score` matches the full head to 3.5e-5 per sum.

## 2–6. Conversion, merge, compression and alignment

Conversion and merge (`logs/02-*`, `03-*`):

- Four FP32 packages were converted: `fp32-logits`, `fp32-scored`, `fp32-score`, `fp32-score-c4`.
- They were merged into `fp32-merged` and `fp32-merged-c4`.

FP32 alignment against PyTorch (`fp32-merged-align.json`, `fp32-merged-c4-align.json`):

| Function | Max abs error |
| --- | ---: |
| `logits` | ≤ 4.20e-5 |
| `scored` | ≤ 2.68e-5 |
| `score` (74 continuations), c1 | 2.46e-5 per sum |
| `score` (74 continuations), c4 | 2.44e-5 per sum |

INT8 compression and merge (`logs/05-*`, `06-*`):

- Each function was compressed separately (`int8-logits`, `int8-scored`, `int8-score`, `int8-score-c4`), then merged into `int8-merged` and `int8-merged-c4`.

INT8 gate against stateless `ios18-int8-block32` (`int8-merged-align.json`, `int8-merged-c4-align.json`):

| Function | `int8-merged` | `int8-merged-c4` |
| --- | ---: | ---: |
| `logits` | 4.15e-5 | 4.15e-5 |
| `scored` | 2.18e-5 | 2.18e-5 |
| `score` (per sum) | 3.70e-5 | 3.99e-5 |
| rows outside tolerance | 0 | 0 |

### Package sizes (bytes)

| Package | Merged | Sum of separate function packages |
| --- | ---: | ---: |
| FP32 | 80,697,965 | 241,122,305 |
| FP32 -c4 | 99,593,322 | 259,968,446 |
| **INT8** | **23,139,825** | 68,400,205 |
| INT8 -c4 | 28,470,632 | 73,681,796 |

Per INT8 function package:

| Function package | Bytes |
| --- | ---: |
| `logits` | 22,812,208 |
| `scored` | 22,814,444 |
| `score` | 22,773,553 |
| `score-c4` | 28,055,144 |

Shipped packages for comparison:

| Package | Bytes |
| --- | ---: |
| stateless | 22,792,958 |
| kvbatch-v2 | 22,774,484 |
| kvsequence | 22,772,997 |
| v2 pair | 45,547,481 |
| v2 pair + stateless | 68,340,439 |

The merge shares one weight copy across all three functions: the merged package is only ~0.3 MB larger than a single function package.

**Why -c4 is bigger (`logs/06b-c4-constants.log`)**

- The chunked head stores four extra (3072, 384) INT8 constexpr weights beside the full (12288, 384) tied embedding. In effect it holds a second copy of the head (+5.3 MB).
- Its op counts are 51 constexpr vs 47, and 49 linear vs 46.
- The slices are not deduplicated against the embedding.

## 7. Latency on the Mac (`latency.json`, `latency-c4.json`, `latency-table.txt`, `logs/07-*`)

Setup:

- The `measure` stage with `--prompts 16 48 --own 1 8 16 --score-tokens 8 16 --rows 16 --repeats 100`.
- v3 is interleaved with the shipped kvbatch-v2 (decode) and kvsequence (score) in one process.
- Values are the median of 3 interleaved passes, p50 in ms, B=16.

`int8-merged`:

| prompt | own | T | v2 decode | v3 decode | kvsequence score | v3 score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 8 | 2.99 | 3.18 | 10.21 | 9.57 |
| 16 | 1 | 16 | 2.95 | 2.91 | 17.30 | 16.37 |
| 16 | 8 | 8 | 2.96 | 3.02 | 10.04 | 9.24 |
| 16 | 8 | 16 | 2.98 | 3.03 | 17.19 | 16.22 |
| 16 | 16 | 8 | 3.09 | 3.10 | 10.08 | 9.34 |
| 16 | 16 | 16 | 3.10 | 3.11 | 17.18 | 16.25 |
| 48 | 1 | 8 | 3.22 | 3.01 | 10.60 | 10.21 |
| 48 | 1 | 16 | 3.22 | 3.02 | 17.79 | 16.91 |
| 48 | 8 | 8 | 3.30 | 3.15 | 10.62 | 10.24 |
| 48 | 8 | 16 | 3.51 | 3.30 | 17.80 | 17.11 |
| 48 | 16 | 8 | 3.90 | 3.34 | 11.23 | 9.85 |
| 48 | 16 | 16 | 3.51 | 3.31 | 17.59 | 16.13 |

`int8-merged-c4`, `score` column only (decode is the same function):

| prompt | own | T | kvsequence | v3c4 score |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1/8/16 | 8 | 9.87–11.34 | 8.84–9.17 |
| 16 | 1/8/16 | 16 | 17.04–17.35 | 14.93–15.32 |
| 48 | 1/8/16 | 8 | 10.61–10.69 | 9.33–9.58 |
| 48 | 1/8/16 | 16 | 17.68–17.73 | 15.38–15.84 |

What this shows:

- v3 decode is on par with kvbatch-v2 at prompt 16, and 0.2–0.6 ms faster at prompt 48. The v2 rows carry the shared prompt in every row's cache; v3 broadcasts it.
- v3 `score` beats kvsequence by 4–12%.
- -c4 `score` is another 6–10% faster than c1, so it is within the brief's 10% latency bound and better.

## 8. Memory

### `memory` stage as briefed (`memory.json`, `memory-T32.json`, `memory-T64.json`, `logs/08-memory*.log`)

Setup:

- `--prompt 48 --steps 16 --score-tokens 16|32|64 --rounds 20 --runs 3`, 3 fresh processes per configuration, medians, MiB over `start`.
- v2-shipped is stateless + kvbatch-v2 + kvsequence. `-alone` means without stateless.

| Config | T16 load | T16 after rounds | T16 peak | T32 peak | T64 peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2-shipped | 164.6 | 178.9 | 203.1 | 232.7 | 312.0 |
| v3 | 164.5 | 166.6 | 180.8 | 192.4 | 217.7 |
| v3-alone | 159.9 | 162.9 | 176.4 | 195.7 | 238.7 |
| v3c4 | 168.7 | 170.4 | 186.0 | 233.5 | 266.9 |
| v3c4-alone | 166.1 | 168.9 | 183.9 | 233.9 | 265.1 |

**These absolute numbers are polluted.** The relative pattern holds: v2-shipped's peak grows much faster with T. The pollution has three causes:

1. The child imports coremltools, which pulls in torch, inside `load_set`, after the `start` snapshot. That puts ~130–160 MiB of Python imports into every `load`.
2. It builds the inputs for both v2 and v3 (~1.2 GB) before `start`. On this macOS, freed large allocations stay in `phys_footprint` until reused, which adds ±30 MiB of noise.
3. Loading an `.mlpackage` compiles it in-process; the compiler's temporary buffers add more noise.

The output is kept unchanged as the brief's stage result.

### Clean attribution (`memory-clean4-T16/32/64.json`, `tools/memory_clean4.py`, `logs/08e-*`)

Each configuration ran in 5 fresh processes. In each child:

1. Import coremltools.
2. Build only the inputs that configuration uses, kept alive for the whole run.
3. Take the `start` snapshot.
4. Load each model or function one at a time with `CompiledMLModel` from `.mlmodelc` precompiled outside the measured process (`compiled-for-memory/`), as an app does. Take a snapshot after each.
5. Run 20 rounds of the stage workload: prompt, 16 decodes with own caches 1..16, `logits` (v3), one `score` of T tokens, at B=16.

Values are `phys_footprint` (proc_pid_rusage V4) minus `start`, in MiB, medians of 5. Most runs are within ±1 MiB of the median; per-run values are in the JSON.

| Config | T | after stateless | after load | after rounds | peak | min peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2-shipped | 16 | 1.9 | 2.2 | 16.3 | **40.5** | 40.5 |
| v3 | 16 | 1.9 | 2.3 | 4.3 | **18.5** | 18.4 |
| v3-alone | 16 | – | 1.4 | 4.3 | **17.8** | 17.7 |
| v3c4 | 16 | 1.9 | 2.4 | 6.1 | 21.6 | 20.1 |
| v3c4-alone | 16 | – | 1.5 | 4.4 | 19.4 | 19.3 |
| v2-shipped | 32 | 1.9 | 2.2 | 23.2 | **70.2** | 70.1 |
| v3 | 32 | 1.9 | 2.4 | 4.6 | **31.4** | 31.3 |
| v3-alone | 32 | – | 1.4 | 5.0 | **31.2** | 30.6 |
| v3c4 | 32 | 1.9 | 2.4 | 4.5 | 35.1 | 34.9 |
| v3c4-alone | 32 | – | 1.4 | 4.4 | 34.5 | 26.9 |
| v2-shipped | 64 | 2.0 | 2.3 | 57.3 | **150.0** | 149.7 |
| v3 | 64 | 1.9 | 2.4 | 4.6 | **56.8** | 27.6 |
| v3-alone | 64 | – | 1.4 | 4.5 | **56.2** | 56.0 |
| v3c4 | 64 | 2.0 | 2.4 | 4.6 | 65.1 | 65.1 |
| v3c4-alone | 64 | – | 1.4 | 4.4 | 64.4 | 64.4 |

How to read this:

- **Load.** Loading a compiled model adds only ~0.1–2 MiB of `phys_footprint` in every configuration. The weights are file-backed and mapped, so `phys_footprint` does not count them. The real weight cost is the mapped files:
  - v2-shipped: 68.3 MB (three weight files);
  - v3: 45.9 MB (stateless + merged);
  - v3-alone: 23.1 MB (merged only).
  
  Per-function snapshots show that the three functions of the merged package share that one mapping: `after_fn_scored`, `after_fn_score` and `after_fn_logits` differ by ≤ 0.2 MiB.
- **Retained after the workload.**
  - v2-shipped keeps 16 → 23 → 57 MiB as T grows from 16 to 64. kvsequence's `[B,T,12288]` logits buffers stay allocated.
  - v3 keeps ~4.5 MiB at every T. `score` returns `target_logprobs` only.
- **Peak.** v3 is 2.2–2.6× lower than v2-shipped and grows ~0.8 MiB per T token from T16 to T64, against v2-shipped's ~2.3 MiB per T token.
- **-c4.** Its peak is **higher** than c1 at every T: +3, +4 and +8 MiB at T16, T32 and T64. Its load is the same, and its mapped weights are 5.3 MB larger. The chunked head does not lower the scoring peak in Core ML's CPU execution.
- **Outliers.** The two "min peak" outliers (v3 T64 27.6; v3c4-alone T32 26.9) are single runs. In those runs the allocator reused memory freed during setup, so both `after_load` and peak went negative by ~30 MiB. The medians are unaffected.

Superseded attempts, kept as evidence:

- `memory-clean-T*.json` (`tools/memory_clean.py`): freed inputs stayed in the footprint, so v3 loads showed −340 MiB.
- `memory-clean2-probe.json`: `malloc_zone_pressure_relief` released 0, so the artifact remained.
- `memory-clean3-T*.json` (all inputs kept, `.mlpackage` loads): bimodal ±35 MiB from in-process compilation. This motivated `.mlmodelc` loading in v4.

## 9. Search (`search/`, `logs/09-search.log`)

The check ran `$C search $O/int8-merged outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl $O/search` over 125 cases.

| Against | exact | same set | top-1 equal | max common-score diff |
| --- | ---: | ---: | ---: | ---: |
| v2 package (kvbatch-v2) | **125** | 125 | 125 | 1.7e-5 |
| FP32 fixture | 46 | 86 | 125 | 0.56 |

- Both rows are as expected: exact against v2, and the same numbers as v2 against FP32 (the INT8 effect).
- Search p50 was 85.0 ms for v3 vs 82.7 ms for v2, interleaved. p95 was 184.8 vs 185.9.
- The memory runs were in progress in the background at the time, so treat these timings as indicative only.

## 10. Fixture (`fixture/`, `logs/10-fixture.log`)

- The fixture was built from `int8-merged`, the chosen package.
  - Steps: 16 rows through `scored` and `logits`, prefix from a B=1 prompt call.
  - Scores: 8 cases through `score`, up to 16 rows per call.
- Against stateless INT8: max **3.24e-5** per step value and **1.70e-5** per score sum, well inside 3e-4 / 1e-3.
- Files: `step.jsonl`, `step.npz`, `score.jsonl`, plus meta with the package manifest `7e4c01d2…` and the tokenizer `68bd37c4…`.

## 11. Client resource and release copy

- `$O/ios-kvbatch-v3/` (`logs/11-ios.log`) contains:
  - `Resources/TinyChineseV1KVBatchV3INT8.mlmodelc`;
  - `Resources/VimeLMManifestZhV1KVBatchV3.json` (section `kv_batch_v3`, with per-file hashes of the compiled model);
  - `KVBatchV3Fixture-step.jsonl`, `-step.npz`, `-score.jsonl`, `-meta.json` and `manifest.json`.
- The earlier v1, v2 and kvsequence resources are untouched.
- Release copy: `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch-v3/`.
  - It is a byte-identical copy of `$O/int8-merged`: `diff -r` is clean, 22 MiB.
  - `RELEASE.json` and `SHA256SUMS.txt` have the same hashes before and after (`logs/12-release-copy.log`). Registering the copy in the release inventory is left to Windows, as before.

## 12. iPhone

Not done in this session. Vime should check on device:

- the peak with v3 loaded alone at T = 16 and 32;
- that the `logits` function (no prefix, empty row cache) covers the stateless package's use, with the latency of that path.

## Decision

**1. Ship v3: yes, `int8-merged`.**

- Both alignments pass with zero rows outside tolerance.
- Search is identical to the v2 package (125/125 exact), and the fixture agrees with stateless to 3e-5.
- Latency is equal to or better than v2 for decode, and 4–12% better for scoring.
- Memory is clearly lower:
  - mapped weights: 45.9 MB vs 68.3 MB (23.1 MB without stateless);
  - retained after work: ~4.5 MiB vs 16–57 MiB;
  - peak: 18.5 / 31.4 / 56.8 MiB vs 40.5 / 70.2 / 150.0 MiB at T 16 / 32 / 64.
- With v3, the kvsequence T cap from the last report can be relaxed. T ≤ 16 costs a ~18 MiB peak and ~4 MiB retained. T ≤ 32 costs ~31 MiB peak.

**2. Choose -c4: no.**

- The brief's rule is to choose it only if it lowers the scoring peak by more than its extra size. It does not: its peak is higher at every T (+3 to +8 MiB), and the package is 5.3 MB bigger because of the duplicated head constants.
- Its 6–10% faster scoring does not outweigh that.

**3. Drop the stateless package: yes on the Mac evidence; confirm once on device.**

- The `logits` function with no prefix and no row cache passed the INT8 gate against stateless (4.15e-5) and the fixture's `logits` step check.
- v3-alone has the same retained and peak memory as v3 at every T, and maps 22.8 MB less.
- What is not measured here: the latency of full-prompt logits through `logits` vs the stateless package. Vime should check this on device before removing the stateless resource.

## Code changes

None. `src/vimeml/deployment/zh_v1_batch3.py` and `scripts/zh/deployment/coreml_zh_v1_batch3.py` are at `586a04e`. The only deviation from the brief is the unit-test invocation (`PYTHONPATH=tests`, section 0). The memory helpers are new files under `$O/tools/` and are not repository code.

## Files

| Path | What |
| --- | --- |
| `torch-check.json` | PyTorch check |
| `fp32-{logits,scored,score,score-c4}/`, `fp32-merged{,-c4}/`, `fp32-merged{,-c4}-align.json` | FP32 packages and alignment |
| `int8-{logits,scored,score,score-c4}/`, `int8-merged{,-c4}/`, `int8-merged{,-c4}-align.json` | INT8 packages and Vime's gate |
| `latency.json`, `latency-c4.json`, `latency-table.txt` | latency |
| `memory.json`, `memory-T32.json`, `memory-T64.json` | `memory` stage as briefed |
| `memory-clean4-T16/32/64.json`, `compiled-for-memory/` | clean memory attribution (used for the decisions) |
| `memory-clean-T*.json`, `memory-clean2-probe.json`, `memory-clean3-T*.json` | superseded attempts, kept as evidence |
| `search/`, `fixture/`, `ios-kvbatch-v3/` | search, fixture, client resource |
| `tools/` | `run_build.sh`, `run_measure_memory.sh`, `memory_clean{,2,3,4}.py` |
| `logs/` | every stage |

## Windows follow-up (2026-11-18)

- The results ZIP was checked against its SHA256SUMS. The release copy is byte-identical to `int8-merged` and passes `zh_v1_batch3.verify_merged`. Its manifest records the conversion source `0a05cef6…`, which is `zh_v1_batch3.py` at `586a04e`.
- Its `weight.bin` is byte-identical to the sequence package's: the three functions share the one INT8 weight file that every v2 package already carried.
- The release inventory now lists the four `ios18-int8-block32-kvbatch-v3/` files and `coreml_manifest_sha256.kvbatch_v3_int8` = `7e4c01d2…`.
- The release README describes the v3 package. The v2 package is marked as superseded for beam search.
- `verify_release.py --hashes` passes on 382 files.
- The Hugging Face release was updated in place at revision `32766748114c38ffc2252b645a51758e066e80b8` (parent `ff3dab5f…`). A fresh download of that revision passes `verify_release.py --hashes`.
- Section 0: the v3 test module now puts `tests/` on `sys.path`, so `python -m unittest tests.test_deployment_zh_v1_batch3` works from the repository root.
- Section 8: the `memory` stage now follows `memory_clean4.py`:
  - coremltools is imported and all inputs are built before `start`;
  - packages are compiled once outside the measured processes, and `CompiledMLModel` loads them;
  - there is a snapshot after each model or function;
  - the default is 5 runs, reporting medians and the minimum peak.
