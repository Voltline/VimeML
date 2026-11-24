# vimeml-tiny-zh-v1 sentence scoring on Core ML (sequence head) — 2026-11-16

This follows the [sequence-head brief](../../mac/mac-zh-v1-kvsequence.md). It continues in the checkout from the earlier tasks (`~/Documents/Sources/VimeML-zh-kvbatch`), pulled fast-forward to `main` `fcc1750`.

**Before the pull**

- The checkout still held the v2 task's uncommitted changes: the `zh_v1_batch2.py` shape fix and the untracked v2 report. Upstream had already taken both (`a1a3a63`).
- To let the pull fast-forward, the local copies, a diff and the original untracked report were saved to `outputs/deployment/mac-20261115-v2-local-backup/`. The file was then restored to `HEAD`.
- Nothing local was lost. The upstream v2 report differs from the Mac's by a few lines of editing.

**Unchanged**

- The checkpoint, the tokenizer, the INT8 recipe (block32, threshold 2048, 47 matrices) and the scoring rule.
- `RELEASE.json` and `SHA256SUMS.txt` (still `899ca51a…` / `58096d45…`, recorded in `logs/10-release-meta-before.sha`).
- `src/vimeml/deployment/zh_v1_batch2.py` (no conversion or alignment problem needed a fix).
- Nothing was committed on the Mac.

**Environment**

- Hardware and OS: Apple M3, 16 GiB, macOS 27.3 (26C5035r), Xcode 27.3 (27C5020n).
- Python: 3.11.17 in a uv venv, with PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5 and sentencepiece 0.2.1.
- Core ML: mlprogram, iOS 18 target, FP32 compute, `CPU_ONLY`.
- Power and load: AC power, and `pmset -g therm` recorded no warning. Load average was about 4 during the measurements.

All outputs are under `outputs/deployment/zh-v1-kvsequence/` (`$O`): stage logs in `logs/`, helper scripts in `tools/`. No output was overwritten, and no stage failed.

## Summary

- **Gates:** every stage passed on the first try.
  - `check` lists `sequence-Tflex` as traced and checked.
  - FP32 alignment against PyTorch, and Vime's INT8 gate against the stateless package, both have max 3.34e-5 (tolerance 3e-4 per position, 1e-3 per sum).
- **The two questions:**
  1. **B=16, T=16 latency:** one sequence call takes **17.5–18.4 ms**. 16 chained `unified-scored` decode calls at B=16 take **54.5–62.2 ms**. That is **3.1–3.4× faster** for prefixes of 16–64 slots, Mac p50, timed in one process with the two packages interleaved.
  2. **Memory over T:** it **does grow** with T, so the curve is below. These are Core ML numbers with inputs preallocated, MiB added over stateless + inputs.
     - T ≤ 8: within ~1 MiB of `unified-scored` (34–40 vs 40.8).
     - T=16: +9 MiB (49.9), peak 66.7.
     - T=32: +23 MiB (63.4), peak 103.
     - T=64: +63 MiB (104), peak 169.
- **Fixture:** 8 cases, 39 candidates, 13 calls.
  - Package vs stateless INT8: max 3e-5 per sum.
  - Package vs PyTorch FP32: max 0.17 per sum, which is INT8 quantization.
  - The candidate ranking equals FP32's in 8/8 cases.
- **Decision: ship as an additional resource, with T capped per call.**
  - At **T ≤ 8** the footprint meets the brief's "within a few MiB" rule.
  - **T=16** costs ~9 MiB more retained and ~25 MiB more at peak.
  - Recommended cap: **8**, or 16 if Vime accepts that extra memory.
  - Even split into T=8 calls, a B=16 batch of 16-token continuations costs ~22 ms on the Mac, against ~55 ms for 16 decode calls.
  - Not measured: both batch packages loaded together. That is the shipping configuration if sentence scoring and search coexist, and it should be checked on device.

## 1. PyTorch check (`torch-check.json`, `logs/01-check.log`)

| Check | Result |
| --- | ---: |
| Decode (eager and traced) | max 5.01e-5 |
| Scored head vs log-softmax | 1.91e-6 |
| Sequence head per position | max 2.19e-5 |
| `score_continuations` sums | max 1.91e-5 |
| Traced graphs | `logits-T1`, `logits-Tflex`, `scored-T1`, `sequence-Tflex`: traced and checked |

The run printed `passed`. The sequence-head numbers agree with the Windows checks (1.1e-5 per position, 2.7e-5 per sum).

## 2–5. Conversion, compression and alignment

- `convert $O/fp32-seq --tokens flex --head sequence` built the package with B 1..16, T 1..128 and S 1..128 (all RangeDim) and the new input `targets` int32 `[B,T]` (`logs/02-convert-fp32-seq.log`).
- The INT8 package (`int8-seq`, 22 MiB) was compressed from it with the unchanged recipe (`logs/04-compress-int8-seq.log`).

Max abs error per check. FP32 align is against PyTorch FP32; the INT8 gate is against the stateless INT8 package, Vime's gate. Positions or continuations outside tolerance: 0 in every row.

| Check | Rows | Positions | FP32 align | INT8 gate |
| --- | ---: | ---: | ---: | ---: |
| Sequence, per position, B=4 (prompts 2–17, continuations 1–7) | 4 | 53 | 1.05e-5 | 9.5e-6 |
| Sequence, per position, B=1 | 1 | 3 | 1.34e-5 | 5.7e-6 |
| Sequence, per position, B=16 (padded rows, different positions) | 16 | 217 | 1.53e-5 | 1.43e-5 |
| `score_continuations`: 37 continuations in 4 calls, per sum | — | — | 3.34e-5 | 3.34e-5 |
| **Result** | | | **pass, max 3.34e-5** | **pass, max 3.34e-5** |

Logs: `logs/03-align-fp32-seq.log`, `logs/05-align-int8-seq.log`; JSON: `fp32-seq-align.json`, `int8-seq-align.json`.

## 6. Latency on the Mac

Settings: `measure` with the stateless package loaded, 200 repeats, `CPU_ONLY`. Values are p50 in ms; p95 is in `latency-seq.json`.

- A scoring call is B rows × T continuation tokens over S cached slots, with `position = S`.
- `measure`'s grid skips S + T > 128, which removes S=128 for every T.

### Sequence package, scoring call

| B \ (T, S) | T4 S16 | T4 S64 | T8 S16 | T8 S64 | T16 S16 | T16 S32 | T16 S64 | T32 S16 | T32 S64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.13 | 3.53 | 3.60 | 3.55 | 3.27 | 2.92 | 3.06 | 4.38 | 4.79 |
| 4 | — | — | — | — | 6.37 | 6.28 | 6.65 | — | — |
| 8 | — | — | — | — | 10.10 | 10.15 | 10.60 | — | — |
| 16 | 6.75 | 7.39 | 10.53 | 11.22 | **17.75** | **17.81** | **18.36** | 31.18 | 32.86 |

Prompt prefill in one call (B=1): 8 tokens 2.63 ms, 16 tokens 3.03 ms, 32 tokens 4.25 ms. `unified-scored`, measured fresh the same day: 2.04 / 2.32 / 3.25 ms.

At B=16 the cost grows roughly linearly with B×T: about 0.07 ms per scored position. That fits the head matmul and log-softmax over 12,288 logits running at every position.

### `unified-scored`, decode at B=16 (`latency-unified-scored.json`, measured fresh)

| S | 16 | 32 | 64 | 128 |
| --- | ---: | ---: | ---: | ---: |
| K=149 | 3.10 | 3.21 | 3.67 | 4.04 |
| K=450 | 3.04 | 3.28 | 3.60 | 4.12 |

### The comparison: one sequence call vs 16 decode calls, B=16, T=16 (`latency-seq-vs-decode16.json`)

Both packages are loaded in one process after the stateless package, and inputs are preallocated (`tools/compare_seq_vs_decode.py`).

- **Sequence side:** one call with `input_ids`/`targets` `[16,16]` at `position = P`.
- **Decode side:** 16 chained `unified-scored` calls, `[16,1]`, K=149. Call i sits at `position = P+i` with S = P+i, so the cache grows one slot per call.
- **Timing:** only `predict` is timed. The client's cache writes between decode calls are not counted, which can only favour the decode side.
- **Rounds:** each round alternates the two sides; 3 passes of 100 rounds.

| Shared prefix P | Sequence, one call (p50 per pass) | median | 16 decode calls (p50 per pass) | median | Ratio |
| ---: | --- | ---: | --- | ---: | ---: |
| 16 | 17.64 / 17.41 / 17.50 | **17.50** | 54.78 / 53.52 / 54.50 | **54.50** | **3.12×** |
| 32 | 17.90 / 17.63 / 17.69 | **17.69** | 57.28 / 56.52 / 56.79 | **56.79** | **3.21×** |
| 64 | 18.38 / 18.66 / 18.28 | **18.38** | 62.21 / 62.73 / 61.73 | **62.21** | **3.38×** |

These agree with the separate `measure` runs: sequence B16 T16 is 17.8–18.4 ms, and 16 × the `unified-scored` B16 decode time is 50–65 ms.

Smaller T trades latency for memory (see 7):

- B=16, T=8 is 10.5–11.2 ms per call, so two T=8 calls cost ~22 ms.
- That is still 2.5× better than 16 decode calls.

## 7. Memory

### `memory` stage (3 fresh processes per configuration, medians; `memory.json`, `logs/07-memory.log`)

Values are added over the process after the stateless package was loaded and run, in MiB. Both packages run the v2 call sequence (T=1); the sequence package then adds a B=16, T=32, S=64 scoring call.

| Config | Load | Prefill 24 | First B16 S32 | Sweep to S=64 | B16 S128 | Score B16 T32 S64 | Peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seq` | 39.0 | 40.3 | 60.4 | 99.2 | 180.0 | 220.2 | 266.6 |
| `unified-scored` | 39.1 | 40.4 | 60.5 | 98.9 | 180.0 | — | 181.5 |

As the v2 report showed, this stage creates each call's NumPy inputs inside the measured window, so the growth after load is mostly Python's inputs; one B=16, S=128 cache pair is 56 MiB. The two packages are identical through the v2 sequence. The final scoring call adds 40 MiB to the current footprint and 85 MiB to the peak.

### Memory over T, inputs preallocated (`memory-over-t/`, `tools/memory_over_t.py`)

Per package, in 3 fresh processes:

1. load and run the stateless package;
2. allocate every input of the run;
3. load the package;
4. run prefill 24, then the v2 sequence;
5. sequence package only: B=16, S=64 scoring calls with T rising 1 → 64, then T=16/32 at S=16 and S=96;
6. repeat everything once.

Values are medians in MiB added over "stateless loaded + inputs allocated". "Peak" is the lifetime maximum up to that point.

| After | `seq` footprint | `seq` peak | `unified-scored` footprint | `unified-scored` peak |
| --- | ---: | ---: | ---: | ---: |
| load | 32.8 | 32.8 | 39.5 | 39.6 |
| prefill 24 | 34.1 | 36.4 | 40.9 | 41.7 |
| v2 decode sequence (T=1, to B16 S128) | 34.0 | 36.9 | 40.8 | 42.3 |
| score T=1 | 34.0 | 36.9 | | |
| score T=2 | 34.6 | 38.0 | | |
| score T=4 | 36.3 | 41.9 | | |
| score T=8 | 39.7 | 50.1 | | |
| **score T=16** | **49.9** | **66.7** | | |
| score T=32 | 63.4 | 103.2 | | |
| score T=64 | 104.0 | 169.2 | | |
| T=16/32 at S=16 and S=96 | 90.7 | 182.9 | | |
| whole sequence repeated | 90.1 | 195.7 | 40.8 | 42.5 |

Reading the curve:

- **T ≤ 8 is free.** The sequence package's footprint stays at or below `unified-scored`'s ~40 MiB.
  - The 33 vs 39.5 MiB load difference between the two packages is within the spread between processes. In the `memory` stage, both loaded at 39 MiB.
- **From T=16 on, the footprint grows with the largest T seen**, at about 1–1.5× the size of the `[16,T,12288]` FP32 logits (12 MiB at T=16, 24 at T=32, 48 at T=64).
  - Core ML keeps that memory after the call.
  - It only partly gives it back after smaller calls (104 → 90.7 MiB).
  - The repeat adds nothing further.
- **The peak rises faster**, to about twice the logits size above the retained footprint at the call. That is the transient logits and log-softmax buffers.
- `footprint` shows the same weight mapping as `unified-scored` (154 MB mapped file, clean). Unlike the T=1 packages in the v2 report, the weights are not copied to the heap. The extra memory is `Malloc Large`, which fits buffers sized by B×T×12,288.

**Per-call cap for Vime:**

| Cap on T | Footprint over `unified-scored` | Peak | B=16 call |
| ---: | ---: | ---: | ---: |
| 8 | about 0 | ~50 MiB | ~11 ms |
| 16 | +9 MiB | ~67 MiB | ~18 ms |

Longer continuations can be split into several calls, as the brief allows: the second call reads the first call's `new_key`/`new_value` from the cache.

## 8. Fixture (`fixture/`, `fixture-summary.txt`, `logs/08-fixture.log`)

`fixture $O/int8-seq $O/fixture` wrote `sequence.jsonl` with 8 cases. Each case has its candidates, the shared prefix ids, each candidate's continuation ids, and the per-candidate sums from the package, the stateless INT8 package and PyTorch FP32. The rule is the production one, `score_item`.

| Case | Candidates | Calls | Longest continuation | Package vs stateless | Package vs FP32 | Ranking = FP32 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 3 | 1 | 4 | 0 | 0.079 | yes |
| 2 | 3 | 2 | 6 | 2e-5 | 0.091 | yes |
| 3 | 3 | 1 | 7 | 1e-5 | 0.127 | yes |
| 4 | 3 | 1 | 8 | 1e-5 | 0.073 | yes |
| 5 | 3 | 2 | 5 | 2e-5 | 0.170 | yes |
| 6 | 3 | 1 | 6 | 1e-5 | 0.131 | yes |
| 7 | 3 | 2 | 16 | 2e-5 | 0.084 | yes |
| 8 | 18 | 3 | 9 | 3e-5 | 0.153 | yes |
| **All** | **39** | **13** | | **max 2.36e-5** | **max 0.170** | **8/8 (top-1 and full order)** |

- Package vs stateless agrees well within the brief's ~1e-3 per sum.
- The FP32 difference (≤ 0.17 nats over a whole sentence) is INT8 quantization, the same effect as in the v2 search check. It changes no ranking here.
- A call count of 2 for a 3-candidate case means one prefill plus one scoring call.

## 9. Client resource and release copy

- `$O/ios-kvsequence/` (`logs/09-package-ios.log`):
  - `Resources/TinyChineseV1KVSequenceINT8.mlmodelc`;
  - `Resources/VimeLMManifestZhV1KVSequence.json` (section `kv_sequence`);
  - `KVSequenceFixture-sequence.jsonl`, `KVSequenceFixture-meta.json` and `manifest.json`.

  The v1 and v2 resources are untouched.
- Release copy: `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvsequence/`, a byte-identical copy of `$O/int8-seq` (`diff -r` clean, 22 MiB). `RELEASE.json` and `SHA256SUMS.txt` are unchanged.

**Windows follow-up (2026-11-16).** The results ZIP was checked against its SHA256SUMS (65 files). The release copy is byte-identical to `int8-seq` and passes `verify_package` with the v2 format. Its `weight.bin` is byte-identical to the batched KV v2 package's. The release inventory now lists the four `ios18-int8-block32-kvsequence/` files and `coreml_manifest_sha256.kvsequence_int8` = `c7b9e791…`; the README describes the sequence package and the per-call T cap. `verify_release.py --hashes` passes on 378 files. The Hugging Face release was updated in place at revision `ff3dab5f2b730fb8e9a5bd10addc8ac6414a5b6e` (parent `8dd4f197…`); a fresh download of that revision passes `verify_release.py --hashes`.

## 10. iPhone

Not done in this session, as the brief says. Vime should check on device:

- the T curve above (at least T = 8, 16, 32);
- the footprint with `unified-scored` and the sequence package **both loaded**. This was not measured on the Mac; the two are separate weight files, so roughly additive load cost is expected.

## Decision

**Ship `int8-seq` as an additional resource for sentence scoring; keep `unified-scored` for the search.**

- Both alignments pass, the fixture agrees and the latency gain is ~3×.
- The footprint stays within the brief's "few MiB" of `unified-scored` only if each call's T is capped. **Recommended cap: T ≤ 8**, retained cost about 0 and peak ~+10 MiB.
- **T ≤ 16** is the alternative if ~9 MiB retained and ~25 MiB peak are acceptable; it saves one call per 16-token batch (18 ms vs ~22 ms).
- Uncapped T up to 64 costs ~60 MiB retained and ~130 MiB peak, and is not recommended.

## Change to zh_v1_batch2.py

None. The file is at `fcc1750`. The v2 task's local change was already upstream; its backup is in `outputs/deployment/mac-20261115-v2-local-backup/`, which is not part of this task.

## Files

| Path | What |
| --- | --- |
| `torch-check.json` | PyTorch check |
| `fp32-seq/`, `int8-seq/`, `fp32-seq-align.json`, `int8-seq-align.json` | packages and alignments |
| `latency-seq.json`, `latency-unified-scored.json`, `latency-seq-vs-decode16.json` | latency |
| `memory.json`, `memory-over-t/` | memory (`memory` stage; T curve with `footprint`/`vmmap` text per run) |
| `fixture/`, `fixture-summary.txt`, `ios-kvsequence/` | fixture and client resource |
| `tools/` | `run_build.sh`, `run_latency.sh`, `compare_seq_vs_decode.py`, `memory_over_t.py` |
| `logs/` | every stage |
