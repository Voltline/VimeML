# vimeml-zh-v2-20m Core ML Build (Mac) & Vime Integration Verification — 2026-11-21

This follows [mac-zh-v2-20m.md](../../reference/mac-zh-v2-20m.md) (including the updated rescored-order gate in `Deviation`) on `Voltline/VimeML` at `main` `511b798` with `zh-v2-20m-handoff-20261120.zip` (SHA256 `213c3e3995143979bcf1fda51f8d685eb3430d90afcb98178f8e56c0e1cf6e49`, all 10 files verified against `SHA256SUMS`).

**Unchanged**

- Weights, tokenizer (`zh-unigram-12288`, SHA256 `68bd37c4…7e5a`), INT8 recipe (`int8-b32`, `linear_symmetric`, per-block 32), `fp32kv` variant, and search/rescoring rules (`beam=16`, `topk=10`, `lambda=1.6`).
- No VimeML code was modified, committed, or pushed on the Mac.

**Environment**

- OS & Toolchain: macOS 27.3 (`26C5047e`, arm64), Xcode 27.3 (`27C5035i`).
- Python: CPython 3.11.17 in a `uv` venv, PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5, sentencepiece 0.2.1.
- Core ML: mlprogram, iOS 18 target, `CPU_ONLY`.
- Physical devices: iPhone 13 Pro and iPhone 16 Pro Max, same `-O` build comparing v1 vs v2 in clean processes.

## Summary

- **Stateless FP32 & INT8 (`coreml_zh_v1.py`):**
  - `align` vs `vimeml-zh-v2-20m-fp32-input`: **passed** (`max_abs = 3.27e-5` ≤ `3e-4`, 5/5 checks, 0 failed).
  - `compress` (`int8-b32`): 47 parameter matrices quantized, 2 structural constants excluded.
  - `evaluate` (Standard dev): skipped because `outputs/benchmarks/ime-standard-zh-v1/frozen-r1` is not present on this Mac (approved).
- **Batched KV v4 (`coreml_zh_kvbatch4.py`, `fp32kv`):**
  - `check`: **passed** (`eager_fp32kv` `2.15e-5`, `traced_fp32kv` `2.15e-5`, `eager_fp16kv` `4.79e-3`).
  - `build`: source alignment **passed** (`max_abs = 2.48e-5`), INT8 vs stateless alignment **passed** (`max_abs = 2.38e-5`, 0 rows outside tolerance). Merged INT8 package: **23,271,109 bytes** (vs 89,698,481 separate).
  - `fixture`: step vs stateless max `1.53e-5`, score sums max `1.78e-5`, same order **8/8**.
  - `ios`: produced `TinyChineseV2KVBatchV4FP32KVINT8.mlmodelc` alongside v1's `TinyChineseKVBatchV4FP32KVINT8.mlmodelc`.
  - `measure` (Mac, prompt 48 & 80): latency matches v1 within measurement noise.
- **Search & Vime 125-case fixture parity (`zh-generation-v2-20m-task-r1`):**
  - `prompt_ids`: **125 / 125** exact match.
  - Displayed (`rescored`, `gen_score + 1.6 × lm_score`) Top-1: **125 / 125** exact match on both Mac Python and Vime Swift (0 mismatches).
  - Raw `gen_score` Top-1 (before rescoring): **122 / 125** (3 near-tie swaps of `0.012–0.056`, listed below; 2 of 3 already match the FP32 `rescored` Top-1).
  - Top-10 text set: **93 / 125** identical; 32 cases differ at the 10th-place boundary due to INT8 noise (`max |Δgen_score| = 0.1886`, `max |Δlm_score| = 0.2946`, `max |Δfinal_score| = 0.5448`).
- **On-device same-build comparison (iPhone 13 Pro & iPhone 16 Pro Max):**
  - **Rime reranking (online default path, 181 keys):** v2 matches v1 identically — iPhone 13 Pro `p50 4.12 ms / p95 7.47 ms` (**100% ≤ 60 ms**), iPhone 16 Pro Max `p50 2.97 ms / p95 4.74 ms` (**100% ≤ 60 ms**).
  - **Whole-sentence generation (8 sentences, one shot):** v2 task mode + self-rescoring is **2.4× faster** than v1 (`p50 91.4 vs 211.9 ms` on 13 Pro; `59.1 vs 136.6 ms` on 16 Pro Max) and uses **10–29 MB less peak memory** (`75.1 vs 85.1 MB` on 13 Pro; `63.0 vs 91.7 MB` on 16 Pro Max).
  - **Key-by-key sentence generation (125 keys, every keystroke):** v1 incremental LM mode stays 100% within 60 ms on both phones (`13 Pro p50 12.29 / p95 30.97 ms`), whereas v2 task mode recomputes from the key suffix each keystroke (`13 Pro p50 51.80 / p95 134.49 ms`, **56.8% ≤ 60 ms**; `16 Pro Max p50 34.14 / p95 91.69 ms`, **80.8% ≤ 60 ms**).

---

## 1. Stateless & Batched KV v4 Alignment

### Stateless FP32 alignment (`zh-v2-20m-align-fp32/alignment.json`)

| Reference | Tolerance | Checks | Max abs error | Failed | Result |
| --- | :---: | ---: | ---: | ---: | :---: |
| `vimeml-zh-v2-20m-fp32-input` | `3e-4` | 5 | `3.27e-5` | 0 | **passed** |

### Batched KV v4 (`fp32kv`) sizes & alignment

| Stage | Merged bytes | Separate bytes | `logits` max | `scored` max | `score` sum max (`scored` / `prefix`) | `prefix` max (`one vs ext`) | Rows outside | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `src-merged` (vs PyTorch) | 80,822,163 | 316,392,159 | `2.43e-5` | `2.01e-5` | `2.48e-5` / `2.48e-5` | `1.48e-5` (`1.19e-5`) | 0 | **passed** |
| `int8-merged` (vs stateless INT8) | **23,271,109** | 89,698,481 | `2.38e-5` | `1.42e-5` | `2.05e-5` / `2.05e-5` | `1.43e-5` (`1.08e-5`) | 0 | **passed** |

## 2. Mac Latency (`measure-p48.json`, `measure-p80.json`, p50 / p95 ms)

| Prompt | Package | `prefix_via` | `prompt_prefix` | `extend_prefix_1` | `scored_step_b16` | `score_b16` (T=16) |
| ---: | --- | --- | :---: | :---: | :---: | :---: |
| 48 | `v1-fp32kv` | `prefix` | 3.45 / 4.11 | 1.43 / 1.99 | 3.95 / 4.63 | 18.49 / 20.16 |
| 48 | `v2-fp32kv` | `prefix` | 3.41 / 3.67 | 1.24 / 1.36 | 3.85 / 4.55 | 17.93 / 18.33 |
| 80 | `v1-fp32kv` | `prefix` | 5.13 / 5.60 | 1.41 / 1.78 | 3.88 / 4.24 | 18.15 / 18.71 |
| 80 | `v2-fp32kv` | `prefix` | 5.32 / 5.80 | 1.41 / 1.72 | 3.85 / 4.31 | 18.22 / 18.54 |

## 3. Search & Vime 125-Case Fixture Parity (`zh-generation-v2-20m-task-r1`)

| Runtime | `prompt_ids` exact | Raw `gen_score` Top-1 | **Displayed `rescored` Top-1** | Top-10 set identical | Max `|Δgen|` | Max `|Δlm|` | Max `|Δfinal|` |
| --- | :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| Mac Python (`int8-merged`) | 125 / 125 | 122 / 125 | **125 / 125** | 93 / 125 | `0.1886` | `0.2946` | `0.5448` |
| Vime Swift (`TinyChineseV2KVBatchV4FP32KVINT8`) | 125 / 125 | 122 / 125 | **125 / 125** | 93 / 125 | `0.1886` | `0.2946` | `0.5448` |

### Mismatched cases

- **`rescored` Top-1 mismatches:** **none (0 / 125).**
- **Raw `gen_score` Top-1 mismatches (3 / 125, identical on Mac Python and Vime Swift, all near-ties from INT8 quantization):**
  1. `zh-pin-cped-c68c09b368/mixed`: FP32 `#1` `难道帝游没有杀人头` (`-4.39219`) vs INT8 `#1` `你到底有没有生日糖` (`-4.43216`), `#2` `难道帝游没有杀人头` (`-4.44427`). Both select `你到底有没有瘦肉体` after rescoring.
  2. `zh-pin-risawoz-e24c0caece/mixed`: FP32 `#1` `我想买了车` (`-2.35359`), `#2` `我想买辆车` (`-2.40920`) vs INT8 `#1` `我想买辆车` (`-2.35043`), `#2` `我想买了车` (`-2.39643`). Both select `我想买辆车` after rescoring.
  3. `zh-pin-cped-fd787d639d/initials`: FP32 `#1` `他们还以为肚子里的孩子不是流浪的` (`-6.81331`), `#2` `他们还以为肚子里的孩子不是老了的` (`-6.83104`) vs INT8 `#1` `他们还以为肚子里的孩子不是老了的` (`-6.77799`), `#2` `他们还以为肚子里的孩子不是流浪的` (`-6.82235`). Both select `他们还以为肚子里的孩子不是老了的` after rescoring.

## 4. On-Device Benchmarks (Same `-O` Build, Clean Processes)

### Load time & memory (`phys_footprint` MB, 2 ms peak sampler)

| Device | Model | Load `score+prefix` | Load `scored` | Base MB | After `score+prefix` | After `scored` | Peak MB (`Δpeak`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iPhone 13 Pro | `v1` | 921.6 ms | 420.5 ms | 25.5 | 34.8 | 61.2 | 85.1 (`+59.7`) |
| iPhone 13 Pro | `v2` | 837.6 ms | 408.6 ms | 25.6 | 34.7 | 62.5 | **75.1 (`+49.5`)** |
| iPhone 16 Pro Max | `v1` | 683.4 ms | 333.4 ms | 25.8 | 35.1 | 62.8 | 91.7 (`+65.8`) |
| iPhone 16 Pro Max | `v2` | 676.5 ms | 313.9 ms | 25.8 | 35.1 | 35.9 | **63.0 (`+37.2`)** |

### Latency: Rime reranking, key-by-key generation, and whole-sentence generation

| Device | Model | Rime rerank (181 keys) p50 / p95 / ≤60 ms | Key-by-key gen (125 keys) p50 / p95 / ≤60 ms | Whole-sentence gen (8 cases) p50 / p95 / mean |
| --- | --- | :---: | :---: | :---: |
| iPhone 13 Pro | `v1` | 4.15 / 7.43 ms / **100.0%** | 12.29 / 30.97 ms / **100.0%** | 211.9 / 464.5 / 216.9 ms |
| iPhone 13 Pro | `v2` | 4.12 / 7.47 ms / **100.0%** | 51.80 / 134.49 ms / **56.8%** | **91.4 / 158.8 / 89.6 ms** |
| iPhone 16 Pro Max | `v1` | 2.88 / 5.83 ms / **100.0%** | 7.26 / 17.72 ms / **100.0%** | 136.6 / 287.2 / 140.5 ms |
| iPhone 16 Pro Max | `v2` | 2.97 / 4.74 ms / **100.0%** | 34.14 / 91.69 ms / **80.8%** | **59.1 / 104.4 / 57.0 ms** |

### Whole-sentence generation breakdown (ms, top-1 identical and correct for all 8)

| Keys | Top-1 | 13 Pro v1 | 13 Pro v2 | 16 PM v1 | 16 PM v2 |
| --- | --- | ---: | ---: | ---: | ---: |
| `nihao` | 你好 | 41.6 | 35.2 | 24.9 | 22.4 |
| `zwsrf` | 中文输入法 | 66.6 | 45.3 | 42.8 | 27.6 |
| `wmmtjm` | 我们明天见面 | 105.3 | 52.0 | 67.8 | 32.0 |
| `tuijiancaishishane` | 推荐菜是啥呢 | 256.1 | 85.9 | 180.9 | 53.7 |
| `zhongwenshurufa` | 中文输入法 | 198.0 | 91.4 | 120.9 | 59.1 |
| `jintiantianqihenhao` | 今天天气很好 | 211.9 | 103.0 | 136.6 | 66.6 |
| `dajiayiqiqugongyuansanbuba` | 大家一起去公园散步吧 | 391.6 | 145.1 | 263.3 | 89.9 |
| `taolunxiayigejidudegongzuojihua` | 讨论下一个季度的工作计划 | 464.5 | 158.8 | 287.2 | 104.4 |

## 5. Ship-Gate Assessment

1. **Mac alignment & search gates:** **PASSED** (`3.27e-5` stateless, `2.48e-5` source, `2.38e-5` INT8, `125/125` rescored Top-1).
2. **Vime 125-case fixture gate:** **PASSED** (`125/125` exact `prompt_ids`, `125/125` rescored Top-1, `max |Δfinal| = 0.5448`).
3. **On-device 60 ms gate (iPhone 13 Pro):**
   - **Rime reranking (the current online candidate path):** **PASSED** (`p50 4.12 ms`, `p95 7.47 ms`, **100% ≤ 60 ms**, identical to v1).
   - **Key-by-key task-mode generation:** **DOES NOT FIT 60 ms on iPhone 13 Pro** (`p50 51.80 ms`, `p95 134.49 ms`, **56.8% ≤ 60 ms**; 54/125 keys exceed 60 ms because task mode cannot reuse the per-key beam tree across keystrokes when keys are in the prompt).
   - **Recommendation:** Both v1 and v2 are bundled behind `vime.chineseModelVersion`. For tomorrow's submission, v2 is safe as the default model for Rime reranking/correction (100% ≤ 60 ms on 13 Pro), while keeping sentence generation behind its development switch until task-mode keystroke latency is brought within 60 ms on 13 Pro (or keeping v1 as default if the team prefers to ship v2 only alongside live sentence generation).
