# vimeml-tiny-zh batched KV v4 (`prefix` function, FP16 KV): Mac brief

This is a self-contained Mac task. It follows the v3 task ([mac-zh-v1-kvbatch-v3.md](mac-zh-v1-kvbatch-v3.md), report [mac-20261118/zh-v1-kvbatch-v3.md](../reports/mac-20261118/zh-v1-kvbatch-v3.md)); nothing from that session's directories is needed. It uses the zh-v1 weights. vimeml-zh-v2-20m has the same shape, so the chosen variant is rebuilt with its weights later by swapping the bundle.

## Why

Vime profiled the shipped v3 package (`TinyChineseV1KVBatchV3INT8.mlmodelc`) on an iPhone 13 Pro and an iPhone 16 Pro Max (iOS 18, CPU only, 2 ms sampling of `phys_footprint` and its kernel peak). Both devices agree:

| Load order | After the first function | After the second | Then 16 long sentences scored |
| --- | --- | --- | --- |
| `score`, then `scored` | +1.4 MB steady (+20.7 MB transient) | **+27.3 MB steady, not released** | peak 72.2–72.8 MB |
| `scored`, then `score` | +1.8 MB steady, but a **+48.1 MB transient** (77.9 MB total) | +0.1 MB | steady ~31.7 MB |

The client builds the shared prefix KV, and extends it key by key, with `scored` and a dummy `allowed: [0]`. That call needs no head and no logsumexp. Two questions follow:

1. **Does a function without the head avoid the extra 27 MB?** The weights are file-mapped and shared by the three v3 functions (v3 report: one 23.1 MB package). The extra 27 MB is about one copy of the weights, so it may come from Core ML's CPU backend repacking weights per function, and a `prefix` function may not avoid it. The load-order matrix below tests this on the Mac. Vime repeats it on the iPhones.
2. **Does FP16 KV I/O cost accuracy?** It halves the prefix KV (27 KB per token in FP32) and every row cache. On Windows, PyTorch with every KV input and output rounded to FP16 gave at most 3.6e-3 per logit and 1.8e-3 per sentence sum, and the beam search returned the same ranking in all 125 fixture cases (exact order 125/125, top 1 125/125). FP16 compute is the third variant; it is tested here for the first time.

## What changed: `vimeml.deployment.zh_v1_batch4`

v3's three functions (`logits`, `scored`, `score`, unchanged; see the [v3 brief](mac-zh-v1-kvbatch-v3.md)) plus:

| Function | Inputs | Outputs |
| --- | --- | --- |
| `prefix` | `input_ids` [1,T], `prefix_length` [1], `prefix_key`/`prefix_value` (9,1,6,P,64) | `new_key`/`new_value` (9,1,6,T,64) only |

- No final norm, no head, no logsumexp. Token t sits at `prefix_length + t`.
- First call: `prefix_length` 0 with one zero slot. Later calls extend: append `new_key`/`new_value` after the valid slots (`zh_v1_batch4.prefix_call`).
- `score_continuations(..., prefill="prefix")` uses it for sentence scoring.

Three variants, each with all four functions and compressed to the deployed INT8 block-32 recipe:

| Variant | KV inputs/outputs | Compute | Tolerance (per value / per sum) |
| --- | --- | --- | --- |
| `fp32kv` | FP32 | FP32 | 3e-4 / 1e-3 (v3's) |
| `fp16kv` | FP16 | FP32 | 1e-2 / 1e-2 |
| `fp16kv-fp16c` | FP16 | FP16, except reductions, exp/log and norms | 1e-1 / 2.5e-1 (catches only a broken conversion) |

The KV tensors are `key_cache`, `value_cache`, `prefix_key`, `prefix_value`, `new_key` and `new_value`. Logits, `logsumexp`, `allowed_logits` and `target_logprobs` stay FP32. For the fp16 variants the deciding gate is the search check, not the tolerance.

Windows checks (`check` stage, zh-v1 weights): eager and traced FP32 max 3.24e-5; FP16-rounded KV max 3.57e-3; unit tests `tests/test_deployment_zh_kvbatch4.py` 5/5. `coremltools` was not run on Windows.

## Setup

The handoff ZIP holds the files that are not in git: the zh-v1 release subset (inference bundle, stateless INT8, v3 INT8), the readings dictionary and the generation fixture. Check it against its `SHA256SUMS` first.

```bash
git clone https://github.com/Voltline/VimeML.git VimeML-kvbatch4 && cd VimeML-kvbatch4
git log --oneline -1                     # needs the commit that adds zh_v1_batch4
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e . torch==2.7.1 coremltools==9.0 numpy sentencepiece==0.2.1
unzip /path/to/kvbatch4-handoff.zip -d .   # creates artifacts/ and outputs/ at these paths:
#   artifacts/releases/vimeml-tiny-zh-v1-hf-v1/{inference,coreml/ios18-int8-block32,coreml/ios18-int8-block32-kvbatch-v3}
#   outputs/zh-readings/dict/vime_pinyin.dict.yaml
#   outputs/vime-fixtures/zh-generation-v1-lm-r2/{cases.jsonl,meta.json}
O=outputs/deployment/zh-kvbatch-v4
C="python scripts/zh/deployment/coreml_zh_kvbatch4.py"
python -m unittest tests.test_deployment_zh_kvbatch4 tests.test_deployment_zh_v1_batch3
```

The v3 task used Python 3.11, PyTorch 2.7.1, coremltools 9.0 and sentencepiece 0.2.1. If `pip install -e .` pulls something else, record the versions. Every stage refuses to overwrite an output.

## Stages

1. **Check.** `$C check $O/torch-check.json` must print `passed`.
2. **Build** the three variants (convert, merge, align against PyTorch, compress, merge, align against the stateless INT8 package):

   ```bash
   for v in fp32kv fp16kv fp16kv-fp16c; do $C build $O --variant $v 2>&1 | tee $O-build-$v.log; done
   ```

   - Each variant writes `$O/<variant>/`: `src-<function>`, `src-merged`, `src-merged-align.json`, `int8-<function>`, `int8-merged` and `int8-merged-align.json`.
   - If a conversion fails (for example over the FP16 input/output types or the FP16 op selection), fix it in `zh_v1_batch4.py`, rerun `check` and the unit tests, rebuild that variant, and record the change.
   - If a variant's source alignment fails, stop that variant. Its failure and the alignment file are the result.
   - Record the package bytes. Four functions should still store one weight copy, about 23 MB like v3.
3. **Load order (memory), the main question.**

   ```bash
   $C loadorder $O/loadorder.json v3=artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch-v3 \
     fp32kv=$O/fp32kv/int8-merged fp16kv=$O/fp16kv/int8-merged fp16kv-fp16c=$O/fp16kv-fp16c/int8-merged
   ```

   - Each run is a fresh process. coremltools is imported and the inputs are built before `start`, and the models load from `.mlmodelc` compiled outside the process.
   - Orders: `score>scored`, `scored>score`, `score>logits`, `logits>score`, `score>prefix`, `prefix>score`, `score>prefix>scored`. Orders with functions a package lacks are skipped.
   - The workload is 10 rounds. Each round is a 48-token prefix and 8 one-token extensions, through `prefix` if loaded, else `logits`, else `scored`; then, only if `scored` is loaded, 16 B=16 beam steps; then one B=16, T=16 `score` call.
   - Report, per package and order, the steady value after each load, `load_peak`, `after_rounds` and `peak`.
   - The question is whether the Mac shows Vime's effect: `score>scored` keeps about +27 MB more than `scored>score`. Then check whether `score>prefix` or `score>logits` avoids it.
4. **Latency.** Compare the same four packages:

   ```bash
   $C measure $O/latency.json v3=artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch-v3 \
     fp32kv=$O/fp32kv/int8-merged fp16kv=$O/fp16kv/int8-merged fp16kv-fp16c=$O/fp16kv-fp16c/int8-merged
   ```

   It measures a 48-token prefix, a one-token extension, a B=16 beam step and a B=16, T=16 scoring call. v3 builds the prefix through `scored`.
5. **Search**, for every variant that built:

   ```bash
   $C search $O/fp32kv/int8-merged $O/search-fp32kv
   for v in fp16kv fp16kv-fp16c; do $C search $O/$v/int8-merged $O/search-$v --reference $O/fp32kv/int8-merged; done
   ```

   - `fp32kv` is compared with v3. Expect exact order in all or nearly all of the 125 cases; differences are allowed only between near-tied neighbours.
   - The fp16 variants are compared with `fp32kv` and with the FP32 fixture.
6. **Fixture**, for every variant that passed search: `$C fixture $O/<variant>/int8-merged $O/fixture-<variant>`.
7. **Client resources**, for every variant with a fixture:

   ```bash
   $C ios $O/<variant>/int8-merged $O/ios-<variant> --alignment $O/<variant>/int8-merged-align.json \
     --search $O/search-<variant> --fixture $O/fixture-<variant>
   ```

   Vime measures these on the iPhones; the Mac result does not choose the variant on its own.

## Decision rules

These are set before any result. The final choice uses Vime's iPhone numbers for the same orders.

- **`prefix` function.** Keep it if, on the iPhone, the steady footprint of `score>prefix` (both orders) is at least 10 MB below `score>scored`, and its load peak is not higher. If `score>logits` gives the same saving, the client switches to `logits` and `prefix` is dropped (one function fewer). If no order avoids the extra memory, the cause is in Core ML's loading, not in the head, and `prefix` is dropped.
- **`fp16kv`.** Adopt it if its INT8 stateless alignment passes; search against `fp32kv` has top 1 equal in all 125 cases and exact order in at least 120, with the others only near-tied swaps; the fixture's score order matches in every case; and the iPhone peak in the beam workload is lower.
- **`fp16kv-fp16c`.** Adopt it only if it meets every `fp16kv` condition and is faster or lighter than `fp16kv` on the iPhone. A failed conversion or alignment is a result, not something to work around.
- **No further variants in this task.** INT4 weights, ANE and other layouts are out of scope.

## Report and return

Write `docs/zh/reports/mac-<date>/zh-kvbatch-v4.md` in the format of the earlier reports. It should cover:

- the environment;
- every stage, with failures kept as evidence and any code change;
- package sizes per variant (merged against separate);
- the alignment tables;
- the load-order table;
- the latency table;
- search and fixture summaries;
- what the Mac numbers say for each decision rule.

Bring back one ZIP with SHA256SUMS. It should contain:

- `$O` without the `src-*` packages;
- the build logs;
- the report;
- `ios-*/`;
- any change to `zh_v1_batch4.py` or `coreml_zh_kvbatch4.py` as a patch, or a note that there was none.

Do not commit or push from the Mac.

## Notes for Vime (not for this session)

- Repeat the load-order matrix on both iPhones with the `ios-*` resources: the same seven orders and the same workload.
- For each order, report the steady value after each load, the load transient and the peak.
- Feed KV as Float16 `MLMultiArray`s for the fp16 variants. Logits, log-probs and sentence scores stay Float32.
