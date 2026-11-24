# vimeml-tiny-zh-v1 batched KV decoding: Mac handoff

This is the task brief for the macOS session. Vime now runs pinyin-constrained beam search with the beam kept per key position. Its runtime uses the existing batch-1 chunk-KV INT8 package, so a bucket of 16 hypotheses costs 16 `predict` calls. That is about 3.8 ms each on an iPhone 13 Pro, and 492–724 ms p50 per generation, about 10× too slow.

This task builds a package that advances a whole bucket in one call, gates it, measures it, and prepares the client resource and fixtures. It does not change the checkpoint, the tokenizer, the INT8 recipe or the search rules.

Templates:

- [zh-v1 Core ML brief](mac-zh-v1-coreml.md)
- [zh-v1 Core ML report](../reports/mac-20261111/zh-v1-coreml.md)

## Targets (from Vime)

On iPhone 13 Pro, `CPU_ONLY`, with the stateless package also loaded:

- **Latency:** one bucket (B=16) in one batched one-token step should take at most 8 ms. B=8 should take at most 5 ms.
- **Memory:** report the keyboard-extension footprint added on top of today's 43 MB.

Vime will reuse buckets across keystrokes and show generation asynchronously, so the per-step cost of one bucket is what matters.

## Interface

`vimeml.deployment.zh_v1_batch.KVBatchGraphZh`:

| Name | Type | Meaning |
| --- | --- | --- |
| `input_ids` | int32 `[B,1]` | one new token per row |
| `position` | int32 `[B]` | row cache length before this token; `position[b] < S` |
| `key_cache`, `value_cache` | float32 (or float16) `(9,B,6,S,64)` | row b's slots `>= position[b]` are masked; they may hold any finite value, but NaN/Inf reaches the output (zero new buffers) |
| → `logits` | float32 `[B,12288]` | the new token only; no softmax |
| → `new_key`, `new_value` | `(9,B,6,64)` | the new token's key/value; the caller writes them into slot `position[b]` of row b |

Rows never attend across each other. Padding rows (token 0 at position 0) do not affect real rows.

**Two deliberate differences from what Vime asked for.** Explain them in the report.

1. **`S` is flexible (1..128).** The caller passes only `max(position)+1` slots.
   - With 128 fixed slots, one B=16 cache is 28 MiB, so a call would move about 56 MiB of input.
   - Real rows are a short prompt plus at most about 20 tokens.
2. **Only the new slot is returned,** not full caches. Writing two full cache copies per call would cost more than the step itself.
   - With caller-owned buffers, nothing is copied except the new slot.
   - Vime can still back the inputs with its own buffers.

**B variants:**

| Variant | B | Why |
| --- | --- | --- |
| flex | 1..16 (`RangeDim`) | the default Vime asked for |
| fixed | 16 | padding cost, if `RangeDim` turns out slow |
| fixed | 8 | the B=8 latency Vime asked for |

**The prompt stays on the existing batch-1 chunk package.** Run it once, then copy its cache rows into the batch buffer.

The `merge` stage also builds one multifunction package (`prefill` = chunk, `decode` = batch). Its weights are stored once, so loading both functions should cost one copy of the weights instead of two. Check whether this holds on the Mac and in the footprint.

**Allowed-token logits (Vime's question 7).** Returning only the allowed tokens' logits cannot skip the head matmul without changing the scores. The search needs log-softmax over all 12,288 tokens, and the normalizer needs every logit.

- The head is about a quarter of the per-token compute (4.7M of about 20M multiply-adds).
- What could be saved is only the `[B,12288]` output copy and the client-side softmax.
- If the measurements show the output copy matters, one option is a variant that returns `logsumexp` plus gathered logits. It is not built yet.

**MLState.** It is not used.

- A bucket gathers hypotheses whose parents sit in different earlier buckets, so each step would need its rows rebuilt from arbitrary parents.
- Explicit caller-owned buffers with `outputBackings` do this with no extra copies.

## Setup

```bash
unzip vimeml-zh-kvbatch-mac-<date>.zip -d vimeml-zh-kvbatch-mac
python3 vimeml-zh-kvbatch-mac/setup_mac.py            # default ~/Documents/Sources/VimeML-zh-kvbatch
cd ~/Documents/Sources/VimeML-zh-kvbatch
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e . && python -m pip install coremltools==9.0 psutil
```

Use the zh-v1 pins (Python 3.11, PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5). Record the actual versions, the macOS build and the hardware.

## Payload

| Path | Contents |
| --- | --- |
| `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/inference/` | FP32 inference bundle (weights, config, tokenizer, manifest) |
| `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32/` | shipped INT8 stateless package (alignment reference) |
| `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvchunk/` | shipped INT8 chunk-KV package (prompt / `prefill`) |
| `outputs/zh-readings/dict/vime_pinyin.dict.yaml` | reading dictionary used by the search |
| `outputs/vime-fixtures/zh-generation-v1-lm-r2/` | Vime's generation fixture: 125 cases, beam 16, top 10, FP32 PyTorch reference |

No benchmark items, training data, full checkpoints or credentials are included. The generation fixture comes from the public set's development split plus five hand-written keys.

## Tasks

All commands are `python scripts/zh/deployment/coreml_zh_v1_batch.py <stage> …`. Each stage writes a fresh output under `outputs/deployment/zh-v1-kvbatch/`; never overwrite an output.

1. **PyTorch check.**
   - Run `check out/check.json`. It must pass, as it does on Windows (max 3.96e-5).
   - It steps 16 rows of 3–25 tokens. Rows start at different steps, so positions differ in every call. Each step must match each row's full forward.
2. **FP32 flex.**
   - `convert out/fp32-flex`.
   - `align out/fp32-flex out/fp32-flex-align.json`: against PyTorch FP32, tolerance 3e-4.
   - If conversion fails on the dynamic slot count or the one-hot write, fix it in `zh_v1_batch.py`, rerun `check`, and record the change.
3. **INT8 flex.**
   - `compress out/fp32-flex out/fp32-flex-align.json out/int8-flex`.
   - `align out/int8-flex out/int8-flex-align.json --stateless artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32`.
   - This is Vime's gate: every step within atol = rtol = 3e-4 of the INT8 stateless package. The earlier chunk-KV INT8 package was within 4.2e-5.
4. **Fixed B=16 and B=8.** Repeat steps 2–3 with `--batch 16` and `--batch 8`. The B=8 alignment skips the 16-row cases; the B=16 one also checks padding rows.
5. **FP16 cache (exploratory).**
   - `convert --cache-dtype float16`, then align and compress.
   - The strict gate may fail. If it does, report the error distribution and argmax agreement, and keep the package as evidence only.
   - Measure it with `--compute-units CPU_ONLY`, `CPU_AND_NE` and `ALL`. Vime saw no ANE gain and a 2× GPU loss with the chunk package.
6. **Merged package.**
   - `merge out/int8-flex out/int8-merged` (chunk = shipped INT8 chunk package).
   - Confirm both functions load (`function_name="prefill"` / `"decode"`).
   - Compare the package bytes and RSS against loading the two packages separately.
7. **Latency on the Mac.**
   - Run `measure <pkg> out/<pkg>-latency.json` for INT8 flex, fixed 16, fixed 8, the merged package and the FP16 variant.
   - Grid: B ∈ {1, 4, 8, 16}, S ∈ {16, 32, 64, 128}, 200 repeats after 20 warm-ups, with the stateless INT8 package loaded alongside.
   - Report p50/p95 per call, and for fixed packages the padding cost (B=1 rows in a B=16 call).
8. **Vime fixture.**
   - `fixture out/int8-flex out/fixture`.
   - It contains 16 sentence prefixes of different lengths. Each is pre-filled with the chunk package, then all are advanced in one decode call.
   - The per-row top-5 comes from the package, the INT8 stateless package and PyTorch FP32.
9. **Search check.**
   - `search out/int8-flex outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl out/search-r2`.
   - The fixture's `meta.json` expected the same text sets for INT8. The 2026-11-15 run showed otherwise: INT8 itself (batched or stateless) keeps the same text set in 86/125 cases and Top-1 in 125/125, with drops mostly at ranks 9–10. Compare an INT8 package with a stateless-INT8 search, not with the FP32 fixture.
   - Also run it with the FP32 flex package and `--chunk` set to an FP32 chunk package. There the order should be exact and scores within 1e-3.
   - Report mean calls/rows per generation and the Mac p50 per generation.
10. **Client resource.** Pick the INT8 variant to ship from the measurements; default to flex unless it is clearly slower than fixed 16. Then run:

    ```bash
    python scripts/zh/deployment/package_zh_v1_kvbatch_ios.py --batch out/int8-flex --alignment out/int8-flex-align.json \
        --search out/search-r2 --fixture out/fixture --output out/ios-kvbatch
    ```

    This writes `Resources/TinyChineseV1KVBatchINT8.mlmodelc` and a `kv_batch` manifest section with the existing kv entry's fields plus the batch range. The existing zh-v1 resources stay untouched.
11. **iPhone (if the test host is available).**
    - Per-predict latency for B = 1, 4, 8, 16 on the iPhone 13 Pro and the iPhone 16 Pro Max, at the S values a real bucket uses (16 and 32).
    - The footprint added with the stateless package loaded.
    - Vime owns the client code. Do not commit Vime-side changes from this session.
12. **Release copy.** Copy the chosen INT8 package to `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch/`. Do not edit `RELEASE.json` or `SHA256SUMS.txt`; the Windows side updates them.

## Report

Write `docs/zh/reports/mac-<date>/zh-v1-kvbatch.md`, in the format of the zh-v1 Core ML report. It should cover:

- the environment;
- every stage result, including failures kept as evidence;
- the interface and the two deliberate differences;
- the latency tables (Mac, plus iPhone if run);
- the memory and footprint;
- the merged-package result;
- the FP16/ANE result;
- the search check;
- which package the client resource uses, and why.

No benchmark text appears in the report.

Bring back:

- `outputs/deployment/zh-v1-kvbatch/` (without the FP32 packages if space is short);
- the chosen release copy;
- the report;
- the resource directory.

Package them with `scripts/zh/deployment/package_zh_v1_mac_results.py` or a plain zip with SHA256SUMS.
