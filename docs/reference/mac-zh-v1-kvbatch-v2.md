# vimeml-tiny-zh-v1 batched KV decoding v2: Mac brief

This is the second Mac task for the batched KV package. The first one is [mac-zh-v1-kvbatch.md](mac-zh-v1-kvbatch.md), and its report is [mac-20261115/zh-v1-kvbatch.md](../reports/mac-20261115/zh-v1-kvbatch.md).

Vime has integrated the v1 package on device, and latency meets the target: per keystroke p50 35 ms on the iPhone 13 Pro, 23 ms on the 16 Pro Max. Memory does not. The test process grows from 48 MiB to 89 MiB when the batch package is loaded and called once. Vime's own KV buffers account for 13.5 MiB of that; the remaining ~27 MiB is Core ML. The merged package saves nothing on device (43.7 MiB vs 40.5 MiB for two separate packages).

## What changed: `zh_v1_batch2.KVBatchGraphZhV2`

1. **Read-only caches.**
   - v1 merged each layer's new key into the cache with `cache*(1-write) + key*write`. That makes a full `(B,6,S,64)` copy of each cache in every layer: 18 cache-sized temporaries, which Core ML may allocate at the largest flexible shape.
   - v2 attends to the cache (slots `>= position[b]` masked) and to the new tokens separately, then combines the two with one softmax normalizer. No cache-sized intermediate exists.
   - The Mac run did not confirm this: flexible-T packages use ~40 MiB as v1 did, and T=1 graphs use far more. See the [v2 report](../reports/mac-20261115/zh-v1-kvbatch-v2.md).
   - `S` only needs to be `>= max(position)` (and at least 1). The v1 rule, `max(position)+1`, still works.
2. **T new tokens per row** (`input_ids [B,T]`, causal among themselves).
   - T=1 is the decode step. B=1 with T = prompt length is the prompt prefill, so the chunk package is no longer needed.
   - Logits are returned for the last new token only. `new_key`/`new_value` are `(9,B,6,T,64)`.
3. **Optional scored head.**
   - Every row of a search bucket sits at the same key position, so all rows share one allowed-token list: median 149 ids, p90 449, max 536 on the r2 fixture.
   - Input: `allowed` int32 `[K]` (1..1024).
   - Outputs: `logsumexp` `[B]` over all 12,288 logits, and `allowed_logits` `[B,K]`. log P = `allowed_logits − logsumexp`.
   - The head matmul still runs in full. What is saved is the `[B,12288]` output and the client-side softmax.

Windows checks (PyTorch FP32, real zh-v1 weights):

| Check | Result |
| --- | --- |
| Decode, eager and traced | max 3.96e-5 |
| Prefill, then a multi-token call from different positions | max 2.9e-5 |
| Scored head vs log-softmax | 1.9e-6 |
| r2 search, logits head and scored head | 125/125 exact order |

The Core ML side code paths ran on Windows against a fake `predict` backed by the PyTorch graph: flexible, enumerated and fixed shapes, prefill, the scored head, and the search. `coremltools` itself was not run on Windows.

## Setup

Use the existing checkout from the first task (`~/Documents/Sources/VimeML-zh-kvbatch`). Its payload and its first-task outputs stay where they are.

```bash
cd ~/Documents/Sources/VimeML-zh-kvbatch
git pull --ff-only origin main        # needs the commit that adds zh_v1_batch2.py
source .venv/bin/activate
```

Two first-task outputs are used again: `outputs/deployment/zh-v1-kvbatch/int8-flex/` (the v1 baseline) and the shipped chunk package. All new outputs go under `outputs/deployment/zh-v1-kvbatch-v2/` (`O` below). Never overwrite an output.

## Variants

All variants are built by `python scripts/deployment/coreml_zh_v1_batch2.py convert $O/fp32-<name> <flags>`:

| Name | Flags | Question it answers |
| --- | --- | --- |
| `dec-flex` | (defaults: B 1..16, T=1, S 1..128, logits) | Do read-only caches alone fix the memory? |
| `dec-enum` | `--batch 1,4,8,16 --slots 16,32,64,128` | Vime's request: enumerated shapes instead of RangeDim |
| `dec-b16` | `--batch 16 --slots 32,64,128` | Vime's second option |
| `scored` | `--head scored` | Smaller output, no client softmax |
| `unified` | `--tokens flex` | Prefill and decode in one package (no chunk package) |
| `unified-scored` | `--tokens flex --head scored` | Both |

`dec-enum` and `dec-b16` put enumerated shapes on several inputs at once. If coremltools refuses, keep the error as evidence and fall back to fixed shapes:

- B=16 with S fixed at 32, 64 or 128 (`--batch 16 --slots 32`, and so on);
- report each fixed package's memory.

For each variant:

1. `align $O/fp32-<name> $O/fp32-<name>-align.json`: vs PyTorch FP32, tolerance 3e-4. If conversion or alignment fails, fix it in `zh_v1_batch2.py`, rerun `check`, and record the change.
2. `compress $O/fp32-<name> $O/fp32-<name>-align.json $O/int8-<name>`.
3. `align $O/int8-<name> $O/int8-<name>-align.json --stateless artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32`: Vime's gate.

## Tasks

1. **PyTorch check.** Run `check $O/check.json`. It must pass, as it does on Windows.
2. **Build and gate** all six variants as described above.
3. **Memory**, the main question.

   ```bash
   python scripts/deployment/coreml_zh_v1_batch2.py memory $O/memory.json \
     v1=outputs/deployment/zh-v1-kvbatch/int8-flex  v1+chunk=outputs/deployment/zh-v1-kvbatch/int8-flex+chunk \
     dec-flex=$O/int8-dec-flex  dec-enum=$O/int8-dec-enum  dec-b16=$O/int8-dec-b16  scored=$O/int8-scored \
     unified=$O/int8-unified  unified-scored=$O/int8-unified-scored  dec-flex+chunk=$O/int8-dec-flex+chunk
   ```

   - Each configuration runs in 3 fresh processes. The stateless INT8 package is loaded and run first, as in the keyboard.
   - The stage records `phys_footprint` and its lifetime peak (`proc_pid_rusage` V4):
     - after the stateless package;
     - after the chunk package (`+chunk`);
     - after loading;
     - after a 24-token prefill (flexible-T packages);
     - after the first B=16, S=32 call;
     - after a sweep of B 1/4/8/16 × S 16/32/64;
     - after B=16, S=128.
   - Check once that the V4 `phys_footprint` field equals the V2 value used in the first report.
   - Explain where the remaining memory goes, if it can be seen (for example with `vmmap --summary` or `footprint` on a running process).
4. **Latency.** Run `measure $O/int8-<name> $O/int8-<name>-latency.json` for every INT8 variant.
   - Decode grid: B ∈ {1,4,8,16} × S ∈ {16,32,64,128}, with the stateless package loaded.
   - Scored variants are measured at K=149 and K=450.
   - Prompt prefill at 8/16/32 tokens: one call for flexible-T packages, one token per call otherwise.
   - Measure `int8-dec-flex` and the v1 `int8-flex` interleaved, three times, at S=16/32, so the v1→v2 difference is not noise. Measure v1 with `coreml_zh_v1_batch.py measure`.
5. **Search.** Run `search $O/int8-<name> outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl $O/search-<name> --reference-batch outputs/deployment/zh-v1-kvbatch/int8-flex` for every INT8 variant.
   - The prompt is prefilled with the package itself. Pass `--prompt chunk` to use the chunk package instead, for non-flexible-T variants.
   - Expected `vs_reference_batch`: the v1 INT8 batch run, which equals the stateless-INT8 search. Exact order in all or nearly all cases; differences only between near-tied neighbours.
   - The scored head changes only where the normalizer is computed, so it should behave like the logits head.
   - Also run `fp32-dec-flex` and `fp32-unified`. Their prompts are FP32 too now, so `vs_fp32_fixture` should be exact, with scores within 1e-3.
6. **Choice.** Pick the INT8 variant to ship. Prefer, in order:
   1. lowest added footprint with the stateless package loaded;
   2. no chunk package needed;
   3. latency within Vime's targets: B=16 ≤ 8 ms and B=8 ≤ 5 ms on the 13 Pro (the Mac is roughly 2× faster).

   State the trade-off in the report. Then:

   ```bash
   python scripts/deployment/coreml_zh_v1_batch2.py fixture $O/int8-<chosen> $O/fixture
   python scripts/deployment/package_zh_v1_kvbatch_ios.py --batch $O/int8-<chosen> --alignment $O/int8-<chosen>-align.json \
       --search $O/search-<chosen> --fixture $O/fixture --output $O/ios-kvbatch-v2
   ```

   This writes `Resources/TinyChineseV1KVBatchV2INT8.mlmodelc` and `VimeLMManifestZhV1KVBatchV2.json` (section `kv_batch_v2`). The v1 resources stay untouched.
7. **Release copy.** Copy the chosen INT8 package to `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch-v2/`. Do not edit `RELEASE.json` or `SHA256SUMS.txt`.
8. **iPhone.** Not in this session. Vime measures with its own test host.

## Report and return

Write `docs/reports/mac-<date>/zh-v1-kvbatch-v2.md` in the format of the first report. It should cover:

- the environment;
- every stage, including failures kept as evidence;
- a memory table over all configurations;
- the latency tables;
- the search results;
- the choice and why.

No benchmark text appears in the report.

Bring back one ZIP with SHA256SUMS. It should contain:

- `outputs/deployment/zh-v1-kvbatch-v2/`, without the FP32 packages if space is short;
- the report;
- the release copy;
- `ios-kvbatch-v2/`;
- any change to `src/vimeml/deployment/zh_v1_batch2.py`, or a note that there was none.

Do not commit from the Mac.
