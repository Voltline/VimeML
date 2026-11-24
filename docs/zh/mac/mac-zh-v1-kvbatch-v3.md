# vimeml-tiny-zh-v1 batched KV v3 (shared prefix, lean scoring, one package): Mac brief

This is the fourth Mac task for the zh-v1 batched KV packages. It follows [mac-zh-v1-kvsequence.md](mac-zh-v1-kvsequence.md); that task's report is [mac-20261116/zh-v1-kvsequence.md](../reports/mac-20261116/zh-v1-kvsequence.md).

## Why

Vime's 16 Pro Max run peaked at 65.7 MB, and the keyboard extension has a hard memory limit that is lower on 6 GB iPhones. Today Vime loads up to three packages: the stateless one, `kvbatch-v2` (`unified-scored`) and `kvsequence`. Each one holds its own copy of the weights. On top of the weights, three things cost memory:

1. **The prompt is copied into every row.** A beam bucket of 16 rows carries the same BOS + context in all 16 caches. At 27 KB per token, a 48-token prompt is 1.3 MB per row and 21 MB per B=16 call, for each of `key_cache` and `value_cache`.
2. **The sequence head materializes `[B,T,12288]` logits.** At B=16, T=16 that is 12.6 MB inside the graph, and it also returns `new_key`/`new_value` that scoring never uses. On the Mac, memory grew with T: +9 MiB at T=16 and +63 MiB at T=64.
3. **Separate packages hold separate weight copies.** The v1 kvbatch task showed that a merged multifunction package stores the weights once (22.96 MB vs 45.59 MB on disk) and saved about 9.5 MiB on the Mac.

Activation and weight quantization below INT8 are out of scope for the 20M.

## What changed: `vimeml.deployment.zh_v1_batch3`

**One multifunction package** with three functions over the same weights:

| Function | Inputs | Outputs |
| --- | --- | --- |
| `logits` | `input_ids` [B,T], `position` [B], `key_cache`/`value_cache` (9,B,6,S,64), `prefix_length` [1], `prefix_key`/`prefix_value` (9,1,6,P,64) | `logits` [B,12288] of the last new token, `new_key`/`new_value` (9,B,6,T,64) |
| `scored` | as `logits`, plus `allowed` [K] | `logsumexp` [B], `allowed_logits` [B,K], `new_key`/`new_value` |
| `score` | `input_ids` [B,T], `targets` [B,T], `prefix_length`, `prefix_key`/`prefix_value` | `target_logprobs` [B,T] only |

**Shared prefix.**
- The prompt (BOS + context) is passed once, as `prefix_key`/`prefix_value`, with batch 1.
- Every row reads it through a broadcast. No row holds a copy.
- Each row's own cache holds only the tokens that row added after the prompt.
- `position[b]` counts those tokens. The absolute position is `prefix_length + position[b] + t`.
- Attention uses one softmax over three parts: the prefix, the row's cache and the new tokens.
- Prefix slots `>= prefix_length` and row slots `>= position[b]` are masked. As before, they must hold finite values.

**Lean scoring.**
- `score` has no row cache and no KV outputs. Every row reads only the shared prefix.
- It never builds `[B,T,12288]` log-probs. It computes logsumexp over the logits and picks the target logit from the tied head row.
- With `--vocab-chunks 4`, the head runs in four 3072-wide chunks with a running max and sum. This bounds the largest intermediate to `[B,T,3072]`.
  - The chunked slices may duplicate the head weight inside the package. Measure the size and the peak; see below.
- The sentence-scoring rule is the one in [mac-zh-v1-kvsequence.md](mac-zh-v1-kvsequence.md), with the prompt call's KV as the prefix (`zh_v1_batch3.score_continuations`).

**Search.**
- `zh_v1_batch3.BatchBackendV3` plugs into `zh_search`. The prompt call's `new_key`/`new_value` become the prefix; the rest of the search is unchanged.
- Ranking equals the v2 backend in the unit tests.

Shapes are flexible: B 1..16, T, S and P 1..128, with `P + S + T <= 128`. The package targets iOS 18 with FP32 compute and `CPU_ONLY`.

Loading on iOS 18: `MLModelConfiguration.functionName = "logits" | "scored" | "score"`, one `MLModel` per function from the same compiled package. The weights should be shared between them; measure whether they are.

## Windows checks

PyTorch FP32 with the real zh-v1 weights (`check`):

| Check | Result |
| --- | --- |
| `logits` and `scored` after a shared prefix: B = 16, 1, 5 (T = 3), 16 with a 64-token prefix; junk ±30 in every masked slot | passed, max 3.24e-5 over eager and traced |
| `score`, 74 continuations, prefix of 9 and of 1 token | passed |
| `score` with 4 vocab chunks (traced) | max 2.14e-5 per sum |
| Unit tests (`tests/test_deployment_zh_v1_batch3.py`): flexible shapes after tracing, inputs not modified, search equal to the v2 backend | 5/5 |

`coremltools` was not run on Windows.

## Setup

```bash
cd ~/Documents/Sources/VimeML-zh-kvbatch
git pull --ff-only origin main        # needs the commit that adds zh_v1_batch3
source .venv/bin/activate
O=outputs/deployment/zh-v1-kvbatch-v3
S=artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32
C="python scripts/zh/deployment/coreml_zh_v1_batch3.py"
```

All new outputs go under `$O`. Never overwrite an output.

## Stages

1. **Check.** `$C check $O/torch-check.json`. It must print `passed`.
2. **Convert.** Make four FP32 packages:

   ```bash
   $C convert $O/fp32-logits --function logits
   $C convert $O/fp32-scored --function scored
   $C convert $O/fp32-score --function score
   $C convert $O/fp32-score-c4 --function score --vocab-chunks 4
   ```

   If conversion fails, fix it in `zh_v1_batch3.py`, rerun `check` and the unit tests (`python -m unittest tests.test_deployment_zh_v1_batch3`), and record the change.
3. **Merge FP32.** Make two variants:

   ```bash
   $C merge $O/fp32-merged logits=$O/fp32-logits scored=$O/fp32-scored score=$O/fp32-score
   $C merge $O/fp32-merged-c4 logits=$O/fp32-logits scored=$O/fp32-scored score=$O/fp32-score-c4
   ```

   Record the printed package bytes against the separate packages.
4. **FP32 alignment**, against PyTorch FP32. Every function is checked, with junk in the masked slots. Tolerance is 3e-4 per value and 1e-3 per sum.

   ```bash
   $C align $O/fp32-merged $O/fp32-merged-align.json
   $C align $O/fp32-merged-c4 $O/fp32-merged-c4-align.json
   ```

5. **Compress.** Run it per function. The gate is the merged FP32 alignment that contains that exact package.

   ```bash
   for f in logits scored score; do $C compress $O/fp32-$f $O/fp32-merged-align.json $O/int8-$f; done
   $C compress $O/fp32-score-c4 $O/fp32-merged-c4-align.json $O/int8-score-c4
   ```

6. **Merge INT8 and align against the stateless package.** This is Vime's gate.

   ```bash
   $C merge $O/int8-merged logits=$O/int8-logits scored=$O/int8-scored score=$O/int8-score
   $C merge $O/int8-merged-c4 logits=$O/int8-logits scored=$O/int8-scored score=$O/int8-score-c4
   $C align $O/int8-merged $O/int8-merged-align.json --stateless $S
   $C align $O/int8-merged-c4 $O/int8-merged-c4-align.json --stateless $S
   ```

   - Compare `package_bytes` with the sum of the shipped INT8 packages (`ios18-int8-block32-kvbatch-v2` + `-kvsequence`).
   - If the merged package is not about one weight copy, note which constants are duplicated. `-c4` may carry an extra copy of the head.
7. **Latency.** `$C measure $O/int8-merged $O/latency.json`. Repeat with `-c4` if it is kept.
   - It interleaves v3 with the shipped v2 packages on the same work. The grid is prompt 16/48 × own tokens 1/8/16 × scoring T 8/16, at B = 16.
   - v2 copies the prompt into each row (S = prompt + own); v3 passes it once.
   - Expect the decode call to be no slower. The scoring call should be faster than `kvsequence`, since it has no KV outputs and no full log-softmax.
8. **Memory.** This is the main question.

   ```bash
   $C memory $O/memory.json v3=$O/int8-merged v3c4=$O/int8-merged-c4 --without-stateless
   ```

   - Each configuration runs in a fresh process, three runs each, and reports medians.
   - The workload per round is: a 48-token prompt; 16 decode steps at B = 16 with own caches growing 1..16; one `logits` call (v3 only); one B = 16, T = 16 scoring call. There are 20 rounds.
   - Configurations:
     - `v2-shipped`: stateless + `kvbatch-v2` + `kvsequence`, as Vime runs today;
     - `v3` and `v3c4`, with the stateless package;
     - `v3-alone` and `v3c4-alone`, without it. Here the `logits` function replaces the stateless package.
   - Report `load_mib`, `peak_mib`, `after_rounds_mib` and the growth after the first round.
   - Also run the T curve for scoring: `--score-tokens 32` and `--score-tokens 64` as extra outputs (`memory-T32.json`, `memory-T64.json`). This shows whether `-c4` flattens it.
9. **Search.** `$C search $O/int8-merged outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl $O/search`.
   - Expected `vs_v2_package`: exact order in all or nearly all cases. Differences are allowed only between near-tied neighbours.
   - `vs_fp32_fixture` should match the v2 task's numbers.
10. **Fixture.** `$C fixture $O/int8-merged $O/fixture` (or `-c4` if that is chosen). It writes:
    - `step.jsonl` and `step.npz`: 16 rows after one shared prompt, one `scored` and one `logits` call;
    - `score.jsonl`: the 8 sentence-scoring cases of the kvsequence task, run through v3 calls.

    Expect package vs stateless within about 3e-4 per value and 1e-3 per sum.
11. **Client resource.**

    ```bash
    $C ios $O/int8-merged $O/ios-kvbatch-v3 --alignment $O/int8-merged-align.json --search $O/search --fixture $O/fixture
    ```

    This writes `Resources/TinyChineseV1KVBatchV3INT8.mlmodelc` and `VimeLMManifestZhV1KVBatchV3.json` (section `kv_batch_v3`), plus `KVBatchV3Fixture-*`. Existing resources stay untouched.
12. **Release copy.** Copy the chosen INT8 merged package to `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch-v3/`. Do not edit `RELEASE.json` or `SHA256SUMS.txt`.
13. **iPhone.** Not in this session. Vime measures with its own test host.

## Decision rules

- **Ship v3** if:
  - both alignments of the chosen variant pass;
  - the search matches v2;
  - the fixture agrees;
  - `v3-alone` or `v3` uses clearly less memory than `v2-shipped`, at load and at peak.
- **Choose `-c4`** only if it lowers the scoring peak by more than its extra package size, and its latency stays within about 10% of `-c1`.
- **Stateless.** If `v3-alone` covers what Vime uses the stateless package for (the `logits` function with no prefix and no row cache is the same forward pass), report it. Vime can then drop the stateless resource.

## Notes for Vime (not for this session)

- Pass the prompt once as `prefix_*`, and keep each row's own tokens in its own cache. Do not copy the prompt into rows.
- Use `score` for sentence scoring and cap T per call (8 or 16), as before.
- Check `os_proc_available_memory()` before a large call, and size B and T from it. This applies on 6 GB devices in particular.

## Report and return

Write `docs/zh/reports/mac-<date>/zh-v1-kvbatch-v3.md` in the format of the earlier reports. It should cover:

- the environment;
- every stage, including failures kept as evidence;
- package sizes, merged against separate and against the shipped v2 pair;
- alignment tables;
- the latency table;
- the memory tables, including the T curve;
- search and fixture summaries;
- the decision on v3 and on `-c4`, and the stateless question.

Bring back one ZIP with SHA256SUMS. It should contain:

- `$O`, without the FP32 packages if space is short;
- the report;
- the release copy;
- `ios-kvbatch-v3/`;
- any change to `src/vimeml/deployment/zh_v1_batch3.py` or `scripts/zh/deployment/coreml_zh_v1_batch3.py`, or a note that there was none.

Do not commit from the Mac.
