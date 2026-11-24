# vimeml-tiny-zh-v1 sentence scoring on Core ML (sequence head): Mac brief

This is the third Mac task for the zh-v1 batched KV package. It follows [mac-zh-v1-kvbatch-v2.md](mac-zh-v1-kvbatch-v2.md); that task's report is [mac-20261115/zh-v1-kvbatch-v2.md](../reports/mac-20261115/zh-v1-kvbatch-v2.md).

## Why

Vime now scores whole sentences: the literal conversion plus a few dozen near-miss candidates, each differing from the literal in one place. It shares prefixes in a token tree and sends each tree depth through the `unified-scored` package. That package returns the next-token distribution of the **last** new token only, so a candidate with n tokens after the shared part still costs n calls. Long sentences reach a p95 of 310 ms on the Mac, mostly from the number of sequential calls.

The new **sequence head** returns the log-probability of a given next token at **every** new position. A known continuation is then scored in one call. A batch of up to 16 candidates costs one call after the shared prefix, however long they are.

## What changed: `zh_v1_batch2.KVBatchGraphZhV2(model, "sequence")`

- The graph, inputs and cache rules are those of v2.
- **New input** `targets` int32 `[B,T]`: the token that follows `input_ids[b,t]`.
- **Output** `target_logprobs` float32 `[B,T]`: log P(`targets[b,t]` | row b through `input_ids[b,t]`), normalized over all 12,288 logits.
- `new_key`/`new_value` are unchanged: `(9,B,6,T,64)`.
- **Padding.** Rows of different lengths are right-padded with any token and any target. Padded positions come after the real ones, so causality keeps them out of the real scores. The caller ignores their outputs.
- **Scoring rule** (`zh_v1_batch2.score_continuations`, the reference algorithm). With `prefix` the ids every candidate shares (BOS + context, cut where the candidates diverge):
  1. One prefill call over `prefix[:-1]`.
  2. One call per 16 candidates: `input_ids = [prefix[-1]] + c[:-1]`, `targets = c`, `position = len(prefix) − 1`, every row reading the same cache.
  3. A candidate's score is the sum of its `target_logprobs`.

  This is the production rule of `benchmarks.zh13_lm_scores.score_item`: BOS + context + candidate, with targets from the shared prefix on.
- The sequence head is built with a flexible T only, with no enumerated shapes. `position[b] + T <= 128` as before.
- **Cost.** The head matmul runs for every new position, `[B,T,12288]`. At B=16, T=32 that is 25 MB of logits inside the graph. Measure it; see below.

## Windows checks

PyTorch FP32 with the real zh-v1 weights:

| Check | Result |
| --- | --- |
| Per-position log-probs after a per-row prompt, then a padded multi-row call from different positions (B = 4, 1, 16) | max 1.1e-5, 0 outside 3e-4 |
| Traced graph, same checks | passed |
| `score_continuations`, 37 candidates of 1–12 tokens | sums within 2.7e-5; 4 calls |
| Fixture cases under the production rule (`score_item`) | max 1.9e-5; 18 candidates in 3 calls |
| Existing decode, prefill and scored-head checks | unchanged (max 3.96e-5) |

`coremltools` was not run on Windows.

## Setup

Use the checkout from the earlier tasks (`~/Documents/Sources/VimeML-zh-kvbatch`):

```bash
cd ~/Documents/Sources/VimeML-zh-kvbatch
git pull --ff-only origin main        # needs the commit that adds the sequence head
source .venv/bin/activate
O=outputs/deployment/zh-v1-kvsequence
S=artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32
U=outputs/deployment/zh-v1-kvbatch-v2/int8-unified-scored
```

All new outputs go under `$O`. Never overwrite an output.

## Stages

1. **Check.** `python scripts/zh/deployment/coreml_zh_v1_batch2.py check $O/torch-check.json`. It must print `passed` and list `sequence-Tflex` as traced.
2. **Convert.** `python scripts/zh/deployment/coreml_zh_v1_batch2.py convert $O/fp32-seq --tokens flex --head sequence`. Batch and slots stay flexible (B 1..16, S 1..128), as in `unified`.
3. **FP32 alignment.** `python scripts/zh/deployment/coreml_zh_v1_batch2.py align $O/fp32-seq $O/fp32-seq-align.json`, against PyTorch FP32.
   - It runs the per-position checks (B = 4, 1, 16) and the `score_continuations` sum check.
   - Tolerance is 3e-4 per position and 1e-3 per sum.
   - If conversion or alignment fails, fix it in `zh_v1_batch2.py`, rerun `check`, and record the change.
4. **Compress.** `python scripts/zh/deployment/coreml_zh_v1_batch2.py compress $O/fp32-seq $O/fp32-seq-align.json $O/int8-seq`.
5. **INT8 alignment.** `python scripts/zh/deployment/coreml_zh_v1_batch2.py align $O/int8-seq $O/int8-seq-align.json --stateless $S`. This is Vime's gate.
6. **Latency.** `python scripts/zh/deployment/coreml_zh_v1_batch2.py measure $O/int8-seq $O/latency-seq.json`.
   - The default grid is B 1/4/8/16 × T 4/8/16/32 × S 16/32/64/128 (skipping S + T > 128), plus prompt prefill at 8/16/32 tokens, with the stateless package loaded.
   - For comparison, also run `measure $U $O/latency-unified-scored.json`; it may already exist from the v2 task.
   - The numbers to compare: one sequence-head call at B=16, T=16 against 16 `unified-scored` decode calls at B=16.
7. **Memory.** `python scripts/zh/deployment/coreml_zh_v1_batch2.py memory $O/memory.json seq=$O/int8-seq unified=$U`.
   - The sequence run adds a B=16, T=32, S=64 scoring call at the end (`after_score_B16_T32_S64`).
   - Watch whether the `[B,T,12288]` logits raise the footprint beyond the ~40 MiB of the flexible-T packages.
8. **Fixture.** `python scripts/zh/deployment/coreml_zh_v1_batch2.py fixture $O/int8-seq $O/fixture`.
   - It writes `sequence.jsonl`: 8 cases, 3–18 candidates each. Every case has the shared prefix ids, each candidate's continuation ids, and the expected sums from the package, the stateless INT8 package and PyTorch FP32.
   - Expect package vs stateless within about 1e-3 per sum.
9. **Client resource.**

   ```bash
   python scripts/zh/deployment/package_zh_v1_kvbatch_ios.py --batch $O/int8-seq --alignment $O/int8-seq-align.json \
       --fixture $O/fixture --output $O/ios-kvsequence
   ```

   This writes `Resources/TinyChineseV1KVSequenceINT8.mlmodelc` and `VimeLMManifestZhV1KVSequence.json` (section `kv_sequence`), plus `KVSequenceFixture-*`. The v1 and v2 resources stay untouched.
10. **Release copy.** Copy `$O/int8-seq` to `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvsequence/`. Do not edit `RELEASE.json` or `SHA256SUMS.txt`.
11. **iPhone.** Not in this session. Vime measures with its own test host.

## Decision rules

- **Ship** the sequence package if:
  - both alignments pass;
  - the fixture agrees;
  - its footprint stays within a few MiB of `unified-scored`.
- **If the footprint grows a lot at large T**, report the curve over T. Vime can cap T per call (for example 16) and split longer continuations into two calls; the cache rules allow that.
- **Do not replace `unified-scored`.** The search keeps using it. The sequence package is an additional resource for sentence scoring.

## Report and return

Write `docs/zh/reports/mac-<date>/zh-v1-kvsequence.md` in the format of the v2 report. It should cover:

- the environment;
- every stage, including failures kept as evidence;
- alignment tables;
- the latency table, with the comparison in stage 6;
- the memory table;
- the fixture summary;
- the decision.

Bring back one ZIP with SHA256SUMS. It should contain:

- `$O`, without `fp32-seq` if space is short;
- the report;
- the release copy;
- `ios-kvsequence/`;
- any change to `src/vimeml/deployment/zh_v1_batch2.py`, or a note that there was none.

Do not commit from the Mac.
