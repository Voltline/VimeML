# V2.1 KV cache — 2026-10-09

The independent experiment starts from VimeML main `0a85a36` and Vime main `1ea4cf8`.
It reuses the bound V2.1 extend5 best / step40000 inference bundle, tokenizer and original
INT8 block32 recipe. No training, label changes, blind selection or V3 model substitution.

## Interface and reuse

CPU_ONLY, FP32 computation/cache, minimum iOS18, batch1, maximum128 tokens.
The new graph takes `input_ids [1,Q]`, `cache_length [1]`, and two FP32 caches
`[6,1,5,128,64]`; it returns new-token `logits [1,Q,16384]` and two new cache snapshots.
K/V together occupy1.875MiB per snapshot before runtime buffers/branch copies.
Absolute learned positions and a gathered static causal mask preserve the original attention contract.
Twelve scatter_nd operations update the six layers' K/V without empty slice concatenation.

The client shares an exact joint-tokenized common prefix within one scoring request, then scores
each candidate suffix from the immutable prefix snapshot. Next-word and beam branches share
read-only parent snapshots and process only newly appended tokens. Finished beams drop cache
references. Requests do not retain caches across calls: deletion/context changes start fresh,
cancellation releases local state, and there is no sliding window or position reassignment.
Full-vocabulary suffix logP sum, no final EOS/truncation, stable ties and dictionary fallback remain.

One-token decode returns64KiB logits. Prefill still computes all new prefix rows; the Swift wrapper
copies the final row and releases the rest for generation. The final graph has no second dynamic
output slice. Models use new names; `.v21` remains the default, with `.v21KV` or the client's
`Configuration/KVCache.xcconfig` enabling the experiment. V1 and V2 baseline resources stay intact.

## Numerical and task checks

| Check | Result |
| --- | --- |
| FP32 KV → frozen Windows FP32 | Passed; max absolute logits error0.0000514984 |
| INT8 KV → same-weight stateless INT8 | Passed; max error0.0000476837 |
| Fresh prefill, single-token/chunk decode, capacity128, A/B/A branches | Passed |
| Right PAD, future token mutation, dirty unused cache tail | Prefix max difference0 |
| Input branch snapshots | Unmodified |
| Three targeted Python tests | Passed |
| Optimized simulator functional tests | 13 passed |
| Separate baseline/KV simulator benchmarks | Both passed |
| Default Release build-for-testing without KV opt-in | Passed |

Unchanged tolerance: atol=rtol=0.0003. Cache equivalence does not eliminate existing weight
quantization error against FP32; the new INT8-vs-FP32 report separately preserves that failure.
Original valid-prefix fixtures reach approximately1.00949 absolute error; the added branch
diagnostics reach1.78766. This broader maximum is not an additional KV-versus-INT8 error.
Native checks include6755 tokenizer fixtures, joint boundary retokenization, cancellation/reset,
context deletion, metadata/script fallback,20 beam prefixes, all five suggestions, next words and session commit.

| Frozen candidate set | Stateless INT8 / KV Top-1 | Changed Top-1 / full order | Maximum suffix score-sum error |
| --- | ---: | ---: | ---: |
| AJIMEE | 144/200 /144/200 | 0 /0 | 0.0000551492 |
| Historical development | 122/137 /122/137 | 0 /0 | 0.0000345111 |
| Expanded development draft | 1481/2000 /1481/2000 | 0 /0 | 0.0000459552 |

Twelve32-token greedy continuations match the stateless INT8 version12/12 using actual incremental
decode. They remain8/12 equal to the original FP32 trajectories; next-token Top-1 matches FP32
12/12. Expanded draft labels remain provisional. The original FP32 expanded Top-1 was1487/2000;
KV does not recover or worsen the recorded INT8 difference.

## Performance and memory

Apple M3 Mac,16GiB, macOS27.2, Xcode27.1, Core ML Tools9.0/Python3.11.17/torch2.7.1.
Python measurements include cache validation, owned-array copies and Python overhead. Warm
decode uses an already prepared cache; total generation includes prefill. Synthetic token prefixes
measure computation, not input-method quality. No benchmark ran concurrently with quality evaluation.

| Prefix tokens | Stateless / KV single decode p50(ms) | Stateless / KV8-token total p50(ms) |
| --- | ---: | ---: |
| 7 /8 | 2.221 /2.248 | 17.835 /16.929 |
| 31 /32 | 3.201 /1.869 | 29.779 /17.349 |
| 63 /64 | 4.510 /2.109 | 36.359 /18.656 |
| 119 | 6.443 /1.766 | — |

Optimized iOS27.0 Simulator test hosts on the same Mac, separate PIDs67143 and67417, same
binary/configuration/workload, CPU_ONLY; one backend loaded per benchmark process.
These values are not physical iPhone or keyboard-extension measurements.

| End-to-end request | n | Baseline p50/p95(ms) | KV p50/p95(ms) | p50 speedup |
| --- | ---: | ---: | ---: | ---: |
| Four short scoring fixtures | 80 | 6.94 /17.09 | 6.47 /9.53 | 1.07× |
| 17-token context, four candidates | 30 | 41.93 /43.22 | 24.54 /30.36 | 1.71× |
| Next words,20 prompts | 60 | 57.75 /97.02 | 29.96 /42.02 | 1.93× |
| Beam,20 prompts | 60 | 455.19 /539.10 | 189.68 /207.70 | 2.40× |

| Simulator test-host footprint(MiB) | Baseline | KV |
| --- | ---: | ---: |
| Before model load | 34.221 | 33.518 |
| After model load | 37.752 | 37.799 |
| After workload | 41.502 | 42.096 |
| After50 further next-word requests | 41.502 | 42.081 |
| Kernel lifetime peak | 48.034 | 48.299 |

First process load371.16/417.85ms includes identity checks; OS/Core ML caches may be warm.
Different process baselines and allocation high-water marks limit interpretation of small differences.
Logical cache bytes, process footprint, resident memory and device memory budgets are distinct.
This bounded check does not certify long-term stability, fully cold startup or real-extension limits.
The user chose Mac/simulator-only verification; prior day's phone values are not reused here.

## Preserved attempts and transfer

Initial dynamic mask/select conversion failed(max36.85). A gathered static mask passed FP32,
but the concat/cache-update variant later produced incomplete full-length outputs in INT8 comparison
(max25.875). Removing dynamic logit slicing alone did not resolve it(FP32 max24.687).
An attempted scatter_along_axis export failed shape inference. The final index_put/scatter_nd graph
passed all comparisons and native tests. These observations do not isolate a single runtime cause;
failed reports/logs and exact accepted exporter source remain under `v21-kv-conversion-diagnostics-v1`.
The final source adds input/cache validation after export without changing the accepted graph.

The first simulator Release build failed because KV files were absent from the build-script sandbox
input list. An opt-in xcconfig and explicit KV resource lists resolve this without making default
builds require KV payloads. Failed and successful xcresults remain separate.

Accepted artifacts: `tiny-ja-v2.1-extend5-kv-fp32-v5`, `kv-int8-b32-v2`, and `kv-client-resources-v2`
under `artifacts/deployment/` with the common `tiny-ja-v2.1-extend5-` prefix.
FP32/INT8 packages contain50,382,727/14,355,753 logical bytes;32 finite learned matrices are
compressed, while structural mask constants and norms remain unchanged.
All new reports are under `outputs/deployment/v21-kv-*`.

`vimeml-v21-kv-mac-results.zip` transfers accepted models, client compiled resources, all KV reports
and `handoff.json` with both source commits/PRs, identities and path/size inventory. Original V2
tokenizer/baseline payload is reused from the previous results ZIP. Source changes travel through
separate linked PRs. No model binaries or client source snapshots are committed or included as backups.
