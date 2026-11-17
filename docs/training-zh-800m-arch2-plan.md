# zh-800m second architecture (arch2): design

This document designs the second architecture of the 800M Chinese desktop model, internal code name `zh800m-arch2`. The 800M now training ([training plan](training-zh-800m-plan.md), called arch1 below) is not affected. Work on arch2 starts after arch1 has finished training and been evaluated.

## Why change

arch1 trains with the packing code of [8903f17](../src/vimeml/training/joint_zh200m.py). Measured:

- about 0.26 s per update in steady state (first-step compilation excluded), about 106k supervised tokens per update;
- about 3.6 days for 1,198,964 updates;
- useful MFU about 14% against the measured matmul peak;
- padding ratio about 1.37.

Goals:

- about 800M parameters, the same 12K vocabulary, the same data, data order and token budget;
- updates k times as large as arch1's (k set by a 200M precheck, 4 or 8; see Larger updates);
- one pass in under 1.5 days on the same hardware, i.e. useful MFU of 30–40%;
- quality no worse than arch1 (decision rule below).

## What we take from the cluster proposal

The cluster proposed the following for arch2:
- sequence packing with a longer context;
- RoPE;
- GQA;
- optional QK-norm;
- four candidate shapes.

This design decides as follows.

| Proposal | Decision | Reason |
| --- | --- | --- |
| Sequence packing | **Adopted, isolated per window** | See the next section |
| Rebuild c2048 windows per document | **Rejected** | Changes the content and distribution of the training data; see the next section |
| RoPE | Adopted | Packed windows each start at position 0, which relative positions handle naturally. Learned positions are only 0.8M parameters, so saving parameters is not the reason |
| GQA with 8 KV heads | Adopted with 7 KV heads | See Architecture |
| QK-norm | Adopted | Steadier at the unchanged 3e-4 peak learning rate |
| Shape B2 (d1792 / 23 layers) | Adopted, with 7 KV heads | 15% fewer sequential layers than 27, and an ordinary aspect ratio. 28 query heads do not split into 8 groups (see Architecture) |
| `lm_batch` counted in tokens | Rejected | Still counted in windows, now 2048 × k; data order and loss normalization match arch1 (see Larger updates) |

### Packing isolates windows; no document-level long context

- arch1 windows are cut at paragraphs and average about 34–42 tokens (Shape section of the [training plan](training-zh-800m-plan.md)).
  - Rebuilding c2048 windows per document would teach long cross-paragraph context.
  - That changes the training distribution and breaks the "data content unchanged" constraint.
- IME contexts are short: the in-field context plus the pinyin.
  - Paragraph windows match that use.
  - The 800M is also the round 4 distillation teacher, and the teacher sees inputs as short as the student's.
- A per-document rebuild means new window indexes, a new mixture, a new `dataset_sha256` and a new step count. arch2 would then no longer be comparable with arch1 at equal tokens.
- The speed comes from packing itself, not from attention across windows.
  - Several windows share one row, each as its own segment.
  - Positions restart at 0 for each segment, and attention is causal within the segment only (a block-diagonal causal mask).
  - The training objective is then identical to arch1, token by token.

### Measure the bottleneck before changing the architecture

The cluster's diagnosis is "sequences are too short to fill the matmuls". That is only half right:

- The matmul M dimension is the number of positions in a microbatch, tens of thousands now, whatever the row length. Short rows cost through padding, attention kernels on very short sequences, and the fixed cost of each forward/backward pass.
- arch2 has almost the same parameter count as arch1 (777M vs 784M), so the per-update gradient all-reduce is the same size, about 3.1 GB in FP32. If most of an update is communication or host synchronization, no architecture reaches 1.5 days.

At 0.26 s per update and about 20k padded positions per GPU, pure compute is about 60 ms. The other ~200 ms must be accounted for first, hence step 0.

## Step 0: locate the bottleneck (on arch1 code, no training)

1. **Single-process control**: run one GPU with the same per-GPU load as the multi-GPU run (same rows and lengths) and compare the update time. The difference is the cost of communication and synchronization.
2. **All-reduce bandwidth**: all-reduce a 3.1 GB FP32 tensor in the training environment and measure the effective bandwidth. Check in NCCL's INFO log that the transport is direct GPU-to-GPU, not shared memory or network sockets.
3. **Profiler**: capture 5 updates with `torch.profiler` on rank 0. Split the time into compute, communication, idle and host work (packing, collation, `.item()` syncs).

Decision:

- **Communication above 30% of an update**: fix communication before arch2. Options:
  - overlap the all-reduce with the backward pass when there is a single microbatch;
  - tune the DDP buckets;
  - if needed, communicate gradients in BF16. This changes gradient rounding and is recorded as a deviation.
- **Host work above 15%**: move packing into the DataLoader workers and merge the per-update syncs into one.
- Step 0 ends with a short report that fixes the scope of the execution changes.

## Architecture

| Item | arch1 | arch2 (frozen) |
| --- | --- | --- |
| Vocabulary | 12,288 | 12,288 (unchanged, aligned with the 20M for distillation) |
| d_model | 1536 | 1792 |
| Layers | 27 | 23 |
| Attention | 24 heads × 64 | Q 28 heads × 64, KV 7 heads × 64 (GQA, 4 query heads per group) |
| SwiGLU d_ff | 4096 | 4608 |
| Normalization | RMSNorm, pre-norm | RMSNorm, pre-norm; per-head RMSNorm on Q and K (QK-norm) |
| Positions | learned, 512 | RoPE, θ = 10000, all 64 dimensions |
| Segment length limit | 512 | 512 (one window, as in arch1) |
| Training row length | padded per window | 2048, several windows packed |
| Bias / tied embedding | none / tied | none / tied |
| Parameters | 784,157,184 | **776,524,928** |
| Training FLOP / token | about 4.7 GFLOP | about 4.66 GFLOP |

- **KV heads**: the cluster's B2 pairs 28 query heads with 8 KV heads. 28 is not divisible by 8, so GQA has no whole groups; arch2 uses 7 KV heads instead.
  - 8 KV heads with 32 heads of 56 was the alternative. A head_dim of 56 is not a multiple of 64 and runs the attention kernels less efficiently, so it is not used.
  - 4 KV heads (groups of 7) compress KV too far; 14 halve the saving.
- **Per-layer parameters**: Q and O projections 1792 × 1792 each, K and V projections 1792 × 448 each, SwiGLU 3 × 1792 × 4608, two RMSNorms and two 64-wide QK-norm weights.
- **Total parameters** = 12,288 × 1792 (tied embedding) + 23 × per layer + 1792 (final norm).
- **FLOP** are counted as 6N. Windows average about 40 tokens, so the quadratic attention term is negligible.

**What GQA actually buys**

- KV cache per token drops from 27 × 2 × 1536 = 82,944 values to 23 × 2 × 7 × 64 = 20,608, about a quarter.
- Desktop decoding on short contexts is dominated by reading the weights once per step (about 1.56 GB in BF16), so single-sentence latency does not drop severalfold.
- The gain is in beam search: with beam 16 over several buckets, KV memory and traffic drop markedly.
- Latency is measured on the Mac under the release gates as usual.

## Execution

- **Packing**:
  - Each update's 2048 × k LM windows and 384 × k pinyin samples are sorted longest first and packed first-fit into rows of 2048. Each window or sample is a segment.
  - Attention uses `torch.nn.attention.flex_attention` with a block-diagonal causal mask built from segment ids. RoPE positions restart for each segment.
  - Loss and `per_kind` weights follow arch1: each token is weighted by its task's weight ÷ that task's global supervised tokens.
- **Shapes**: rows are always 2048 wide, so only the row count varies. Compilation marks only the batch dimension dynamic, with at least 2 rows, which avoids recompiles.
- **Microbatch budget**: rows × 2048 ≤ `lm_microbatch` × 512. `lm_microbatch` is set to the largest value that fits by the smoke run's peak memory (try 128 first, i.e. 32 rows per microbatch). At k times the batch an update is about 400–800k tokens, so each GPU runs several forward/backward passes; all but the last accumulate under `no_sync`, so there is one all-reduce per update.
- **Data parallelism balanced by tokens**: segments are assigned to ranks greedily by token count instead of in contiguous rows. Normalization still uses global token counts, so the summed gradient is unchanged.
- **Synchronization**: one merged sync per update (task loss sums, gradient norm, stop flag) instead of separate `.item()` calls.
- **Validation, inference and export are not packed**: they keep the ordinary per-window forward and the KV-cache path.

## More speedups

These four items change only how training runs, not the objective. The biggest lever, larger updates, changes the recipe and has its own section (Larger updates).

1. **Remove per-update host syncs so the CPU prepares the next update early.**
   - Each update now has four points where the CPU waits for the GPU:
     - the timing `torch.cuda.synchronize`;
     - the task losses' `.tolist()` with its all-reduce;
     - `float(clip_grad_norm_)`;
     - the per-update stop-flag all-reduce (`ctx.any`).
   - As a result, packing, collation, copies and kernel launches for the next update cannot start until this one has finished.
   - Changes:
     - time with CUDA events, read at log intervals;
     - keep loss sums and the gradient norm accumulating on the GPU and read them at log intervals;
     - check the stop flag every 10 updates, still stopping every rank after the same update;
     - when gradients are non-finite, skip the update on the GPU (fused AdamW supports the `found_inf` mechanism GradScaler uses) and fail at the next log check. This relies on a PyTorch internal interface and needs unit-test coverage.
2. **CUDA Graphs.**
   - On top of packing, fix each microbatch at R rows × 2048. Missing rows are filled with empty rows, and the variable number of segments goes into a fixed-length segment-id tensor.
   - Record forward and backward with `torch.compile(mode="reduce-overhead")`. Each update then only replays the graph, and launch overhead nearly disappears.
   - Do this when step 0's profile shows launch or scheduling overhead.
3. **Overlap communication with compute.**
   - Gradients sync only in the last microbatch's backward pass, and DDP sends bucket all-reduces during that pass.
   - Enable `gradient_as_bucket_view`, and `static_graph` if the graph is fixed; tune `bucket_cap_mb` from the profile.
4. **GEMM tuning.**
   - Compile with `max-autotune-no-cudagraphs` (or `max-autotune` together with item 2) so GEMMs get better tiles.
   - Use a chunked fused cross-entropy, which avoids round-tripping the full logits through memory. The speed gain is small; it mainly frees memory for larger microbatches.
**Not done**:

- FP8 and any other matmul precision below BF16: prohibited; none is introduced.
- More GPUs. With a fixed global batch, compute per GPU shrinks as GPUs are added while per-update communication and syncs do not, so more GPUs barely help.
- Activation recomputation. Memory is ample, so recomputation would only slow training down.

## Larger updates (adopted)

An arch1 update is only about 100k tokens (about 79k LM supervised tokens plus pinyin). That is small for 800M; public models of this size usually take 0.5M–2M tokens per update. The fixed per-update costs (a ~3.1 GB all-reduce, the optimizer, syncs) do not grow with the batch, so k times the batch spreads them over k times the tokens. arch2 uses k times the batch, with k = 4 or 8 set by the precheck below.

### Recipe as a function of k

| Item | arch1 | arch2 (k×) | k = 4 | k = 8 |
| --- | --- | --- | --- | --- |
| `lm_batch` (windows) | 2048 | 2048 × k | 8192 | 16384 |
| Pinyin mix full/mixed/initials/error | 96/128/112/48 | each × k | 384/512/448/192 | 768/1024/896/384 |
| Tokens per update (approx.) | 100k | 100k × k | 400k | 800k |
| Updates | 1,198,964 | 1,198,964 ÷ k (dry run exact) | about 299,741 | about 149,871 |
| Peak learning rate | 3e-4 | 3e-4 × √k | 6e-4 | 8.5e-4 |
| Final learning rate | 3e-5 | peak ÷ 10 | 6e-5 | 8.5e-5 |
| Warmup (updates) | 2000 | 2000 ÷ k (equal tokens) | 500 | 250 |
| `bucket_multiplier` | 64 | 64 ÷ k | 16 | 8 |
| Milestones | 20 | 20, aligned by tokens | | |

- **Data order**: with `bucket_multiplier` at 64 ÷ k, each length pool is still 131,072 windows and the sequence of pools is identical to arch1. Within a pool, windows are cut by length into k-times-larger batches whose order is then shuffled, so each large batch equals k length-adjacent arch1 batches from the same pool.
- **Pinyin passes unchanged**: the mix scales with the LM batch and the update count divides by k, so the four pinyin kinds still get 1.42/1.35/1.38/1.75 passes.
- **Unchanged**: AdamW (β 0.9/0.95, wd 0.1), gradient clip 1.0, seed, `per_kind` normalization, the shape of the cosine schedule.
- **Learning rate**: the square-root rule is the usual starting point under AdamW. QK-norm makes arch2 steadier at higher learning rates. The precheck tests the scaling as well.

### Precheck (200M, before arch2 work starts)

Three arms on the 12K joint-pinyin 200M recipe (`configs/train-zh200m-12k-pinyin-pt.toml`). Each arm is a complete short cosine schedule on the same token budget, 10% of the full 200M run:

| Arm | `lm_batch` | Pinyin mix | Updates (approx.) | Peak → final LR | Warmup | `bucket_multiplier` |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| 1× | 512 | 48/64/56/24 | 84,500 | 6e-4 → 6e-5 | 1000 | 64 |
| 4× | 2048 | 192/256/224/96 | 21,125 | 1.2e-3 → 1.2e-4 | 250 | 16 |
| 8× | 4096 | 384/512/448/192 | 10,563 | 1.7e-3 → 1.7e-4 | 125 | 8 |

- The arms share data, order and seed; the dry run gives the exact update counts.
- Compare end-of-training web BPC and the four pinyin validation losses, plus the milestone curves at equal tokens.
- **Decision**: take the largest k that passes. An arm passes if its web BPC is at most 0.01 above the 1× arm, each of the four pinyin validation losses is at most 0.01 above the 1× arm, and training shows no loss spike (0.5 or more above the mean of the previous 1000 updates).
- If 8× fails and 4× passes, use 4×. If 4× also fails, run a 2× arm (1024 windows, LR 8.5e-4) and decide then.
- The precheck runs as cluster jobs; its report freezes the arch2 recipe.

### Expected speed

With larger updates the compute per update grows k-fold and the fixed costs stay. From arch1's measurements (about 60 ms of compute and about 200 ms of other costs per update), k = 4 alone brings one pass to about 1.5 days, and the execution changes and the four items above make it faster. The acceptance thresholds scale with k (see Acceptance before training).

## Inference and export follow

- **Model**: a new `model_zh800m_arch2` with RoPE, GQA and QK-norm. It provides a packed training forward (taking segment ids), the ordinary forward and `forward_cached`.
- **Evaluation**: `zh_torch_backend`, `zh13_lm_scores`, pinyin generation (`pinyin_gen_zh.py`) and Standard load arch2.
- **Distillation**: `joint_zh_pinyin` can load arch2 as the teacher. The 12K vocabulary is shared, so token-level logits align directly.
- **Desktop export**: the Core ML package gains RoPE, GQA and QK-norm, numerically matching the PyTorch forward.

## Data and recipe

Same as arch1:

- mixture `zh800m-12k-c512`, same `dataset_sha256`, windows not rebuilt;
- pinyin store `zh-pinyin-pt-12k`, `per_kind` normalization, the pass count of each pinyin kind;
- the sequence of length pools (see Larger updates) and the total token count;
- AdamW (β 0.9/0.95, wd 0.1), gradient clip 1.0, seed;
- 20 milestones (aligned by tokens) and checkpoint retention.

Different from arch1: the `[model]` section, the model code, the execution, and the k-scaled batch, pinyin mix, learning rate, warmup, update count and `bucket_multiplier`. arch2 and arch1 web/chat/probe BPC compare at equal-token milestones. The checkpoint interval stays equal in tokens, i.e. the update interval divides by k.

## Acceptance before training

1. **Unit tests**:
   - parameter count and shape;
   - GQA grouping equals per-head repetition;
   - RoPE KV-cache decoding matches the full forward;
   - **packing equivalence**: for the same windows, the packed forward's per-token loss and gradients match separate per-window forwards (tolerances as in `tests/test_zh800m_pinyin.py`); the data-parallel split matches one device.
2. **Dry run**:
   - 776,524,928 parameters and `max_steps` about 1,198,964 ÷ k;
   - pinyin passes as in arch1 (1.42/1.35/1.38/1.75);
   - the plan equals arch1's except for the model, the code signature and the items in the Larger updates table.
3. **20-update smoke** reporting:
   - steady-state update time, target at most 0.11 × k s (0.43 s at k = 4, 0.86 s at k = 8, i.e. within 1.5 days);
   - `compiled_graphs`, and whether any update after the first takes more than 2 s;
   - peak memory;
   - padding ratio `padded_positions ÷ (LM input tokens + pinyin input tokens)`, target at most 1.1;
   - the loss at update 20, as a sanity check only.
4. **Go condition**: start the formal run only at or below 0.18 × k s per update (0.72 s at k = 4, 1.44 s at k = 8; about 2.5 days or less). Otherwise return with the profiler results and decide then.

## Quality decision (preregistered)

arch2 and arch1 use the same data and the same number of tokens (batch and learning rate scaled as in Larger updates), and are compared on development:

1. Primary metric: public pinyin set Top-1 after self-rescoring (λ = 1.6), with a sentence-cluster sign-flip paired test.
2. arch2 replaces arch1 as the 800M only if both hold:
   - the primary metric is not significantly below arch1;
   - Standard full-pinyin rerank Top-1 is not significantly below arch1.
3. Prediction (tracks A–D) and milestone BPC are descriptive.
4. Round 4 distillation uses whichever of the two has the higher primary metric as the teacher.

## Timeline and release

- arch2 starts after arch1 has trained and been evaluated: step 0 and the 200M batch precheck first (independent, so they can run in parallel), then the code.
- The release name `vimeml-zh-v2-800m` refers to arch1. If arch2 is ready and passes before v2 is final, it replaces arch1; otherwise it goes into the next release.

## Out of scope

- Document-level long context, a new vocabulary, rebuilt windows.
- MoE and FSDP (the model and optimizer state fit on one GPU).
- FP8 and any other matmul precision below BF16 (prohibited).
- Changing the loss normalization. Batch, learning rate, warmup and step count change only by the Larger updates scaling, with no separate tuning.

## Deviations

(None yet.)
