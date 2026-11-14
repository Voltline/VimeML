# zh-200m six-arm training record (2026-11-13)

All six arms of the [200M training plan](../../training-zh-200m-plan.md) trained as planned. This page records the training process, integrity issues and validation results; scoring on the prediction and reranking benchmarks is in the [judgement report](../zh-200m-20261114/judgement.md). Blind was not read.

## Shapes and scale

| Arm | Vocabulary | Context | Layers | Parameters | Updates | Training time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16k-c512 | 16,384 | 512 | 15 | 198,212,608 | 783,884 | 13.6 h |
| 24k-c512-s1 | 24,576 | 512 | 14 | 194,540,544 | 782,279 | 12.8 h |
| 24k-c512-s2 | 24,576 | 512 | 14 | 194,540,544 | 782,284 | 12.8 h |
| 24k-c512-s3 | 24,576 | 512 | 14 | 194,540,544 | 782,280 | 14.2 h |
| 32k-c512 | 32,768 | 512 | 14 | 202,929,152 | 781,380 | 13.0 h |
| 24k-c128 | 24,576 | 128 | 14 | 194,147,328 | 828,250 | 10.1 h |

All six share one shape family: d1024, 16 heads, SwiGLU d_ff 2560, RMSNorm, no bias, tied embedding and output layer, learned positions. Data and recipe as in the plan: the P1 pool (21,280,000,190 characters) once plus LCCC at a 5% supervised-token quota; cosine 6e-4→6e-5, warmup 1000, wd 0.1, β 0.9/0.95, clip 1.0, prefix crop 0.30, BF16.

## Results (20th measurement point, full FP32 validation)

| Arm | Web BPC | Chat BPC | P20 memorization sample BPC |
| --- | ---: | ---: | ---: |
| **16k-c512** | **3.8251** | 5.3731 | **3.8083** |
| 24k-c512-s1 | 3.8641 | 5.4210 | 3.8469 |
| 24k-c512-s2 | 3.8750 | 5.4133 | 3.8577 |
| 24k-c512-s3 | 3.8767 | 5.4046 | 3.8601 |
| 32k-c512 | 3.9153 | 5.4446 | 3.8995 |
| 24k-c128 | 3.8910 | 5.4037 | 3.8752 |

Descriptive observations (not used for the decision; the decision's primary metric is the prediction benchmark, scored separately):

- **Scale**: from v1 (20M, 12K, web BPC 4.3533) to the six 200M arms, web BPC improves by 0.44–0.53. The memorization gap (training-document BPC minus validation BPC) is 0.007–0.030 across the six arms; its scale depends on vocabulary, so it is not compared in absolute terms across vocabularies.
- **Vocabulary**: the three 24K seeds have web BPC 3.8641/3.8750/3.8767, a run-to-run range of 0.0126; 16K is about 0.04–0.05 below 24K and 32K about 0.04–0.05 above. 16K has 8.2M fewer embedding parameters than 24K and gets a whole extra layer for them (15 vs 14), consistent with the "smaller vocabulary buys layers" trade-off in the vocabulary report.
- **Context**: 24k-c128 (3.8910) is about 0.019 above the mean of the three c512 seeds with the same vocabulary (3.8719). Both use 512 windows per update, about 16k supervised tokens; the difference is only the attention span and how many long sentences are cut.
- These are development-style validation numbers and do not represent prediction or reranking performance.

## Issues found and fixed along the way (recorded as they happened)

1. **Incomplete window indexes (data side).** The window indexes in all three encoding outputs were incomplete when uploaded: the 16K c512 index had only 634,285 entries (1,062,337 expected, 60%), the 24K and 32K indexes fewer, and the chat index was empty. The token stores themselves checked out byte for byte (16K 30,075,892,282 bytes, 24K 28,406,081,214, 32K 27,411,228,006, matching the output manifests). The indexes were rebuilt deterministically from the token stores: 16K c512 331,553,667 windows, 24K c512 331,436,403, 24K c128 353,916,830, 32K c512 400,066,540; the first two equal the counts derived independently from the offsets, and the chat side matches the 16,326,323 declared in the encoding manifest.
2. **Hard-coded context limit.** The batch collator hard-coded the training width limit to 128 (v1's context), so c512 arms failed on data loading with `Task sample outside the fixed context`. It now takes the model's context, still defaulting to 128 so v1/v4 behavior is unchanged, with a unit test added. The error happened twice: first in the main training loop, then in a second, unpatched loader used by the fixed-subset validation every 5,000 steps.
3. **Batch strategy correction** (see deviation 2 in the plan). The registration was "65,536 capacity tokens per update" (128 windows × 512 for c512 arms). Windows turned out to average only about 34 tokens, so 128 windows gave only about 4,320 supervised tokens per update, a quarter to a fifth of v1. With the original setting the four c512 arms had web BPC 6.46 at the 2/20 point against 4.47 for 24k-c128, a gap that could not come from parameters or data. The four c512 arms were rerun with 512 windows (about 16.4k supervised tokens, the same order as v1's about 21.5k); 24k-c128 already used 512 windows and was not rerun. After the rerun, the same arm's web BPC at the 1/20 point fell from 6.6078 to 4.5449.
4. **Local compute unusable.** Under sustained full load the development machine's compute processes were heavily throttled (16 processes got about 0.35 cores in total), so SentencePiece training and window index rebuilding were not feasible locally; these steps moved to compute machines. The local disk was used only for transfer and checks.

## Integrity checks

All six arms pass every item:

- `summary.json` status `complete`, each arm's `step` equal to `max_steps`;
- `final.pt` present and loadable, checkpoint header step equal to total steps, optimizer state non-empty;
- `milestones.json` with exactly 20 measurement points, numbered 1–20 with none missing;
- at training start every dependency is checked file by file by SHA256, and each arm's mixture `dataset_sha256` matches the value pinned in its config;
- checkpoint choice is the preregistered final checkpoint, not chosen by BPC or any downstream result.

## Outputs

Each arm in `artifacts/models/tiny-zh200m-<arm>-d1024/`: `final.pt`, `milestones.json` (full validation at 20 points), `metrics.jsonl`, `plan.json`, `config.json`, `environment.json`, `summary.json`, and the two milestone checkpoints at 1/20 and 1/5.

## Next (outside this record)

Development scoring on the prediction benchmark `ime-predict-zh-v1-r1` and the reranking benchmark, and the decision on the plan's three contrasts (16K vs 24K, 32K vs 24K, c128 vs c512); done in the [judgement report](../zh-200m-20261114/judgement.md).
