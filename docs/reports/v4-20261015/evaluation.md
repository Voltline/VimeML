# First V4 run and local evaluation — 2026-10-15

V4 improves general Japanese LM but does not establish a stable V2.1 reranking replacement gain. Primary full FP32 BPC falls from 3.080191 to 3.051781 (0.922% relative). The LM-selected checkpoint scores 492/728 on Standard development versus 505/728 for V2.1; development-selected epoch 8 scores 503/728. No blind data participates.

## Training and archival

Fixed 8K, d384,7 layers, 6 heads, SwiGLU 960, context 128,15,070,848 parameters. The twelve-epoch ceiling ends through normal plateau stopping at nine complete passes, step 443,115,5,174,759,074 effective targets (8.0203 uncropped-store token exposures after prefix crops). Measured-update throughput is 441,043.7 effective tokens/s, excluding validation, saving and initial warmup timing.

Downloaded 62 files/1.078 GiB with per-file SHA256 verification. Full best (step 372000), best-epoch (epoch 8, step 393880), and last (epoch 9, step 443115) retain optimizer/RNG state. All nine epoch inference weights, tokenizer, configurations, reports and logs are archived. The three corresponding inference packs are bitwise identical to their full model tensors; inference packs are not resumable training checkpoints.

`best.pt` is selected by fixed-subset validation loss; `best-epoch.pt` is selected by complete-epoch BPC. Full best SHA256: `f12fa61230222485339af9609748f5f074f17a134b35d7d6a4310b6c2db35326`.

## Data and protocol

V4 and deployed V2.1 use identical original train/validation text, with different tokenizers. Both train TXT identities are `9c1b4585adaf0518dc0275e7f8086e967db26f9797fb571e1ff481aeadf2ff8f`:24,938,228 FineWeb and 247,140 Tatoeba sentences. The eight-shard JpnMix addition used by V3 is absent from this V4 run. Domain/repetition effects remain possible, but a shared newly added batch is not established as the explanation.

The evaluation plan, checkpoint identities, inputs, and selection rule were frozen before scoring. Complete validation contains the same 260,337 sentences/11,110,489 Unicode characters. All prediction targets including EOS are retained:5,716,082 for V2.1 and 6,627,568 for V4. Compare BPC rather than token PPL. V2/V4 row offsets address JSONL/TXT respectively; original TXT SHA and source ordering establish alignment.

Local RTX 5060 Laptop, FP32, TF32 disabled. Fresh V2.1 Standard evaluation exactly reproduces previous candidate orders and token log probabilities (maximum drift 0). Existing scoring is unchanged: joint-tokenized suffix full-vocabulary logP sum, no EOS/truncation, stable ties, whole-case overflow fallback, full denominator including pool misses. No Standard fallback occurs for any evaluated checkpoint.

Standard development 728 has AI expert review, not native-human gold; legacy 500 remains historical draft. The original Standard export manifest is unavailable locally. The exact candidate snapshot was recovered from completed historical records, with retained references, context, source/provenance, and file hashes verified. Candidate pools and labels were preserved. AJIMEE's retained export is independently verified by its existing loader. Historical 137 references also match their retained source hash. Report tracks separately; they are not an independent held-out confirmation. Blind is not read, scored or inferred.

## Full validation BPC

| Checkpoint | Local full BPC (FP32) |
| --- | ---: |
| V2.1 | 3.080190570 |
| V4 best / 372000 | 3.051781125 |
| V4 epoch 8 / 393880 | 3.057868989 |
| V4 last / 443115 | 3.058944878 |

Local FP32 and remote BF16 training measurements are distinguished; remote best/last are 3.051885722/3.058956864.

## Frozen-pool reranking Top-1

| Track | AzooKey | V2.1 | V4 best | V4 epoch 8 | Pool recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard dev / 728 | 453 | 505 | 492 | 503 | 630/728 |
| Legacy regression / 500 | 317 | 374 | 372 | 367 | 443/500 |
| AJIMEE / 200 | 87 | 144 | 142 | 144 | 162/200 |
| Historical dev / 137 | 111 | 122 | 121 | 120 | 136/137 |

| Standard source | V2.1 | V4 best | V4 epoch 8 |
| --- | ---: | ---: | ---: |
| wrime | 298/483 | 288/483 | 292/483 |
| jmultiwoz | 207/245 | 204/245 | 211/245 |

On Standard, V2.1 corrects 100/damages 48 relative to AzooKey; primary V4 corrects 100/damages 61. Epoch 8 is closer on development without a consistent advantage across the other tracks. Paired p-values below are exploratory, unadjusted descriptive statistics, not significance claims.

| Track | Checkpoint | Improved vs V2.1 | Damaged vs V2.1 | Net | Exact McNemar p |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard dev | V4 best / 372000 | 33 | 46 | -13 | 0.176611 |
| Standard dev | V4 epoch 8 / 393880 | 39 | 41 | -2 | 0.911072 |
| Legacy regression | V4 best / 372000 | 13 | 15 | -2 | 0.850554 |
| Legacy regression | V4 epoch 8 / 393880 | 11 | 18 | -7 | 0.264931 |
| AJIMEE | V4 best / 372000 | 7 | 9 | -2 | 0.803619 |
| AJIMEE | V4 epoch 8 / 393880 | 9 | 9 | +0 | 1.000000 |
| Historical dev | V4 best / 372000 | 1 | 2 | -1 | 1.000000 |
| Historical dev | V4 epoch 8 / 393880 | 2 | 4 | -2 | 0.687500 |

## Epoch trajectory and selection

| Epoch | Step | Remote full BPC (BF16) | Local Standard dev Top-1 (FP32) |
| ---: | ---: | ---: | ---: |
| 1 | 49235 | 3.294657 | 477/728 |
| 2 | 98470 | 3.206939 | 483/728 |
| 3 | 147705 | 3.115083 | 498/728 |
| 4 | 196940 | 3.113201 | 479/728 |
| 5 | 246175 | 3.100866 | 475/728 |
| 6 | 295410 | 3.079426 | 484/728 |
| 7 | 344645 | 3.061917 | 481/728 |
| 8 | 393880 | 3.058166 | 503/728 |
| 9 | 443115 | 3.058957 | 493/728 |

The predeclared Standard-development Top-1 rule selects epoch 8 from best and nine epochs; ties prefer best, then the earlier epoch. Other tracks do not change selection. Keep this selection separate from LM-best step 372000. Falling BPC does not produce monotonic reranking gains. The original 4→8→12 continuation schedule differs from planning twelve epochs initially; stopping does not establish saturation under another schedule.

## Homophone and suggestion diagnostics

| Diagnostic | V2.1 | V4 best | V4 epoch 8 |
| --- | ---: | ---: | ---: |
| Contextual homophones / 32 | 31 | 31 | 32 |
| Greedy reference coverage / 14 | 6 | 8 | 5 |
| Sample Top-5 reference coverage / 14 | 9 | 11 | 9 |
| Beam Top-1 reference coverage / 14 | 8 | 9 | 5 |
| Beam Top-5 reference coverage / 14 | 10 | 11 | 11 |

The 14 suggestion references are non-exhaustive and are not real-world prediction accuracy. Preserve 6-token/beam 8/alpha 0.7/eight-seeded-sample settings; also archive twelve fixed prompts with 16-token greedy/sample output. Character budgets differ across tokenizers. Each model’s24 primary/V2.1 qualitative outputs preserve prompts without replacement characters; texts remain in local results.

## Tokenizer diagnostics

Mean candidate lengths are 11.482/10.049 tokens for 8K/16K; readings 20.444/19.346; candidate characters/token 1.555/1.777. Both have zero UNK, roundtrip mismatches and whole-case 128-token overflow. Reading byte-fallback is 0.383%/3.962%. Candidate token inflation is about 14.26%; a possible score-preference effect remains a hypothesis, not a causal result. Scoring rules are unchanged. The 8K goal remains increased backbone capacity.

## Next experiment

A separate V4-B run keeps the same data, 8K tokenizer, fixed model, seed 42, batch 512, BF16, AdamW, and prefix crop. Fresh initialization uses a complete twelve-epoch/590,820-step cosine from 6e-4 to 6e-5 with 1,000 warmup updates; plateau delta 0.01 and patience 2 remain fixed. The configuration is `configs/train-v4-b12cosine.toml`, with independent output under `artifacts/models/tiny-ja-v4-b12cosine-e8k-d384-l7/`. This tests the early-annealing hypothesis without adding data or task objectives. The completed results are in the [V4-B comparison](b12cosine-evaluation.md).

Local evidence: [summary](../../../outputs/evaluation-v4-20261015/comparison-summary.json), [frozen plan](../../../outputs/evaluation-v4-20261015/plan.json), [download hashes](../../../outputs/training-v4-20261015/download-manifest.json). Binary/per-case artifacts are excluded from Git. This report does not establish Apple deployment performance.
