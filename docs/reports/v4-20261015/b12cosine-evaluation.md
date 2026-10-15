# V4-B twelve-epoch cosine comparison — 2026-10-15

The full twelve-epoch schedule improves LM further: local FP32 BPC **3.027777**, 0.787% below V4-A's 3.051781 and 1.702% below V2.1. LM-best scores 498/728 on Standard development; the predeclared development rule selects **best, step 584000, 498/728**. LM and candidate ordering are assessed separately below.

A denotes the first 4→8→12 budget-extension experiment; B denotes a twelve-epoch cosine chosen at initialization. Both are base-LM experiments. Original data, 8K tokenizer and model dimensions are unchanged.

## Training facts and controls

Completed twelve epochs/590,820 steps/6,899,499,874 effective targets without early stopping. Trainer-recorded session elapsed is 15,941.1 seconds; measured update throughput 439,458.1 effective tokens/s. A completed nine epochs/5,174,759,074 targets; B consumes **33.33% more effective tokens**. The comparison includes schedule and realized training budget, so gains cannot be attributed solely to learning-rate shape.

Fixed 15,070,848 parameters, 8K/context 128, d384/layers 7/heads 6/SwiGLU 960; fresh seed 42, batch 512, BF16, compiled backbone, same AdamW and 30% prefix crop. B chooses one twelve-epoch cosine 6e-4→6e-5, warmup 1000, plateau min_delta 0.01/patience 2. Checkpoints confirm identical model/tokenizer/token and window manifests/shared trainer-model-data hashes and training settings. A's continuation helper signature is absent when unused by fresh B. FineWeb/Tatoeba text is unchanged; no data additions/removals.

Full best is step 584000 (fixed-subset-loss selection). Last and best-epoch are epoch 12/step 590820,  with bitwise equal model tensors. Downloaded 45 files/1.373 GiB with SHA256 verification. Three full checkpoints retain optimizer/RNG; all twelve epoch inference weights are archived. Corresponding best/epoch 12 inference packs are bitwise equal to full model tensors. Full-best SHA256: `18c49f1d4de5b2852f98c9bc30bf48ca7c84077cfb09a25e3c9002cedf5d1106`.

## Complete validation BPC

| Checkpoint | Full FP32 BPC |
| --- | --- |
| V2.1 | 3.080190570 |
| V4-A LM best | 3.051781125 |
| V4-A epoch 8 | 3.057868989 |
| V4-B LM best | 3.027777023 |
| V4-B last / epoch 12 | 3.032232369 |

Same 260,337 sentences/11,110,489 Unicode characters/6,627,568 V4 prediction targets including EOS, without truncation. All table entries use local RTX 5060 Laptop/PyTorch 2.10/FP32 with TF32 disabled. Remote BF16 best/last are 3.028079685/3.032557366 and are distinguished from local values. V2.1/A reuse the previous complete FP32 results after input/model/tokenizer identity checks.

## Frozen candidate reranking

| Track | V2.1 | V4-A LM best | V4-A epoch 8 | V4-B LM best |
| --- | --- | --- | --- | --- |
| Standard dev / 728 | 505 | 492 | 503 | 498 |
| Legacy / 500 | 374 | 372 | 367 | 377 |
| AJIMEE / 200 | 144 | 142 | 144 | 144 |
| Historical dev / 137 | 122 | 121 | 120 | 120 |

| Source | V2.1 | V4-A LM best | V4-A epoch 8 | V4-B LM best |
| --- | --- | --- | --- | --- |
| wrime | 298/483 | 288/483 | 292/483 | 297/483 |
| jmultiwoz | 207/245 | 204/245 | 211/245 | 201/245 |

Evaluation uses identical frozen 728/500/200/137 inputs and AzooKey pools; no blind access or label/pool/scoring changes. Primary score remains joint-tokenized suffix full-vocabulary logP sum, no EOS/truncation, stable ties, whole-case fallback and full denominator. Standard labels are AI-reviewed, not native-human gold; legacy 500 is historical draft. Original Standard export manifests remain unavailable; use previously verified historical candidate snapshots with unchanged provenance limitations. Baseline reuse verifies per-case frozen equality, precision and checkpoint identities; source hashes are recorded.

Paired p-values are exploratory and unadjusted, not multiplicity-corrected significance claims. Development selection compares best and twelve epochs by Standard contextual-sum Top-1; ties prefer best, then the earlier epoch. Other tracks do not alter selection.

| V4-B | Track | Comparator | Improved | Damaged | Net | Exact p |
| --- | --- | --- | --- | --- | --- | --- |
| best | Standard dev / 728 | v21 | 42 | 49 | -7 | 0.529602 |
| best | Standard dev / 728 | v4a-best | 38 | 32 | +6 | 0.550413 |
| best | Standard dev / 728 | v4a-epoch-8 | 35 | 40 | -5 | 0.644464 |
| best | Legacy / 500 | v21 | 18 | 15 | +3 | 0.728332 |
| best | Legacy / 500 | v4a-best | 15 | 10 | +5 | 0.424356 |
| best | Legacy / 500 | v4a-epoch-8 | 21 | 11 | +10 | 0.110184 |
| best | AJIMEE / 200 | v21 | 7 | 7 | +0 | 1.000000 |
| best | AJIMEE / 200 | v4a-best | 10 | 8 | +2 | 0.814529 |
| best | AJIMEE / 200 | v4a-epoch-8 | 7 | 7 | +0 | 1.000000 |
| best | Historical dev / 137 | v21 | 2 | 4 | -2 | 0.687500 |
| best | Historical dev / 137 | v4a-best | 2 | 3 | -1 | 1.000000 |
| best | Historical dev / 137 | v4a-epoch-8 | 4 | 4 | +0 | 1.000000 |

## Training trajectory and realized budget

| Epoch | A full BF16 BPC | B full BF16 BPC | A dev Top-1 | B dev Top-1 |
| --- | --- | --- | --- | --- |
| 1 | 3.294657 | 3.310575 | 477 | 473 |
| 2 | 3.206939 | 3.270015 | 483 | 478 |
| 3 | 3.115083 | 3.244630 | 498 | 494 |
| 4 | 3.113201 | 3.218378 | 479 | 498 |
| 5 | 3.100866 | 3.196170 | 475 | 481 |
| 6 | 3.079426 | 3.168985 | 484 | 490 |
| 7 | 3.061917 | 3.142492 | 481 | 482 |
| 8 | 3.058166 | 3.116429 | 503 | 498 |
| 9 | 3.058957 | 3.086446 | 493 | 495 |
| 10 | — | 3.057157 | — | 490 |
| 11 | — | 3.037641 | — | 496 |
| 12 | — | 3.032557 | — | 489 |

At the same ninth epoch, B BPC 3.086446 trails A3.058957; B overtakes A-best only during epochs 11–12. This supports budgeting through the planned anneal, while interim loss alone cannot determine final quality. BPC and development Top-1 remain non-monotonic. One seed 42 run per schedule does not establish a general causal 8K/15M architecture advantage.

| Diagnostic | V2.1 | V4-A LM best | V4-A epoch 8 | V4-B LM best |
| --- | --- | --- | --- | --- |
| Homophones / 32 | 31 | 31 | 32 | 32 |
| Greedy coverage / 14 | 6 | 8 | 5 | 9 |
| Sample Top-5 coverage / 14 | 9 | 11 | 9 | 8 |
| Beam Top-1 coverage / 14 | 8 | 9 | 5 | 8 |
| Beam Top-5 coverage / 14 | 10 | 11 | 11 | 11 |

The 32 homophone and 14 suggestion cases are small synthetic diagnostics with non-exhaustive references. Greedy/sample/beam can move differently and are not real-world prediction accuracy. Equal token budgets differ in character coverage across 8K/16K. Archive twelve fixed qualitative prompts without tuning to outputs.

## Assessment and retained artifacts

The complete twelve-epoch design has clear LM gains and is a candidate foundation for later task training. On Standard, primary B corrects 42/damages 49 versus V2.1 (net-7, descriptive p=0.529602); no stable replacement gain is established. Relative to AzooKey, B corrects 98/damages 53 versus 100/48 for V2.1. On 407 contextual cases, B scores 272 versus 275 for V2.1 and 254 for B without context. Context helps without an overall lead. A useful next question is whether reading-conditioned training improves the same pools; these results do not establish a need to immediately extend pure-LM training. A/B LM-best and development-best checkpoints remain archived with separate selection rationales.

[Local summary](../../../outputs/evaluation-v4-b12cosine-20261015/comparison-summary.json) · [Frozen plan](../../../outputs/evaluation-v4-b12cosine-20261015/plan.json) · [Download hashes](../../../outputs/training-v4-b12cosine-20261015/download-manifest.json) · [Control audit](../../../outputs/evaluation-v4-b12cosine-20261015/experimental-controls.json)
