# Sarashina2.2-1B local candidate-reranking comparison

Completed 2026-10-16 on an RTX 5060 Laptop 8GB across four frozen development/regression sets, 1,565 items in all. Sarashina scores 518/728 on Standard development, 13 more than V2.1 and 9 fewer than TinySwallow; on the other three sets it beats TinySwallow by 4, 6 and 1. The result supports keeping two teacher candidates and is not enough to call either one stronger across the board.

## Model and protocol

The official [sbintuitions/sarashina2.2-1b](https://huggingface.co/sbintuitions/sarashina2.2-1b) base model, pinned revision `3bb836ad7475ba192926be66651e4730825df7da`. The official "1B" count excludes the embedding and output head; the loaded total is **1,407,538,944** parameters, 102,400-token vocabulary, 24 layers, hidden 1792, native BOS 1. This is not the separately published Instruct version.

Weight SHA256 `d23ae09c0c8e1f161bea04640359f27f706514016b65a22ad726e8b0b1624420`, matching the official release pointer. Only safetensors and the built-in Transformers Llama implementation are loaded, `trust_remote_code=False`. Tokenizer roundtrip compatibility for Japanese, spaces, newlines and full-width characters was checked.

The raw-text probability protocol of [TinySwallow](external-1p5b.md) and [Rakuten](external-rakuten-1p5b.md) is reused: `context + candidate` tokenized jointly, full-vocabulary log-probability sum after the common token prefix; native BOS start, with no candidate-final EOS, chat template, task instruction or READ condition. The window stays at 128, not extended to the native 8,192; no trimming, stable ties, whole-item fallback and the full denominator are all kept.

BF16 weights and forward, FP32 CE, no quantization or training updates. Candidates go in batches of eight, and only scored positions are projected onto the full vocabulary in blocks of 128 positions; the scoring formula is unchanged. Mean and empty left context are predeclared auxiliary diagnostics only; nothing was tuned on the scores.

## Full results

Top-1 correct counts. V2.1 is the official extend5 baseline and V4-B the step-584,000 LM best; all reuse identity-checked historical per-item results.

| Set | Items | AzooKey | V2.1 | V4-B | TinySwallow | Sarashina |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard development | 728 | 453 | 505 | 498 | **527** | 518 |
| Legacy regression | 500 | 317 | 374 | 377 | 377 | **381** |
| AJIMEE | 200 | 87 | 144 | 144 | 144 | **150** |
| Historical development | 137 | 111 | 122 | 120 | 127 | **128** |

The four sets are not added up as independent test samples. Standard accuracy is 71.15%, 1.79 points above V2.1 and 1.24 below TinySwallow. Against the original AzooKey order it fixes 108 items and breaks 43, net +65. The pool still covers the correct answer on 630/728 items; no candidates were generated or added.

| Standard source | Items | V2.1 | TinySwallow | Sarashina |
| --- | ---: | ---: | ---: | ---: |
| WRIME | 483 | 298 | 319 | 313 |
| JMultiWOZ | 245 | 207 | 208 | 205 |

The net gain over V2.1 comes from +15 on WRIME, with JMultiWOZ 2 lower. The ranking signal of external Japanese models on these natural expressions is of research value, but the differences cannot be attributed to parameter count, training data or architecture alone.

## Paired comparisons

"Original model wrong, Sarashina right / original right, Sarashina wrong"; p is a two-sided exact McNemar descriptive value without multiple-comparison correction.

| Set | vs V2.1 gain/loss | p | vs TinySwallow gain/loss | p |
| --- | ---: | ---: | ---: | ---: |
| Standard development | 58 / 45 | 0.23690 | 32 / 41 | 0.34918 |
| Legacy regression | 23 / 16 | 0.33678 | 22 / 18 | 0.63583 |
| AJIMEE | 13 / 7 | 0.26318 | 10 / 4 | 0.17957 |
| Historical development | 7 / 1 | 0.07031 | 5 / 4 | 1 |

A 9-item total difference does not establish TinySwallow as reliably stronger, and the point estimates on the other three sets do not establish Sarashina as reliably stronger either. On Standard, Sarashina fixes 32 of TinySwallow's errors while losing 41 of its correct items, so their per-item ranking signals differ; this run did not use the answers to tune fusion weights or train a selector.

| Set | Sum with original context, primary | Sum, empty context | Mean with original context, auxiliary |
| --- | ---: | ---: | ---: |
| Standard development | 518 | 492 | 297 |
| Legacy regression | 381 | 349 | 298 |
| AJIMEE | 150 | 139 | 105 |
| Historical development | 128 | 125 | 40 |

## Resources, outputs and limits

The full evaluation including model load took **158.97 s**, excluding download, dependency setup and result checks. PyTorch allocated after model load 2,725.17 MiB, peak allocated 2,861.55 MiB, peak reserved 3,174 MiB (about 3.10 GiB). These are allocator records that exclude the desktop and other processes and are neither physical footprint nor iOS latency. Runtime: PyTorch 2.10.0+cu128, Transformers 4.57.6, reusing the isolated external-model dependencies.

The entry point `scripts/benchmarks/evaluate_external_lm.py` gained an explicitly pinned Sarashina registry entry; the scoring implementation is unchanged. Source snapshot, source and model file hashes, protocol, environment, dependencies, per-token scores, per-item paired results and check records are in `outputs/evaluation-sarashina-20261016/`. `acceptance.json` confirms that candidate sets, input and three baseline hashes, supervised position counts, finite scores and log-probability sums passed on all 1,565 items; no set had window or roundtrip fallbacks.

No test or blind data was read, no references changed, no training updates run, and nothing done to the remote V4.1. Full LM validation BPC, reading-conditioned and instruction-style conversion were not evaluated. Overlap between the external pretraining text and the public benchmarks is unknown; the existing development/regression label and snapshot source limitations still apply. This is a teacher-candidate screening result; transferring the gain to a 15M model needs a separate distillation experiment.
