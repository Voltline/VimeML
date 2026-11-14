# RakutenAI-2.0-mini-instruct local reranking comparison

2026-10-16. Reuses the four frozen development/regression sets and the raw-text probability scoring interface of the [TinySwallow comparison](external-1p5b.md); 1,565 items evaluated locally on an RTX 5060 Laptop 8GB. Rakuten scores 462/728 on Standard development, not reproducing TinySwallow's 527/728; the other three sets are close to V2.1. The result compares suitability for candidate reranking and does not represent the model's overall Japanese or dialogue ability.

## Model and fixed protocol

The official [Rakuten/RakutenAI-2.0-mini-instruct](https://huggingface.co/Rakuten/RakutenAI-2.0-mini-instruct), revision `6d902489587d324b7d5e201299e4e1a169f3a40b`. The model card describes it as the instruction-tuned version of RakutenAI-2.0-mini, whose [base model](https://huggingface.co/Rakuten/RakutenAI-2.0-mini) is a Japanese language model trained from scratch. This run uses the instruction-tuned weights, not the base weights.

The model is a Mistral architecture: 1,534,683,136 parameters, 48,000-token vocabulary, 22 layers, hidden 2048, native BOS 1. BF16 weights and forward, FP32 full-vocabulary CE, no quantization or training updates. The isolated Transformers 4.57.6 and PyTorch 2.10.0+cu128 are reused; safetensors are loaded with remote code disabled.

The original candidates, references, left contexts and input file hashes are unchanged. The primary score is still the full-vocabulary log-probability sum after the common token prefix, with `context + candidate` tokenized jointly; scoring starts at the native BOS with no candidate-final EOS. The window is fixed at 128, ties keep AzooKey's order, and context is not trimmed; any misfit would fall back for the whole item, and none of the four sets had any fallback. The native 128K context was not used to extend the window.

No chat template, instruction prompt or READ condition was added. This measures how the instruction-tuned weights transfer to the existing raw-text LM reranking interface and cannot be extrapolated directly to instruction-style conversion. Mean and empty-left-context scores were predeclared as auxiliary diagnostics; nothing was tuned on them.

## Full results

Top-1 correct counts; V2.1, V4-B and TinySwallow reuse identity-checked historical per-item results.

| Set | Items | AzooKey | V2.1 | V4-B | TinySwallow | Rakuten Instruct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard development | 728 | 453 | 505 | 498 | **527** | 462 |
| Legacy regression | 500 | 317 | 374 | **377** | **377** | 375 |
| AJIMEE | 200 | 87 | **144** | **144** | **144** | **144** |
| Historical development | 137 | 111 | 122 | 120 | **127** | 122 |

Standard is 63.46%, 5.91 points below V2.1 and 8.93 below TinySwallow. Against the original AzooKey order it fixes 91 items and breaks 82, net +9; the same pool covers the correct answer on 630/728 items.

| Standard source | Items | V2.1 | V4-B | TinySwallow | Rakuten Instruct |
| --- | ---: | ---: | ---: | ---: | ---: |
| WRIME | 483 | 298 | 297 | 319 | 282 |
| JMultiWOZ | 245 | 207 | 201 | 208 | 180 |

Both sources fall below the existing plain-LM baselines. The comparison does not hold everything except parameter count fixed: architecture, vocabulary, data and training all differ, so the difference cannot be attributed to instruction tuning or the Mistral architecture alone.

## Paired comparisons and auxiliary diagnostics

Against TinySwallow, on Standard it fixes 31 and loses 96, net −65. Against V2.1, 48/91, net −43; against V4-B, 46/82, net −36. Two-sided exact McNemar descriptive p values without multiple-comparison correction: `6.33e-9`, `0.000333`, `0.001862`. These remain exploratory comparisons on reused development/regression sets.

Against TinySwallow, gain/loss is 20/22 on Legacy, 10/10 on AJIMEE and 5/10 on Historical. The AJIMEE tie still has per-item differences.

| Set | Sum with original context, primary | Sum, empty context | Mean with original context, auxiliary |
| --- | ---: | ---: | ---: |
| Standard development | 462 | 416 | 337 |
| Legacy regression | 375 | 333 | 335 |
| AJIMEE | 144 | 122 | 102 |
| Historical development | 122 | 116 | 69 |

The left context gives a positive total gain on all four sets but does not close the main development gap to TinySwallow. Based on these results, the primary scoring was not changed, no other prompt was added and no other model version was selected.

## Run and audit

The full evaluation including model load took 168.10 s, excluding weight download, dependency setup and result checks. PyTorch weight allocation 2,928.18 MiB, peak allocated 3,000.22 MiB, peak reserved 3,050 MiB (about 2.98 GiB). The records exclude the desktop and other processes and are neither physical footprint nor deployment latency.

The evaluator `scripts/benchmarks/evaluate_external_lm.py` gained an explicitly pinned model registry entry; the scoring implementation is unchanged. The raw source bytes of each run's entry point are kept in its result directory, and the SHA256 of the old TinySwallow plan and source can still be checked. The math consistency check of blocked projection against full logits passes; the output projection path of the native Mistral forward was checked.

Outputs are in `outputs/evaluation-rakuten-instruct-20261016/`. `model-source.json` and `plan.json` record the source, model and data SHA256 and the protocol; `environment.json` and `dependencies.json` the environment; per set, `scores.jsonl`, `metrics.json` and `paired-*.json` the per-token scores and per-item comparisons against the three baselines; `summary.json`, `comparison-summary.json` and `acceptance.json` completion and audit. Checks of candidate sets, finite scores, target counts and log-probability sums passed on all 1,565 items, and the evaluation inputs and three baseline files were unchanged.

No test or blind data was read, no labels changed, no evaluation answers used to train a model or write a task prompt, and nothing was done to the remote V4.1 training. Overlap between the external pretraining/fine-tuning text and the public evaluation content is unknown; the source limitations of the development/regression references and snapshots still apply.

The result supports keeping TinySwallow as the preferred teacher candidate for the current probability-reranking task. Japanese fine-tuning labels and general dialogue metrics cannot replace an actual IME evaluation; whether TinySwallow's gain can be distilled into a 15M model is still a separate open question.
