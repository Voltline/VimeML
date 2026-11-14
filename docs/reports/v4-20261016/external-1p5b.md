# TinySwallow-1.5B local candidate-reranking comparison

Completed 2026-10-16 on an RTX 5060 Laptop 8GB across four frozen development/regression sets, 1,565 items in all. TinySwallow scores 527/728 on Standard development, 22 more than V2.1 and 29 more than V4-B. It ties V4-B on the old 500-item set and AJIMEE and gains 7 on historical development. The sets are not added up as independent test sets.

## Model and protocol

The official [SakanaAI/TinySwallow-1.5B](https://huggingface.co/SakanaAI/TinySwallow-1.5B), pinned revision `535d1322615c9f8171c3d60b10ef6955cc896fe9`. The model card describes Qwen2.5-32B-Instruct as teacher and Qwen2.5-1.5B-Instruct as student, trained with TAID and continued Japanese pretraining. This run uses the plain TinySwallow release, not the separately published Instruct version.

Loaded: 1,543,714,304 parameters, 151,936-token vocabulary. BF16 weights/forward, FP32 full-vocabulary cross entropy, no quantization, no training updates, no remote inference; safetensors with the built-in Transformers Qwen2 implementation, `trust_remote_code=False`.

The primary score keeps the existing plain-LM rule: joint tokenization of the raw `context + candidate`, summing full-vocabulary log probabilities after the common token prefix of all candidates and the original context. No candidate-final EOS, no truncation, no task prompt, chat template, reading condition or answer. Scoring starts from the model's native BOS 151643. Mean and empty-context scores are predeclared auxiliary diagnostics only; no score or mixing weight was changed based on the results.

The scoring window stays at 128 predicted positions; the model's native 32K context was not used to extend the left context. Each tokenizer may cover a different number of characters within the same window, but input text, candidates, all acceptable answers, the stable tie rule and the whole-item fallback rule follow the frozen protocol. No item in any of the four sets fell back for window or roundtrip reasons, and the full denominator is kept.

Candidates go in batches of at most eight. Only the positions to be scored are extracted and projected onto the full vocabulary in blocks of 128 positions, which avoids producing large unused logits; the vocabulary is neither pruned nor renormalized. CPU checks cover equivalence of blocked and full logits, varying candidate lengths, padding, boundary retokenization and block size. After the run, candidate sets, input and baseline hashes, target counts, finite token scores and log-probability sums were checked item by item; all 1,565 passed.

## Main results

| Frozen set | Items | AzooKey | V2.1 | V4-B | TinySwallow-1.5B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard development | 728 | 453 | 505 | 498 | **527** |
| Legacy regression | 500 | 317 | 374 | 377 | **377** |
| AJIMEE | 200 | 87 | 144 | 144 | **144** |
| Historical development | 137 | 111 | 122 | 120 | **127** |

All figures are Top-1 correct counts against the original frozen references. V2.1 is the official extend5 baseline; V4-B is the original step-584,000 LM best. The completed FP32 per-item baseline scores were reused after verifying their file identities; no baseline checkpoint was reselected.

On Standard development TinySwallow is 72.39% correct, V2.1 69.37% and V4-B 68.41%, gains of 3.02 and 3.98 points. Against the original AzooKey order it fixes 112 items and breaks 38, net +74. The candidate pool covers the correct answer on 630/728 items; no candidates were generated or added.

| Standard source | Items | V2.1 | V4-B | TinySwallow |
| --- | ---: | ---: | ---: | ---: |
| WRIME | 483 | 298 | 297 | **319** |
| JMultiWOZ | 245 | 207 | 201 | **208** |

The net gain over V2.1 comes mostly from WRIME; JMultiWOZ gains one item. Over V4-B, WRIME gains 22 and JMultiWOZ 7. This supports a stronger ranking signal on these natural expressions, but not equal gains in every input domain.

## Paired comparisons and auxiliary scores

Below are "baseline wrong, TinySwallow right / baseline right, TinySwallow wrong" on the same items. p is a two-sided exact McNemar value without multiple-comparison correction, descriptive only, not a confirmatory claim across the four sets.

| Set | vs V2.1 gain/loss | p | vs V4-B gain/loss | p |
| --- | ---: | ---: | ---: | ---: |
| Standard development | 61 / 39 | 0.03520 | 61 / 32 | 0.00346 |
| Legacy regression | 18 / 15 | 0.72833 | 15 / 15 | 1 |
| AJIMEE | 14 / 14 | 1 | 13 / 13 | 1 |
| Historical development | 7 / 2 | 0.17969 | 9 / 2 | 0.06543 |

Equal totals do not mean identical items. On AJIMEE in particular, against V2.1 there are 14 gains and 14 losses, so the teacher's outputs cannot be taken as correct labels.

| Set | Sum with context, primary | Sum, empty context | Mean with context, auxiliary |
| --- | ---: | ---: | ---: |
| Standard development | 527 | 504 | 345 |
| Legacy regression | 377 | 361 | 323 |
| AJIMEE | 144 | 139 | 127 |
| Historical development | 127 | 121 | 100 |

Keeping the left context raises the total on all four sets. Mean is clearly weaker in this run and is not used to reselect the primary scoring rule. Some per-item differences involve notation preferences such as 「つらい／辛い」 and 「なぜ／何故」, so not every change in answer hits can be read as a change in homophone understanding; the existing acceptable answers are unchanged.

## Local resources and outputs

| Item | Recorded in this run |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8GB |
| PyTorch / CUDA | 2.10.0+cu128 |
| Transformers | 4.57.6, separate dependency directory |
| Weight allocation | 2,945.28 MiB, about 2.88 GiB |
| PyTorch peak allocated | 3,144.79 MiB, about 3.07 GiB |
| PyTorch peak reserved | 3,486 MiB, about 3.40 GiB |
| Four sets including model load | 195.83 s, excluding download, dependency setup and result checks |

These are local PyTorch allocator records; they exclude the desktop display and other processes and are neither physical footprint nor iOS latency. Nothing was installed into or upgraded in the training virtual environment; the optional packages are in `artifacts/external-eval/site-packages/`. Nothing was done to the remote V4.1 run.

Entry point `scripts/benchmarks/evaluate_external_lm.py`, optional versions in `requirements/external-eval.txt`, math consistency checks in `tests/test_external_lm_scoring.py`. The official model and tokenizer files are kept in `artifacts/models/external/TinySwallow-1.5B/`; this run's records are in `outputs/evaluation-tinyswallow-20261016/`:

- `model-source.json`, `plan.json`: pinned revision, SHA256 of model files and evaluation inputs, scoring and precision protocol.
- `environment.json`, `dependencies.json`: hardware and runtime dependencies.
- Per set `scores.jsonl`, `metrics.json`, `paired-v21.json`, `paired-v4b.json`: candidate and token scores, summaries, paired changes.
- `summary.json`, `acceptance.json`: completion and per-item checks.

## Interpretation limits

This run read no blind or test data, changed no references, ran no full LM validation BPC, and evaluated neither reading-conditioned nor instruction-style conversion. Overlap between the external model's pretraining/distillation data and these public evaluation texts is unknown; the source limitations of the existing development/regression labels and candidate snapshots still apply.

TinySwallow has about 102 times V4's parameters, and its data, vocabulary, training pipeline and precision also differ, so this run cannot estimate the causal gain of parameter count alone. The result shows research value as a future teacher candidate, especially for the currently weak WRIME. Whether its ranking gain transfers to a 15M model needs a separate training experiment.
