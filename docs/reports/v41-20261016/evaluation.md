# V4.1 mixed pretraining evaluation

2026-10-16. V4.1 completed 16 epochs and 865,557 updates normally. Plain-LM reranking peaks at 509/728 on Standard development, four more than V2.1; the old web set and AJIMEE show trade-offs, and the result cannot yet justify replacing the deployed model. Reading-conditioned scoring remains clearly weaker than plain-LM scoring of the same weights.

The key weights, full training state, per-epoch evaluations, configs and the actual source were downloaded; all 178 files passed remote/local SHA256 checks. Later analysis ran locally on an RTX 5060 Laptop and no longer needed the remote instance. The instance was not shut down or released; no extra training, no test/blind reads and no reference changes were made.

## Experiment identity and comparison scope

The model keeps the 8K vocabulary, context 128, d384, seven layers, six heads, SwiGLU 960: 15,070,848 parameters. A seed drawn from the operating system's random source and frozen (1219827974) initializes AdamW from random weights; the full 16-epoch cosine runs from 6e-4 to 6e-5 with 1,000 warmup steps.

Training includes the original FineWeb/Tatoeba, RealPersonaChat/MRMP, and C1r2 conversion examples derived from the original text. Chat is about 5% of actual LM supervised tokens; the two objectives are normalized separately by supervised tokens, with loss weights LM 0.8 and conversion 0.2. Full exposure: 9,683,477,884 LM targets and 157,132,320 conversion targets. Data and config details are in the [training record](../../training-v41.md).

Against V4-B, data, task, seed and budget all changed at once, so differences cannot be attributed to chat, conversion data or epoch count alone. This report trains no new model and changes neither inference scores nor item filtering.

Before looking at the new regression results, the local comparison fixed two plain-LM candidates: epoch 13 `best-ordinary.pt`, selected on Standard development, as primary, and epoch 16 `last.pt` as secondary. The conditional path uses epoch 16 `best-rerank.pt`, whose model tensors equal `last.pt`.

## Full epoch trend

| Epoch | Step | Plain LM BPC ↓ | Chat BPC ↓ | Full conversion NLL ↓ | Plain rerank /728 | Conditional rerank /728 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 324,582 | 3.352653 | 3.306278 | 0.417820 | 502 | 440 |
| 8 | 432,771 | 3.304337 | 3.271272 | 0.369753 | 489 | 429 |
| 10 | 540,966 | 3.246432 | 3.236468 | 0.322366 | 494 | 440 |
| 12 | 649,163 | 3.190037 | 3.213051 | 0.270873 | 504 | 448 |
| 13 | 703,259 | 3.164086 | 3.174325 | 0.236317 | **509** | 459 |
| 14 | 757,360 | 3.141436 | 3.163651 | 0.225616 | 506 | 457 |
| 15 | 811,459 | 3.123016 | **3.153930** | 0.210921 | 507 | 457 |
| 16 | 865,557 | **3.121224** | 3.172254 | **0.207814** | 505 | **461** |

The rerank dip at epochs 7–8 did not turn into an overall decline; the second half brought real learning gains. But plain rerank, chat validation and conditional rerank peak at different epochs; the last epoch lowers plain LM BPC by only about 0.00179 over epoch 15 while chat BPC rises. Continuously falling loss cannot be equated with deployment gains, and this run does not imply that every experiment should train exactly 16 epochs.

On the same original validation text, 11,110,489 characters, uniform FP32:

| Model / weights | Full plain LM BPC ↓ |
| --- | ---: |
| V2.1 extend5 step40000 | 3.080191 |
| V4-B step584000 | **3.027777** |
| V4.1 epoch 13 | 3.164086 |
| V4.1 epoch 16 | 3.121224 |

V4.1's final BPC is about 1.33% higher than V2.1 and about 3.09% higher than V4-B; epoch 13 is about 2.72% higher than V2.1. This is a validation result on the plain web/sentence distribution, not an overall score for all Japanese use. Chat validation was not rerun for the older models, so V4.1's own curve cannot show that it has surpassed them.

## Same-pool IME results

The existing four frozen non-blind snapshots, 1,565 record positions in all, are reported separately and not treated as independent test samples. Standard has been used for per-epoch selection; the old web set is a regression diagnostic; AJIMEE is public and exposed; historical development contains synthetic/AI-assisted labels. These results do not replace the new benchmark being built independently.

| Set | AzooKey | V2.1 plain LM | V4-B plain LM | V4.1 epoch 13 plain LM | V4.1 epoch 16 plain LM | V4.1 epoch 16 CTX/READ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard development /728 | 453 | 505 | 498 | **509** | 505 | 461 |
| Legacy regression /500 | 317 | 374 | **377** | 370 | 376 | 356 |
| AJIMEE /200 | 87 | **144** | **144** | 132 | 137 | 98 |
| Historical development /137 | 111 | 122 | 120 | **124** | **124** | 118 |

All four extra Standard items at epoch 13 come from WRIME: 302/483 versus V2.1's 298; JMultiWOZ is 207/245, the same as V2.1. At epoch 16 the figures are 295 and 210, a trade-off between sources.

At epoch 13, Standard fixes 100 items and breaks 44 against AzooKey, net +56. But V2.1 already reaches 505/728, so any replacement gain must be judged mainly against V2.1, not shown only as a gain over AzooKey.

### Per-item pairing against V2.1

| Candidate | Set | Gain | Loss | Net | Exact McNemar two-sided p |
| --- | --- | ---: | ---: | ---: | ---: |
| Epoch 13 plain LM | Standard | 45 | 41 | +4 | 0.74653 |
| Epoch 13 plain LM | Legacy | 17 | 21 | −4 | 0.62710 |
| Epoch 13 plain LM | AJIMEE | 4 | 16 | −12 | 0.01182 |
| Epoch 13 plain LM | Historical | 4 | 2 | +2 | 0.68750 |
| Epoch 16 plain LM | Standard | 38 | 38 | 0 | 1.00000 |
| Epoch 16 plain LM | Legacy | 18 | 16 | +2 | 0.86417 |
| Epoch 16 plain LM | AJIMEE | 7 | 14 | −7 | 0.18925 |
| Epoch 16 plain LM | Historical | 5 | 3 | +2 | 0.72656 |

p values are exploratory only; repeated epoch selection, multiple sets and item correlation are not corrected for. Epoch 13's Standard advantage is small, and its negative AJIMEE signal must be kept on record rather than reporting only favorable sets. Epoch 16 is more balanced on these regression metrics but shows no evidence of surpassing V2.1 overall.

AJIMEE changes include both notation differences and real homophone errors. For example, epoch 13 fixes 「家系を助ける→家計を助ける」 but also ranks 「短気でわがまま→短期でわがまま」 first. No answers were added, scores changed or items refiltered because of this.

## Reading-path diagnosis

Same epoch 16 weights, same Standard candidates and references:

| Input / scoring path | Correct /728 |
| --- | ---: |
| Plain LM, original left context | **505** |
| Plain LM, no left context | 502 |
| Structured input, CTX kept, READ cleared | 502 |
| Structured input, full CTX/READ | 461 |
| Structured input, CTX cleared, READ kept | 455 |

Full READ against empty READ gains 48 and loses 89, net −41, descriptive paired p=0.000581. The current structured READ path has a negative effect on this development input; its gap to the plain LM cannot simply be read as the model having lost ranking ability overall.

This ablation changes input length, position and content at once, and empty READ itself is about 10% of training views, so it cannot isolate a single mechanism such as reading-label errors, position distribution or the objective, let alone show that reading conditioning is ineffective in general.

Conditional scoring has four structure-length overflows on AJIMEE; per the original protocol these fall back to AzooKey for the whole item, keeping all 200 in the denominator. No other comparison had fallbacks. Four fallbacks cannot explain the overall gap between conditional and plain-LM scoring.

## Prediction and reproduction

On the original fixed twelve prefixes, with the same budget of sixteen new tokens, greedy and temperature 0.8, seed 42, as a qualitative look only: the model produces natural continuations such as 「今日はとても暑いですね。」, but still makes tense conflicts such as 「明日の朝は、お弁当を食べました。」 and repeats after 「この本を読んで」. These examples are not used to rank prediction ability formally.

All local runs use uncompressed primitive FP32 with TF32 off. Plain scoring keeps the full-vocabulary log-probability sum after the jointly tokenized common prefix, no EOS, no truncation, stable ties; conditional scoring sums OUT only, without EOS. Length normalization and fusion weights were not adjusted. Full Standard candidate rankings for epoch 13 plain, epoch 16 plain and conditional all reproduce the training-time records item by item.

Fourteen evaluation configurations were completed, 6,151 repeated configuration-item views in all; these are not 6,151 independent items. Per-item results, pairings, breakdowns and reference identity checks passed, and the fixed SHA256 of the input snapshots was rechecked at the end.

## Outputs and next steps

- Download records: `outputs/training-v41-20261016/remote-inventory.json`, `download-receipt.json`; the actual training source is in its `remote-source/`.
- Evaluation records: `outputs/evaluation-v41-20261016/`, with `plan.json`, `environment.json`, `acceptance.json`, `summary.json`, `comparison-summary.json`, `reading-ablation.json`, `training-curve.json` and per-item scores.
- Repository re-evaluation entry point: `scripts/benchmarks/evaluate_v41.py`, with an explicit separate empty output directory. It was organized from the original local script; the original script snapshot and SHA256 of the completed evaluation stay in the outputs above and are not written back into the historical record.
- Local model: `artifacts/models/tiny-ja-v41-e8k-d384-l7/`. Six role checkpoints plus epoch 14 and 15 weights are kept; epoch 13 is kept as `best-ordinary.pt`, epoch 16 as `last.pt` and other roles. Epoch 1–12 weights were not downloaded; their per-epoch evaluation records are complete.
- Primary plain-rerank candidate: `best-ordinary.pt`, step 703259; secondary: `last.pt`, step 865557. Do not mistake `best.pt` for the best plain LM; it was selected by conditional reranking.
- Training dashboard: [W&B 5uuh8yfb](https://wandb.ai/voltline233/vimeml/runs/5uuh8yfb).

This run shows that mixed training from scratch keeps plain reranking close to the deployed baseline and slightly improves some development metrics. It has not shown an overall replacement gain, and the reading path did not meet expectations. Epochs 13 and 16 should go to an independent benchmark as preregistered candidates, with source, reading length and context gains reported separately; do not pick checkpoints repeatedly on new test results and then claim independent validation. V2.1 stays deployed for now; no training is extended and no new experiment is started automatically.
