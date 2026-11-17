# vimeml-zh-v2 release scope

Drafted 2026-11-15. This page fixes which Chinese v2 models are released, which work counts only as v2 research, and the gates each model passes before release. Training rules stay in each model's training plan; blind and public-test confirmation runs get their own preregistered plans, and this page sets only the principles. The Chinese version, [docs/zh-CN/release-zh-v2-plan.md](zh-CN/release-zh-v2-plan.md), is the reference text; this page is its translation.

## What v2 is

v2 is the Chinese model line after vimeml-tiny-zh-v1. Every v2 model can **generate sentences directly from pinyin**: keys go in and the model writes the text, instead of only reranking Rime's candidates. Following the [three-task plan](benchmarks/predict-zh-plan.md), one model does rerank, prediction and pinyin generation. Pinyin generation is required, and every new model has a KV cache.

### Released models

| Release name | Platform | Size | Tokenizer | Candidate | Status |
| --- | --- | ---: | --- | --- | --- |
| `vimeml-zh-v2-20m` | mobile | ~20M | 12K (same as v1) | distilled `kd-cont` with self rescoring ([evaluation](reports/zh-kd-eval-20261117/report.md), 2026-11-17); no [round 3](reports/zh-kd-r3-eval-20261117/report.md) arm replaced it | candidate, not final |
| `vimeml-zh-v2-200m` | desktop | ~200M | 12K | 12K joint pinyin pretraining `zh200m-12k-pinyin-pt`, rescoring with its own scores | passed development evaluation ([report](reports/zh-200m-pinyin-eval-20261116/report.md)); latency and confirmation run pending |
| `vimeml-zh-v2-800m` | desktop | 784M | 12K | [zh-800m training plan](training-zh-800m-plan.md) plus pinyin ability | data in preparation |

- **20M** is the mobile model and keeps v1's deployment: Core ML INT8, batched KV decoding and bucketed beam search.
  - Online, v1 rescores the top 10 generated candidates (see [inference-time rescoring](zh-CN/reports/zh-pinyin-rescore-20261115/report.md)).
  - **One model on the phone** (decided 2026-11-16): r2-cont needs v1 to rescore its candidates, which means loading two 20M models, and that is not acceptable on a phone. v2-20m must rescore with its own scores and beat v1 alone in LM mode. Pinyin post-training weakens r2-cont's LM ability, so it scores worse as its own prior than v1 does. Distillation from the 200M (same 12K vocabulary) may give generation and LM ability in one model, so the choice of v2-20m waits for the distillation line.
- **200M** is released too. If the 12K joint pretraining fails its evaluation, the fallback is the 200M base plus pinyin post-training (as r2-cont). That counts as a deviation and is recorded in the evaluation report.
- **800M** gets its pinyin ability either from post-training or from switching to joint pretraining. The choice follows the 200M evaluation (see the 800M plan's evaluation section).
- All three models use the 12K tokenizer. They share the pinyin data, `zh_search` and the Vime integration, and the larger models can serve as token-level distillation teachers for the 20M.
- The desktop models are not built for phones and get no Mac INT8 tests. Only FP32/FP16 accuracy and desktop latency are measured.

### Not released: v2 research record

| Work | Outcome | Record |
| --- | --- | --- |
| 20M three-task round 1 (`s1-task`/`s1-tok`) and round-2 retrain (`r2-base`) | superseded by, or indistinguishable from, `r2-cont` | [round-2 evaluation](zh-CN/reports/zh-pinyin-r2-20261114/evaluation.md) |
| 20M joint pretraining from scratch (group 3) | generation worse; 20M stays on continued training | [record](zh-CN/reports/zh-pinyin-pt-20261115/record.md) |
| zh-200m six arms (16K/24K/32K vocabulary, context 128/512, three seeds) | no effective contrast; kept as scaling, vocabulary and context findings | [judgement](zh-CN/reports/zh-200m-20261114/judgement.md) |
| Prediction step-0 scaling survey | 800M not deferred | [survey](zh-CN/reports/zh-predict-step0-20261115/survey.md) |

The six-arm checkpoints are not released: they are LM-only and cannot generate from pinyin.

### Still v1

The batched KV Core ML packages (v1 and v2), bucketed beam search and v1's LM generation mode are deployment updates of `vimeml-tiny-zh-v1`. The model weights have not changed, so they keep shipping under the v1 name. v2-20m reuses this runtime.

## Release gates

These are principles. The exact numbers and tests go into each model's confirmation plan, which is committed before its blind is opened.

1. **Pinyin generation is the primary metric.**
   - Measure it on the private benchmark and on the public set `ime-pinyin-zh-v1`.
   - Report full pinyin, mixed abbreviations and all-initials separately, each with rescoring on and off and context on and off.
   - The 20M is compared with the current online path (Rime 16 sentences + v1 rerank, with context) and must not regress significantly on full pinyin.
   - The desktop models are compared with v2-20m.
2. **Rerank guard.** `ime-standard-zh-v1-r1` full-pinyin Top-1 must not be significantly below v1. If a three-task model's own reranking regresses, the online reranker can stay v1, and the release notes say so.
3. **Prediction is descriptive.**
   - Report `ime-predict-zh-v1-r1` tracks A–D.
   - Compare the desktop models with v1 and the 200M default arm.
   - Prediction results do not decide whether a model ships.
4. **Latency.**
   - The 20M is measured on an iPhone (v1's batched KV packages have not been measured there yet).
   - The desktop models get full-search p50/p95 measured on a Mac.
5. **One confirmation run.**
   - Each released model's candidate and scoring rule are fixed before its blind and public test split are opened, and the run is done once.
   - Settings tuned on development (λ, beam width and so on) are frozen at that point and not changed afterwards.

## Data compliance

The data requirements for the mainland China launch apply only to **v2-20m**.

- **export-r1 and export-r2 meet the same standard.** Both batches went through the same rule filters, term lists and spam-site removal (the site list is the union of both batches). Human and AI review only ever covered the review queue (about 3% of documents); most r1 documents were never read by a person either:
  - r1 kept a reviewed document only if every vote was keep (382K documents), and dropped unreviewed ones;
  - r2 has no review and drops every queued document, which is stricter than r1.
  So "r2 had no human review" is not a compliance difference.
- The 20M's current pretraining and pinyin data come from export-r1. Distillation may use r1 + r2, and a teacher trained on r2 is not a problem (decided 2026-11-16).
- Client-side filtering is handled by Vime and is not part of this repository. It is the backstop: rule filters have limited recall, and both batches may miss a little.
- The desktop models (200M, 800M) are not under this requirement.

## Open decisions

- Final `vimeml-zh-v2-20m`: the current candidate is the distilled `kd-cont` (2026-11-17). Round 3 replaces it if better; iPhone latency and confirmation scoring are still to come.
- ~~Whether 200M uses joint pretraining or post-training~~: decided 2026-11-16, joint pretraining with its own rescoring (see the [evaluation report](reports/zh-200m-pinyin-eval-20261116/report.md)).
- The source of the 800M's pinyin ability, decided after the 200M evaluation and before 800M training starts.
- Default beam width, decided when the public-set beam sweep finishes. Beam 16 and beam 12 results are in so far, and 16 is significantly better.
