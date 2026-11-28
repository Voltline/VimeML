# vimeml-ja-v6 release scope

> [Chinese version](../zh-CN/ja/release-ja-v6-plan.md) (reference text)

Drafted 2026-11-28. This page sets which models the Japanese v6 series releases, which work is only a research record, and the gates each model passes before release. It follows the [vimeml-zh-v2 release scope](../zh/release-zh-v2-plan.md), without an 800M. Training rules stay with the [ja-gen plan](training/training-ja-gen-plan.md); test confirmations get their own preregistered plans, and this page sets only principles.

## Naming

The Japanese line has used V1–V4.x (V2.1 and V4.2 were deployed in turn) and V5 (scale-up pretraining, not deployed; see the [V5 record](training/training-v5.md)). The ja-gen batch is named **vimeml-ja-v6**, continuing that numbering. `vimeml-ja-v2-20m` in earlier ja-gen documents is `vimeml-ja-v6-20m`.

## What v6 contains

Every v6 model **generates text directly from a kana reading**: type kana and the model outputs the conversion, instead of only reordering AzooKey's candidates. The same model generates candidates and scores AzooKey and generated candidates alike (`gen_score + 1.6 × lm_score`). All models use the 12K tokenizer `ja-unigram-12288` and have a KV cache.

### Released models

| Release | Target | Parameters | Source | Status |
| --- | --- | ---: | --- | --- |
| `vimeml-ja-v6-20m` | phone | 20,036,736 | `colloq-c40`: `kd-cont` (distilled from the 200M) plus colloquial post-training ([3.4 report](reports/ja-gen-stage3-seeds-20261128/report.md)), checkpoint `19b04ddf…` | Test confirmation passed 2026-11-28: kana test lenient Top-1 88.00% against 85.00% for the V4.2 deployment ([report](reports/ja-gen-confirmation-20261128/report.md)); Mac / Vime handoff 2026-11-28 ([brief](mac/mac-ja-v6-20m.md)); release after the Mac and device results |
| `vimeml-ja-v6-200m` | desktop | 193,625,088 | 2.2 joint reading pretraining, reranked by its own scores, checkpoint `47721ddd…` | Development evaluation: kana lenient Top-1 89.33, colloquial 89.67, Standard reranking 517 ([report](reports/ja-gen-stage2-200m-20261128/report.md)); release after a test confirmation and a desktop latency test |

- **20M** is the phone model and replaces V4.2: Core ML INT8, batched KV v4, beam search bucketed by reading position. The phone loads one model.
- **200M** is the desktop model: no phone constraints, no Mac INT8 test, only FP32/FP16 accuracy and desktop latency with PyTorch on Windows, as for the Chinese v2 desktop models.
- **No 800M.** The Chinese-Japanese-English 800M is a separate experiment, stage 6 of the ja-gen plan, and not part of v6.

### Not released; research record only

| Work | Outcome | Record |
| --- | --- | --- |
| 2.1 20M LM and reading post-training arms (4 seeds each) | The reading arm became the pre-distillation baseline | [2.1 report](reports/ja-gen-stage2-20m-20261127/report.md) |
| 2.3 distilled 20M `kd-cont` and its seed replicas | Superseded by `colloq-c40`; the stage 4 comparison | [2.3 report](reports/ja-gen-stage2-kd-20261128/report.md) |
| 3.3 `colloq-ctrl`, `colloq-c40-p85`; 3.4 seed replicas | Used for selection and decisions, not released | [3.3 report](reports/ja-gen-stage3-colloq-20261128/report.md), [3.4 report](reports/ja-gen-stage3-seeds-20261128/report.md) |

## Release gates

Principles below. Numbers and tests go into each model's confirmation plan, committed before test is opened.

1. **Kana conversion is the primary metric.** Lenient Top-1 on `ime-kana-ja-v1` test, with the final list AzooKey N-best 20 plus the generated top 10, ordered by the model's own scores. Exact Top-1 and the colloquial `ime-kana-ja-colloq-v1` test are reported alongside.
2. **Baselines.** 20M against the deployed V4.2 setup (AzooKey N-best 20 + V4.2 reranking). 200M against `vimeml-ja-v6-20m`.
3. **Reranking guard.** Standard development reranking at or above 501, the low end of the V4.2 recipe seed band (the ja-gen stage 2 guard).
4. **Latency.** 20M measured per keystroke on iPhone (see the handoff brief); 200M full-flow p50/p95 with PyTorch on Windows.
5. **Confirm once.** Settings fixed on development (λ 1.6, beam 16, top 10, AzooKey top 20) stay frozen. Both test splits were used once by the 20M confirmation; the 200M confirmation also runs once on them, as the Chinese v2-200m was confirmed on the same public test after the 20M.

## How it is released

- Hugging Face repository `Voltline/vimeml-ja-v6` with `20m/` and `200m/` directories, laid out as `Voltline/vimeml-zh-v2`: inference bundle, tokenizer, reading index and model card.
- `NOTICE.md` lists training data sources and licenses. WRIME (CC BY-NC-ND 4.0) and JMultiWOZ (CC BY-SA 4.0) are used only for evaluation; their items are not shipped.
- 20M is released after the Mac and device results pass; 200M after its test confirmation and desktop latency test.

## Open items

- `vimeml-ja-v6-200m` test confirmation plan: draft, commit, then run once.
- Mac and Vime results for `vimeml-ja-v6-20m`.
