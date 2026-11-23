# zh-800m arch2 evaluation and decision (2026-11-23)

arch2 (k = 4) is compared with arch1 on development, under the "Quality decision" preregistered in the [arch2 plan](../../training-zh-800m-arch2-plan.md). Development only; blind and test were not read. The Chinese version, [docs/zh-CN/reports/zh-800m-arch2-eval-20261123/report.md](../../zh-CN/reports/zh-800m-arch2-eval-20261123/report.md), is the reference text.

**Conclusion: arch2 replaces arch1 as the `vimeml-zh-v2-800m` candidate.** Both conditions hold:

| Condition | arch1 | arch2 | Paired difference (arch1 → arch2) | Decision |
| --- | ---: | ---: | --- | --- |
| Primary: public pinyin set, self-rescored Top-1 (λ = 1.6) | 65.3% | **65.9%** | +0.6 points (+195/−175), sentence-cluster sign-flip p 0.49 | not significantly lower: holds |
| Standard full-pinyin rerank Top-1 | 95.40% (1,389 / 1,456) | 94.99% (1,383 / 1,456) | net −6 (+8/−14), exact McNemar p 0.29, 95% CI [−15, +3] | not significantly lower: holds |

Pinyin generation is level. On prediction (descriptive) arch2 is significantly better: track A Top-3 0.3394 → 0.3604, net +151 items, p 5e-5. By rule 4 of the plan, the round-4 distillation teacher also becomes arch2, the one with the higher primary metric.

## Model and training

- Config `configs/train-zh800m-arch2-k4.toml`: d1792, 23 layers, 28 query / 7 KV heads (GQA), RoPE, QK-norm, 776,524,928 parameters, context 512. 8,192 windows per update (4× arch1), learning rate 3e-4 → 3e-5, warmup 500.
- Same data as arch1: identical `dataset_sha256`, 2,455,478,177 LM examples, 94.33 billion supervised tokens, and identical sample counts for all four pinyin kinds.
- 299,741 updates, 230,916 s (64.1 h) of training. This run used the code before the speed fix. The fix changes only code, not the recipe, and is handled separately.
- The evaluated checkpoint is the preregistered `final.pt`, SHA256 `8af7dab5e134964678de99ac44aa671282683aaac82c0f83100dd10b44f2d58b`. Every evaluation output records it.

## BPC and pinyin validation (20 milestones, FP32)

Milestones are taken at equal token counts. arch2 has lower web, chat and probe BPC than arch1 at all 20 milestones. The gap is widest mid-run (web about −0.17) and narrows by the end:

| Milestone | arch1 web | arch2 web | Diff | arch1 chat | arch2 chat | Diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (5%) | 3.8202 | 3.7139 | −0.106 | 5.4919 | 5.4375 | −0.054 |
| 5 (25%) | 3.6526 | 3.4839 | −0.169 | 5.3929 | 5.2741 | −0.119 |
| 10 (50%) | 3.5417 | 3.3713 | −0.170 | 5.3089 | 5.2038 | −0.105 |
| 15 (75%) | 3.3650 | 3.2592 | −0.106 | 5.2280 | 5.1352 | −0.093 |
| 20 (100%) | 3.2451 | **3.1861** | −0.059 | 5.1726 | **5.1010** | −0.072 |

Final probe (memorization sample) BPC: arch1 3.2374, arch2 3.1809.

Final pinyin validation (teacher-forced exact match / loss):

| Kind | arch1 | arch2 |
| --- | --- | --- |
| Full pinyin | 0.804 / 0.153 | 0.815 / 0.133 |
| Mixed abbreviation | 0.529 / 0.464 | 0.547 / 0.422 |
| All initials | 0.312 / 0.936 | 0.337 / 0.868 |
| Typo | 0.676 / 0.292 | 0.694 / 0.263 |

## Pinyin generation: `ime-pinyin-zh-v1-r1` development (3,444 inputs)

Same settings as arch1: task mode, beam 16 per key position, top 10, rescored by the model itself (λ = 1.6), in-field context.

| Model | Kind | Top-1 | Top-3 | Top-10 | **Rescored Top-1** |
| --- | --- | ---: | ---: | ---: | ---: |
| arch2 | all | 57.3 | 71.9 | 80.7 | **65.9** |
| arch2 | full / mixed / initials | 89.4 / 55.3 / 33.4 | 97.2 / 71.2 / 49.3 | 99.1 / 81.6 / 59.0 | 91.9 / 65.3 / 42.4 |
| arch1 | all | 56.2 | 71.1 | 79.9 | 65.3 |
| arch1 | full / mixed / initials | 88.6 / 54.3 / 31.5 | 97.1 / 70.5 / 47.6 | 99.3 / 80.7 / 57.4 | 91.7 / 65.0 / 40.3 |

Paired, arch1 → arch2 (sentence-cluster sign-flip):

| Kind | Rescored Top-1 diff | +/− | p | Generation-order Top-1 diff | p |
| --- | ---: | --- | ---: | ---: | ---: |
| All | +0.6 | +195/−175 | 0.49 | +1.1 | 0.13 |
| Full pinyin | +0.2 | +13/−12 | 1.0 | +0.9 | 0.46 |
| Mixed abbreviation | +0.3 | +140/−133 | 0.78 | +1.0 | 0.29 |
| All initials | +2.1 | +42/−30 | 0.19 | +1.9 | 0.24 |

Against the 200M (`zh200m-12k-pinyin-pt`, the 2026-11-16 development rows registered in the v2-200m confirmation plan, rescored Top-1 58.8%): arch2 +7.1 points (+390/−144, p 5e-5), arch1 +6.6 points (+386/−160, p 5e-5).

## Standard full-pinyin rerank: `ime-standard-zh-v1-r1` development (1,456 items)

| Model | Top-1 | Against v1 (net, McNemar p, 95% CI) |
| --- | ---: | --- |
| arch2 | 94.99% | +20 (+33/−13), p 0.0045, [7, 33] |
| arch1 | 95.40% | +26 (+37/−11), p 2.2e-4, [13, 39] |

arch2 against arch1: net −6 (+8/−14, p 0.29), not significant.

## Prediction: `ime-predict-zh-v1-r1` development (descriptive)

Decoding as in the arch1 evaluation: beam 16, candidates of 2–6 characters, top 5, context 512, FP32. Pairing is clustered by dialogue (2,400 clusters).

| Track A (7,200 items) | arch1 | arch2 | Net (+/−) | Cluster sign-flip p | 95% CI (items) |
| --- | ---: | ---: | --- | ---: | --- |
| Top-1 | 0.2107 | 0.2224 | +84 (+383/−299) | 0.0023 | [34, 136] |
| **Top-3** | 0.3394 | **0.3604** | +151 (+478/−327) | 5e-5 | [96, 206] |
| Top-5 | 0.3929 | 0.4268 | +244 (+506/−262) | 5e-5 | [190, 298] |

- Track A Top-3 by source: CPED 0.244, CrossWOZ 0.357, KdConv 0.411, RiSAWOZ 0.431 (arch1 0.220, 0.349, 0.377, 0.412).
- Track B keystroke savings 0.238 (arch1 0.221), click rate 0.190 (arch1 0.176).
- Track C first syllable Top-1/3/5: 0.168 / 0.229 / 0.262 (arch1 0.143 / 0.194 / 0.234); initials Top-1/3/5: 0.710 / 0.823 / 0.840 (arch1 0.670 / 0.760 / 0.774).
- Track D Top-1: 20% prefix 0.406 (arch1 0.375), 50% prefix 0.293 (arch1 0.275).

## Typo correction (descriptive)

`ime-typo-zh-v1-r1` development, LM display rule, calibrated with `typo_zh calibration` (λ 1.6, self prior, clean false display limit 10%):

| Model | Margin | typo@2 | Clean false display |
| --- | ---: | ---: | ---: |
| arch2 | 3.7674 | 34.52 | 9.0% |
| arch1 | 3.5048 | 34.08 | 9.0% |

As before, typo correction does not grow with model size.

## Execution record

- The evaluation ran in the training GPU environment from main 1ddbfae, with the same items and settings as the arch1 evaluation; pinyin generation ran in 5 shards and was merged. Every step finished; none failed.
- Every output records checkpoint SHA256 `8af7dab5…d58b`. The readings dictionary, pinyin set manifest, typo set manifest and prediction frozen files have the same SHA256 as in the arch1 evaluation.
- Paired tests and the decision were computed locally. arch1's per-item results are those of its 2026-11-20 evaluation, not rerun.
- Benchmark files and per-item results are not in the repository. This page contains only statistics.

## Next

- Register v2-800m (arch2) in the [v2 confirmation plan](../../benchmarks/confirmation-zh-v2-plan.md): the H1 baseline is the finished v2-200m test run, and the registered development rows are this arch2 run. Test opens only after that amendment is committed.
- After the confirmation passes, export the FP32 inference bundle and publish it in `800m/` of `Voltline/vimeml-zh-v2`.
