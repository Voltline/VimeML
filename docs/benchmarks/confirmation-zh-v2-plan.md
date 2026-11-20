# vimeml-zh-v2 test-split confirmation plan (preregistered)

> v2-20m ran on 2026-11-20 (run-02) and passed; results in the [confirmation report](../reports/zh-v2-confirmation-20261120/report.md). v2-200m has not run yet.

Drafted 2026-11-20. The plan counts as registered only once it and the run code are committed; the test split of `ime-pinyin-zh-v1-r1` stays closed until then. It covers two models, `vimeml-zh-v2-20m` and `vimeml-zh-v2-200m`, each run once, 20M first. Entry point [confirm_zh_v2.py](../../scripts/benchmarks/confirm_zh_v2.py); every fixed item is in [zh_v2_confirm.py](../../src/vimeml/benchmarks/zh_v2_confirm.py). The principles of the release gates are in the [v2 release scope](../release-zh-v2-plan.md#release-gates); this plan gives the exact numbers and tests. The registered text is the Chinese version, [docs/zh-CN/benchmarks/confirmation-zh-v2-plan.md](../zh-CN/benchmarks/confirmation-zh-v2-plan.md); this page is its translation.

## Question

Both models were chosen on development: the 20M through four rounds of distillation decisions ([distillation plan](../training-zh-kd-plan.md)), the 200M through the [joint pinyin pretraining evaluation](../reports/zh-200m-pinyin-eval-20261116/report.md). λ = 1.6 and beam 16 were also set on development and the private benchmark. On the unopened public test split this plan asks:

- **v2-20m**: the phone loads only this model, rescoring with its own scores. Is its pinyin-generation Top-1 above the current online path (Rime 16 sentences + v1 rerank, with the in-field context)? Does it regress significantly on full pinyin or on Standard reranking?
- **v2-200m**: the desktop model, rescoring with its own scores. Is its pinyin-generation Top-1 above v2-20m's?

It also gives each model's accuracy estimate on test.

## Models

| Item | v2-20m | v2-200m |
| --- | --- | --- |
| Arm | `kd-cont` (from r2-cont, 30,000 updates distilled from the 200M teacher, α 0.5, T 1, r1 data) | `zh200m-12k-pinyin-pt` (12K joint pinyin pretraining) |
| Checkpoint | `artifacts/models/tiny-zh13-12k-pinyin-kd-cont/final.pt`, step 30,000, SHA256 `412aee1d1c2d73c9f2ff52533188055e4655575472fe3be136e048fd240268a4` | `artifacts/models/tiny-zh200m-12k-pinyin-pt-d1024/final.pt`, SHA256 `90e48fe4cbaeacf5c58774ded243e57d94c86923a51d191555fbbfc7954150f8` |
| Development result it was judged on | self-rescoring run, `rows.jsonl` SHA256 `0003a012b4b750bccc26c8013ada2aec089b15b8a6dacf0d0f92256b2f9b6a46`, Top-1 54.82% | self-rescoring run, `rows.jsonl` SHA256 `9e1be73704483e0b298c2b1a18c03de24ea88de1bdb7b32e48b3e0e32b7a397c`, Top-1 58.77% |
| H1 baseline | the online path: Rime 16 sentences + v1 rerank, with the in-field context | v2-20m's test result (the first run under this plan) |

Shared by both:

- tokenizer `zh-unigram-12288/tokenizer.model`, SHA256 `68bd37c45bcb018b32b2eef5b241561ab5b6dc1b77c1c3aadcc1b41c4bf47e5a`;
- reading dictionary `vime_pinyin.dict.yaml`, SHA256 `3a1654cd5ce20ecacd88046186c34ad87e748a494f7078f23c1ad15492eb0c8a`;
- the frozen `ime-pinyin-zh-v1-r1` manifest, SHA256 `19133b0afaed162034277fe3bbe72b195b46a101462c65697a1a1ec64589248f`.

**Why these models**:

- `kd-cont` became the candidate in the 2026-11-17 [distillation evaluation](../reports/zh-kd-eval-20261117/report.md): public development, self-rescored 54.8%, +2.1 over v1 alone in LM mode (p 0.041). No arm of rounds 3 and 4 ([distillation plan](../training-zh-kd-plan.md)) replaced it.
- The 200M was chosen on 2026-11-16: joint pretraining, rescoring with its own scores.
- Both choices were made before test was opened and did not use test. The other arms do not see test, and test is not used to reselect.

## Hypotheses and decision

Every comparison uses all inputs of the public test split: 576 sentences, 3,436 inputs (576 full pinyin, 2,284 mixed abbreviations, 576 all-initials). The test is a sentence-clustered sign-flip test, 20,000 permutations, two-sided, α = 0.05.

### v2-20m

**Primary hypothesis H1**: all inputs, online path → `vimeml-zh-v2-20m` (self-rescored), exact Top-1.

Passing requires all of:

1. **H1 holds**: difference > 0 and p < 0.05.
2. **No significant full-pinyin drop**: on full-pinyin inputs alone, the same test does not give difference < 0 with p < 0.05.
3. **Standard rerank guard**: `ime-standard-zh-v1-r1` blind full-pinyin rerank, v1 → v2-20m, does not give net < 0 with exact McNemar p < 0.05. The v1 side uses the scores of the 2026-11-11 confirmation run (run-01), not recomputed.

There is only one primary hypothesis, so no multiplicity correction. Conditions 2 and 3 are guards, not further hypotheses. The phone loads one model, so if Standard reranking regresses there is no v1 to fall back to; condition 3 is therefore a hard condition.

### v2-200m

**Primary hypothesis H1**: all inputs, v2-20m's test result → `vimeml-zh-v2-200m` (self-rescored), exact Top-1. Pass: difference > 0 and p < 0.05.

**Standard rerank**: the same test as v2-20m's condition 3, but it does not decide the pass. Under [release gate](../release-zh-v2-plan.md#release-gates) 2 the desktop can load v1 as well: if the 200M drops significantly (net < 0 and p < 0.05), the desktop keeps v1 as its reranker and the release notes say so.

The 200M runs only after v2-20m's run has finished. It has no full-pinyin guard: its baseline is v2-20m, and the full-pinyin guard protects the online path.

### How to read it

- On development (computed through this plan's code path), v2-20m is 22.5 points above the online path (54.8 vs 32.4, +869/−95, p 5e-5), full pinyin 85.8 vs 81.8 (+40/−17, p 0.003); Standard reranking against v1 is −4 items (p 0.63). So v2-20m's H1 and full-pinyin guard are almost certain to pass, and the Standard guard is the only one of the three with real risk.
- On development, the 200M is 3.9 points above v2-20m (+292/−156, p 1e-4) and +23 items over v1 on Standard reranking (p 0.0002). H1 will very likely pass and reranking will very likely not fall back to v1.
- **Pass**: the model is confirmed on test; the release notes report test Top-1 with its confidence interval in place of the development numbers.
- **Fail**: this checkpoint is not released under the name, and test is not reopened for another arm. The analysis goes into the report, and test can no longer be called an unused independent set. If v2-20m fails, the phone keeps v1; if the 200M fails, the plan is discussed separately.
- Passing does not mean ready to ship. iPhone latency (v2-20m), client keyword blocking and the other release gates are verified separately.

## Descriptive reports (no decision)

- Each model's Top-1, Top-3 and Top-10 coverage, by input kind (full pinyin, mixed abbreviations, all-initials), with rescoring on and off and in generation order.
- Strata by sentence length (5–7, 8–12, 13–20 characters) and by source (CrossWOZ, RiSAWOZ, KdConv, CPED).
- The same tables for the baseline.
- Differences between development and test are described only, not tested.
- v1 alone in LM mode is not run on test. The only comparison of v2-20m with v1 alone is development's +2.1 (p 0.041).

## Scoring settings

- Search: bucketed search in [zh_search.py](../../src/vimeml/deployment/zh_search.py), beam 16 per key position, task mode, with the in-field context.
- Rescoring: the top 10 candidates are ordered by genScore + λ × prior, λ = 1.6, with the model's own LM score as prior.
- In-field context: the test split's contexts are rebuilt during the confirmation run by `contexts_pinyin_ime_zh.py`, with the development rules (Vime's `sentenceContext`: the last 64 characters, cut after the last sentence-final punctuation or newline).
- Online path: Rime 16 sentences plus v1 rerank (Vime #70 logic), with the same contexts, produced by the local Rime harness. The harness is not in this repository; it reads the test items and contexts that the confirmation run writes out and writes each input's candidate order.
- FP32, TF32 off. The Standard guard scores as in the v1 confirmation plan: `BOS + left context + candidate` tokenized jointly, full-vocabulary log-probability sum, no EOS, ties keep the pool's original order.

## Running

Run locally; test text stays in ignored local directories.

1. Commit this plan, and register both models' checkpoints, judged development results and their SHA256s in `zh_v2_confirm.MODELS`. At run time the script checks that the plan and the code involved are committed and unmodified and that every registered file matches its SHA256.
2. **Preflight**: regenerate development with the registered settings and compare each input's rescored Top-1 hit with the judged result. If more than 0.5% of inputs flip, or Top-1 moves by more than 0.2 points, test is not opened. The preflight can run on its own (`--preflight-only`) without opening test.
3. Write a consumption record to `outputs/benchmarks/zh-v2-confirmation/consumption.json`, then rebuild the test contexts and generate test.
4. v2-20m: after the local Rime harness has produced the online path's test results, run `finish`, which computes H1, the full-pinyin guard and the Standard guard and writes `summary.json`. v2-200m uses v2-20m's finished test result as its baseline and computes H1 and Standard reranking.
5. Each model runs once. If a technical fault interrupts a run before `summary.json` is written, it can be rerun under the same plan and recorded in the consumption record; never rerun to change the result. A finished model is never run again.
6. Any change after the first run is reported as a deviation, and the original outputs are kept.

The report publishes only summary numbers and item ids, never test item text or candidates.

## Limits

- Test readings are automatic readings on which three tools agree, not human-reviewed; the keys are synthetic, not real typing.
- All sources are dialogue corpora, which differ from real keyboard input.
- Standard blind was used for the v1 confirmation on 2026-11-11; here it is a fixed reference test set, not a new blind.
- The private pinyin benchmark was used to set λ; it is descriptive only and does not enter the decision.
