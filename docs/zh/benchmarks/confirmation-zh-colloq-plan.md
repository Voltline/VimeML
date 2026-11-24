# Test confirmation of the colloquial post-training candidate colloq-c40 (preregistered)

> Run on 2026-11-23 (run-02): passed; see the [confirmation report](../reports/zh-colloq-confirmation-20261123/report.md).

Drafted 2026-11-23. This plan counts as registered only once it and its code are committed; until then, the test split of `ime-pinyin-zh-colloq-v1-r1` stays closed. The Chinese version, [docs/zh-CN/zh/benchmarks/confirmation-zh-colloq-plan.md](../../zh-CN/zh/benchmarks/confirmation-zh-colloq-plan.md), is the reference text.

The entry point is [confirm_zh_colloq.py](../../../scripts/zh/benchmarks/confirm_zh_colloq.py), and every fixed setting is in [zh_colloq_confirm.py](../../../src/vimeml/benchmarks/zh_colloq_confirm.py). Search, rescoring, context and Standard scoring are those of the [v2 confirmation plan](confirmation-zh-v2-plan.md).

## Questions

`colloq-c40` was chosen on development (see Deviations in the [colloquial post-training plan](../training/training-zh-colloq-plan.md)). This plan answers two questions on test:

- Is it better than the confirmed `vimeml-zh-v2-20m` (`kd-cont`) at colloquial full-sentence pinyin conversion?
- Does it drop significantly on public pinyin, full pinyin or the Standard rerank?

If it passes, `colloq-c40` replaces `kd-cont` as `vimeml-zh-v2-20m` and goes on to Mac alignment and iOS integration.

## Models

| Role | Arm | Checkpoint SHA256 |
| --- | --- | --- |
| Candidate | `colloq-c40` (30,000 more updates from `kd-cont`, colloquial 208/512, seed 20261122) | `0ec5ffb110fc3959534858fe1a931f4c177878daa8db33bea861c843ca1c1027` |
| Baseline | `kd-cont` (`vimeml-zh-v2-20m`, confirmed 2026-11-20) | `412aee1d1c2d73c9f2ff52533188055e4655575472fe3be136e048fd240268a4` |
| Descriptive | `colloq-c20` (round 1 winner) | `658e6c0e600eec400657a9f61546f48994596a42ddc585c33509d1672d6f0e29` |

Judged development result: the `colloq-c40` self-rescored colloquial run, `rows.jsonl` SHA256 `70a1f5be08aa8fc38e52131ed6f0cb4b8d3f4aca51b14ade473560faf0d2036d`, Top-1 42.3%.

## Test sets

- **`ime-pinyin-zh-colloq-v1-r1` test**: 775 sentences, manifest SHA256 `f94a76df…`.
  - It has not been opened since the freeze; every training round read development only.
  - Contexts come from the freeze-time `test-contexts.jsonl` (SHA256 `8275d961…`).
- **`ime-pinyin-zh-v1-r1` test**: 576 sentences, 3,436 inputs. This is its **second use**.
  - The v2-20m confirmation of 2026-11-20 (run-02) already opened it.
  - Neither `kd-cont`'s result there nor this split played any part in choosing `colloq-c40`.
  - Here it serves only as a guard. It reuses run-02's contexts file and `kd-cont`'s test rows (`rows.jsonl` SHA256 `6f6a31d2…`).
- **`ime-standard-zh-v1-r1` blind full-pinyin rerank**: as in the v2 confirmation, with v1's scores from the 2026-11-11 confirmation run.

## Hypothesis and decision

Pinyin tests use a sentence-clustered sign-flip test with 20,000 permutations, two-sided, α = 0.05. Standard uses exact McNemar.

**Primary hypothesis H1**: on all inputs of the colloquial test, exact Top-1 from `kd-cont` to `colloq-c40` rises, both self-rescored: difference > 0 and p < 0.05.

**Guards** (all must hold):

1. Public pinyin test, all inputs, `kd-cont` → `colloq-c40`: no difference < 0 with p < 0.05.
2. The same on full-pinyin inputs only.
3. Standard blind full-pinyin rerank, v1 → `colloq-c40`: no net < 0 with exact McNemar p < 0.05.

There is one primary hypothesis, so there is no multiple-comparison correction.

**Reading the result**:

- On development, `colloq-c40` is 2.7 points above `kd-cont` on the colloquial set (42.3 vs 39.6) and 1.5 points above on public pinyin (p 0.011). Its Standard net against v1 is 0.
- **Pass**: `colloq-c40` becomes `vimeml-zh-v2-20m`. The release notes report its Top-1 on both test splits.
- **Fail**: `kd-cont` stays `vimeml-zh-v2-20m`. From then on the colloquial test no longer counts as unused, and it is not reopened for another arm.

## Descriptive report (not part of the decision)

- Top-1, Top-3, Top-10 and rescored Top-1 of all three models on the colloquial test, by input kind.
- Paired colloquial test differences: `colloq-c20` vs `kd-cont` and `colloq-c40` vs `colloq-c20`.
- The same table for `colloq-c40` and `kd-cont` on the public pinyin test.

## Execution

The run is local. Test text stays in ignored local directories.

1. Commit this plan and its code. At run time the script checks that both are committed and unmodified, and that every registered file has its SHA256.
2. **Preflight**: regenerate the `colloq-c40` colloquial development run and compare each input's rescored Top-1 hit with the judged rows. The thresholds are those of the v2 confirmation: if more than 0.5% of inputs flip, or Top-1 moves by more than 0.2 points, test stays closed. The preflight can run on its own (`--preflight-only`).
3. Write the consumption record `outputs/benchmarks/zh-colloq-confirmation/consumption.json`. Then generate both test splits and the Standard blind scores, compute the decision and write `summary.json`.
4. Run once. If a technical failure stops the run before `summary.json` is written, it may be rerun under the same plan and the rerun is recorded in the consumption record. It is never rerun to change the result.

Reports give summary numbers and item ids only, never test text or candidates.

## Limitations

- Readings in the colloquial set are automatic, with the three tools in agreement, and were not reviewed by hand. The context is only the part of the same message before the cursor.
- This is the public pinyin test's second use, so its guard is less independent than on a first use.
- `colloq-c40` was chosen after the fact (see Deviations in the colloquial post-training plan). This confirmation is the independent check of that choice.
