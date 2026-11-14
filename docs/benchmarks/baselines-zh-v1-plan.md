# vimeml-tiny-zh-v1 baseline comparison plan (preregistered)

Drafted 2026-11-11. Benchmark version `ime-standard-zh-v1-r1`. Run on blind only after this plan and the run script were committed. Entry point [compare_zh_baselines.py](../../scripts/benchmarks/compare_zh_baselines.py), implementation in [zh_baselines.py](../../src/vimeml/benchmarks/zh_baselines.py), Rime configurations in `VARIANTS` of [rime_candidates.py](../../scripts/benchmarks/rime_candidates.py). The registered text is the Chinese version, [docs/zh-CN/benchmarks/baselines-zh-v1-plan.md](../zh-CN/benchmarks/baselines-zh-v1-plan.md), which the scoring script checks via `--blind-plan`; this page is its translation. Results: [blind confirmation and baselines](../reports/zh-v1-20261111/blind-and-baselines.md).

## Status of blind

Blind was used once in the [confirmation](confirmation-zh-v1-plan.md) (run-01, passed). Since then it is only a fixed reference test set, no longer an unused independent set. This plan's blind results only describe where v1 stands relative to other systems; they are **not** used to select, tune or change v1, and do not change the confirmation's conclusion. The report publishes only summary numbers.

## Questions

References for v1's score:

1. How much does a stronger Rime configuration (extra dictionaries and a grammar model) raise Rime's own first candidate, and does v1 still lead?
2. How good is Rime's default scheme (Luna Pinyin)?
3. How well do 0.5B-class general Chinese LMs rerank the same pool with the same method, and how far is v1 (about 20M parameters) from them?

## Comparison systems

| Name | Type | Contents |
| --- | --- | --- |
| `rime-ice` | Engine | librime 1.17.0 + rime-ice `99662ba`, configured as for the frozen pool (user dictionary off, Lua translators removed, emoji and Traditional off) |
| `rime-ice-gram` | Engine | The above + the Wanxiang grammar model `wanxiang-lts-zh-hans.gram` (418,071,596 bytes, SHA256 `3457b69f5fe0fc4f5d211c878bbfa2e2361b90639952c9845c8e31e9d047cde6`), parameters copied from rime-ice's `others/recipes/grammar.recipe.yaml`: `collocation_max_length` 6, `collocation_min_length` 3, `collocation_penalty` −14, `non_collocation_penalty` −6, `weak_collocation_penalty` −100, `rear_penalty` −20, `contextual_suggestions` false, `max_homophones` 8 |
| `rime-luna` | Engine | librime 1.17.0 + `luna_pinyin_simp` (rime-luna-pinyin `56b934b`, rime-prelude `082425e`, rime-essay `054920d`, rime-stroke `1e8fff9`, OpenCC data bundled with librime), only the user dictionary off |
| `qwen2.5-0.5b` | External LM | `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| `qwen3-0.6b-base` | External LM | `Qwen/Qwen3-0.6B-Base` revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| `vimeml-tiny-zh-v1` | Model under comparison | Development uses the `12k-fresh-s1` results of the 2026-11-11 decision; blind uses the confirmation run-01 results; no rescoring |

The Qwen base versions are used, not instruction-tuned ones, because the task scores continuations. transformers 4.57.6.

## Scoring

**Engines**: each configuration runs on its own and takes the first 20 candidates for the whole input that are all Han characters and as long as the target, in engine order, without lattice padding and without left context (Rime has none). This is the frozen pool's construction rule minus the padding step. So `rime-ice` on full pinyin is identical to the pool's original order (development 1,001). On abbreviated pinyin there are 3 items where Rime gives no candidate and the answer appears only in the lattice padding, so the engine scored alone gets 139 against 142 for the pool's original order; the Rime baseline in the decision and the confirmation still means the pool's original order.

Engine ranking can use candidates outside the pool, so comparisons among engines, and between engines and v1, compare "each system's first candidate", not reranking of one shared pool.

**External LMs**: rerank the frozen pool with the same method as v1: `<|endoftext|>` as start token, `start + left context + candidate` tokenized jointly, scored from the first token where any candidate differs from the left context alone, FP32 full-vocabulary log-probabilities summed, no EOS, TF32 off, ties keep pool order. Roundtrip must hold. Items where it does not, or that error, fall back to pool order and are counted in `fallbacks`. Empty-pool items stay in the denominator. So differences between an LM and v1 come from the language model alone.

## Comparisons and decision

The main comparison is full-pinyin exact Top-1: v1 paired with each of the five systems, each with an exact two-sided McNemar test; the five p values are Holm-corrected, with bootstrap net intervals also reported. The abbreviated track, Top-3, per source, length strata, pool-conditional Top-1 and MRR are descriptive.

Development is an open set and can be rerun and inspected at any time. On blind, each system runs once. A run interrupted by a technical fault can be rerun under the same plan; never rerun to change the result. Results are reported as they come in either direction: if a system beats v1, the report says so, without reselecting systems or changing the scoring.

## Running

1. Deploy the three Rime configurations: `rime_candidates.py deploy --variant {ice,ice-gram,luna}`.
2. Development: `compare_zh_baselines.py engine|lm <system> OUT --track {full,abbrev}`, then pair with v1 using `score_standard_ime_zh.py compare`.
3. Commit this plan and the script.
4. Blind: the same commands with `--split blind --blind-plan docs/zh-CN/benchmarks/baselines-zh-v1-plan.md`. The scoring script checks that the plan is committed, unmodified and names `ime-standard-zh-v1-r1`. Pair with the v1 results of confirmation run-01.
5. Outputs go to `outputs/benchmarks/zh-v1-baselines/{development,blind}/`.

## Limits

- The engines see only pinyin, not left context; v1 and the external LMs see left context. This is a real difference between systems, not a bias to remove. The empty-left-context control in the [leakage check](../reports/zh-v1-20261110/ime-benchmark.md#leakage-check) (development 81.9%) serves as a reference for "v1 without left context".
- External LMs only rerank the frozen pool, so the pool's ceiling (about 97.5% on full pinyin, about 41% on abbreviated) limits them too.
- Whether Qwen's training data includes the public CrossWOZ, RiSAWOZ and KdConv datasets is unknown, so leakage cannot be ruled out. A high Qwen score should be read with that in mind.
- The Wanxiang grammar model uses rime-ice's recommended parameters, untuned.
- Only accuracy is compared, not speed; whether a 0.5B model can run on device is outside this plan.
