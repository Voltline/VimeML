# Public Chinese keyboard-typo set ime-typo-zh-v1

**What it is.** The typo-correction benchmark of the [keyboard typo plan](../training-zh-typo-plan.md), version `ime-typo-zh-v1-r1`, development split only. The Chinese version is [docs/zh-CN/benchmarks/typo-ime-zh.md](../zh-CN/benchmarks/typo-ime-zh.md).

**Code and text.** Code: [typo_zh.py](../../src/vimeml/benchmarks/typo_zh.py), CLI [scripts/benchmarks/typo_zh.py](../../scripts/benchmarks/typo_zh.py), tests [test_benchmark_typo_zh.py](../../tests/test_benchmark_typo_zh.py). The set is built from the development text of [ime-pinyin-zh-v1](pinyin-ime-zh.md) and Vime's readings dictionary, so like them it stays in ignored directories (`outputs/benchmarks/ime-typo-zh-v1/`) and is not committed.

## Items

Every development sentence of `ime-pinyin-zh-v1-r1` (578), full-pinyin input only, in three variants: 1,734 inputs.

| Variant | Input |
| --- | --- |
| `clean` | the sentence's full pinyin |
| `typo1` | one typo |
| `typo2` | two typos in different syllables |

Typos follow the training rules ([zh_typo.py](../../src/vimeml/posttraining/zh_typo.py), weights from [data-zh-pinyin-typo.toml](../../configs/data-zh-pinyin-typo.toml)): QWERTY-neighbour substitution 40, deletion 25, insertion 20 (half a repeated letter), adjacent swap 15. Seeds are `ime-typo-zh-v1/<id>/<count>`, independent of the training data. Each item keeps the context from the pinyin set.

## Frozen candidate pool

Built once by vimeml-tiny-zh-v1 (`tiny-zh13-12k-fresh-s1-d384-l9/final.pt`) with Vime's bucketed search, lm mode, beam 16, in-field context. Every model is scored on the same pool.

- **base**: top 10 for the typed input.
- **corrections**: every single edit that undoes one typo of the data rules and leaves valid full pinyin (delete a letter, insert any letter, swap two adjacent letters, replace a letter by a QWERTY neighbour), each with its search top 1. Texts equal to `base[0]` and duplicates are dropped. Clean inputs get corrections the same way, which is what the false-display rate measures.

A sample of 40 sentences gave about 65 correction spellings per one-typo input, and the typed-correct spelling was among them every time. At about 140 s per input, the build runs in shards (`--shard I/N`, sentence index mod N) that `merge` joins in the pinyin set's order.

## Scoring rules

For each input, `base[0]` and every correction text are scored by:

| Rule | Score | Models |
| --- | --- | --- |
| `lm` | LM score after the context (`zh13_lm_scores.score_item`, the production rule) | all |
| `task` | `log P(output + EOS \| [2,4] + context + [5] + typed keys + [6])` | three-task models |

The context is trimmed from the left, once per input, to fit the longest candidate. Display: with `delta = best correction − base[0]`, the correction is shown second when `delta ≥ margin`, otherwise the list is `base[0], base[1]`. A correction is never shown first.

## Metrics

- **Typo inputs**: @2 (reference among the first two), shown-correct (the shown correction is the reference) and base Top-1, per `typo1`, `typo2` and both.
- **Clean inputs**: false display (a correction is shown) and @2.
- **Margins**: the `lm` rule is reported at the production margin 3. Every rule is also reported at its calibrated margin: the margin with clean false display ≤ 10% that maximizes typo @2, ties to the larger margin. This is the value that goes into an export's `calibration.json`. The `calibration` command writes that file from a model's `lm` rows: format `vimeml_zh_calibration_v1`, the model's checkpoint sha256 and step, search (beam 16, top 10), rescoring (λ and prior, `self` or `v1`) and the correction rule with its calibrated margin, typo @2 and clean false display at that margin and at margin 3 (no margin meets the cap: `enabled` false).
- **Pool coverage**: share of inputs whose reference is `base[0]`, in `base[:2]` or among the corrections.
- **Tests**: paired typo @2, sign-flip over sentence clusters, 20,000 permutations, seed 1118.

The primary metric is @2 on `typo1`. Single edits cannot undo two typos, so `typo2` mostly measures whether the second candidate already holds the sentence and is descriptive.

## Commands

```
python -X utf8 scripts/benchmarks/typo_zh.py build --out OUT/shard-I --shard I/N
python -X utf8 scripts/benchmarks/typo_zh.py merge OUT/shard-0 ... OUT/shard-(N-1)
python -X utf8 scripts/benchmarks/typo_zh.py score --checkpoint CKPT --out DIR [--rules lm task]
python -X utf8 scripts/benchmarks/typo_zh.py report NAME=DIR:RULE ... [--base NAME/fixed|NAME/calibrated] --out FILE
python -X utf8 scripts/benchmarks/typo_zh.py calibration --rows DIR --lambda 1.6 --prior self --out calibration.json
```

The test split of `ime-pinyin-zh-v1` is never read.
