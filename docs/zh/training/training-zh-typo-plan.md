# Chinese keyboard typos: data, scoring and evaluation plan

Drafted 2026-11-17. The Chinese version, [docs/zh-CN/zh/training/training-zh-typo-plan.md](../../zh-CN/zh/training/training-zh-typo-plan.md), is the reference text.

Vime shipped sentence-level correction on 2026-11-17:

- it finds the token windows the model scores weakest and makes single-letter edits to the spelling inside them (substitute, insert, delete, swap);
- Rime converts each edited spelling as a whole sentence, and its first candidate is kept;
- the LM score compares original and corrected sentences; a correction is shown only if it wins by at least showMargin = 3, and never in first place.

The client side is close to its limit, so further gains have to come from the model. This plan covers three model-side changes and one public benchmark:

1. **Keyboard typos in training data**: a new pinyin task kind, `typo`.
2. **Scoring against the keys actually typed**: task-mode P(sentence | typed keys) replaces "LM score plus hand-tuned penalties".
3. **Calibration shipped with each export**.
4. **A public typo benchmark**: `ime-typo-zh-v1`, built from public text only.

## Where things stand

- The existing `error` kind covers reading errors only: listed misreadings and fuzzy sounds (see [zh_pinyin_data.py](../../../src/vimeml/posttraining/zh_pinyin_data.py)). The training data has no keyboard typos, so the model has never seen "wrong keys, right sentence".
- `zh_search` requires candidates to match the keys exactly, so generation has no typo tolerance. Every correction candidate comes from the client's edit enumeration.
- Edit penalties and showMargin are hand-tuned values on the log-probability scale, and they need retuning whenever the model changes.

## 1. `typo` data

**Source and split**:

- The paragraphs are the existing stores' pinyin paragraphs (the pinyin share of P1 web/review documents and LCCC); validation uses the same validation paragraphs.
- The random seed uses its own label, `<key>#typo`.
- Output goes to a separate directory, `zh-pinyin-typo-12k`, with its own manifest. The existing five stores stay byte-identical, so the 800M already in training is unaffected.

**Sample format** is the same as the other pinyin kinds: `[2,4]+context+[5]+keys+[6]+output+[3]`. The keys carry the typos; the output is the original sentence.

**Typo rules**:

- **Base keys**: full pinyin 70%, mixed abbreviations 30%. All-initials input gets no typos, because a slip on a single letter is essentially unrecoverable.
- **Typo count**: one typo 85%, two typos 15%. Two typos never fall in the same syllable.
- **Edit types**: letters only, never the `'` separator, in these proportions:

  | Type | Share | Rule |
  | --- | ---: | --- |
  | Adjacent-key substitution | 40% | Replace a letter with a neighbouring key on the 26-key QWERTY layout (left/right on the same row, diagonal on the rows above and below) |
  | Omission | 25% | Delete one letter |
  | Extra letter | 20% | Insert one letter: half are neighbours of the keys on either side, half repeat the previous letter |
  | Swap | 15% | Swap two adjacent letters |

- **Valid spellings**: a typo can still produce a valid pinyin string, for example `shi` typed as `si`. These samples stay in, because that is the case the client handles worst. The manifest counts valid and invalid spellings separately.
- **Where the proportions come from**: the edit-type shares are priors, not estimated from any private data. The private corpus is used only for the final check.
- **Keyboard scope**: 26-key only; the 9-key layout is out of scope.

**Training mix**: `typo` becomes a fifth pinyin task kind under `per_kind` normalization.

- The 20M takes 48 per update, the same order as `error`'s 32.
- The trainer gains support for reading an extra kind from a second directory, through the `[pinyin]` section in `joint_zh_pinyin` / `joint_zh200m`.

## 2. Scoring against the keys typed

During correction, the original and every candidate face **the same typed keys**, so the score can be:

```
taskScore(sentence) = log P(output + EOS | [2,4] + context + [5] + typed keys + [6])
```

- The prefix (context plus keys) is shared by every candidate, which fits the existing sequence package: prefill the prefix once, then score each candidate's output in a batch with `score_continuations`. **No new export graph is needed**; VimeML provides a reference algorithm, PyTorch checks and a fixture.
- The display rule becomes `taskScore(corrected) − taskScore(original) ≥ margin`. A model trained on `typo` data learns how plausible an edit is as a real slip, so the reliability of edit types such as omissions no longer needs hand tuning.
- This works only for three-task models. v1 has no task mode and keeps the LM rule.

## 3. Calibration

Each export ships a `calibration.json` next to `RELEASE.json`, containing:

- the rescoring λ;
- the correction margin;
- per-edit-type penalties, if any are still needed.

The values are fitted on `ime-typo-zh-v1` development: first cap the false-display rate on clean sentences at 10%, then take the setting with the highest typo @2. Vime reads this file when it switches models.

## 4. The `ime-typo-zh-v1` benchmark

**Source**: development sentences of the public pinyin set `ime-pinyin-zh-v1-r1`, full-pinyin inputs only. The test split stays held back and is not read.

**Three versions per sentence**:

- clean;
- one typo;
- two typos (same rules as training, separate seed).

**Candidate pool**: for each typo input, enumerate every single-letter edit of the spelling and take v1 bucketed search's first candidate for each; add the top 10 for the original keys. The pool is built once and frozen in the benchmark. Every model scores the same pool, so the pool does not depend on the model under test.

**Metrics**:

- with typos: corrected Top-1 and @2, separately for one and two typos;
- clean sentences: false-display rate (how often a correction appears) and how often the first candidate is made wrong.

**Tests**: sentence-cluster sign-flip, 20,000 permutations.

Generator and data go in the repo, under the same rules as the public pinyin set.

## Decision rules

All on development.

**20M**: continue from the arm the distillation evaluation picks, in two arms:

- with `typo`;
- without `typo`, as the control.

Both arms share step count, seed and data order.

1. **Primary**: `ime-typo-zh-v1` typo @2. The `typo` arm uses the taskScore rule; the control uses today's LM rule (margin 3). The primary must be significantly higher.
2. **Guard 1**: public pinyin set self-rescored Top-1 must not be significantly below the control.
3. **Guard 2**: clean false-display rate must not exceed the control's, and Standard full-pinyin Top-1 must not be significantly below v1.
4. **Descriptive**:
   - all two-typo metrics;
   - valid-spelling typos reported on their own;
   - the private benchmark (local only).

**800M**: once the 20M passes, add `typo` to the mixture at one of the 800M's resumes, recorded as a deviation of the 800M plan. There is no separate post-training, because pinyin post-training weakened LM ability in r2-cont.

**200M**: v2-200m is final and is not retrained.

## Out of scope

- Typo-tolerant matching inside `zh_search` itself (for example, key-level edit-distance search); candidates still come from the client's enumeration.
- 9-key typos.
- Further client latency work.

## Deviations

Recorded 2026-11-18, as the work started.

- **Insert rule.** An extra letter is a repeat or a neighbour of the letter it follows, not of the keys on either side.
- **Trainer support.** The typo store is read as a `[[data.extra]]` directory (added in round 3) instead of a new `[pinyin]` section, and `distillation.ce_only_kinds = ["typo"]` trains it on cross-entropy only, because the teacher never saw typos. Only `joint_zh_pinyin` (20M) has it; `joint_zh200m` gets it with the 800M resume.
- **Control arm.** The control is the existing r3-ctrl (kd-cont plus 30,000 updates, seed 20261117), evaluated in round 3. The typo arm differs only by the 48 typo samples per update, so every other kind reads the same batches; the update grows from 512 to 560 samples.
- **Benchmark pool.** Corrections are the single edits that undo one typo of the data rules (substitution only by QWERTY neighbours, insertion of any letter), not every single-letter edit; this keeps the typed-correct spelling for every one-typo input in a 40-sentence sample at about 65 spellings per input. Clean inputs get corrections too, since the false-display rate needs them. At about 140 s per input the pool is built in shards.
- **Benchmark text.** The set derives from the pinyin set's development text, which stays out of git (CPED is local use only), so only the generator is committed.
- **Clean Top-1.** A correction is never shown first, so "how often the first candidate is made wrong" is zero by construction and not reported; Top-1 on typo inputs is the pool's base Top-1, the same for every model.

## Typo store (built 2026-11-18)

[data-zh-pinyin-typo.toml](../../../configs/zh/data-zh-pinyin-typo.toml), label `zh-pinyin-v1`: 7,730,142 train samples (242.8M tokens) and 17,009 validation samples. Bases: full 70.0%, mixed 30.0%. Two typos: 15.0%. Edits: substitution 41.0%, deletion 25.7%, insertion 20.5%, swap 12.7% (a swap that does not fit is redrawn). 26.4% of full-pinyin bases stay valid pinyin after the typos. The five existing stores are unchanged.

The arm trains on a rebuild of the same config with sentencepiece 0.2.2 (manifest `3bc0164b…`). Its row structure, lengths and statistics equal the 0.2.1 build above, but 21 of 17,009 validation rows hold other token ids: the two versions break ties in runs of a repeated character differently (`...` + `.` against `.` + `...`, the same for `---`, `??` and doubled key letters such as `xx`). Re-encoding the 0.2.1 rows with 0.2.2 reproduces the rebuild's validation file byte for byte.

## Result (2026-11-19)

**The 20M typo arm does not pass** ([report](../reports/zh-typo-eval-20261119/report.md)).

- Primary: typo @2 with the arm's task rule is 32.2, against 34.3 for the control's LM rule at margin 3 (−2.2, p = 0.0001). Typo training lifts the task rule from 13.0 to 32.2, but not past the LM rule, which gives 34.0–34.3 with v1, r3-ctrl and the arm alike.
- The guards pass: self-rescored Top-1 54.2 vs 54.6, Standard 0.9354 vs v1 0.9361, clean false display 9.7 vs 14.2.

Consequences:

- Vime keeps the LM rule. `calibration.json` ships the rescoring λ and the LM-rule margin calibrated on this benchmark (margin 3 shows corrections on 12.6–14.2% of clean sentences; calibrated 3.5–3.9 keeps it under 10% at −0.3 typo @2).
- No taskScore reference or fixture.
- The 800M does not get `typo` data, and neither does round 4.
