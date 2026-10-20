# Native-speaker Chinese IME re-evaluation — 2026-10-17

All 100 blinded items were returned and confirmed by the user, who identified themself as a native Chinese speaker. The returned inputs match the frozen packet exactly. On the 35 scorable new diagnostics, the fixed Rime/Octagram n-best baseline matches a listed acceptable answer on 22 cases; gated 4M also matches 22, and gated 25M matches 23. These results do not support extrapolating the earlier small sentence-set scores to general Chinese input.

The 60-case legacy audit remains easier and scores higher after human relabeling. The two source groups are reported separately. No model, threshold, dictionary, source gate or decoding setting was selected or revised using the returned annotations.

## Frozen annotation and prediction identity

- Dataset: `zh-human-100-20261017-4c0543baa2d76573`.
- Returned export SHA256: `c5531d74f54d6b65cb3047755591f748f72e422e278614b0ca1fabf129151643`.
- Export time: 2026-10-17 18:44:02 Asia/Shanghai.
- Legacy labels: 40 single-answer and 20 multiple-answer cases, all scorable.
- New labels: 22 single-answer, 13 multiple-answer, 4 insufficient-information and 1 invalid-input case. The invalid case includes a guessed answer, but its explicit invalid decision is respected: it is not silently put back into the denominator.

The original export was copied byte-for-byte to the ignored `artifacts/chinese-human-eval-20261017-184402/` directory. Inputs, source mapping, cached-file hashes and the evaluation policy were frozen before new inference. The original download, prior labels, candidate pools, scores and reports were not modified.

All 60 legacy predictions in each lane were reused from saved pre-label scores after checking input, context, model and cache identity. Only the 40 new diagnostic inputs were decoded and scored. Their model-facing records contain ID, input and context, followed by engine-generated candidates; they contain no human answer or authored construction seed. The same fixed 4M and 25M weights and reference executable were used.

Two unchanged lanes were evaluated: ordinary one-sentence Rime/Octagram and Rime/Octagram with `max_sentences=9`, `sentence_cutoff_threshold=1.0`. Candidate generation still uses `rime-pinyin-simp` and the old published simplified-Chinese Octagram model. This is not a test of all Rime configurations or modern Wanxiang dictionaries. User learning and completion remain disabled. The prior dictionary-membership proxy and actual selection-based full-key coverage flags are retained.

## Metric definition

The main metric is exact membership of a complete-key output in the reviewer's **listed acceptable answer set**. The first listed answer, which the review page described as the reviewer's preference, is scored separately. Only leading/trailing whitespace is trimmed. Pronouns, punctuation, spelling, character variants and pinyin are not automatically normalized.

Multiple-answer lists are not guaranteed to enumerate every acceptable output. A string outside the list is recorded as unlisted, not automatically certified as a semantic error. Some annotations explicitly accept common nonstandard spellings or fuzzy interpretations; these choices are preserved. The metric is individual listed-answer agreement, not a strict normative spelling or raw-pinyin-consistency score, and not recovery of a known real user's intended text.

Insufficient-information and invalid cases are disclosed and excluded from scorable denominators, rather than counted as correct or forced into an invented gold label. No pooled 95-case score is used to hide the difference between the old reference set and new diagnostics.

## Main results: same n-best candidate configuration

| Source group | Scorable / returned | Rime + Octagram | + 4M, fixed gate | + 25M, fixed gate | Candidate ceiling |
|---|---:|---:|---:|---:|---:|
| Legacy audit | 60/60 | 48/60 (80.0%) | 53/60 (88.3%) | 56/60 (93.3%) | 58/60 |
| New diagnostics | 35/40 | 22/35 (62.9%) | 22/35 (62.9%) | 23/35 (65.7%) | 28/35 |

First-listed-answer agreement is 47 → 52 → 55 on the legacy 60, and 21 → 21 → 22 on the new scorable 35. These are preferences on the supplied artificial tasks, not observed intended-text recovery in actual use.

Relative to the engine, legacy 4M rescues 8 listed-answer misses and introduces 3; 25M rescues 9 and introduces 1. On the new diagnostics, 4M rescues 1 and introduces 1, yielding no net improvement; 25M rescues 1 and introduces 0. One additional match out of 35 is not enough evidence of a stable model advantage.

The ordinary one-sentence lane scores 48/60 for all three systems on the legacy audit, and 22/35 for all three on the new diagnostics. Its candidate ceilings are respectively 48 and 25. Exposing n-best candidates raises new-case recall to 28, but this does not automatically improve the selected output.

## New diagnostics by input type

| Diagnostic group | Scorable / returned | Engine | + 4M | + 25M | Candidate ceiling |
|---|---:|---:|---:|---:|---:|
| Context-free ambiguous words/short phrases | 10/10 | 10 | 10 | 10 | 10 |
| Context-conditioned homophones | 10/10 | 7 | 7 | 7 | 10 |
| Conversational full-pinyin sentences | 5/5 | 2 | 1 | 2 | 4 |
| First-letter abbreviations | 2/5 | 0 | 1 | 1 | 1 |
| Mixed abbreviations | 3/5 | 1 | 1 | 1 | 1 |
| Authored key errors | 5/5 | 2 | 2 | 2 | 2 |

The ten ambiguous short inputs have generous, explicitly human-approved answer sets and all three fixed systems hit one. They should not be mistaken for sentence conversion success. Across the other 25 scorable new diagnostics the counts are 12, 12 and 13. Across the 15 scorable new sentence/abbreviation/key-error items, excluding the two homophone-word groups, the counts are 5, 5 and 6. These subgroups are tiny, diagnostic slices, not population estimates.

Seven of the new 35 have no listed acceptable answer in the full-key top-nine pool, putting them beyond any pure reranker. Of the 28 reachable cases, gated 25M still misses 5. Candidate generation and ranking policy both contribute to the remaining gap.

The fixed dictionary gate retains the engine's choices on the context-word group. Ungated diagnostic scoring hits 8/10 for 4M and 10/10 for 25M there, but loses one listed-answer match on the context-free ambiguity group. This is a post-label policy diagnostic, not a selected replacement policy or new deployment result. No gate was changed in the main table.

## What the human audit changed

Twenty of the old 60 have multiple accepted answers, extending beyond the previously flagged gender cases to names, places and other homophones. Three original single-reference labels are not present in the native answer sets; the raw inputs support details that the original labels omitted or changed. Those original files remain unchanged, and their disagreements are retained locally for audit.

Under the native sets, the old n-best 4M/25M counts increase from the old-reference 49/52 to 53/56. That does not validate the old set as representative. Even the 40 native-labeled single-answer legacy cases remain easy for these systems: engine/4M/25M counts are 33/34/37. The new conversational, abbreviation and typo diagnostics expose substantially different behavior.

Some listed-answer misses are plausible alternate wordings and need final human adjudication, while others are visibly poor conversions. A local per-case audit shows exactly what was output; this report does not silently approve additional strings or revise the user's decisions.

## PinyinGPT: cached outputs only

The existing segmented-input outputs were rejudged against native legacy labels without new model inference. Full-pinyin accepted top-1/top-10 is 42/60 and 43/60; abbreviated input is 15/60 and 30/60. First-listed top-1 counts are 41 and 14. These runs still use reference-derived syllables, with seven raw-reading mismatches, and cannot be compared directly with the raw-keystroke Rime lanes. No new PinyinGPT task was constructed from the returned human answers.

## Validation and local artifacts

Four targeted scoring tests pass with `PYTHONPATH=src`: alternative answers versus first preference, excluded decisions even when an answer is supplied, complete-key coverage and rescue/break bookkeeping, and absence of implicit punctuation/pronoun normalization. An independent row-wise count agrees with both lane summaries. Input/cache identity, the 60 cached + 40 new split, label-free model inputs, and the prior configuration digests were checked. Existing model inference parity evidence remains in the earlier experiment; no training or client runtime was changed.

Entry point: [`evaluate_chinese_human_review.py`](../../scripts/benchmarks/evaluate_chinese_human_review.py). Implementation: [`chinese_review_eval.py`](../../src/vimeml/benchmarks/chinese_review_eval.py). Tests: [`test_chinese_review_eval.py`](../../tests/test_chinese_review_eval.py).

```powershell
.\.venv\Scripts\python.exe scripts/benchmarks/evaluate_chinese_human_review.py all --annotations <native-export.json>
```

A revised annotation export must use a new `--artifacts` and `--output` directory. The current frozen revision is protected against replacement. Derived output files are in `outputs/chinese-human-eval-20261017-184402/`: aggregate `summary.json`, joined per-case records, new candidate/score dumps, legacy-reference disagreements, cached PinyinGPT rejudgment, independent verification/fingerprints, and a Chinese per-case audit. Labels and individual predictions stay outside Git.

The next research work should improve candidate recall and correction coverage, then validate context-aware gating on a separate human-labeled set. These diagnostic results support continuing investigation; they do not support treating the old 60-case score as overall Chinese IME accuracy.
