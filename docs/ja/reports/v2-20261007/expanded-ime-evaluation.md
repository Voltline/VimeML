# Expanded draft IME evaluation — 2026-10-07

V1 and V2.0 best receive FP32 scoring on the same 2,000 development cases and actual AzooKey candidates. The 1,000-case blind export is not LM-scored. Reference labels remain drafts.

| Ranking | Top-1 / 2,000 | Top-5 | MRR | Engine corrections / regressions |
| --- | ---: | ---: | ---: | ---: |
| Engine | 1,224 | 1,711 | 0.71462 | — |
| V1 | 1,453 | 1,745 | 0.79248 | 306 / 77 |
| V2.0 | 1,463 | 1,750 | 0.79542 | 323 / 84 |

V2 relative to V1 corrects 92 and regresses 82 cases, net +10 (0.50 percentage points), exact paired p=0.49517. Candidate reference coverage is 1,756/2,000; covered Top-1 is 82.74% / 83.31%. There are no empty pools or recorded length/round-trip fallbacks.

Contextual cases score 1,066/1,439 versus 1,076/1,439; context-free cases score 387/561 for both. Removing context from the same full set gives 1,420/1,425. Token-mean diagnostics give 1,057/1,082. These descriptive subgroup differences are not causal evidence. Tool elapsed time includes loading/scoring and is not iOS latency.

## Post-hoc label sensitivity

A UniDic-based diagnostic proposes 2,934 aliases for 1,393 cases under matching lemma, reading, POS, and inflection rules. It raises engine/V1/V2 Top-1 to 1,532/1,755/1,769; V2 relative to V1 has 43 corrections and 29 regressions, p=0.12492. This analysis follows score inspection, can introduce false equivalences, and does not modify original labels or determine model selection.

Original scores reside in `outputs/ime-eval/expanded-v21-dev-draft-{v1,v2}/`, comparisons and label-sensitivity records in adjacent versioned directories. Candidate export identity and full denominators remain fixed. [Label audit](label-error-audit.md) and [restart evaluation](v21-evaluation.md) preserve subsequent diagnostics.
