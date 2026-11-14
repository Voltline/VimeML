# Standard IME v2 blind

2026-10-21. The v1 blind set was consumed by the V2.1/V3 A comparison, and model selection since then used development only, which left no fresh confirmation set. v2 blind redraws blind-role items with the v1 protocol from the part of the same draft pool that v1 did not select. The frozen version is `ime-standard-ja-v2-blind-ai-expert` (release `5bd7c490…`). **No candidates were exported and no model was run before or after freezing**; scoring must happen once, after a model plan is committed, and is reported under [metrics v2](metrics-v2.md). Entry point: [`prepare_standard_ime_v2.py`](../../scripts/benchmarks/prepare_standard_ime_v2.py).

## Construction

The v1 draft pool holds 4,000 conversion spans from blind-role WRIME authors and JMultiWOZ users, extracted without LM scores or candidate coverage and already screened for overlap with the old training text (see the [Standard IME benchmark](standard-ime.md)). v1 used 750 of them.

| Step | Excluded |
| --- | ---: |
| Documents already used by v1 (same post or dialogue) | 750 |
| Overlap with v1 training text (v1 audit) | 2 |
| Overlap with the V4.1+ chat corpus (482,320 sentences; 3 fully contained, 16 with similarity ≥ 0.85) | 19 |
| Remaining usable | 3,229 |
| More than 20 per author/user, or duplicates | 1,199 |
| Character-trigram Jaccard ≥ 0.5 with an already selected target (JMultiWOZ templates) | 45 |

Items were drawn round-robin over the v1 strata (with/without left context × reading length) with the v1 quotas: WRIME 500 and JMultiWOZ 250, from 62 authors/users. Authors and users are disjoint from v1 development, and no input/left-context pair overlaps v1 development.

## Review and freezing

All 750 items were reviewed one by one by AI without access to model outputs or candidates (`ai_expert`, `native_japanese_review=false`):

| Result | Count |
| --- | ---: |
| Removed | 21 (WRIME 14, JMultiWOZ 7) |
| Reading corrected | 26 |
| Several acceptable notations | 289 |
| Frozen | **729** (WRIME 486, JMultiWOZ 243; 518 with left context) |

Removal rules: the span starts inside a word or name that the left context cuts; the span starts with a counter whose reading depends on the preceding number (人, 月, 泊, 分, 杯 and similar); private personal or shop names whose spelling cannot be determined; joke spellings with no definite conversion. Acceptable notations add only natural spellings of the same word with the same reading, checked against UniDic/Sudachi readings; five earlier variants whose readings actually differed were removed before freezing.

## Limitations

- Labels are AI expert review, not native-speaker gold; acceptable notations are not exhaustive.
- Authors and users come from the same blind role as the consumed v1 blind. Only documents, messages and input/left-context pairs are guaranteed not to be reused; author style is not independent of v1 blind.
- Overlap screening covers the old training text and the chat corpus, not the web shards added for V4.3 (not available locally).
- Independent collection of the sources cannot prove that the text never entered the web training corpus.

Candidate export: [candidate export v2](candidates-v2-handoff.md). Scoring followed the [confirmation plan v2](confirmation-v2-plan.md); the set was used on 2026-10-21, see the [confirmation v2 results](confirmation-v2-results-20261021.md). Item text, review sheets and candidates stay in a local ignored directory and never enter Git.
