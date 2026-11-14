# Segmented-input benchmark: design and prototype

2026-10-21. Every existing track converts one span per item, and the left context is always the correct original text. In real use people type and commit segment by segment, so each conversion sees **the text they committed earlier**, and an error in one segment flows into the context of the next. This page gives the design of a long-term benchmark and a prototype on exposed development data that checks the protocol. The prototype supports no capability conclusions.

## Design

| Level | Input | Context | Metrics |
| --- | --- | --- | --- |
| 1 Segmented (teacher-forced) | Messages cut into consecutive conversion segments | Original prefix | Segment Top-1 |
| 2 Segmented (free-running) | Same | The model's own earlier segment choices | Segment Top-1, whole-message exact rate, message character error rate |
| 3 Prefix input (long term) | Reading prefix (typing in progress) | Committed text | Predicted-conversion hit rate, edits before commit |

Levels 1 and 2 share one candidate pool and differ only in the context used for reranking, so candidates are exported once. This assumes AzooKey's pool does not depend on the left context when Zenzai is off; otherwise free-running would need candidates re-exported for each model's actual context, at a cost that grows with the number of models. The empty-left-context control in candidate export v2 (the same 728 items with the left context blanked) tests this assumption directly.

Level 3 needs readings cut mid-word plus "acceptable candidates at this point of typing", which the current dictionary-agreement rules cannot label. It waits for real input logs or human labels.

## Prototype `ime-incremental-ja-proto1`

Entry point: [`prepare_incremental_ime.py`](../../scripts/benchmarks/prepare_incremental_ime.py). Source: the 728 messages of Standard v1 development (already exposed). Each continuous Japanese run is cut at word boundaries that are boundaries for both UniDic and Sudachi, not inside compounds, and not starting with a particle, auxiliary verb, suffix or non-independent word. A cut is made at the next allowed position once 8 characters have accumulated; a final segment of fewer than 4 kana merges into the previous one. Both dictionaries must agree on every segment's reading standalone and in context. Digits, Latin letters and emoji count as text committed as is.

| Statistic | Value |
| --- | ---: |
| Messages that segment completely | 587 (141 more have a segment with disagreeing readings and are dropped whole) |
| Segments | 2,000 (1,440 with left context) |
| Segments per message | 1: 98, 2: 149, 3: 133, 4: 80, 5: 44, ≥ 6: 83 |

Labels are dictionary-generated drafts (`dictionary_draft`), not reviewed. The extra acceptable notations added in the development review do not carry over to segments, so segment-level exact match is on the strict side. The prototype is for checking whether the candidate pool depends on context, how large the gap between free-running and teacher-forced is, and whether segment metrics separate models better than single-span metrics. If these turn out useful, a formal version should be redrawn from unexposed sources and reviewed.

## Open items

- Mac export of n-best 20/50 candidates (included in [candidate export v2](candidates-v2-handoff.md)).
- Local scoring script: teacher-forced reuses the existing reranking; free-running appends the model's Top-1 to the left context segment by segment in message order, then reranks.
