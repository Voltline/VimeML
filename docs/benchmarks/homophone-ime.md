# Written-text homophone challenge

2026-10-21. The [error decomposition](../reports/analysis-20261021/ime-errors.md) showed that semantic errors on the conversational sets are down to a dozen or so, and that the remaining semantic gap sits in encyclopedic written text such as AJIMEE. AJIMEE has only 200 items from a public source. This set tests homophone choice in written text specifically. The frozen version is `ime-homophone-ja-v1-ai-expert` (release `b0dd1161…`) with **619 items**. No model was used for selection, and no candidates were exported or scored before freezing. Entry point: [`prepare_homophone_ime.py`](../../scripts/benchmarks/prepare_homophone_ime.py).

## Source

One shard of the Japanese Wikipedia 2023-11-01 dump (Hugging Face `wikimedia/wikipedia`, pinned revision `b04c8d1c`, `train-00014-of-00015`, the shard with the newest article IDs, 92,631 articles). License CC BY-SA 4.0; every item keeps its article title and URL for attribution. Wikipedia is not a direct training source, but the web corpus may contain reposts.

## Construction

1. **Homophone groups**: UniDic counts of common nouns, verbs and adjectives on the shard. Words with the same reading, part of speech and conjugation type but different lexemes form a group. Each lexeme has at least 2 characters and at least 30 occurrences in the shard; proper nouns, numerals, non-independent verbs and rendaku readings are excluded. Words with the same UniDic lexeme count as notation variants (such as 聞く/聴く), not homophones, consistent with the notation-lenient rule. 1,924 groups.
2. **Spans**: in sentences containing a group word, a 6–20 character conversion span that contains the word and starts and ends on word boundaries; the left context is the preceding text of the same paragraph (at most 200, at least 10 characters). UniDic and Sudachi must agree on the reading both standalone and in context, the same extraction rule as Standard.
3. **Paired design**: 320 groups drawn at random; for each, the two most frequent lexemes get one item each, from different articles. A strategy that always picks the more frequent word can only get about half right.
4. **Screening**: sentences were checked against the old training text (25,185,368 and 1,563,989 lines) for full containment or similarity ≥ 0.85, which removed 14 candidates; 1 was removed for any 12-character overlap with AJIMEE.

| Statistic | Value |
| --- | ---: |
| Candidates passing extraction | 14,260 |
| Draft | 640 (320 groups; 606 nouns, 34 verbs; 417 targets of 6–10 characters, 223 of 11–20) |
| Removed in review | 21 |
| Reading corrected | 8 |
| Several acceptable notations | 5 |
| Frozen | **619**, 320 groups, 299 of them with both items kept |

## Review

AI reviewed every item (`ai_expert`, not native-speaker human review), checking the reading, the integrity of the source text, and whether the left context and span determine the reference. Main removal reasons: the dictionary reads another sense, so the word is no longer a homophone of its counterpart (for example 前世 read ゼンセ in Buddhist contexts, 4 items); personal or royal names whose reading cannot be determined or was wrong; source text missing link words or with broken sentence splits; classical Chinese and prewar katakana legal text; spans starting inside a word. One typo in the source (same reading) was corrected. Acceptable notations were added only where both spellings are in common use.

## Scoring

Scoring followed the [confirmation plan v2](confirmation-v2-plan.md); the set was used on 2026-10-21, see the [confirmation v2 results](confirmation-v2-results-20261021.md). Under [metrics v2](metrics-v2.md): exact Top-1 and notation-lenient Top-1; diagnostics are accuracy when the reference is the more versus the less frequent word of its group, and the number of groups with both items correct. Statistics cluster by homophone group. The candidate pool is n-best 20, with n-best 50 reported alongside.

## Limitations

- One Wikipedia shard only, encyclopedic expository style; the paired design splits items evenly between more and less frequent words, which is not the distribution of real use.
- Labels are AI review, not native-speaker gold.
- Screening does not cover the web shards added for V4.3; Wikipedia text may appear in the web corpus as reposts.

Item text, the homophone group list and review sheets stay in a local ignored directory and never enter Git.
