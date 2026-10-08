# Standard Japanese IME benchmark

Version: `ime-standard-ja-v1-ai-expert-r1`; frozen release ID: `d9c60e0e8cb44481ae97c2cea16cfdca`. The benchmark evaluates kana-to-kanji candidate reranking, not unrestricted conversational generation. Labels are AI-expert-reviewed, not native-speaker gold.

## Sources and split roles

| Track | WRIME | JMultiWOZ | Total | Role |
| --- | ---: | ---: | ---: | --- |
| Development | 483 | 245 | 728 | Model/scoring selection |
| Originally blind, now fixed reference | 475 | 250 | 725 | Consumed once in the initial comparison; subsequent registered reference comparisons |
| Historical web regression | — | — | 500 | Exposed old FineWeb development subset; draft labels |

Total size is 1,953. New-source and old-web scores remain separate; source-macro and micro averages are both reported. Development/reference scores are not combined for selection. The historical 1,000-case blind set is a separate artifact and remains unscored.

WRIME writer identities are disjoint across the original development/reference draft. JMultiWOZ user/dialogue groups are disjoint; wizard and topic independence is not guaranteed. Draft sampling caps WRIME at 20 cases per writer and JMultiWOZ at one case per dialogue. JMultiWOZ uses USER turns only. Source separation and finite overlap screening do not prove complete independence from web pretraining.

## Cleaning and label review

Conversion targets are contiguous original spans, 4–40 characters, with 4–64 kana reading characters and at most 64 characters of same-message left context. NFC/control cleanup removes obvious URL/mention/HTML/redaction noise. Target spans containing Latin characters, digits, or emoji are excluded. UniDic/Sudachi agreement supplies draft readings; expert review resolves linguistic acceptability without manufacturing unsupported readings.

All 1,500 new-source draft cases were reviewed before initial candidate/model scoring. The release excludes 47 cases, corrects 50 readings and four homophonic text errors, and contains 538 multi-reference cases. Maximum observed accepted-reference count is 32, below the configured cap of 256. Frozen references do not expand in response to model outputs.

Overlap screening scans approximately 5.42 GB of old/new prepared text using normalized exact containment and 12-character anchors with similarity at least 0.85 for sufficiently long spans. Four matching cases were excluded. Short commonplace phrases, paraphrases, and broader semantic near-duplicates remain limitations. V2 tokenizer reference round trips have no recorded failures; maximum BOS-inclusive reference lengths are 56/63/35 tokens for development/reference/regression.

## Candidate export and scoring

Converter revision: `d59a28e4c7ca049aef04f29a91eae9677a7753f2`; dictionary revision: `4d418525b090cf49c219819d05a7e3cc2a4346eb`; emoji revision: `67b822603391b01238d7b80b8b61b63f966cf357`. Export uses N-best 20, typo mode off, Zenzai disabled, and stable mode off. Reference answers are not added to candidates.

Private candidates reside in `artifacts/benchmarks/ime-standard-ja-v1-candidates/{development,blind,regression}`. The historical directory name `blind` remains for artifact identity; its evaluation role is now exposed reference. Scoring uses the fixed joint-tokenization suffix-sum protocol in [evaluation](../evaluation.md), with whole-case fallback and full-denominator recall misses.

Preparation/export/evaluation entry points reside in `scripts/benchmarks/{download_standard_ime,prepare_standard_ime,standard_ime,evaluate_standard_ime,compare_standard_ime}.py` and `run_standard_ime_mac.sh`. The [Mac exchange template](../../templates/benchmarks/standard-ime-mac.md) specifies candidate export only. Candidate identity, source attribution, labels, and consumption receipts remain with private manifests.

## Consumption and statistical interpretation

The initial V2.1/V3 A comparison was preregistered under plan `8ac20582a3ca4dce9c79eb09342b836c`. The 725-case split was consumed in that comparison. Later models require a frozen selection and a registered reference-test plan before a single fixed reference evaluation; the reference result does not guide checkpoint reselection or label edits. Independent future generalization claims require a new unseen release.

Paired comparisons require identical case IDs, references, readings, context, provenance, candidates, and engine order. Case-level exact McNemar statistics are exploratory; author/user dependence, multiple comparisons, and development selection are not fully corrected. [Initial results](standard-ime-results-20261008.md) and [V3 B results](../reports/v3-20261008/b-evaluation.md) retain source-level outcomes.

## Redistribution

WRIME uses CC-BY-NC-ND 4.0 and JMultiWOZ uses CC-BY-SA 4.0. Text and derived items remain private research materials with source notices; this repository and public model releases contain aggregate results only. AI review does not override source licensing.
