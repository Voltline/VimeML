# Expanded IME candidate-set construction — 2026-10-07

The historical expanded set contains 2,000 development and 1,000 blind cases sampled from old corpus validation/test. Labels are dictionary-checked drafts. Actual Mac candidates are imported; the 1,000 blind cases have not received LM scoring.

There are 2,160 contextual and 840 context-free cases, 2,914 FineWeb and 86 Tatoeba cases, and 421 multi-reference cases. Readings, source groups, original sentences, and URLs are unique across the 3,000 cases. Maximum BOS-inclusive reference length is 38 V1 tokens or 37 V2 tokens.

Sampling seed is 20261007. Targets preserve contiguous source text; UniDic/Sudachi agreement determines draft boundaries/readings. Filters exclude observed OOV, high-risk ambiguous readings, noise, and obvious overlap with old 200/137-case inputs. Source groups and URLs are split-disjoint. Selection does not depend on model scores or candidate recall. Dictionary agreement is not linguistic gold review; semantic category balance and comprehensive near-duplicate auditing remain incomplete.

Inputs reside in `artifacts/benchmarks/ime-expanded-v21-final/`; candidates in `artifacts/benchmarks/ime-expanded-v21-candidates-v1/`. Preparation/transfer/import tools are `prepare_expanded_ime.py`, `build_v21_handoff.py`, and `import_expanded_ime.py`. The historical exchange archive includes source notices and maps.

Export fixes converter `d59a28e4c7ca049aef04f29a91eae9677a7753f2`, N-best 20, and typo/Zenzai/stable off. Returned inputs, order, settings, logs, and candidate identity are retained. Recall misses remain in the full denominator. [Initial development evaluation](expanded-ime-evaluation.md) retains draft-label limitations.
