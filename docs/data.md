# Data and provenance

## Frozen V1/V2 corpus

The sentence corpus combines nine FineWeb2-Edu Japanese `small_tokens_cleaned` Parquet shards and Tatoeba Japanese sentences. The selected web shards are `train-00010`, `00030`, `00050`, `00070`, `00090`, `00110`, `00130`, `00150`, and `00170` of 283, totaling approximately 2.43 GB compressed.

Deterministic cleaning processes 2,289,346 source records into 25,713,003 unique sentences. Unicode/basic-noise cleanup, conservative line joining, sentence segmentation, quarantine, document grouping, and exact deduplication precede splitting. Approved decisions reside in `annotations/approved-v1.jsonl`; 638 quarantined blocks remain excluded.

| Split | Sentences |
| --- | ---: |
| Train | 25,185,368 |
| Validation | 260,337 |
| Test | 267,298 |

The target ratio is 98/1/1 with seed 42. Source documents and exact duplicate text share a split. Semantic near-duplicate and public-benchmark overlap audits are incomplete. Text cleaning does not establish factual quality, unbiased content, or absence of personal information.

The frozen exports reside in `outputs/corpus-fast-v1/`. The original merge manifest retains `stage=merged_corpus_staging` and `ready_for_lm_training=false`; later review/training decisions are separate records rather than edits to the original fingerprint. [Pipeline methods](reference/data-pipeline.md) describe reconstruction.

## V3 sources and selected pool

V3 adds [JpnMix](https://huggingface.co/datasets/AdaMLLab/JpnMix) `minhash_deduped`, [RealPersonaChat](https://github.com/nu-dialogue/real-persona-chat), and the [multi-relational multi-party chat corpus](https://github.com/nu-dialogue/multi-relational-multi-party-chat-corpus).

Sixteen JpnMix shards were downloaded, but the experiments use only the current eight-shard cleaned sample. Those eight shards contain 971,513 source documents, approximately 2.038 GB compressed. The selected web pool contains 349,384 documents and 499,999,997 prediction pairs before the subsequent pricing/catalogue exclusions. Observed upstream proportions are C4 28.39%, CulturaX 22.60%, FineWeb2 22.97%, and HPLT2 26.05%; all five nominal JpnMix sources are not represented in this sample.

The dialogue sources contain 14,543 conversations and 509,299 turns in total. Chat validation/test are held out by whole conversation, not individual utterance. Speaker independence is not established. The raw dialogue content contains 5,222,806 tokens excluding BOS/EOS.

V3 A uses the selected new pool plus chat. V3 B adds the old V2 training corpus. Both use frozen virtual mappings and a 5% chat share measured in effective post-crop tokens. This share involves repeated dialogue exposure, not equivalent amounts of new unique chat text. A 49,205-block pricing/catalogue exclusion affects training only; broad validation/test distributions remain intact. V3 B additionally excludes 1,062 old and 16,854 new overlapping sequences and 67 exact cross-pool web duplicates.

Overlap screening uses exact identity and anchor/containment rules. It is not a comprehensive semantic decontamination audit. WRIME and JMultiWOZ remain evaluation sources. [V3 design](plan_v3.md) records budgets and normalization.

## Storage and licensing

Sentence exports retain source spans and provenance. Encoded storage uses little-endian uint16 tokens, uint64 sequence offsets and JSONL row offsets, and uint8 source identifiers. A sequence is BOS + complete sentence + EOS. Long-sequence windows cover every prediction target once; later windows do not introduce artificial BOS tokens.

Data and token stores remain in ignored `datasets/`, `outputs/`, and `artifacts/`. Source-level licenses and attribution are retained locally; [NOTICE](../NOTICE.md) distinguishes project licensing from dataset terms.
