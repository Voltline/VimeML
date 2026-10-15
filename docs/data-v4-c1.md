# V4-C1 conversion dataset — 2026-10-15

The first conversion dataset contains **2,841,003 training and 18,956 validation examples**, prepared from the original LM train/validation text with the frozen 8K tokenizer. It uses automatic dual-dictionary silver labels. Formal post-training has not started. [Chinese report](zh-CN/data-v4-c1.md) · [Training design](posttraining-v4.md).

## Frozen counts

| Metric | Train | Validation |
| --- | ---: | ---: |
| Examples | 2,841,003 | 18,956 |
| OUT/EOS targets per pass | 11,486,846 | 75,686 |
| Mean reading characters | 6.765 | 6.674 |
| Mean surface characters | 5.186 | 5.104 |
| Mean full-READ serialized tokens | 23.284 | 22.649 |
| Kana-only targets | 568,200 | 3,791 |
| Fixed empty-READ views | 284,062 | 0 |
| Left-context crops | 564 | 2 |
| FineWeb examples | 2,814,085 | 18,794 |
| Tatoeba examples | 26,918 | 162 |

The initial three-million/20,000 selection was reduced by shared-reading, exact task-overlap, and compound-boundary filters while retaining the 20% kana-only cap. Removed rows were not backfilled; the final counts do not establish exhaustion of eligible examples in the corpus. At 256 conversion examples per update, twelve passes require 133,176 updates and 137,842,152 conversion targets. Ordinary-LM replay is counted separately.

## Annotation and isolation

Sudachi and fugashi/UniDic analyze complete parent sentences. Selected spans use shared boundaries and aligned lexical readings, retain original surface/inflection, and contain at most six morphemes per analyzer, 1–24 surface characters, and 1–48 reading characters. Kana surfaces retain keyboard orthography, including particles は/へ/を. Stored dual readings are normalized span readings; per-token raw dictionary features are not all stored. Source offsets and frozen dictionary identities support reproduction.

Each parent contributes at most one span, sampled from its available reading-length, context-length, and kanji buckets. This is not globally equal-sized stratification. Target proper-noun/OOV flags, unsupported characters, unreliable readings, and boundary mismatches are excluded. A conservative rule also removes single-kanji nouns adjacent to other kanji. Analyzer agreement cannot eliminate every proper noun, source typo, or ambiguous reading.

Original train parents are checked against the complete original validation set. Exact conversion field triples and serialized sequences shared with frozen conversion validation are removed from train. Known nonblind development/regression records serve only full-case identity decontamination; common surface words are not independently blacklisted. No benchmark text enters training and no test/blind data is accessed. Document IDs are unavailable, limiting the isolation claim to parent sentences and exact task cases. Conversion validation parents previously participated in base-LM selection.

Tokenizer SHA256: `296a2bdf4f4862be8bda848ed1cd261f0b0fd43064181fad5a513c6ce73081b5`. Original train/validation TXT identities were rechecked at delivery; no corpus additions were made.

## Encoding and acceptance

```text
[BOS]<CTX>context<READ>reading<OUT>surface[EOS]
```

Fields are separately encoded, markers use IDs 4/5/6, and at most 129 serialized tokens produce 128 input/target positions. Only left context is cropped. Labels are shifted once and supervise OUT/EOS only. Approximately 10% of train examples carry a fixed empty-READ flag while preserving the original reading.

The parent selection passed full source/span/hash, codec, mask, dropout-view, and split checks. Final filters only remove rows and copy retained records/tokens/flags unchanged, inheriting those checks. Final SHA256 identities, offsets, counts, and kana caps were rechecked. The mmap dataset reopens files by path for spawned workers.

Codex AI reviewed 200 stratified parent examples and identified shared 家 readings, single-kanji compound boundaries such as 岩石棒, and name fragments such as 春 in 春也. Systematic filters quarantine these patterns, including some valid compound fragments. A final 200-example review packet includes complete parent sentences. No native-human review or corpus-wide label-accuracy estimate is claimed.

## Artifacts

Directory: `artifacts/posttrain-data/v4-c1/`, excluded from Git.

| Artifact | Purpose |
| --- | --- |
| `manifest.json`, `acceptance.json`, `delivery.json` | Frozen identity, inherited checks, and source verification |
| `*.tokens.bin` | uint16 serialized tokens |
| `*.offsets.bin`, `*.out-start.bin`, `*.reading-dropout.bin` | uint64 offsets and uint8 target/view flags |
| `*.records.jsonl` | Parent hashes, source offsets/IDs, spans, fields, and normalized readings |
| Quarantine records and historical manifests | Exclusion history and audit lineage |
| `stratified-review-200.json` | Final review packet with complete parents |
| `preparation-source/`, `refinement-source/` | Exact hashed preparation source snapshots |

Manifest SHA256: `3eb58a9dba365e96f094bfb22950c66a9d06cddeb1cb7f5e4db2e78bf59532c0`.

Local GPU capacity and the prepared cloud configuration are documented separately in the [training design](posttraining-v4.md).
