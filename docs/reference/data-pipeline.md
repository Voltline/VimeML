# Corpus processing reference

The frozen corpus is `outputs/corpus-fast-v1/`: 2,289,346 source records and 25,713,003 unique sentences. Input selection and split counts appear in [data](../data.md). No API is required for the deterministic full build.

## Processing stages

```text
Parquet / TSV → Unicode and deterministic cleanup → keep/drop/review
              → conservative within-sentence line restoration → segmentation/quarantine
              → document grouping, exact deduplication, split assignment → text/JSONL/provenance
```

Whitespace, decoration, and isolated URLs can be dropped; ambiguous navigation tails remain review items. Adjacent lines are not joined solely because punctuation is absent. Source text and modification reasons remain auditable. Approval decisions are stored in `annotations/approved-v1.jsonl`. No reading augmentation is generated.

## Parallel and sharded reconstruction

The single-host entry uses independent worker SQLite databases followed by bucketed merging:

```bash
python scripts/corpus/build.py --config configs/corpus-parallel.toml --output outputs/corpus-new --workers 8
```

Cross-host partitions use the same source, code, configuration, and review decisions:

```bash
python scripts/corpus/preprocess.py --config configs/corpus-sharded.toml --part 0/2 --output outputs/corpus-part-0
python scripts/corpus/preprocess.py --config configs/corpus-sharded.toml --part 1/2 --output outputs/corpus-part-1
python scripts/corpus/merge.py --config configs/corpus-sharded.toml --parts outputs/corpus-part-0 outputs/corpus-part-1 --output outputs/corpus-new --workers 8
```

Global deduplication and split assignment occur at merge. Resume uses compatible completed partitions; deleted worker outputs are not recoverable inputs. Source-code fingerprints in frozen V1 records remain unchanged by later documentation edits.

## Review and exports

`scripts/review/prepare.py` produces stratified review samples and quarantined targets. `run.py` optionally requests DeepSeek review suggestions using external environment credentials; API suggestions do not rewrite or automatically approve source text. `approve.py` records explicit selected decisions. Completed review excludes 638 quarantined blocks. Model review is not comprehensive quality certification.

Exports include ordered `train/validation/test.txt` and `.jsonl`, `provenance.jsonl`, document/review indices, quarantine files, manifests, counts, and original integrity records. The original merge staging flag is preserved; later review and baseline-use decisions are separate records. Exact deduplication does not establish semantic independence or absence of benchmark contamination.

`scripts/tools/check_corpus.py` provides metadata checks; full large-file reads are a separate optional operation. Existing completed build records are sufficient for routine artifact reuse, without repeated full-corpus hashing.
