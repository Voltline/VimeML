# V4 8K token data and context-128 windows

2026-10-14. The V4 experiment uses vocabulary 8192 to reduce embedding allocation and increase Transformer backbone capacity. Vocabulary 12K is not a future V4 model option; the earlier 12K/16K tokenizer runs remain historical diagnostics. Throughput is a measured constraint, not the motivation for the 8K selection.

[Chinese report](../../zh-CN/reports/v4-20261014/token-data.md). [Configuration](../../../configs/data-v4.toml), [entry point](../../../scripts/tokenizer/encode_v4.py), [implementation](../../../src/vimeml/tokenizer/encode_v4.py), [tokenizer record](tokenizer.md).

## Scope and input identity

The pipeline encodes the complete frozen `outputs/corpus-fast-v1/train.txt` (25,185,368 sentences) and `validation.txt` (260,337 sentences), preserving row order, whitespace, and the existing split. It uses the accepted `artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model`; it does not refit a tokenizer or use the 2M tokenizer-training sample as the LM dataset. Test/blind text, benchmarks, reference answers, and model scores are not consumed. No Transformer training is launched.

The original full train JSONL is absent after prior local cleanup. The encoder therefore reads TXT directly and uses the frozen V2 `sources.bin` files only to retain per-sentence primary source IDs. Corpus TXT hashes, manifest identity, sentence/character/source counts and source-file hashes must agree before these indices can be reused. V2 token sequences/offsets are not reused.

All encoded texts must decode exactly, use IDs within 8192, and contain no raw reserved IDs 0–6. Every stored sequence is BOS + full text + EOS. Long sentences are retained; no text is dropped, truncated, or concatenated. Text partitioning records exact complete-line byte boundaries and starting row numbers, so the independent workers preserve source alignment.

Unlike legacy stores, V4 `rows.bin` contains uint64 byte offsets into the corresponding **TXT**, not the removed JSONL. This distinction is explicit in the manifest (`row_offset_source=corpus_txt`). Full document-level origins cannot be reconstructed from these primary source IDs alone; archived corpus build records remain separate.

## Outputs and reproduction

```powershell
.venv/Scripts/python.exe -X utf8 scripts/tokenizer/encode_v4.py --dry-run
.venv/Scripts/python.exe -X utf8 -u scripts/tokenizer/encode_v4.py
# Reuse verified committed parts after interruption, or finish index/loader checks.
.venv/Scripts/python.exe -X utf8 -u scripts/tokenizer/encode_v4.py --resume
```

The fixed configuration uses 8 workers, one native encoder thread per worker, batch4096, and 128 MiB text chunks. Existing outputs are never overwritten without explicit `--resume`; resume verifies matching signatures and part checksums. Input SHA256 values are calculated before/after encoding. Part and final binary fingerprints are retained. Temporary parts are removed only inside this experiment's output, after final token verification. V1/V2 source and artifacts remain unchanged.

| Path | Purpose |
| --- | --- |
| `artifacts/token-data/corpus-v4-8k/` | Train/validation uint16 tokens, uint64 offsets/TXT row positions, uint8 sources, checksummed manifest/stats and part audit |
| `artifacts/training-data/corpus-v4-c128/` | Compact NPZ window indices, manifest and loader verification |
| `outputs/tokenizer/v4-data-20261014.log` | Run log |
| `runs/tokenizer/corpus-v4-8k-encode/` | TensorBoard run progress |

Window indices preserve every next-token pair, including EOS. At boundaries, adjacent windows share one input token; prediction targets neither overlap nor disappear. Later windows do not receive an artificial BOS. PAD=0 and padded labels=-100; collators mask padding from loss. These artifacts use the existing `TokenStore`, `SentenceWindowDataset` and CPU PyTorch DataLoader contracts without changing V1/V2 code.

V4 indices contain train/validation only. Shared `prepare_indexes` now accepts explicit splits, and the V4 training entry selects these two splits. Legacy CLI defaults remain unchanged. The full training configuration and launch commands are in [the V4 training guide](../../training-v4.md).

## Verification and limitations

Eleven tokenizer tests pass, including TXT partition/source alignment, exact native token sequences, reconstructed text positions, complete long-sentence target coverage, padded loss masks, and completed-store resume. Real-data checks include all-sentence roundtrip/ID validation, exact validation agreement with tokenizer-stage totals, merged offset/count/size/source checks, file hashes, selected long-window checks, and CPU PyTorch batches with two spawn workers. These checks are data acceptance, not LM evaluation or training-throughput measurements.

The corpus's inherited `ready_for_lm_training=false` flag and cleaning/near-duplicate limitations remain explicit. The new token data does not constitute a new Japanese quality review or a release-level corpus-quality approval. No blind access or benchmark label modifications occur.

## Measured output

Subsequent model preparation added an explicit split option to shared index preparation and a V4 training entry. Legacy commands still default to three splits. See [model/local acceptance and launch](model.md).

Encoding, merging, and token-store fingerprints completed in 158.44 seconds; index/loader checks ran afterward. The eight binary files total 1,787,137,101 bytes (1.664 GiB); window NPZ indices total 523,408 bytes.

| Metric | Train | Validation |
| --- | ---: | ---: |
| Sentences | 25,185,368 | 260,337 |
| Content tokens | 620,021,409 | 6,367,231 |
| Stored tokens (BOS/EOS) | 670,392,145 | 6,887,905 |
| Prediction pairs | 645,206,777 | 6,627,568 |
| Context-128 windows | 25,208,300 | 260,591 |
| Additional long-sentence windows | 22,932 | 254 |
| Longest sequence (BOS/EOS) | 709 | 506 |
| Characters/token | 1.7434 | 1.7449 |
| Byte-token fraction | 0.2615% | 0.2709% |
| UNK / roundtrip errors | 0 / 0 | 0 / 0 |

All 645,206,777 train and 6,627,568 validation prediction pairs are retained. Core checks exercised 60 train windows / 3022 target pairs and 59 validation windows / 2618 target pairs, including long-sentence boundaries. CPU PyTorch checks passed four batches / 256 windows per split with two spawn workers, validating 6429 train and 6522 validation target pairs. Unbucketed diagnostic padding ratios are not formal training efficiency estimates.

Final token-store manifest SHA256: `c988f8eabfb903de25bc6d526f9ea42e686771814f7e33da56e0e6dc17931d5b`. Final binary/source hashes, window hashes, tokenizer identity, split list, prediction-pair counts, and absence of test outputs/temporary parts were verified after completion.

The token store, window indices, model and local checks are complete. The next step is direct full base-LM training. The original corpus quality metadata remains unchanged; re-encoding is not a new corpus-quality review.
