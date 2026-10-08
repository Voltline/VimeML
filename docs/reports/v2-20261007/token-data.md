# V2 token store and window index — 2026-10-07

Eight workers encode the frozen corpus in batches of 4,096, preserving sentence order. Encoding/merge/new-artifact identity takes 89.39 seconds; window indexing takes approximately 1.42 seconds.

| Metric | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Sequences | 25,185,368 | 260,337 | 267,298 |
| Stored tokens including BOS/EOS | 581,648,025 | 5,976,419 | 6,079,517 |
| Prediction pairs | 556,462,657 | 5,716,082 | 5,812,219 |
| Context-128 windows | 25,196,843 | 260,464 | 267,421 |
| Additional long-sentence windows | 11,475 | 127 | 123 |

Binary assets total 1,624,528,997 bytes (1.513 GiB); compressed window indices total 264,360 bytes. Targets are covered once without overlap or artificial BOS on later windows. Offset, size, and count checks reuse frozen corpus/tokenizer records.

Assets reside in `artifacts/token-data/corpus-v2-16k/` and `artifacts/training-data/corpus-v2-c128/`. Training reads binary assets directly; JSONL row offsets support provenance lookup. Batch 256 gives 98,426 updates per epoch and 393,704 across four epochs. Effective post-crop token counts are reported separately.

```bash
python scripts/tokenizer/encode.py --tokenizer artifacts/tokenizers/ja-unigram-16k-v2 --output artifacts/token-data/NEW-VERSION --workers 8 --batch-size 4096 --verification metadata
```

Original preparation records remain under `outputs/history/v2-20261007-session/` and `outputs/model-checks/v2-phase-c.json`.
