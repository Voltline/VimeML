# V2.1 restart1 evaluation — 2026-10-07

The two-epoch restart completes 98,426 updates from V2.0 step 375,000 weights with fresh AdamW. Formal throughput is 273,368 effective tokens/s; total elapsed time is 3,739.8 seconds. All original artifacts remain frozen.

| Checkpoint | Old validation BPC ↓ | AJIMEE / 200 | Historical dev / 137 | Draft dev / 2,000 |
| --- | ---: | ---: | ---: | ---: |
| V2.0 best | 3.2598221 | 125 | 124 | 1,463 |
| Restart epoch 1 | 3.1859122 | 136 | 125 | 1,462 |
| Restart epoch 2, step 98,426 | 3.0811788 | 137 | 121 | 1,476 |

The final checkpoint improves BPC and AJIMEE while historical development falls below the first epoch. Draft development gains 13 cases relative to V2.0. Different task subsets do not move monotonically with language-model loss. Draft labels, small sets, and repeated development inspection limit generalization conclusions.

The second-epoch checkpoint initializes the subsequent extension with optimizer state preserved. Models reside in `artifacts/models/tiny-ja-v2.1-e16k-d320-l6-restart1/`; FP32 expanded caches are `outputs/ime-eval/expanded-v21-dev-draft-restart1-{epoch1,epoch2}/`. The old 1,000-case blind set is not scored. [Extension selection](v21-extend.md) records the later deployed checkpoint.
