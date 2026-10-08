# V3 A adaptation results — 2026-10-08

V3 A adapts V2.1 extension step 40,000 to the selected JpnMix web pool and two real-dialogue corpora. The two-epoch run completes 25,146 updates and 952,851,047 effective tokens, including 47,642,545 chat tokens (5%). Approximately 9.08 effective chat exposures reflect repeated sampling, not additional unique conversations.

The run uses fresh AdamW, learning rate 1e-4 → 1e-5, warmup 500, batch 512, BF16, compiled backbone, and fixed `epoch_mean_tokens` loss scaling. RTX 4090 D elapsed time is approximately 27.59 minutes; update throughput is 614,843 effective tokens/s. Allocated/reserved memory is approximately 14,831/19,232 MiB. Initial updates, validation, and saving are excluded from the reported throughput.

## Domain validation

| Domain BPC ↓ | V2.1 initial | A epoch 1 | A epoch 2 |
| --- | ---: | ---: | ---: |
| Held-out chat | 4.392485 | 3.035096 | 3.006127 |
| Old corpus | 3.0802686 | 3.434576 | 3.425275 |
| New pool | Not listed | 3.090339 | 3.032179 |

Chat BPC improves approximately 31.56% while old-corpus BPC worsens approximately 11.20%. Conversations, rather than utterances, are held out, but speaker independence is not established. Domain BPC values are only comparable within the same text/protocol. The adaptation changes multiple source and optimization variables; the result does not isolate chat content or education filtering as the sole cause.

## Fixed candidate evaluation

| Checkpoint | AJIMEE / 200 | Historical dev / 137 |
| --- | ---: | ---: |
| V2.1 initial | 144 | 122 |
| A epoch 1 | 134 | 121 |
| A final, step 25,146 | 136 | 123 |

The final model scores 1,462/2,000 on expanded draft development versus V2.1 1,487. It corrects 46 and regresses 71 cases, net -25, exploratory paired p=0.0261. AJIMEE has six corrections/fourteen regressions, net -8, p=0.1153; historical development has four corrections/three regressions, net +1, p=1.0. Draft labels and small exposed sets limit interpretation.

The preselected final checkpoint subsequently receives the preregistered Standard IME comparison: development 508/728 versus V2.1 505, originally blind reference 546/725 versus 535, and historical regression 368/500 versus 374. Source-level effects differ; [initial benchmark results](../../benchmarks/standard-ime-results-20261008.md) retain the paired comparison. This evaluation consumes the 725-case split, so later evaluations are fixed-reference comparisons.

## Conclusion and archival

A learns the new chat distribution but regresses old-corpus likelihood and historical draft reranking. The new-source results do not establish a stable deployment replacement advantage. V2.1 remains deployed; the final A model is retained as a research checkpoint. Qualitative continuations show limited generation capability rather than validated conversational quality.

Model and original reports reside in `artifacts/models/tiny-ja-v3-a-chat05-d320-l6/` and `outputs/training-v3/`. Initial domain validation, full validation, epoch IME results, fixed benchmark caches, and stage records remain frozen. Historical 1,000-case blind candidates receive no LM scoring. AI-reviewed labels are not native-speaker gold; case-level statistics are exploratory.
