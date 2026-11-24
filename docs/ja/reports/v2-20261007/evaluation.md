# V2.0 final evaluation — 2026-10-07

V2.0 completes four epochs and 393,704 updates from random initialization. The selected subset-loss best is step 375,000. The run uses the frozen V1/V2 corpus, V2 tokenizer, 12.54M architecture, BF16, and batch 256.

| Fixed metric | V1 | V2.0 best |
| --- | ---: | ---: |
| Complete validation BPC | 3.4736559 | 3.2598221 |
| AJIMEE Top-1 / 200 | 124 | 125 |
| Historical development / 137 | 122 | 124 |
| Expanded draft development / 2,000 | 1,453 | 1,463 |

BPC improves while small-set IME gains are limited. The expanded comparison has 92 corrections and 82 regressions relative to V1, net +10, exact paired p=0.49517. Draft label sensitivity and recall constraints prevent treating this as a robust generalization advantage.

The original FP32 same-pool evaluations use matching V2 tokenization, unchanged labels, stable ties, and full denominators. Historical model/validation artifacts remain in `artifacts/models/tiny-ja-v2.0-e16k-d320-l6/`. [Expanded results](expanded-ime-evaluation.md) and [label audit](label-error-audit.md) retain task-level diagnostics. Training throughput is approximately 121,228 effective tokens/s over the recorded update interval.
