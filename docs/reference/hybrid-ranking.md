# Historical AzooKey–LM hybrid ranking

The frozen V1 prototype uses `AzooKey score + 2 × contextual LM logP sum`. Lambda 2 is selected on the independent historical development set, not AJIMEE. Current Core ML deployment uses pure LM reranking; this hybrid policy is a historical baseline and requires new calibration for another model or quantization.

## Scoring and selection

`scripts/benchmarks/hybrid.py` provides `prepare-dev`, `score`, `tune`, and `evaluate`. The component scores are combined without extra EOS, length normalization, candidate removal, or lexical rules. Ties retain engine order; whole-case fallback handles incomplete scoring. Lambda must be finite and nonnegative; lambda zero exactly recovers exported engine order.

The fixed grid is 0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3. Selection maximizes development Top-1, then minimizes engine regressions, then chooses the smaller lambda. Lambda 2 and 3 both reach 123/137, with three and four regressions respectively; the recorded rule selects 2. A manually specified lambda is diagnostic rather than calibrated.

## Dataset and outcomes

The AI-reviewed development set has 137 cases, 136 with 20 candidates and one with 18; reference coverage is 136/137. Query/context overlap checks with AJIMEE exclude obvious normalized matches, without establishing semantic or training independence. Review did not use Tiny LM scores.

| Set | Engine Top-1 | Pure LM | Hybrid lambda 2 | Hybrid corrections / regressions |
| --- | ---: | ---: | ---: | ---: |
| Development / 137 | 111 | 122 | 123 | 15 / 3 |
| AJIMEE / 200 | 87 | 124 | 118 | 38 / 7 |

AJIMEE hybrid Top-5 is 153/200, versus pure LM 151 and engine 143. Lower engine regressions do not imply higher Top-1. The 38 reference recall misses cannot be repaired by reranking.

The policy is stored in `artifacts/ranking-policies/tiny-ja-v1-hybrid-dev-v2/policy.json`; score/report caches reside under `outputs/ime-eval/`. Evaluation checks checkpoint, tokenizer, precision, converter/dictionary settings, and development/test identity before reusing caches. Previously inspected AJIMEE results are fixed comparisons rather than fresh blind evidence. [Development generation](development-generation.md) and [AJIMEE export](ajimee-benchmark.md) retain provenance.
