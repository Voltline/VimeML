# Evaluation methodology and results

## Fixed-pool IME scoring

AzooKey supplies the same frozen candidates for every compared model. Context and candidate are jointly tokenized with BOS. The primary score is the sum of full-vocabulary log probabilities after the pool's common token prefix. No EOS, truncation, or candidate-specific length normalization is added. Token mean is a secondary diagnostic. Boundary retokenization makes this a token-suffix likelihood surrogate rather than an exact conditional string probability.

Ties retain engine order. Any candidate overflow or non-round-tripping input causes whole-case fallback. Empty pools, recall misses, and fallbacks remain in the full denominator. Accepted outputs use exact reference matching; references are not injected into candidate pools. Top-1, Top-5, recall, MRR, minimum reference CER, corrections, regressions, and subgroup results are reported separately.

## Dataset roles

| Set | Role and limitations |
| --- | --- |
| AJIMEE JWTD v2, 200 | Public fixed comparison, previously inspected; not a fresh blind test |
| Historical development, 137 | AI-reviewed generated cases; used for historical ranking calibration |
| Expanded development, 2,000 | Draft labels, predominantly held-out old web text; development/diagnostic evidence |
| Historical blind, 1,000 | Candidate export completed; no LM scoring |
| Standard development, 728 | Frozen AI-reviewed WRIME/JMultiWOZ cases; model selection allowed |
| Standard reference, 725 | Originally blind, already consumed; subsequent registered fixed-reference comparisons only |
| Standard legacy regression, 500 | Exposed old web subset, historical draft labels; separate regression track |

Standard labels are not native-speaker gold. Source-held-out texts and limited overlap screening do not prove complete independence from web pretraining. Documentation never treats the consumed reference set as newly blind. [Benchmark protocol](benchmarks/standard-ime.md) records exact identities and consumption rules.

## Language-model metrics

NLL is measured over valid prediction targets, including EOS. BPC is total NLL divided by Unicode character count times ln(2), excluding special tokens from the character denominator. Perplexity/NLL from different tokenizers are not directly comparable. BPC values on different corpora are not directly comparable either. Full-corpus and fixed-subset selection measurements are labeled separately.

## Recorded task outcomes

| Model | AJIMEE / 200 | Historical dev / 137 | Standard dev / 728 | Exposed reference / 725 | Legacy / 500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 FP32 | 124 | 122 | Not measured | Not measured | Not measured |
| V2.0 FP32 | 125 | 124 | Not measured | Not measured | Not measured |
| V2.1 release FP32 | 144 | 122 | 505 | 535 | 374 |
| V3 A final | 136 | 123 | 508 | 546 | 368 |
| V3 B selected epoch 2 | 134 | 120 | 507 | 533 | 364 |

V2.1 quantized scores are 144/200, 122/137, and 1,481/2,000 on the historical sets; Standard scores above are FP32. V3 results do not establish a stable deployment advantage. [V2 summary](reports/v2-summary.md) and [V3 B evaluation](reports/v3-20261008/b-evaluation.md) retain detailed measurements.

## Statistical and deployment interpretation

Paired comparisons require identical case/reference/context/provenance/candidate identity. Exact case-level McNemar p-values are exploratory; they do not fully correct author/user correlation, multiple comparisons, or development checkpoint selection. Nonsignificance is not an equivalence certificate. Post-hoc dictionary aliases remain sensitivity analyses, never revised primary labels.

Strict quantized logits alignment and task accuracy are distinct criteria. Fixed-pool hit counts do not prove numerical equivalence, unchanged rankings, generation quality, or device latency. Python timing, host-app tests, simulator memory, and real keyboard-extension measurements retain separate scopes. Existing score caches are reused when identities match; completed fixed-reference evaluations are not repeated.
