# V1 FP32 baseline results

The V1 run completes one epoch and 196,853 updates. Validation NLL is 4.5415709, perplexity 93.8381, and BPC 3.4736559. Configuration and timing appear in [V1 training](../training.md). The corpus test split is not used for checkpoint selection.

## Real-candidate ranking

| Method | AJIMEE Top-1 / 200 | Top-5 / 200 |
| --- | ---: | ---: |
| AzooKey exported order | 87 | 143 |
| Contextual V1 suffix logP sum | 124 | 151 |
| Historical hybrid lambda 2 | 118 | 153 |

The pool covers 162/200 exact references. Pure LM corrects 48 engine errors and introduces 11 regressions. The no-context diagnostic scores 116/200. Recall misses remain in the denominator. Context/candidate scoring is jointly tokenized, with no EOS or silent truncation and stable ties.

Historical development results are engine 111/137, pure LM 122/137, and calibrated hybrid 123/137. Lambda selection occurs on development only. That set is AI-reviewed generated text, not native-speaker gold.

V1 scores 1,453/2,000 on the later expanded draft development set. Those labels remain drafts; exposed public/development comparisons do not establish unseen generalization. [AJIMEE](ajimee-benchmark.md), [hybrid ranking](hybrid-ranking.md), and [expanded evaluation](../reports/v2-20261007/expanded-ime-evaluation.md) describe measurement scope.

## Generation and deployment

The twenty-prefix phrase demo has mixed qualitative results and is not an open-generation accuracy benchmark. The Core ML INT8 release retains AJIMEE/historical-development hit counts while failing strict logits alignment. [V1 Core ML](coreml-v1.md) distinguishes quantization diagnostics, runtime timing, and device memory.
