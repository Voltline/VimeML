# Historical label and error audit — 2026-10-07

AI review of the 137 historical development cases hides candidate/model scores during review, retains 135 cases, and quarantines ambiguous `シャシンテン` and `オオサカテン`. Sixty-three cases receive 86 additional accepted spellings in a separate `ime-dev-label-reviewed-v3` artifact. Original labels/scores remain unchanged. Prior exposure to four disagreement cases prevents describing this as independent blind native-speaker review.

| Label version | V1 Top-1 | V2.0 Top-1 | V1 Top-5 | V2.0 Top-5 |
| --- | ---: | ---: | ---: | ---: |
| Original | 122/137 | 124/137 | 134/137 | 136/137 |
| Separate reviewed diagnostic | 131/135 | 131/135 | 134/135 | 134/135 |

The original net gain disappears under the revised diagnostic label set. Further missing-reference concerns remain unreviewed; main metrics are not retroactively altered. Dictionaries provide reading evidence rather than automatic correctness decisions.

Among AJIMEE 200 plus reviewed development 135, 93 cases have at least one model error. AJIMEE has 38 reference recall misses for both models and 38/37 in-pool ranking errors, yielding 124/125 correct first choices. Categories include homophone sense, compound/counter segmentation, proper names, terminology, inflection, and missing spellings. Not every exact-reference miss is a semantic retrieval failure.

Examples include V2's correction of `直行座標` to `直交座標` and regression from `盗難` to `東南`. Contextual AJIMEE scores are 59/58 and context-free scores 65/67; no clear contextual advantage is established here. All recorded length fallbacks are zero. Error categories are AI diagnostics, not gold causal labels.

Separate label decisions reside in `artifacts/benchmarks/ime-dev-label-reviewed-v3/`; cached audit scores and errors remain in `outputs/ime-eval/`. Tools `audit_v2_labels.py` and `analyse_v2_errors.py` reuse existing scores. [Expanded draft evaluation](expanded-ime-evaluation.md) remains the unchanged primary historical result.
