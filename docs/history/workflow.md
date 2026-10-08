# Historical V1 workflow status

The original pipeline covers source download, deterministic cleaning, grouped deduplication/splits, review, tokenizer training, binary encoding, sentence-window training, fixed-candidate evaluation, Core ML conversion, and device measurement.

The corpus and V1 baseline are frozen. Later V2/V3 experiments use separate configurations and artifacts. Historical build/review flags preserve their original stage identity instead of being rewritten to describe later training completion. Model suggestions do not automatically approve or rewrite corpus text.

Completed scores and original reports remain reusable evidence. Documentation-only maintenance does not require rebuilding corpora, rerunning LM evaluations, or repeating full-file checksums. [Documentation index](../index.md) provides the current experiment status and measured deployment results.
