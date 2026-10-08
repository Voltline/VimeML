# V1 training baseline

V1 is a frozen, randomly initialized 7,386,624-parameter Japanese decoder-only model. Architecture: vocabulary 16,384, context 128, hidden width 256, four layers, four attention heads, GELU feed-forward width 1,024, pre-LayerNorm, causal SDPA, learned absolute positions, and tied embeddings. Dropout is zero.

## Data and optimization

The V1 Unigram tokenizer uses identity normalization, preserved whitespace, byte fallback, and PAD/UNK/BOS/EOS IDs 0/1/2/3. Training uses a two-million-sentence train-only tokenizer sample and the [frozen corpus](data.md). Independent sentence windows retain EOS targets; padding labels are -100.

| Setting | Value |
| --- | --- |
| Epochs / updates | 1 / 196,853 |
| Batch / precision | 128 / BF16 |
| Optimizer | AdamW, betas 0.9/0.95, weight decay 0.1 |
| Learning rate | 3e-4 to 3e-5, cosine decay, 2,000 warmup updates |
| Gradient clipping | 1.0 |
| Training windows / targets | 25,197,117 / 573,295,237 |

Configuration and entry points reside in `configs/` and `scripts/training/`. CLI defaults refer to V1 artifacts. New runs require distinct output directories; the completed baseline does not require reconstruction for subsequent comparisons.

## Completed run

An RTX 5060 Laptop run completed in 3,469.8 seconds (57 minutes 50 seconds), at approximately 169,587 training tokens/s. Reported peak allocated memory was 2,427.54 MiB and reserved memory 10,846 MiB; these allocator measurements are not total physical device use.

`best.pt` and `last.pt` correspond to step 196,853. Validation NLL is 4.5415709, perplexity 93.8381, and BPC 3.4736559 over 5,890,326 targets. Test text was not used for model selection. Fixed-pool IME results and baseline limitations appear in [V1 results](reference/results.md).

Checkpoints contain model/optimizer state, configuration, tokenizer/data identity, and training position. Local model assets reside in `artifacts/models/tiny-ja-v1/`. Inference-only release weights omit optimizer state. Dependencies and the recorded platform pins are defined separately from model version labels.
