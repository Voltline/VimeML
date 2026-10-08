# Local phrase-completion demo

`scripts/tools/phrase_demo.py` runs a local web demo using a frozen TinyGPT checkpoint. The default is V1, CPU FP32, four threads, and localhost port 8765. Model loading occurs once. Input text is not stored.

```bash
python scripts/tools/phrase_demo.py
```

## Generation behavior

The interface accepts a single Japanese within-sentence prefix and requests three to five suggestions with a four-to-sixteen-token continuation limit. Sentence-final punctuation in the initial prefix is rejected, including quoted punctuation. Excessively long context is rejected rather than silently truncated. A token is not equivalent to a word.

The ranking mode uses batched beam search with width 8 and alpha 0.7. Sampling uses greedy output plus eight batched trajectories, temperature 0.8, top-k 50, and top-p 0.9. Generation normalization does not change the raw suffix-sum candidate-reranking rule. Empty, duplicate, prefix-changing, or damaged outputs are filtered without template padding. Truncation/repetition markers are diagnostics, not grammatical certification.

An embedded second sentence is clipped at the first sentence end for display; original trajectories remain in fixed-suite outputs. HTTP requests use same-origin checks and inference serialization. Reported generation time includes tokenization/decoding but excludes model loading and does not represent iPhone latency.

## Historical suite

Twenty fixed prefixes reside in `configs/phrase-demo-prompts.json`. `--suite` and `--suite-mode sample` produce separate versioned outputs, preserving parameters, identity, raw generation, and stop reasons. Existing beam/sample reports are in `outputs/phrase-demo/tiny-ja-v1/` and `tiny-ja-v1-sample/`.

AI-assisted qualitative review identifies ten prefixes with clearly natural suggestions, three with weaker semantic/factual quality, and seven with unsuitable outputs. This is not unique-answer accuracy. Warm local median latency is approximately 50 ms for beam and 116 ms for sampling. Repetition, drift, and truncation remain; candidate reranking is the primary deployment capability.
