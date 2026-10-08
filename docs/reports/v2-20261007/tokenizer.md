# V2 tokenizer preparation — 2026-10-07

The 16K Unigram tokenizer trains on two million train sentences with seed 42, identity normalization, preserved whitespace, byte fallback, `add_dummy_prefix=false`, and PAD/UNK/BOS/EOS IDs 0/1/2/3. Configuration: [tokenizer-v2.toml](../../../configs/tokenizer-v2.toml). Preparation and full-split validation complete in 313.97 seconds.

| Metric | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Sentences | 25,185,368 | 260,337 | 267,298 |
| Content tokens | 531,277,289 | 5,455,745 | 5,544,921 |
| Token count relative to V1 | -3.071% | -3.095% | -3.122% |
| Characters/token | 2.0346 | 2.0365 | 2.0373 |
| Mean tokens/sentence | 21.0947 | 20.9565 | 20.7443 |
| Context-128 coverage | 99.9560% | 99.9551% | 99.9547% |
| Byte-fallback fraction | 0.3030% | 0.3137% | 0.3108% |

All 25,713,003 sentences have zero UNK and zero round-trip mismatches. Coverage uses content+BOS+EOS ≤ context. Token reduction reflects the entire retrained vocabulary, not solely removal of the dummy prefix. Two of six IME boundary probes still retokenize at the context/candidate seam, so joint encoding remains required.

Assets reside in `artifacts/tokenizers/ja-unigram-16k-v2/`, with model, vocabulary, configuration, manifest, statistics, examples, and V1 comparison. Existing corpus identity is reused; original validation records remain frozen.
