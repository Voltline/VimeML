# V4-a tokenizer initialization and training

2026-10-14. Scope: local tokenizer training and tokenization statistics. The fixed V4-a architecture remains vocabulary 8192, d384, seven layers, six heads, FFN960, context128, and 15,070,848 parameters. This run does not initialize/train the Transformer, export Core ML, or score an LM. [Chinese record](../../zh-CN/reports/v4-20261014/tokenizer.md).

## Implementation and reproduction

Decision on 2026-10-14: V4 uses 8K to move embedding allocation into backbone capacity; 12K is not a future model option. The 12K/16K runs below are historical tokenizer diagnostics. See [full 8K data preparation](token-data.md).

[Configuration](../../../configs/tokenizer-v4.toml), [entry point](../../../scripts/tokenizer/train_v4.py), [implementation](../../../src/vimeml/tokenizer/v4.py), [frozen kana inventory](../../../configs/kana-v4.txt). Existing V1/V2 code and artifacts are unchanged.

```powershell
.venv/Scripts/python.exe -X utf8 -u scripts/tokenizer/train_v4.py --dry-run
.venv/Scripts/python.exe -X utf8 -u scripts/tokenizer/train_v4.py
.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -p 'test_tokenizer*.py'
```

The entry point refuses to overwrite an existing experiment. Use a separate configuration/output directory for another experiment.

8192/12288/16384 Unigram vocabularies share identity normalization, preserved whitespace, byte fallback, no dummy prefix, and an exact vocabulary limit. PAD/UNK/BOS/EOS are 0/1/2/3. Reserved USER_DEFINED markers `<CTX>/<READ>/<OUT>` are 4/5/6; they are not used to construct conversion examples here. Future task formatting must specify how literal marker strings in raw text are handled.

All vocabulary sizes use one persisted 2,000,000-line reservoir sample from the full frozen train TXT, produced with Python `random.Random(42)`. The scan verifies train SHA256, sentence/character counts, maximum UTF-8 sentence length, and reserved strings. Native training uses the entire sample without resampling. SentencePiece 0.2.1's internal reservoir seed is separate; this explicit sample is not claimed identical to V2's native sample.

Full validation SHA256 is verified against the frozen integrity report. Statistics cover the training sample, full validation, and the explicit standard development directory. Test/blind paths are rejected; their text is not read. No LM probability, reranking, or reference accuracy is computed, and benchmark files/references are unchanged. The existing corpus manifest's `ready_for_lm_training=false` is preserved.

## Kana and vocabulary budget

576 assigned Unicode 15.1 characters are frozen in a text file, one character per line. Ranges: U+3041–309F, U+30A0–30FF, U+31F0–31FF, U+FF66–FF9F, U+1AFF0–1AFFF, U+1B000–1B0FF, U+1B100–1B12F, U+1B130–1B16F, excluding unassigned positions. This conservative scope includes historical kana, halfwidth forms, combining/spacing voiced marks, and extended small kana; it does not cover every Japanese symbol or unassigned codepoint.

`required_chars` preserves NORMAL single-character entries while allowing larger learned pieces. Each character must have an exact normal vocabulary entry, encode alone to its single ID, decode unchanged, and avoid UNK. This is not mandatory character-by-character segmentation of every reading. Additional probes cover decomposed voiced forms, halfwidth forms, spaces, tabs, emoji, rare kanji, and markers.

Four base symbols, three reserved markers, and 256 byte pieces consume 263 entries inside the total vocabulary. The 8K model has 7929 NORMAL entries; its 576 required kana and all other required characters consume part of that budget. Training can fail if required alphabet plus meta pieces exceeds the vocabulary, or if the corpus cannot support the exact vocabulary size. No silent vocabulary reduction is allowed.

Sources: [SentencePiece 0.2.1 TrainerInterface](https://github.com/google/sentencepiece/blob/v0.2.1/src/trainer_interface.cc), [Unigram finalization](https://github.com/google/sentencepiece/blob/v0.2.1/src/unigram_model_trainer.cc), and [TrainerSpec](https://github.com/google/sentencepiece/blob/v0.2.1/src/sentencepiece_model.proto). Finalization preserves required characters; meta pieces are deducted from the vocabulary budget.

## Statistics and artifacts

Lengths exclude special tokens. Metrics include tokens/sentence, tokens/candidate, characters/token, byte-token fraction, UNK, roundtrip errors, length percentiles, and the fraction fitting `content+BOS+EOS≤128`. Readings are measured in original benchmark katakana and a diagnostic U+30A1–30F6 hiragana mapping, which neither alters benchmark files nor establishes the final conversion normalization policy.

Corpus character capacities derived from `128 × chars/token` or `126 × chars/token` are estimates. Development prefix coverage uses actual tokenizer offsets for the first 126 content tokens, including short texts; it is not a packed LM window measurement.

Local ignored outputs:

- `artifacts/tokenizers/v4-a/`: shared sample, sampling/config/plan records, comparison, final manifest.
- `artifacts/tokenizers/v4-a/ja-unigram-{8192,12288,16384}/`: model/vocab, native request/log, manifest, exact statistics, and per-character verification.
- `outputs/tokenizer/v4-a-20261014.log`: run log.
- `outputs/tokenizer/v4-a-v2-reference.json`: existing V2 tokenizer development statistics and historical full-validation statistics.
- `runs/tokenizer/v4-a-*`: TensorBoard events.

The primary model is `artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model`. Full train/validation token data has been regenerated with this frozen artifact/hash, as recorded in [token data](token-data.md). V2 token IDs and checkpoint embeddings are incompatible. Tokenizer-stage train-sample statistics remain distinct from full-train statistics.

Nine tokenizer tests passed, including actual native training that retains extended kana absent from its input, normalization/IDs/markers, deterministic sampling, and held-out path rejection.

## Measured results

Run completed in 799.51 seconds (13.33 minutes). All three exact vocabulary sizes passed 576 single-character kana checks. The 2,000,000-line train sample and 260,337-line full validation have zero UNK and roundtrip errors for every model. Standard development statistics cover 728 cases and 8790 actual candidates; all candidate/context+candidate texts fit 128 with BOS/EOS, and UNK/roundtrip errors are zero.

| Metric | V4 8K | V4 12K | V4 16K | Existing V2 16K |
| --- | ---: | ---: | ---: | ---: |
| Validation tokens/sentence | 24.4576 | 22.2564 | 21.0468 | 20.9565 |
| Validation chars/token | 1.7449 | 1.9175 | 2.0277 | 2.0365 |
| Validation content tokens | 6,367,231 | 5,794,172 | 5,479,254 | 5,455,745 |
| Validation byte tokens | 17,252 | 17,252 | 17,252 | 17,113 |
| Validation byte-token fraction | 0.2709% | 0.2977% | 0.3149% | 0.3137% |
| Validation p50 / p95 / p99 | 21 / 54 / 78 | 19 / 49 / 71 | 18 / 46 / 68 | 18 / 46 / 67 |
| Validation fits 128 + BOS/EOS | 99.9070% | 99.9416% | 99.9547% | 99.9551% |
| Estimated chars / 126 content tokens | 219.86 | 241.61 | 255.49 | 256.60 |
| Development tokens/candidate | 11.4820 | 10.4761 | 10.0470 | 10.0494 |
| Development candidate chars/token | 1.5552 | 1.7046 | 1.7774 | 1.7769 |
| Development candidate byte fraction | 1.3723% | 1.5040% | 1.5683% | 1.5781% |
| Development tokens/context+candidate | 18.8307 | 17.1592 | 16.4230 | 16.3642 |
| Reading tokens (original katakana) | 20.4437 | 19.5481 | 18.9066 | 19.3462 |
| Reading tokens (hiragana diagnostic) | 16.8146 | 15.8489 | 15.0096 | 14.9409 |

8K increases validation content tokens by 16.21% against the controlled V4 16K tokenizer, and 16.71% against existing V2 16K. Candidate tokens increase 14.26% against V2. The V2 comparison includes sampling and inventory differences and is not a vocabulary-size-only ablation. All three controlled tokenizers have 17,252 validation byte tokens; their different byte-token fractions reflect different token denominators. Reading fallback is limited to 19 fullwidth parentheses (57 byte tokens), not missing kana. These statistics establish tokenization cost, not an LM/reranking advantage.

Primary model SHA256: `296a2bdf4f4862be8bda848ed1cd261f0b0fd43064181fad5a513c6ce73081b5`. Shared sample SHA256: `0a7aa1eeaa8abf160d34d8338deb5d4b8386ed1a0bf1a7bc8c4c3e388c538e3d`. Artifact/sample/code/config and development-input hashes were checked after completion.

The fixed 15,070,848-parameter model and local checks are now complete; see [model preparation](model.md). The next step is [direct full base-LM training](../../training-v4.md).
