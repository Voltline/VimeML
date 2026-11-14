# vimeml-tiny-zh-v1 tokenizer: vocabulary size and character coverage (2026-11-06)

An exploratory comparison that trained tokenizers only: no language model, no BPC, no IME accuracy. The aim was to learn before training how much compression improves from 8K to 32K, how many layers that costs at a fixed parameter budget, and how character coverage affects rare characters.

Conclusions first:

- **The 20M model tries only 8K and 12K, both at coverage 0.9995.** 12K is the most balanced choice, 8K the control. Which is better can only be decided by BPC and IME results at equal compute.
- **The 200M model considers 16K, 24K and 32K**, one arm per vocabulary.
- **Coverage 0.9999 is not used.** It does cut byte fallback for rare characters, but single characters take almost all the multi-character slots, and 12K's compression falls back to 8K's level.
- **The formal tokenizer must make `\n` a user_defined_symbol.** Nine in ten byte tokens counted here are newlines (see below).

## Setup

Entry points: [probe_zh_v13_vocab.py](../../../scripts/tokenizer/probe_zh_v13_vocab.py) (training and statistics) and [probe_zh_v13_bytes.py](../../../scripts/tokenizer/probe_zh_v13_bytes.py) (byte fallback with newlines removed). Outputs in `outputs/tokenizer/zh-v13-vocab-probe/`, not in Git. Each variant has a directory with `tokenizer.model`, `tokenizer.vocab` and `stats.json`.

```powershell
.venv/Scripts/python.exe -I -X utf8 scripts/tokenizer/probe_zh_v13_vocab.py . outputs/tokenizer/zh-v13-vocab-probe
.venv/Scripts/python.exe -I -X utf8 scripts/tokenizer/probe_zh_v13_bytes.py . outputs/tokenizer/zh-v13-vocab-probe
```

The actual run also listed 8K / 0.9999, which exited with an error: coverage 0.9999 needs 10,417 required characters, more than the 8,192 vocabulary. 12K / 0.9999 was trained afterwards on the same sample with the same settings. The repository script no longer lists 8K / 0.9999.

**Training sample**: from the train split of [zh-v13-export-r1](../../data-zh-v1.md).

- Web: 38 web files, 2 random row groups per file, split into paragraphs on newlines (at most 16 KB UTF-8 each), 39,474 paragraphs per file, 1,500,000 in all.
- Chat: 80,000 random LCCC train utterances, about 5%.
- Both parts shuffled into one file, `train-sample.txt`. Every variant reads the same file; no resampling.
- 97,396,787 characters, of which 80,500,741 Han. Seed 20261106, file SHA256 `2ed06667da76300b1ca5b144c1b5bb448aed8ae9fe8ac707bcf0990fde4484de`.
- The review queue part had not been exported yet, so the sample holds only automatically kept web text and LCCC.

**SentencePiece 0.2.1 settings**: Unigram, `normalization_rule_name=identity`, `byte_fallback=true`, `add_dummy_prefix=false`, `remove_extra_whitespaces=false`, `split_digits=false`, `hard_vocab_limit=true`, `seed_sentencepiece_size=1,000,000`. PAD/UNK/BOS/EOS = 0/1/2/3. No task markers such as `<CTX>/<READ>/<OUT>` were added, so these tokenizers have 3 more free slots than the formal one; this does not affect the comparison.

**Evaluation data**:

| Set | Size | Use |
| --- | --- | --- |
| export-r1 validation, web | 14,681 documents, 19,184,275 characters (15,622,687 Han), 281,051 paragraph newlines | Compression, byte fallback, context capacity |
| export-r1 validation, LCCC | 56,620 utterances, 677,064 characters | Compression of chat text |
| Standard Chinese IME benchmark, development | 1,456 items, 20 full-pinyin candidates each, 29,120 in all | Candidate length, candidate byte fallback, left-context segmentation stability |

Only `split == development` benchmark items are read; blind items are discarded on load and take no part in any computation. Validation takes no part in tokenizer training.

## Results

Unless noted, figures come from web validation. "Byte fallback" is byte tokens as a share of all tokens, newlines excluded.

| Vocabulary / coverage | Tokens per Han (web) | Tokens per Han (chat) | Characters in 126 tokens | Byte fallback | Single / multi-char / other | Mean multi-char length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K / 0.9995 | 0.989 | 0.949 | 156 | 0.19% | 5623 / 1468 / 841 | 2.06 |
| 12K / 0.9995 | 0.878 | 0.855 | 176 | 0.22% | 5623 / 5108 / 1297 | 2.19 |
| 16K / 0.9995 | 0.834 | 0.809 | 186 | 0.23% | 5623 / 8805 / 1696 | 2.28 |
| 24K / 0.9995 | 0.787 | 0.764 | 197 | 0.24% | 5623 / 16168 / 2525 | 2.46 |
| 32K / 0.9995 | 0.758 | 0.738 | 204 | 0.25% | 5623 / 23570 / 3315 | 2.61 |
| 12K / 0.9999 | 0.977 | 0.935 | 159 | 0.05% | 8341 / 1678 / 2009 | 2.06 |

"Other" is pieces containing non-Han characters: punctuation, letters, digits and mixed pieces; byte and special tokens are not counted in these three columns. "Characters in 126 tokens" is estimated as 126 × characters / tokens (context 128 minus BOS/EOS), not measured on packed windows.

Statistics on benchmark development:

| Vocabulary / coverage | Tokens per candidate | Tokens per reference | Candidates with byte fallback | Left-context segmentation unchanged |
| --- | ---: | ---: | ---: | ---: |
| 8K / 0.9995 | 4.47 | 4.20 | 0.38% | 98.0% |
| 12K / 0.9995 | 4.06 | 3.67 | 0.38% | 96.3% |
| 16K / 0.9995 | 3.89 | 3.43 | 0.38% | 95.3% |
| 24K / 0.9995 | 3.73 | 3.24 | 0.38% | 93.8% |
| 32K / 0.9995 | 3.67 | 3.16 | 0.38% | 93.0% |
| 12K / 0.9999 | 4.42 | 4.12 | 0.03% | 98.0% |

"Left-context segmentation unchanged" means that when left context and candidate are encoded together, the leading tokens equal the left context encoded alone. When they differ, the candidate's first piece has merged with the end of the left context, and scoring as "left-context tokens + candidate tokens" needs separate handling at the boundary.

## Parameter allocation at the 20M budget

Computed for the vimeml-tiny-zh-v1 model family: d384, SwiGLU d_ff 960, learned positions 128, tied embedding and output layer, about 20M parameters total. Each layer has 4·d² + 3·d·d_ff + 2·d = 1,696,512 parameters. The layer count is the integer closest to the budget.

| Vocabulary | Layers | Embedding | Backbone | Total | Output layer share of per-token compute | Compute per Han (MACs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 10 | 3.15M | 16.97M | 20.16M | 15% | 20.9M |
| 12K | 9 | 4.72M | 15.27M | 20.04M | 23% | 18.3M |
| 16K | 8 | 6.29M | 13.57M | 19.91M | 31% | 17.2M |
| 24K | 6 | 9.44M | 10.18M | 19.67M | 47% | 15.9M |
| 32K | 4 | 12.58M | 6.79M | 19.42M | 64% | 15.0M |

Per-token compute = backbone weights + attention (2·128·d per layer) + output layer (V·d), multiplied by tokens per Han to give compute per Han. A larger vocabulary lowers compute per Han slightly, but the savings come mainly from the backbone, so model capacity shrinks with it. 12K / 0.9999 has the same parameters as 12K / 0.9995, but its worse compression gives 20.4M per Han.

## Observations

**1. Compression gains diminish quickly.** 8K to 12K cuts tokens per Han by 11%; 12K to 16K by 5%; 16K to 32K by only 9% in all. At the 20M budget, 16K costs a layer against 12K for only 5% more compression; at 24K and 32K the backbone has only 6 and 4 layers left and the output layer takes over half the compute.

**2. Coverage decides single characters; vocabulary size decides only multi-character pieces.** At coverage 0.9995, every vocabulary has the same 5,623 single characters; all extra slots go to multi-character pieces, which also get longer (mean 2.06 to 2.61 characters). So rare-character fallback does not change with vocabulary: of the 8,132 distinct Han characters in validation, 2,532 have no single-character piece; they are 0.045% of Han occurrences, for example 抟, 绗, 蛉, 佧, 訇, 钿.

**3. Coverage 0.9999 buys 8K compression with 12K parameters.** At 0.9999 single characters rise to 8,341, and with byte tokens and other fixed entries only 1,678 slots remain for multi-character pieces. Rare-character fallback falls to 0.009% (Han characters in validation without a single-character piece fall to 890), and candidates with byte fallback from 0.38% to 0.03%; but tokens per Han go from 0.878 back to 0.977, almost the same as 8K. 0.38% of candidates falling back is not worth giving up 11% compression.

**4. Larger vocabularies merge left context and candidate more often.** Unchanged left-context segmentation falls from 98.0% at 8K to 93.0% at 32K. This matters for IME scoring: scoring left-context tokens encoded alone followed by the candidate gives a segmentation different from what training saw.

**5. Nine in ten byte tokens are newlines.** The training sample is split into paragraphs and contains no newlines, so the vocabulary has none, and every paragraph newline in validation becomes a `<0x0A>` byte token. For 8K, byte tokens are 2.01% with newlines, 91% of them newlines; 0.19% without. The 2.25% seen in an earlier 12K probe has the same cause. A newline takes one token either as a byte token or as its own piece, so token counts, compression and context capacity are unaffected. But the formal tokenizer should make `\n` a user_defined_symbol so that it has a fixed id and is not mixed with byte tokens.

**6. Training time is not an issue.** Each variant trains in about 200–240 seconds locally (16 threads).

## Follow-up plan (decided 2026-11-06)

**20M model**: compare only 8K / 0.9995 and 12K / 0.9995. 12K is the most balanced: 20 more characters of context, 9% shorter candidates, only one layer fewer. 8K has one more backbone layer and less output-layer compute, which is exactly why Japanese V4 chose 8K. For Chinese the decision rests on BPC and Standard Chinese development results at equal compute, with single-seed differences read against the seed noise band.

**200M model**: one arm each for 16K, 24K and 32K. At this scale the embedding share is much smaller. At d1024, for example, a 32K embedding is 33.6M, about 17% of 200M, whereas at the 20M budget a 32K embedding is 65% of all parameters. So the cost of a large vocabulary is acceptable at 200M; exact d and layer counts follow once the 200M design is fixed.

**How the formal tokenizer differs from these**:

- The training sample includes the exported review-queue part, so it is trained after export-r1 is complete.
- `\n` and task markers (`<CTX>/<READ>/<OUT>` or the Chinese model's counterparts) are user_defined_symbols.
- All other settings stay: Unigram, identity, byte_fallback, coverage 0.9995, hard vocab limit.
- The tokenizers here are exploratory only and are not used to train models.
