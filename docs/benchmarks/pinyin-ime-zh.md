# Public Chinese pinyin-generation test set ime-pinyin-zh-v1

**What it is.** Built and frozen on 2026-11-14 as `ime-pinyin-zh-v1-r1`.

**Purpose.** It rechecks, on public data, the comparison between pinyin generation and the production Rime path. Until now that comparison had only been run on the author's private pinyin benchmark.

**Code and text.** The build script is `scripts/benchmarks/prepare_pinyin_ime_zh.py`. Item text stays in ignored directories and is not committed.

## Composition

| Source | Type | License |
| --- | --- | --- |
| CrossWOZ | Task dialogue, user turns | Apache-2.0 |
| RiSAWOZ | Task dialogue, user turns | MIT |
| KdConv | Knowledge chat, turns that do not quote knowledge attributes | Apache-2.0 |
| CPED | TV-series dialogue | Apache-2.0 (repository license; the scripts' copyright is not covered, so the text is used locally only) |

**Sampling.**

- 300 dialogues per source.
- One sentence per dialogue.
- Split development / test by a dialogue hash, half each.

After freezing:

| Split | Sentences | Full pinyin | Mixed | All initials | 5–7 chars | 8–12 chars | 13–20 chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| development | 578 | 578 | 2,288 | 578 | 234 | 230 | 114 |
| test | 576 | 576 | 2,284 | 576 | 229 | 225 | 122 |

**Test is not read for now.** It is kept for a later confirmation.

## Build rules

**Dialogues and exclusions:**

- Loading and filtering follow `ime-predict-zh-v1`:
  - only people's turns are used;
  - KdConv turns that quote knowledge are skipped;
  - annotated entities and slot values are protected.
- Only dialogues that never entered `ime-standard-zh-v1` (its draft) or `ime-predict-zh-v1` (any split) are used.
- Dialogues flagged by the prediction benchmark's 16-gram leak scan are dropped whole.

**Sentences:**

- A sentence is a maximal Han run bounded on both sides by the message ends or punctuation.
- It must:
  - be 5–20 characters long;
  - not touch a protected value;
  - not start with a particle;
  - not repeat the text of another sentence in the set.
- The target length rotates over 5–7, 8–12 and 13–20 characters. The rule is relaxed when a dialogue has no sentence of the wanted length.

**Readings:**

- A sentence is kept only if pypinyin, pypinyin over jieba words and g2pM agree on every character.
- There is no manual review.
- ü is written v.

**Keys** (same format as the private benchmark):

- One full-pinyin input.
- One all-initials input: the first letter of each syllable, with zh/ch/sh shortened to z/c/s.
- Up to four mixed inputs:
  - words come from jieba with HMM off;
  - each word is abbreviated whole, independently, with probability 0.5;
  - duplicates are dropped, and so is any input that happens to abbreviate every word.
- The private benchmark has an all-initials input for 45% of its sentences. Here every sentence has one, so that kind can be read on its own.

**Overlap with the training corpus:**

- Sentences of 8 or more characters (737 sentences) were searched verbatim in `export-r1/train` (45,925,410 documents).
- 46 were found and dropped, 10–13 per source.
- Shorter sentences are common phrases and are not screened.

## Limitations

- **Synthetic keys.** The keys are not real typing.
  - The abbreviation rule shortens whole words.
  - Real users also shorten only some syllables of a word.
- **Unreviewed readings.** Three readers agreeing does not guarantee a correct reading.
- **Dialogue style.** Dialogue corpora lean formal and still differ from everyday chat input.
