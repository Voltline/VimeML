# Chinese prediction benchmark ime-predict-zh-v1

Frozen 2026-11-11 as version `ime-predict-zh-v1-r1`. Built under the [preregistered plan](predict-zh-plan.md): four sources, 1,200 dialogues each, 600 each for development and blind. Labels are the original text with no human annotation; readings for the prefix-completion track are automatic, keeping only fragments where three reading tools agree. Build scripts are in Git; item text stays in ignored local directories.

Blind stays closed after freezing and is opened once only, in the confirmation plan for the final desktop model. Development is used for every decision in the plan.

## Sources

| Source | Version | Type | License | Candidate dialogues | Leakage hits (removed) | Selected |
| --- | --- | --- | --- | ---: | ---: | ---: |
| CrossWOZ | `df82c9f` | Task dialogue, user turns | Apache-2.0 | 5,012 | 674 (13.4%) | 1,200 |
| RiSAWOZ | `1e2d3be` | Task dialogue, user turns | MIT | 10,200 | 650 (6.4%) | 1,200 |
| KdConv | `653db76` | Knowledge chat, turns without knowledge attributes | Apache-2.0 | 3,500 | 1,563 (44.7%) | 1,200 |
| CPED | `1e4b81c` | TV drama lines, chit-chat | Apache-2.0 (repository license) | 11,835 | 106 (0.9%) | 1,200 |

- CrossWOZ, RiSAWOZ and KdConv are the same versions as the [reranking benchmark](standard-ime-zh.md). Dialogues that entered the `ime-standard-zh-v1` drafts (1,000 per source, covering its development and blind) are not candidates.
- CPED comes from `data/CPED/{train,valid,test}_split.csv` in the GitHub repository `scutcyr/CPED`, not via the Qianyan platform. The repository license is Apache-2.0 and the README has no separate data terms, only a request to cite the paper (arXiv:2205.14727). The lines come from broadcast TV dramas whose script copyright is outside that license, so the text is used locally only and never distributed with the repository or any handoff package.

### Choosing the chit-chat source

Checked in the plan's order:

| Candidate | Condition 1 (license) | Condition 2 (not LCCC, not in training list) | Condition 3 (leakage ≤ 2%) | Outcome |
| --- | --- | --- | --- | --- |
| NaturalConv | Met (non-commercial research only, no redistribution) | Met | 1,202 / 19,919 = 6.0% | Not used |
| DuLeMon | Not accepted (see [deviation](predict-zh-plan.md#deviations) 1) | — | Not scanned | Not used |
| CPED | Met | Met | 106 / 11,835 = 0.9% | **Selected** |

The choice looked only at license texts, source lists and scan totals, never at any model score.

## Leakage filtering

- Method as in v1's [leakage check](../reports/zh-v1-20261110/ime-benchmark.md#leakage-check): keep only Han characters in each turn, take every 16-character n-gram, and search verbatim in the training export `export-r1/train` (60 files, 45,925,410 documents, also Han only). Hashes are used only for prefiltering; every hit is confirmed verbatim.
- Any dialogue with a hit is removed whole, before sampling. Filtering used only n-gram matching; nobody read blind text.
- KdConv's hit rate is high. Its dialogues often cite encyclopedic knowledge about film, music and travel, sentences that are also common on the web, which is probably the main reason (not checked item by item). 1,937 dialogues remain, enough to draw 1,200.
- **CPED's scan sensitivity is lower**: lines are generally short, and only 5,281 of 11,835 dialogues have at least one turn long enough to give a 16-character n-gram. The rest cannot be checked this way.

## Items

Sampling follows the plan: seed 20261111; each source is shuffled by the seed and taken in order until 1,200 usable dialogues; these are sorted by `sha256(seed/dialogue id)` and assigned alternately to development and blind. Details:

- **Usable turns**: the human side's turns (user turns only in task dialogues; KdConv turns citing knowledge attributes excluded), 4 to 120 characters, at least 60% Han, with no links, markup or replacement characters.
- **Left context**: all earlier turns of the dialogue (both sides) plus the committed part of the current turn.
- **Track A**: 3 positions per dialogue, drawn by the seed from all qualifying positions. A position lies between two Han characters, is followed by at least 2 more Han characters before the next non-Han character, and its next 6 characters do not fall on an annotated entity or slot value. The true continuation runs to the next non-Han character, 5.3 characters on average.
- **Track B**: each turn containing a track A position is simulated once in full.
- **Track C**: 1 item per dialogue, the target a 2–6-character fragment ending at punctuation or the end of the turn, drawn as for the reranking benchmark's abbreviated track. Readings are accepted only when pypinyin (whole sentence), pypinyin (jieba-segmented) and g2pM agree; any disagreement anywhere in the clause moves on to the next fragment.
- 4 KdConv and 12 CPED dialogues were skipped for having fewer than 3 qualifying positions, and 1 KdConv dialogue for having no usable completion fragment.

## Size

| Source | Track A (dev / blind) | Track B turns (dev / blind) | Track C (dev / blind) |
| --- | ---: | ---: | ---: |
| CrossWOZ | 1,800 / 1,800 | 1,523 / 1,507 | 600 / 600 |
| RiSAWOZ | 1,800 / 1,800 | 1,475 / 1,468 | 600 / 600 |
| KdConv | 1,800 / 1,800 | 1,489 / 1,478 | 600 / 600 |
| CPED | 1,800 / 1,800 | 1,526 / 1,519 | 600 / 600 |
| Total | 7,200 / 7,200 | 6,013 / 5,972 | 2,400 / 2,400 |

Development and blind share no dialogues.

## Decoding and simulation details

These details were fixed before any model was scored and are recorded as [deviation](predict-zh-plan.md#deviations) 2 of the plan. Code in `src/vimeml/benchmarks/predict_zh.py`; the unit tests in `tests/test_predict_zh.py` check the decoder against exhaustive enumeration (with an unbounded beam, results equal enumerating every token sequence).

**Feeding left context to the model**

- Each model joins left context in its own training format. vimeml-tiny-zh trained each turn as a separate `BOS … EOS` sequence, so each earlier turn becomes `BOS turn EOS` and the current turn `BOS committed part`. External models separate turns with newlines.
- When it does not fit, the earliest whole turns are dropped, leaving 12 tokens for decoding. If the current turn alone is too long, only its end is kept.

**Decoding**

- **Token back-off**: when the committed part ends in a Han character and its last token is all Han, that token is backed off and every hypothesis must first spell it again. This lets the model continue by merging the last character with the continuation into one token, instead of being stuck at a token boundary. Backed-off characters do not count toward the candidate.
- Beam width 16: at each step each hypothesis takes at most 16 extensions, then the 16 highest-scoring overall are kept. Only all-Han tokens are allowed, at most 6 new characters.
- Every hypothesis at a token boundary with 2 to 6 new characters is a candidate; when one string has several paths, the highest-scoring path counts. The score is the sum of FP32 token log-probabilities.
- The top 5 candidates by score, none a prefix of another, are taken; ties sorted by string. Track A's confidence is the first candidate's score.

**Track C keystroke constraint**

- The reading table comes from the rime-ice single-character tables `8105.dict.yaml` and `41448.dict.yaml` (all readings of each character, ü written v).
- First syllable only: the candidate's first character must have a reading equal to that syllable; later characters are free; candidates are 2 to 6 characters.
- Initial of every character: the candidate's length must equal the keystroke count, and the i-th character must have a reading starting with the i-th letter (zh, ch, sh treated as z, c, s).
- The constraint is checked during expansion; incompatible hypotheses are discarded.

**Track B simulation**

- Typing units are jieba words (accurate mode, HMM off). Without a pick, the user types up to the end of the next word.
- Each Han character costs as many keystrokes as its pinyin letters, with pypinyin annotating each contiguous Han run as a whole; each non-Han character costs 1.
- Nothing is offered at the start of a turn, before anything is committed; after that, predictions are offered after every commit. Their left context is the earlier turns plus the committed part of the current turn.
- The top 3 candidates scoring at or above the threshold are shown. If one is a prefix of the remaining text, the user picks the longest (1 keystroke), possibly stopping mid-word; otherwise typing continues as above.
- The threshold grid is minus infinity plus the 5%, 10%, … 95% quantiles of the model's track A first-candidate scores, 20 values; the one with the highest keystroke savings on development is chosen, ties going to the higher threshold.
- Keystroke savings are computed over total keystrokes of all turns (not averaged per turn), with display count, pick count and pick rate also reported.

**Metrics**

- Track A: mean length of hit candidates means the length of the highest-ranked hit within the Top-3, averaged over items with a Top-3 hit.
- Every track's Top-k is reported as an overall per-item rate (micro) and an equal-weight average over the four sources (macro). The primary metric, track A Top-3, uses micro.

## Evaluation entry points

```bash
python -X utf8 scripts/benchmarks/predict_ime_zh.py run --out outputs/benchmarks/ime-predict-zh-v1/runs/<name>
python -X utf8 scripts/benchmarks/predict_ime_zh.py score --run outputs/benchmarks/ime-predict-zh-v1/runs/<name>
```

The step-0 external systems use the same entry point with different adapters (rules in the plan's [deviation](predict-zh-plan.md#deviations) 3):

```bash
python -X utf8 scripts/benchmarks/predict_ime_zh.py build-ngram    # character 5-gram, written to outputs/benchmarks/ime-predict-zh-v1/baselines/
python -X utf8 scripts/benchmarks/predict_ime_zh.py run --model char5 --out .../runs/char5-kn
python -X utf8 scripts/benchmarks/predict_ime_zh.py run --model hf --name Qwen/Qwen2.5-0.5B --out .../runs/qwen2.5-0.5b
python -X utf8 scripts/benchmarks/predict_ime_zh.py run --model hf --name Qwen/Qwen2.5-3B --dtype bfloat16 --out .../runs/qwen2.5-3b
```

`run` first checks the SHA256 of the four frozen files, reads development only, writes one jsonl line per item, and records in `meta.json` the model and tokenizer SHA256s, decoding parameters, rime-ice version and reading-table SHA256, and jieba and pypinyin versions.

## Files

The frozen version is in `outputs/benchmarks/ime-predict-zh-v1/frozen-r1/` (local, ignored):

| File | SHA256 |
| --- | --- |
| `items-a.jsonl` | `493fc4702b9dc37b1f645fcfdcc3efa515f840adcef7e1423c3d233b21eea87f` |
| `messages-b.jsonl` | `2640d6b573fb2e45996a7b6556a2aeafe9af3e42e96603588947ba75fb7a1c71` |
| `items-c.jsonl` | `dc88bc5f0a7e1ed2e3c0ee2e9ae7841ca2712e13a34151e293e2b205df21fb2e` |
| `manifest.json` | `002dc9a39113e1831b5d2352187a079e15705770310dc81de6e06a87e68f6415` |

`manifest.json` records the seed, sources, leakage scan totals, build statistics and each file's count and SHA256. A rebuild from the same raw data reproduces the three item files byte for byte.

## Build entry points

```bash
# Leakage scan: once for the four original candidate sources, once for CPED
python -I -X utf8 scripts/benchmarks/prepare_predict_ime_zh.py scan outputs/benchmarks/ime-predict-zh-v1/leak-scan-r1
python -I -X utf8 scripts/benchmarks/prepare_predict_ime_zh.py scan outputs/benchmarks/ime-predict-zh-v1/leak-scan-r2-cped --sources cped
# Build
python -I -X utf8 scripts/benchmarks/prepare_predict_ime_zh.py build outputs/benchmarks/ime-predict-zh-v1/frozen-r1 \
  --scan outputs/benchmarks/ime-predict-zh-v1/leak-scan-r1 outputs/benchmarks/ime-predict-zh-v1/leak-scan-r2-cped \
  --sources crosswoz risawoz kdconv cped
```

Raw data is in `datasets/ime-standard-zh-v1/raw/` (the first three sources) and `datasets/ime-predict-zh-v1/raw/CPED/`. The first scan's four sources included NaturalConv; it was not selected and is not read by the build.
