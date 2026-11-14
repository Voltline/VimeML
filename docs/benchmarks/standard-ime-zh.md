# Standard Chinese IME benchmark

2026-11-06. Sampling, item-by-item AI expert review, human spot checks and real Rime candidates are all done; the frozen version is `ime-standard-zh-v1-r1`: 1,456 development items and 1,445 blind items. Full pinyin is the primary track; abbreviated pinyin (initials) is a separate track with 2–6-character targets only. The label level is `ai_expert_reviewed`, with no claim of native-speaker gold. Scripts and protocol are in Git; item text, candidates and review records stay in ignored local directories.

Blind was first used on 2026-11-11 in the [vimeml-tiny-zh-v1 confirmation](confirmation-zh-v1-plan.md) (run-01, passed; results in the [report](../reports/zh-v1-20261111/blind-and-baselines.md)). Since then it is only a fixed reference test set: it can no longer be called a new blind test, nor be used for tuning.

## Composition and roles

| Source | Type | License | Development | Blind |
| --- | --- | --- | ---: | ---: |
| CrossWOZ (`df82c9f`) | Task dialogue, user turns | Apache-2.0 | 478 | 479 |
| RiSAWOZ (`1e2d3be`) | Task dialogue, user turns | MIT | 493 | 483 |
| KdConv (`653db76`) | Knowledge chat, turns without knowledge attributes | Apache-2.0 | 485 | 483 |
| Total | | | 1,456 | 1,445 |

Each source and split is stratified by target length (2–3, 4–6, 7–12 characters) and reported separately. The main summaries are micro per source plus a macro that weights the three sources equally. Source quotas and length strata are a fixed experimental design, not the proportions of real keyboard traffic.

All three sources are public datasets released on GitHub; they may also have been crawled from the web into training corpora. The overlap screen is below.

## Sampling and readings

- Fixed seed 20261106, 1,000 draft items per source. At most one item per dialogue; within each source, dialogues are shuffled by the seed and split into development and blind.
- The given left context is only the part of the current message before the target (at most 64 characters); earlier messages are not concatenated.
- The target is the end of a clause: a contiguous Han run starting at a jieba word boundary and ending at punctuation or the end of the message, 2–12 characters long.
- The target covers no annotated entity or slot value: CrossWOZ dialog_act/user_state, RiSAWOZ user_actions/belief_state, KdConv entity names and short attribute values.
- The target does not start with a particle that attaches leftward (的, 了, 呢, …). Each (left context, target) pair occurs once; within a source, the same target occurs at most 3 times.
- Readings come from pypinyin on the whole sentence. Characters where g2pM disagrees are recorded in `reading_conflicts` for review.
- Keys: full pinyin writes ü as `v`; abbreviated pinyin takes the first letter of each syllable and covers only 2–6-character targets.
- Sampling reads no LM scores, candidate lists or candidate difficulty.

## Review and freezing

**AI expert review**
- All 3,000 drafts were reviewed one by one, without seeing candidates or model scores.
- The review sees the left context, the target, a few following characters and the reading. Records in `expert-review-r1/specs/chunk-*.txt` and `chunk-*.jsonl`, one line of reasoning per item.
- Items were deleted when:
  - the target contains an unannotated proper name: branch or shop names, sights, people, works, landmarks, street names;
  - the target boundary cuts a word;
  - the target or source text has an unrecoverable typo.
- Reading fixes include 长 read as chang when it means duration, 得 read as dei when it means "must", and 查查 read as cha.
- Acceptable variants include only same-sound, same-length spellings:
  - 他家/它家 (referring to a shop), and 他/她 when the left context has no referent;
  - 其他/其它, complement 的/得, 预定/预订, 订/定, 那个/哪个 in questions.
- These variants are not claimed to be exhaustive. This version has at most 3 per item.

**Human spot checks**
- Questions the AI could not settle went to the project owner, decided one by one on a local web page. Decisions recorded in `expert-review-r1/human-review-r1.json`.
- 谁 is read shei throughout, 50 items.
- City and district names are no longer deleted wholesale but decided per item: 26 of 34 kept.
- Of 6 individually uncertain items, 2 were deleted.
- Human spot checks are not a full native-speaker review.

**Freezing**
- `freeze` has four preconditions:
  - every draft has exactly one review record;
  - no `unsure` remains;
  - human notes have been merged into the review records;
  - the output directory is new.
- Result: 2,864 kept, 37 readings fixed, 99 deleted (28 of them by human override).
- Deleted items are not replaced.
- 87 items have readings different from the draft, including the 谁 unification.
- 136 items carry acceptable variants.
- The abbreviated track has 1,954 items.
- The manifest records these label fields:
  - `ai_expert_reviewed=true`
  - `human_spot_review=true`
  - `native_chinese_review=false`
  - `labels_formal_gold=false`
- A frozen version is never modified. A fix means a new version with a description of what changed, and every compared model must use the same label version.

## Candidate pools

**Engine**
- librime 1.17.0 (Windows prebuilt `33e7814`) plus rime-ice (`99662ba`), called through ctypes by `rime_candidates.py`.
- Three changes make the engine fixed and input-dependent only:
  - the user dictionary is off, so earlier queries do not affect later rankings;
  - the Lua translators for date, time, lunar calendar, UUID, numbers, calculator and English are removed;
  - emoji and Traditional conversion are off.
- Everything else is unchanged, including the rime-ice dictionary, the spelling algebra with abbreviations and the Lua filters that reorder candidates.

**Pool construction**
- Take Rime's candidates for the whole input, keep those that are all Han and as long as the target, in Rime order.
- If fewer than 20, pad with k-best paths over the rime-ice dictionary from `rime_lattice.py`. Padded candidates are tagged `origin=lattice`.
- The reference is never injected into the pool.

**Reference coverage** (a hit means the target or any acceptable variant is in the pool)

| Track | Split | Items | Rime first candidate hits | Rime candidates only | Pool (20) hits |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | development | 1,456 | 1,001 (.688) | 1,043 (.716) | 1,420 (.975) |
| Full | blind | 1,445 | 1,045 (.723) | 1,065 (.737) | 1,407 (.974) |
| Abbreviated | development | 990 | 142 (.143) | 338 (.341) | 403 (.407) |
| Abbreviated | blind | 964 | 147 (.152) | 321 (.333) | 401 (.416) |

Abbreviated input is highly ambiguous, and a 20-candidate pool covers only about four in ten references. Misses stay in the denominator, so abbreviated Top-1 tops out near .41. This track is compared only with itself and not pooled with full pinyin.

The Rime first-candidate column is the engine's own baseline, not a model score, and does not count as consuming blind.

## Overlap screen

`screen_overlap_zh.py` searches the vimeml-tiny-zh-v1 training corpus for verbatim overlap with item text. Scope:

- the 21 non-dropped stage-a-v2 shards, keep and review parts both, a conservative scope;
- LCCC v2 train and review.

Two patterns per item, matched as exact substrings with Aho-Corasick:

- the whole contiguous Han run containing the target, at least 8 characters, insensitive to surrounding punctuation;
- the whole message, at least 8 characters.

Shorter fragments are mostly common phrases and are not screened. The method cannot find paraphrases or near-duplicates, so no claim of zero contamination is made.

**Results**

| Split | Items screened (Han run / whole message) | Items hit (Han run / whole message) |
| --- | ---: | ---: |
| development | 889 / 1,385 | 57 / 19 |
| blind | 886 / 1,374 | 44 / 27 |

- 128 items hit at least one pattern: CrossWOZ 38, RiSAWOZ 44, KdConv 46.
- The matched fragments are almost all 8–11-character pleasantries or generic questions, which are common on the web and in chat anyway; they do not indicate that a source dataset was included in the corpus.
- Only 1 match reaches 12 characters or more, in development: a KdConv turn paraphrasing encyclopedic text.
- No sign of whole dialogues being reposted on the web.

**Handling**
- No items deleted; `ime-standard-zh-v1-r1` is not revised.
- Hit item ids are in `frozen-r1/overlap/overlap.jsonl`. Reports can list "hits removed" as a separate sensitivity subset.

## Scoring and blind use

- Models only rerank the pool. The main score is the candidate's log-probability sum given the left context, FP32, no EOS, no truncation. Ties keep the pool's original order.
- Metrics follow [IME metrics v2](metrics-v2.md). An item with a window or roundtrip problem falls back to the pool's original order. Items whose reference is not in the pool stay in the denominator.
- Development can be used repeatedly for debugging and selection.
- Blind can be used only when both hold: labels are frozen, and a preregistered plan naming every compared model and the scoring rules has been committed.
- The first opening of blind is recorded as consumption. After that it can continue as a fixed reference test, but can no longer be called a new blind test, and its scores cannot be used for further tuning and then claimed as independent validation.

## Scoring script and Rime baseline (development)

The scoring entry point is `score_standard_ime_zh.py`, with two subcommands.

**score**
- Without `--scores`, scores the pool's original order, which gives the Rime baseline.
- With `--scores`, reads a JSONL with one `{"id", "candidates", "scores"}` per line:
  - `candidates` must match the frozen pool order exactly;
  - candidates are reranked by score from high to low, ties keeping pool order;
  - items with null `scores` fall back to pool order and are counted in `fallbacks`.
- Writes per-item ranks and these summaries:
  - exact Top-1/3/5/10/20, pool-conditional Top-1, MRR;
  - micro per source and the equal-weight macro over three sources;
  - length strata and source × length strata;
  - the sensitivity subset without overlap hits.
- Refuses to run if the output directory exists.
- `--split blind` requires `--blind-plan`. The plan must be committed to Git, unmodified in the working tree, and name `ime-standard-zh-v1-r1`; the result records the plan's commit.

**compare**
- Paired comparison of two scorings via `vimeml.benchmarks.statistics.paired`.
- Each dialogue contributes at most one item, so a dialogue cluster is a single item and cluster and item tests are equivalent.
- Reports Top-1 and Top-3, overall and per source.

**Rime baseline**

The pool's original order; outputs in `outputs/benchmarks/ime-standard-zh-v1/baselines-r1/`:

| Track / scope | Items | Top-1 | Top-3 | Top-10 | In pool | Pool-conditional Top-1 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full, total | 1,456 | 1,001 (.688) | .900 | .963 | 1,420 (.975) | .705 | .800 |
| Full, CrossWOZ | 478 | 310 (.649) | .918 | .977 | 469 | .661 | .788 |
| Full, RiSAWOZ | 493 | 357 (.724) | .913 | .974 | 486 | .735 | .824 |
| Full, KdConv | 485 | 334 (.689) | .870 | .938 | 465 | .718 | .787 |
| Full, 2–3 chars | 523 | 397 (.759) | .950 | .990 | 522 | .761 | .859 |
| Full, 4–6 chars | 467 | 348 (.745) | .927 | .974 | 460 | .757 | .841 |
| Full, 7–12 chars | 466 | 256 (.549) | .818 | .921 | 438 | .584 | .693 |
| Full, overlap hits removed | 1,392 | 950 (.682) | .898 | .963 | 1,358 | .700 | .797 |
| Abbreviated, total | 990 | 142 (.143) | .236 | .345 | 403 (.407) | .352 | .206 |
| Abbreviated, 2–3 chars | 523 | 80 (.153) | .273 | .426 | 254 | .315 | .235 |
| Abbreviated, 4–6 chars | 467 | 62 (.133) | .195 | .255 | 149 | .416 | .173 |

- Full-pinyin three-source macro Top-1 is .687; abbreviated .143.
- Full-pinyin Top-1 equals the hits of the pool's first entry (1,001); likewise for abbreviated (142).
- Reranking was verified with a constructed score that gives the reference the highest score: Top-1 equals exactly the 1,420 in-pool items.
- Long sentences are the main difficulty: full-pinyin Top-1 at 7–12 characters is only .549, but Top-3 is .818. That is where model reranking has room.

## Build entry points

Raw sources are in `datasets/ime-standard-zh-v1/raw/` and the engine in `datasets/ime-standard-zh-v1/engine/`. Outputs under `outputs/benchmarks/ime-standard-zh-v1/`: `draft-r1/`, `expert-review-r1/`, `frozen-r1/` (with `overlap/`), `rime/`.

```powershell
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/prepare_standard_ime_zh.py extract --out outputs/benchmarks/ime-standard-zh-v1/draft-r1
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/prepare_standard_ime_zh.py freeze --human outputs/benchmarks/ime-standard-zh-v1/expert-review-r1/human-review-r1.json --out outputs/benchmarks/ime-standard-zh-v1/frozen-r1
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/rime_candidates.py deploy
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/rime_candidates.py query <F>/rime-full.jsonl <F>/cand-full.jsonl
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/rime_lattice.py index
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/rime_lattice.py pool <F>/rime-full.jsonl <F>/cand-full.jsonl <F>/pool-full.jsonl --track full
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/screen_overlap_zh.py <F>/items.jsonl <F>/overlap <corpus files>...
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/score_standard_ime_zh.py score <F> <B>/rime-full-development --track full
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/score_standard_ime_zh.py score <F> <OUT> --track full --scores <model>.jsonl --name <model>
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/score_standard_ime_zh.py compare <B>/rime-full-development <OUT> <OUT>/compare.json
```

`<B>` is `outputs/benchmarks/ime-standard-zh-v1/baselines-r1`.

For the abbreviated track, replace `full` with `abbrev`. Completed outputs are not rebuilt; any parameter change uses a new output directory.
