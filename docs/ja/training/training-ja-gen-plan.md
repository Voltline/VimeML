# Japanese generation line (ja-gen) plan

> The plan is registered in Chinese, and the [Chinese version](../../zh-CN/ja/training/training-ja-gen-plan.md) is authoritative.

Drafted 2026-11-25. Committing this page registers stages 0 and 1. Those stages build the benchmark, data, readings and tokenizer only; they train no model. For stage 2 onward, this page sets the route. Each stage's arms, configs and decision rules are added here before that stage trains. Registered content is not edited; later changes are recorded as deviations at the end.

## Goals

- One phone-size Japanese model of about 20M parameters does three jobs: **kana → text sentence generation**, reranking AzooKey candidates, and prediction. Generation is required.
- Vime's candidate list is the union of AzooKey candidates and model-generated candidates. The same model scores and orders every candidate in the union.
- There is no separate Japanese 800M. A Chinese–Japanese–English 800M comes last, under its own plan (final item of the [Chinese v2 release plan](../../zh/release-zh-v2-plan.md)).
- [V4.5 ensemble distillation](training-v45-plan.md) is withdrawn and will not run.
- V4.2 epoch 20 stays deployed until a new model passes confirmation.

## Grounds

**The Japanese line is limited by the candidate pool.**

- On Standard development (728 items), the correct answer is missing from AzooKey's top 20 for 98 items, so no reranker can recover them. The remaining Top-1 errors are mostly notation variants. In the [error decomposition](../reports/analysis-20261021/ime-errors.md), pool misses are 48% of errors, notation 35–44% and semantic errors 8%.
- The reranking recipe is saturated. None of the six V4.4 arms had an effect, and the seed band is 501–523 ([V4.4 results](training-v44.md)). The four-seed ensemble reaches 535, so V4.5 could gain about +21 at most.

**What worked on the Chinese line:**

| Finding | Chinese result |
| --- | --- |
| Reading-conditioned generation, then rescoring with the same model's LM score | Public pinyin set, all inputs Top-1 31.0 → 49.1 ([v2-20m confirmation](../../zh/reports/zh-v2-confirmation-20261120/report.md)); rescoring alone 39.5 → 51.3 ([rescoring report](../../zh/reports/zh-pinyin-rescore-20261115/report.md)) |
| At 20M: plain LM pretraining first, then reading post-training; no joint pretraining from scratch | 20M joint pretraining generated worse than post-training (private −2.2, p 0.0004; [joint-pretraining record](../../zh/reports/zh-pinyin-pt-20261115/record.md)) |
| 200M joint reading pretraining, then distillation into 20M | The 200M rescoring with its own scores is clearly stronger ([200M evaluation](../../zh/reports/zh-200m-pinyin-eval-20261116/report.md)); the distilled 20M gains +2.1 on the public set ([distillation evaluation](../../zh/reports/zh-kd-eval-20261117/report.md)) |
| Colloquial post-training | Colloquial set 39.0 → 42.3, public set 49.1 → 51.4 ([colloquial confirmation](../../zh/reports/zh-colloq-confirmation-20261123/report.md)) |
| Beam search bucketed by key position | Only hypotheses that consumed the same number of keys compete, as in Vime; beam 16 is the default ([beam report](../../zh/reports/zh-pinyin-beam-20261115/report.md)) |
| The same 20M architecture as v1 | The Core ML batched-KV v4 package (`prefix`, `score`, `scored`) and the Vime integration are reused as is; the Mac reruns the pipeline on a new bundle ([v2-20m brief](../../zh/mac/mac-zh-v2-20m.md)) |

**Why V4.1's reading task failed while this plan differs:**

| Item | V4.1 | This plan |
| --- | --- | --- |
| Use of the reading | Conditioning input when scoring AzooKey candidates | Generates its own candidates and scores the union |
| Task weight | 0.2 × conversion loss | 320 of 512 samples per update are reading samples (Chinese r2: 352/512) |
| Reading source | Approximate readings from two dictionaries, disagreements kept | Only spans where Sudachi and UniDic agree on the whole reading |
| Ordering | Conditional score only | Generation score + λ × LM score |

V4.1 fell from 505 to 461 on Standard ([V4.1 record](training-v41.md)). In stage 2, the Standard rerank difference between the reading arm and the LM control tests this directly.

## Settings carried over from the Chinese line (not ablated again)

- **Architecture**: vocabulary 12,288, d384, nine layers, six heads, SwiGLU 960, context 128, about 20M parameters. The 200M uses the Chinese 200M shape (d1024).
- **Reserved ids**: pad 0, unk 1, bos 2, eos 3, `<CTX>` 4, `<READ>` 5, `<OUT>` 6, `\n` 7; 256 byte pieces.
- **Task format**: `[BOS, CTX] + context + [READ] + reading + [OUT] + text + [EOS]`, with loss only after `<OUT>`. The reading is a hiragana string, written the way users type it (long-vowel ー, small kana and sokuon as is).
- **Search**: beam 16, bucketed by consumed reading position, top 10, plus log P(EOS) at the end; KV cache required.
- **Rescoring**: `generation score + λ × LM score`. λ starts at 1.6, is tuned on development only, and is frozen afterwards.
- **Recipes**: 20M reading post-training follows Chinese r2-cont (60,000 steps, lr 1.5e-4); distillation follows kd-cont (α 0.5); colloquial post-training follows c40.
- **Not done**: typo correction, ANE, vocabulary-size sweeps, multi-seed ensemble distillation.

## Stage 0: evaluation

**0.1 Candidate-pool coverage.**

- From the existing Standard development export (N-best 20), report pool misses by length and by left context.
- The Mac exports development once more at N-best 50 and 100. The converter, dictionary and emoji revisions are those of v1, and nothing else changes. The results separate items that a deeper pool recovers from items only generation can recover.
- Development only. No other set is exported or read.

**0.2 New public set `ime-kana-ja-v1` (kana → text generation plus reranking).**

- **Sources**: WRIME ver2 and JMultiWOZ. Only authors, users and dialogues that never entered an earlier Japanese benchmark are used (v1 development, v1 reference, the historical regression draft and v2 blind).
- **Span rules** as in Standard: contiguous spans of 4–40 characters within one message, readings of 4–64 kana, left context of at most 64 characters, no Latin letters, digits or emoji. At most 20 items per author and one per dialogue.
- **Strata**: span length 4–10 / 11–20 / 21–40 characters × with or without left context, drawn in rotation.
- **Readings**: an item is kept only if Sudachi and UniDic agree on the whole reading (`src/vimeml/benchmarks/readings.py`). There is no manual review. Readings are normalized to hiragana.
- **Overlap check** against Japanese training text, using exact containment and 12-character anchor similarity as in Standard. Text added in stage 1 is scanned too. Matching items are removed before freezing.
- **Split**: development and test, half each, by an author/dialogue hash, about 600 items each. **Test is sealed.** It is read once, in the preregistered confirmation of stage 5.
- **Candidates**: the Mac exports AzooKey candidates once for both splits with the deployed setting (N-best 20). Test candidates are exported but not scored.
- **Metrics**:
  - primary: Top-1 of the final displayed list, judged notation-leniently with `notation.py`;
  - also reported: exact Top-1, Top-3, and results by length and by left context;
  - statistics clustered by author/dialogue, following [metrics v2](../benchmarks/metrics-v2.md).
- **Licenses**: WRIME is CC BY-NC-ND 4.0 and JMultiWOZ is CC BY-SA 4.0. Item text stays local; the repository and public releases carry aggregate results only.

**0.3 Development baselines** (`ime-kana-ja-v1` development and Standard development):

- AzooKey order;
- V4.2 reranking AzooKey candidates, the current deployment;
- the pool ceiling (share of items whose answer is in the pool).

**0.4 Standard development (728 items)** stays as the reranking guard, for development use only.

**Stage 0 is done when** the coverage report exists; `ime-kana-ja-v1` is frozen with its manifest SHA256 recorded on this page; both baselines are measured; and test is unread.

## Stage 1: data, readings, tokenizer

**1.1 Corpus.**

- The 20M builds on the V4 LM and chat corpora (`corpus-v4`, `corpus-v41-chat`), as V4.2 did.
- The 200M needs more text. Shards of FineWeb-2 `jpn_Jpan` are selected at a fixed revision and stride, then go through the same cleaning and quality screen as the Chinese data. The target is a training token count comparable to the Chinese 200M; the shard list and token count are recorded on this page when built.
- The LLM quality screen covers only data the 20M uses, the same scope as Chinese v2.

**1.2 Readings.**

- Every training span gets readings from both Sudachi and UniDic. It becomes a reading sample only if the two agree on the whole span; otherwise it is used as plain LM text only.
- Spans that cannot be typed in kana (Latin letters, digits, symbols) are not reading samples.
- Before the build, report the agreement rate and the kinds of disagreement on a 10,000-sentence sample. If agreement is below 60%, stop; a different reading method needs a registered deviation first.

**1.3 Sample kinds and starting mix** (512 samples per update; stage 2 does not ablate the mix):

| Kind | Content | Per update |
| --- | --- | ---: |
| `plain` | Plain LM text | 192 |
| `span` | Span conversion with same-message left context, as in real typing | 192 |
| `sentence` | Whole-sentence conversion without context | 128 |

**1.4 Tokenizer**: SentencePiece unigram, 12,288 pieces, trained on the Japanese training sample, with the reserved ids above. Every hiragana and katakana character must be a piece of its own, and validation text must round-trip without loss.

**1.5 Output**: token store `ja-kana-v1-12k`, with its manifest SHA256 pinned in the configs and recorded here; a copy goes to private storage.

**Stage 1 is done when** the reading agreement report exists; tokenizer round-trip and kana coverage checks pass; and the token store is built with its SHA256 pinned.

## Stage 2 onward (route; rules are added before each stage trains)

1. **20M**: pretrain a plain LM from scratch with the 12K vocabulary (V4.2's data and budget), then reading post-training. Compare the reading arm with the LM control on `ime-kana-ja-v1` development generation and on Standard development reranking. Standard reranking guard: the reading arm stays at or above 501, the low end of the V4.2-recipe seed band.
2. **200M**: joint reading pretraining and rescoring with its own scores; then distillation into 20M with the kd-cont recipe.
3. **Colloquial post-training**: c40 recipe. Its data and colloquial development set are registered in that stage.
4. **Confirmation**: preregistered; `ime-kana-ja-v1` test is read once, against the current deployment (AzooKey plus V4.2 reranking).
5. **Mac / Vime**: rerun the batched-KV v4 pipeline. Vime merges AzooKey and generated candidates and scores them together. Generation does not block keystrokes, following the Chinese product decision of 2026-11-21 (show the reranked list at once, generate after a pause or in the background).
6. **Last**: the Chinese–Japanese–English 800M, under its own plan.

## Deviations

**D1 (2026-11-25, 0.2 sources): document-level instead of author-level exclusion.** All 60 WRIME authors and 64 of the 65 JMultiWOZ users had already entered an earlier benchmark or draft candidate pool, so author-level exclusion is not feasible. Instead, every post and dialogue in any earlier Japanese benchmark, draft, review copy or candidate import under `artifacts/benchmarks` is excluded (19 files, 8,000 documents), as is any (left context, reading) pair equal to an earlier item. Development and test are split by a seeded draw of authors and users (WRIME 30/30, JMultiWOZ 33/32), so the two splits share no author; both share authors with earlier sets. Each split has 600 items (WRIME 400 + JMultiWOZ 200). Build script `scripts/ja/benchmarks/prepare_kana_ime_ja.py`, seed 20261125.

**D2 (2026-11-25, 0.2 overlap check): freeze now, deduplicate on the training side in stage 1.** The freeze scanned the existing Japanese training text: the `corpus-fast-v1` training set (25.18M lines) and `corpus-v41-chat` (482K sentences). 45 items matched and were removed before selection. Text added in stage 1 (FineWeb-2 and others) does not change the frozen set. Instead, training documents that overlap a message of this set are dropped at build time, using the same scan, and the count is recorded under stage 1. Messages shorter than 12 normalized characters are not scanned.

**D3 (2026-11-25, 0.2 metric definition, clarification):** notation-lenient judging calls `notation.relation` between the candidate and each accepted answer; `=` or `A` counts as correct. A notation variant counts even when no answer is in the AzooKey pool, which generated candidates (outside the pool) require. The older `notation.error_class` counted such cases as pool misses, so lenient figures on this page run slightly higher than in earlier reports.

**D4 (2026-11-25, 1.3 reading rule): known-difference table.** A whole-span disagreement no longer always keeps a span out of the reading samples. The two readings are compared word by word: both tokenizations are cut at the character boundaries they share, and each minimal segment is normalized (symbol readings and non-kana characters removed, ヂ/ヅ folded to ジ/ズ). A span becomes a reading sample when every segment agrees, or when every differing (surface, UniDic reading, Sudachi reading) triple is in `configs/ja/ja-reading-table.tsv`, which gives the reading to use. The table is built by `scripts/ja/corpus/reading_pairs.py`:

- **Pairs**: seed 20261127, 100,000 sentences each from the V4 LM corpus, V4.1 chat and `ja-gen-export-v1` (same sources and sentence rule as 1.2); 216,348 typeable sentences, 2,065 distinct differing triples.
- **LLM judgment**: two passes, with the two readings shown in opposite order, each with up to three example sentences. Choices: one reading, both (then the one more users type), context-dependent, neither. Consistent verdicts: Sudachi 1,069, UniDic 337, both 30, context 29, neither 23; inconsistent 577.
- **Admission**: only triples seen at least 3 times (357), minus single-kanji words of at most two characters and hand exclusions (37). Every admitted triple was reviewed by hand (`configs/ja/ja-reading-review.toml`): readings of 94 surfaces set by hand, of which 22 correct a consistent verdict (most of them UniDic verdicts without rendaku or with a rarer reading, such as 意気込み イキコミ and 御社 ゴシャ) and 72 resolve inconsistent pairs; 10 surfaces whose reading depends on the sentence or the person are excluded. Result: 303 rows (Sudachi 186, UniDic 11, both 10, hand 96). Sentences with a differing triple outside the table stay plain LM text.
- **Effect** on the 1.2 sample (6,843 typeable sentences): raw agreement 94.8%, normalized 97.0%, with the table 99.0% (LM 99.0, chat 99.7, FineWeb-2 98.5). The 1.2 sample and the pair sample are drawn from the same sources with different seeds and may share sentences.
- **Dictionary pin**: fugashi loads the full `unidic` package instead of unidic-lite whenever it is installed; `readings.py` now refuses to run on any dictionary other than unidic-lite.

**D5 (2026-11-25, 1.3 reading rule): agreed readings users rarely type.** In the 1.3 smoke build both dictionaries agreed on 私 ワタクシ, 日本 ニッポン, 明日 アス, sentence-final 言う ユウ, and 何 ナン even before を/が/か (何をする read なんをする); on the 189K-sample smoke set 私 ワタクシ alone came 1,353 times and 私 ワタシ never. D4 covers only disagreements, so these readings would have become most of the training readings for the most common words. For a segment that is exactly one of these words, the agreed reading is replaced by the typed one: 私 → ワタシ (except before 共/ども, 私共 stays ワタクシドモ), 日本 → ニホン, 明日 → アシタ, 言う ユウ → イウ, 何 → ナニ (except before だ/で/と/の/て/な/じ, where ナン stays). The rules are `AGREED` in `scripts/ja/corpus/kana_samples.py`. They came from reading the 1,500 most frequent readings of kanji segments in the smoke set and spot checks of common words with two readings; other words in that list read as users type them. Compounds such as 日本一 (Sudachi ニッポンイチ, UniDic ニッポン + イチ) now disagree and are left out. `ime-kana-ja-v1` is frozen with the dictionary readings: about 24 development items have an input written with one of these readings (私 8, 明日 6, 何 9, 日本 1, some of them correct as written). The set is not changed; stage 2 reports these items separately.

**Stage 0 done (2026-11-25):** [coverage and baseline report](../reports/ja-gen-stage0-20261125/report.md).

**Freeze record:** `ime-kana-ja-v1` release_id `65a164c1505d4913adf864871143627b`, manifest SHA256 `8d51fc00f6def29cce512e99609d35565ec25ee23a67b5c7c31df143516f9a9d`. Development 600 items (63 authors/users), test 600 (59). Reading dictionaries fugashi 1.5.2 / unidic-lite 1.0.8 / sudachipy 0.7.0 / sudachidict-core 20260723.1. Strata were drawn in rotation; "short × no context" ran out of eligible items at 66 (development) and 65 (test), while every other stratum has 85–125.

**1.1 record (2026-11-25): `ja-gen-export-v1`.**

- **Shards**: FineWeb-2 `data/jpn_Jpan/train/` at revision `af9c1333`, every 8th of the 175 files starting at index 4: 22 files, 82.1 GiB, 49,217,965 documents. Script `scripts/ja/corpus/download_ja_fineweb2.py`.
- **Stage A** (`scripts/ja/corpus/ja_stage_a.py`, rules `src/vimeml/corpus/ja_filter.py` and `configs/ja/ja-filter-terms.toml`): keep 95.66%, drop 3.78%, review 0.57%. Drop reasons, as a share of all documents: adult terms 2.42, short (under 100 Japanese characters) 0.81, spam terms 0.41, gambling terms 0.19, host terms 0.11, low Japanese share 0.06, hate terms 0.03; the rest are under 0.01 each. Exact duplicates: 56. Spam hosts (at least 20 documents and at least 30% term drops): 12,112. Spam farms: 12 domains, after 13 hand-checked sites (personal-homepage hosts, forums, blogs, developer and hobby sites) were added to `spam_allow`. Sixteen shards ran on this machine and six on a second machine with the same code; on the shard run on both, the outputs are identical.
- **Export** (`scripts/ja/corpus/ja_export.py`): drops review-queue documents (279,081), duplicates (56), spam-host documents (236,402), spam-farm documents (1,289) and benchmark documents (14,686, deviation D2: 3,766 messages of at least 12 normalized characters from the 27 `evaluation_items.json` files of the Japanese benchmarks, candidate imports included). Result: train 46,804,382 documents and 55.72G Japanese characters; validation 23,310 documents and 27.8M characters (0.05% by id hash); 78 files, 60.3 GB. Manifest SHA256 `e5b5b1b8a06f8f00fd585b18595376ae8c826e4d4b5f25e60b0fb6fe1e8b8af1`. The token count is recorded with the tokenizer (1.4).

**1.2 record (2026-11-25): agreement 94.8%, gate passed.** `scripts/ja/corpus/reading_agreement.py`, seed 20261126, 10,000 sentences of 4–64 characters (V4 LM corpus 4,000, V4.1 chat 2,000, `ja-gen-export-v1` 4,000).

| Source | Typeable share | Whole-reading agreement | Typeable sentences with an unknown word |
| --- | ---: | ---: | ---: |
| V4 LM corpus | 64.1% | 95.4% | 5.9% |
| V4.1 chat | 92.7% | 98.3% | 3.3% |
| FineWeb-2 export | 60.6% | 91.4% | 8.8% |
| All | 68.4% | 94.8% | 6.2% |

The most frequent disagreements (UniDic / Sudachi) are 日本・日本人 (ニッポン・ニッポンニン / ニホン・ニホンジン), 人 after a noun (一般人 ニン / ジン), 私 (ワタクシ / ワタシ), 身近 (ミヂカ / ミジカ, the same sound in two kana spellings), 世界中 (チュウ / ジュウ), 土曜日 (ヒ / ビ), お母さん・お父さん (ハハ・チチ / カア・トウ) and 家 (イエ / ウチ). In most of these unidic-lite gives the less common reading, so whole-span agreement keeps such sentences out of the reading samples; 日本人 in particular never becomes a reading sample. The rest is partly symbol handling, not reading: Sudachi reads some symbols as キゴウ, and the two analyzers write ～ and … differently. The build compares readings with symbols removed and ヂ/ヅ folded to ジ/ズ.

**1.3 record (2026-11-26): `kana-samples-v1`.** `scripts/ja/corpus/kana_samples.py` (992cfad), 851 jobs split by input file over several CPU jobs, no code changes. Manifest SHA256 `81442c6ed509febaf9d2acf8c6d2f5b2aed9a977e9c6b329a5427ffa44abf939`, 91 GB.

| Source | train | validation | span : sentence (train) |
| --- | ---: | ---: | --- |
| `ja-gen-export-v1` (half of the paragraphs) | 484,505,322 | 240,109 | 74 : 26 |
| V4 LM corpus | 17,942,934 | 184,870 | 82 : 18 |
| V4.1 chat | 393,929 | 4,288 | 62 : 38 |

Web train covered 759M paragraphs; spans average 13.1 characters and 16.4 kana of reading, and 81.9% have left context. Kinds are drawn 3:2, but fewer paragraphs hold a usable whole sentence, so the store has fewer sentence samples. Training draws each kind by its own count, so the per-update mix is unaffected.

**D6 (2026-11-26, 1.5 token store): one store with seven kinds; the 20M uses only the LM and chat reading samples.** `ja-kana-v1-12k` (`scripts/ja/tokenizer/encode_kana.py`, `configs/ja/data-ja-kana-store.toml`) holds:

- `plain` (V4 LM lines), `chat` (V4.1 chat) and `web` (the half of the `ja-gen-export-v1` paragraphs not selected for reading samples, so no paragraph is in both), each as BOS + line + EOS cut into windows of at most 128 tokens;
- `span` and `sentence` (LM and chat reading samples);
- `web_span` and `web_sentence` (web reading samples).

Following 1.1, the 20M builds on the V4 LM and chat corpora, so it uses the first four kinds only; the three web kinds are for the 200M. The 20M's `plain` 192 in the 1.3 table is split into LM 175 + chat 17 (V4.2 had about 9% chat windows). LM and chat train lines identical to any validation line are dropped. Validation reading samples are thinned by `doc_hash` (LM and web 5%, chat all).

**1.4 record (2026-11-26): tokenizer `ja-unigram-12288`.** `scripts/ja/tokenizer/train_kana.py`, model SHA256 `61d71460606eed32f0a976611e5fde0dca12efeecb413c516838911a49d3aa53`.

- **Vocabulary**: 12,288 with the registered reserved ids, 256 byte pieces and 181 single-kana pieces.
- **Training sample**: seed 20261128; web 1,200,000 lines, LM 300,000, chat 80,000 and readings 400,004; 73,018,345 characters, 7,513,344 of them readings.
- **Check**: all 945,577 lines round-trip losslessly (`skipped_meta_space` 0).
- **Compression**: validation documents 34,896,184 characters → 19,550,174 tokens, about 1.785 characters per token; validation readings 990,683 characters → 516,321 tokens, about 1.919.

**1.5 record (2026-11-26): token store `ja-kana-v1-12k`.** `scripts/ja/tokenizer/encode_kana.py`, manifest SHA256 `f5ec6fa71caa410029e9956ea0c4ab5d564eef2d042835660408b047cf6eeab1`. 265,308 validation lines were used for exclusion; 16 `plain` train lines matched and were dropped.

| Kind | Train rows / windows | Train tokens | Validation rows | Passes in 845,000 200M updates |
| --- | ---: | ---: | ---: | ---: |
| `plain` | 25,200,351 | 627,842,532 | 260,496 | 0.805 |
| `chat` | 468,424 | 5,871,887 | 4,980 | 43.3 |
| `web` | 782,198,431 | 19,887,552,580 | 389,838 | 0.501 |
| `span` | 14,899,631 | 476,724,257 | 10,147 | 0.907 |
| `sentence` | 3,437,232 | 119,777,945 | 3,292 | 3.93 |
| `web_span` | 359,005,743 | 10,539,878,249 | 8,818 | 0.414 |
| `web_sentence` | 125,499,579 | 3,615,066,517 | 3,129 | 0.754 |

`web` (the half of the `ja-gen-export-v1` paragraphs not selected for reading samples) is 19.89B tokens.

## Stage 2 registration (2026-11-26)

Trainer `src/vimeml/training/joint_ja_kana.py` (entry `scripts/ja/training/train_ja_kana.py`):

- Each update takes each kind's count of samples. Within a kind the order is a bijection seeded by (seed, kind, pass), so any step's batch depends only on (seed, step).
- Under data parallelism every process takes its rows of the same global batch and normalizes by global supervised-token counts, so the gradient equals the single-device one.
- Checkpoints use the zh13 / zh200m formats.
- `tokens` normalization weighs every token equally; `per_kind` is Σ (kind count / batch) × mean loss of the kind.
- Every run uses final.pt only; no checkpoint is chosen on development.

**2.1 20M** (12K / d384 / L9, 20.0M parameters).

- **Pretraining**: random initialization, V4.2's budget of 1,081,942 updates, each `plain` 466 + `chat` 46, token-normalized, cosine 6e-4 → 6e-5, warmup 1,000, prefix crop 0.30. Four seeds: 1219827974 (V4.2's), 20261126, 20261127, 20261128 (`configs/ja/train-ja-kana-20m-pt-s{1..4}.toml`).
- **Reading arm** (`train-ja-kana-20m-read-s{n}.toml`): continues each seed's pretraining final.pt (weights only, fresh AdamW) with the zh r2-cont recipe: 60,000 updates, cosine 1.5e-4 → 1.5e-5, warmup 1,000, seed 20261130. Each update: `plain` 175 / `chat` 17 / `span` 192 / `sentence` 128, `per_kind`.
- **LM control** (`train-ja-kana-20m-lm-s{n}.toml`): same start and schedule, each update `plain` 466 / `chat` 46, token-normalized.
- **Evaluation** (after training, development only):
  - `ime-kana-ja-v1` development. The final list is AzooKey N-best 20 ∪ beam-16 generation bucketed by key position (top 10), ordered by generation score + λ × LM score; λ starts at 1.6 and is tuned on development only. Primary metric: notation-lenient Top-1. Also reported: exact Top-1, Top-3, results by length and by left context, and the D5 items separately, with author/dialogue-clustered statistics. The LM control cannot generate, so it reranks the AzooKey pool with its LM score.
  - Standard development (728 items): both arms rerank with the LM score only.
  - Each seed is reported, plus the four-seed mean.
- **Decision**: the reading recipe is kept if both hold:
  - (a) the reading arm's four-seed mean lenient Top-1 on kana development is above V4.2 reranking (85.2) and above the LM control (paired by seed);
  - (b) its four-seed mean Standard reranking is at least 501.

  Otherwise the result is recorded and a fix is registered before the 20M distillation. The 200M is unaffected.
- **2.1 result (2026-11-27): the reading recipe is kept.** The reading arm's four-seed mean lenient Top-1 on kana development is 87.17 (V4.2 reranking 85.2, LM control 85.79, higher at 4 of 4 seeds when paired). Standard reranking mean 504.50 (seeds 501–508); LM control 515.25. See the [2.1 report](../reports/ja-gen-stage2-20m-20261127/report.md).

**2.2 200M joint reading pretraining** (`configs/ja/train-ja-kana-200m-joint.toml`, the zh 200M shape 12K / L15 / c128, 193.6M parameters).

- Random initialization, the zh 200M budget of 845,000 updates, cosine 6e-4 → 6e-5, warmup 1,000, seed 20261126, `per_kind`.
- 832 samples per update:
  - plain text 512: `web` 464, `plain` 24, `chat` 24;
  - reading 320: `web_span` 176 + `span` 16, `web_sentence` 112 + `sentence` 16 (the 3:2 span:sentence of 1.3).
- From the 1.3 counts, about 0.41 passes over `web_span` and 0.76 over `web_sentence`. `web` passes are recorded with the 1.5 window count.
- Evaluated as in 2.1 (with its own generation and LM scores), plus Standard reranking. It is the teacher of the 20M distillation (2.3, registered later).

## Stage 3 registration (2026-11-26)

**3.1 Colloquial data** (`ja-colloq-v1-12k`, `ja-colloq-v1-p85-12k`):

- **Classifier**: the zh `zh_colloq.py` rule pre-scan and hashed char 1–3-gram logistic regression (`scripts/ja/corpus/ja_colloq.py`). From the `train` split of `ja-gen-export-v1`, 1,000 paragraphs are drawn at equal probability from each of the four rule bands (`zero/low/mid/high`), 4,000 in all, and labeled by DeepSeek Flash: 1,189 `label=3` (zero 123 / low 140 / mid 347 / high 579), 2,011 `spoken` positives.
- **Export**: from the `train` split, paragraphs with `p_chat ≥ 0.70`, thinned uniformly to 1/3 by `doc_hash("ja-colloq-chat-sub", doc_id)` (`spoken_units = 0`): 25,702,238 paragraphs, 651M Japanese characters (validation 531,653 paragraphs, 13.45M characters). The `heldout` split (2% of documents) never enters training.
- **Quality audit**: 2,000 paragraphs drawn uniformly from the exported train set (seed `20261126`) and checked against the zh quality-screen categories (hate / abuse / sexual / political / junk); no full LLM screen. If the not-ok rate is high and concentrated in one category, a rule or small classifier for that category is added and the store re-encoded.
- **Encoding** (`configs/ja/data-ja-colloq-v1.toml`, `scripts/ja/tokenizer/encode_ja_colloq.py`): one reading pass builds two `vimeml_ja_kana_v1` stores, the full 1/3 pool `ja-colloq-v1-12k` (`p_chat ≥ 0.70`) and its high-threshold subset `ja-colloq-v1-p85-12k` (`p_chat ≥ 0.85`), 30% plain text / 70% readings (`span:sentence = 3:2`, D4/D5 rules). Manifest SHA256 values go into the configs after the build, before training.

**3.2 Colloquial benchmark `ime-kana-ja-colloq-v1`** (registered before labeling):

- **Source**: paragraphs with `p_chat ≥ 0.50` in the `heldout` split of `ja-gen-export-v1` (decoupled from the training threshold), at most one per document, 1,000 candidates from each of the four rule bands, 4,000 in all (`scripts/ja/benchmarks/prepare_kana_colloq_ja.py`, build seed `20261201`). Nothing is drawn from WRIME / JMultiWOZ, so the set does not overlap `ime-kana-ja-v1`.
- **LLM confirmation**: the candidate file keeps only `(id, text)` and is shuffled as a whole (seed `20261203`, no band, score or split), and is labeled with the same DeepSeek Flash prompt and settings as `ja-colloq label-v1`.
- **Inclusion**: `label == 3` only (everyday chat, `include_label2_dialogue = false`). Only if a split has fewer than 600 items after overlap removal are `label == 2` dialogue lines with `kind in {chat, fiction}` added, recorded as a deviation.
- **Split after labeling, independent of labels**: fixed `split_seed = 20261202`; `doc_hash("ime-kana-ja-colloq-v1/id-split/20261202", case_id) % 2` assigns `development` (0) and `test` (1). With at most one item per document, the splits share no document.
- **Overlap check and freeze**: 12-character normalized anchor containment and similarity `≥ 0.85` scans against training text (`corpus-fast-v1` train and the colloquial export train set, skipping the item's own document); hits are removed before freezing. Items are then drawn round-robin by `rule_band × left context × length band` (`selection_seed = 20261202`), 600 each for `development` and `test`. **`test` is sealed** and read once, in the stage 4 preregistered confirmation.

**3.3 Arms and decision rules**:

- **Start**: `final.pt` of the 2.3 `20m-kd-cont`; 30,000 more updates with 200M teacher distillation (`α = 0.5, T = 1.0`), cosine `1e-4 → 1e-5`, warmup 1,000, seed `20261205`, 512 samples per update (`plain 175 / chat 17 / span 192 / sentence 128`, `per_kind`).
- **Three arms**:
  - `colloq-ctrl` (`configs/ja/train-ja-kana-20m-colloq-ctrl.toml`): no colloquial data (continuation control);
  - `colloq-c40` (`configs/ja/train-ja-kana-20m-colloq-c40.toml`, `p70`): 208 colloquial samples per update (about 40.6%) from `ja-colloq-v1-12k`;
  - `colloq-c40-p85` (`configs/ja/train-ja-kana-20m-colloq-c40-p85.toml`, `p85`): 208 colloquial samples per update (about 40.6%) from `ja-colloq-v1-p85-12k`.
  - In the two colloquial arms the 512 per update are 304 from `ja-kana-v1-12k` (`plain 97 / chat 17 / span 114 / sentence 76`) + 208 from the colloquial store (`plain 78 / span 78 / sentence 52`).
- **Primary metric**: notation-lenient Top-1 on `ime-kana-ja-colloq-v1` development after self-reranking (`generation score + λ × LM score`, λ frozen at its 2.3 value), also reported by the four rule bands (`zero/low/mid/high`).
- **Guards**: (a) `ime-kana-ja-v1` development lenient Top-1 not significantly below `kd-cont` (author/dialogue-clustered paired sign-flip, `p < 0.05`); (b) Standard development reranking at least 501 (the low end of the V4.2 seed band) and not significantly below `kd-cont`.
- **Decision**: the default colloquial arm is `colloq-c40` (`p70`). `p85` replaces it only if `colloq-c40-p85` is significantly above `colloq-c40` on the primary metric (clustered paired sign-flip, `p < 0.05`) and passes every guard. The chosen arm replaces `kd-cont` for the stage 4 confirmation only if it is significantly above `colloq-ctrl` on the primary metric and passes every guard; otherwise `kd-cont` stays.

**3.1 record (2026-11-26):**

- **Quality audit**: all 2,000 sampled export rows checked by DeepSeek Flash: ok 1,934 / not ok 66, 3.30% (sexual 24, junk 15, abuse 13, political 9, illegal 3, hate 2). In the `p_chat ≥ 0.85` subset, 25 of 844 (2.96%). No rule added, no re-encoding.
- **Two colloquial stores built** (`encode_ja_colloq.py`, 509,310 validation lines excluded):
  - `ja-colloq-v1-12k` (`p70`, manifest SHA256 `b824b16c2c43bc7f502000cf60a6a96969870f46bfc4aa266244ba41fdcca256`): train `plain 7,379,112 / span 9,238,622 / sentence 3,598,323`; validation `plain 160,070 / span 197,183 / sentence 77,514`.
  - `ja-colloq-v1-p85-12k` (`p85`, manifest SHA256 `8cf9ab8e722749ed1e4181390b000902fc5a24f023bc1de93fdcbfb77e83a5bc`): train `plain 3,158,150 / span 3,870,936 / sentence 1,453,397`; validation `plain 68,190 / span 82,886 / sentence 31,145`.
  - Both `manifest_sha256` values are pinned in `configs/ja/train-ja-kana-20m-colloq-c40.toml` and `configs/ja/train-ja-kana-20m-colloq-c40-p85.toml`.

**3.2 freeze record (2026-11-26):**

- **4,000 candidates labeled** (DeepSeek Flash, labels SHA256 `3d1d1917eb01b58e06677c70787b9f7154e318bcb64dfd3aa3368b1838f2c5f1`): `label=0` 391, `label=1` 683, `label=2` 953 (`chat 750 / fiction 173 / formal 16 / marketing 14`), `label=3` 1,973 (`chat 1,972 / fiction 1`).
- **`label == 3` only** (the dialogue fallback was not needed): the overlap scan removed 275 of the 4,000 candidates (144 of them `label=3`). With `split_seed = 20261202` the usable `label=3` items are `development 911` (`zero 187 / low 202 / mid 235 / high 287`) and `test 918` (`zero 172 / low 216 / mid 270 / high 260`).
- **Frozen**: `ime-kana-ja-colloq-v1` release_id `f0db62b9b21443e39ef38cbb0c296ead`, manifest SHA256 `6538b128474cb938cc942aa645b18ea8282d3218e43a33a77ba0186e6dc66e9f`; `development 600` (`zero 151 / low 134 / mid 141 / high 174`), `test 600` (`zero 142 / low 136 / mid 148 / high 174`); no shared document or group; **`test` sealed**.
- **Development candidate pool** (2026-11-27): Mac handoff for development only, AzooKey export settings as for `ime-kana-ja-v1` (converter `d59a28e4c7ca`, N-best 20, typo off, Zenzai off), all 600 items; pool `kana-ja-colloq-v1-development-n20`, manifest SHA256 `2cfc6f4f76a006a9891dbf317aff39379a969361b47d96e2782563ea98d59759`. No candidates were exported for test.
- **AzooKey order baseline** (development): mean pool size 14.71, exact Top-1 55.50, lenient Top-1 72.83; pool ceiling exact 82.83, lenient 93.50.
