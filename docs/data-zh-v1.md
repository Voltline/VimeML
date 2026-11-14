# vimeml-tiny-zh-v1 corpus (frozen as export-r1, 2026-11-10)

> **Status: frozen.** The corpus was frozen as `zh-v13-export-r1` on 2026-11-10, after the review part was exported and `finish` ran: 104 data files, 45.0 GB, in private storage. Stage B quality scoring was not run; the frozen corpus is stage A plus the review queue. The sections below were written while the work was in progress and keep the intermediate figures; the final export is described under [Final export](#final-export-zh-v13-export-r1) and [Follow-ups](#follow-ups).

vimeml-tiny-zh-v1 is the Chinese model, trained separately at about 20M parameters: d384, context 128; the vocabulary comparison is 8K (10 layers) against 12K (9 layers), see the [tokenizer vocabulary comparison](reports/zh-v1-20261106/tokenizer-vocab.md). This page records the corpus sources, rule filtering (stage A) and the human review queue.

## Sources

| Source | Scope | Use |
| --- | --- | --- |
| FineWeb2 `cmn_Hani` | Revision `af9c13333eb981300149d5ca60a8e9d659b276b9`; every 18th of the 370 train files, starting from the 9th: 21 files, 84.6 GiB, 35.59M documents | Web LM |
| LCCC-base | thu-coai Weibo dialogues, local copy; original train/valid/test split kept | Chat |

- FineWeb2 is downloaded with `scripts/corpus/download_zh_fineweb2.py`, which checks each LFS SHA-256 and writes `datasets/fineweb-2-cmn-hani/manifest.json`.
- FineWeb2 is already globally MinHash-deduplicated per language, so only exact deduplication is done here: comparing full text with whitespace removed, 7,156 duplicates across the 21 shards.

## Traditional script and regional wording

- Detection: first build the set of Han characters that OpenCC `t2s` would rewrite; a document whose Han characters contain more than 8% of these is classed as Traditional.
- Traditional documents are converted to Simplified with OpenCC 1.4.2 `tw2sp`. This also converts Taiwanese wording, for example 軟體→软件, 網路→网络, 影片→视频.
- Converted documents are tagged `script=hant`. 4.45M of the automatically kept documents were converted, about 15%.
- The model learns Simplified output only; Traditional support is left to conversion in the IME.

## Stage A rule filtering

Rules are in `src/vimeml/corpus/zh_filter.py`, term lists and domains in `configs/zh-filter-terms.toml`; running and statistics use `scripts/corpus/zh_stage_a.py run|dedup|report`. Each document gets one of three outcomes:

- **Drop**: any of the following:
  - a blacklisted top-level domain;
  - a tier-1 domain (overseas political media);
  - a host name containing gambling pinyin (`.gov.cn` and `.edu.cn` are exempt);
  - a host name impersonating a government site, for example `x.gov.cn.spam.cn`;
  - one strong term, or two different terms;
  - fewer than 100 Han characters;
  - Han characters below 40% of non-whitespace characters;
  - more than 30% repeated lines.
- **Review**: not dropped, but exactly one weak term, or a tier-2 domain (mainstream Hong Kong and Taiwan media). Weak terms are common words, for example 菠菜, 赌场, 色情, 台独.
- **Keep**: everything else.

The political term list is a v0 draft, not an authoritative list. Before the IME launches, a continuously maintained moderation list should also be added at the IME level.

Gambling-SEO collocations (for example pk10, 彩票平台, 娱乐官网, 备用网址) were chosen as follows:

- Source: a 2026-11-05 trial run on 3 shards (3.8M documents), plus the first batch of human labels.
- Terms that normal text also uses often in the trial were left out, for example 开奖, 快三 ("快三年了"), 充值, 六合, 电投 ("国家电投").
- Four ambiguous terms are weak terms only, for example 娱乐平台, 电子游艺.

**Spam sites** are counted by `report`; either condition lists a site:

- a host with at least 20 documents, at least 30% of them dropped for terms;
- a registered domain with at least 10 hosts whose labels alternate letters and digits at least twice, for example `u07bgqr7k.com`.

Remaining documents on spam sites are dropped wholesale and not reviewed. In the first batch of human labels, not one document of these two kinds was valuable text.

`spam_allow` is a hand-checked allow list: cool3c, 土巴兔, 39 健康网, 摇篮网.

### stage-a-v2 results

| Outcome | Documents | Share | Han characters |
| --- | ---: | ---: | ---: |
| Keep | 30,035,226 | 84.39% | 31.75B |
| Review | 1,025,495 | 2.88% | 1.61B |
| Drop | 4,532,043 | 12.73% | 10.02B |

- Main drop reasons: gambling terms 7.05%, pornography terms 2.48%, blacklisted TLD 2.09%, tier-1 domain 0.51%, political terms 0.44%, too short 0.44%, gambling pinyin in host 0.41%, illegal-service terms 0.37%, government impersonation 0.03%. A document can have several reasons.
- 33,750 spam sites with 629K remaining documents; 1,075 spam-farm domains with 239K remaining documents. The two overlap; these documents are dropped wholesale at export.
- Against v1: v1 lacked the new SEO collocations, `.pw` and government-impersonation rules and kept 30,690,514 documents. v2 keeps 655K fewer (2.1%).

## Human review queue

The review queue is produced by a local queue export tool, 200 items per batch. All human-review code (queue export, review page, label merge) is kept locally and not in the repository. Generation:

- remove exact duplicates and spam-site documents from the review documents;
- sort by id and shuffle with fixed seed 20261105, so a partially reviewed queue is still a random sample;
- export all candidates, no sampling.

Size of review-v2:

- 954,758 items, 4,774 batches;
- by reason: single weak term 663,551, tier-2 domain 298,655, with a small overlap;
- another 70,702 documents on spam sites were dropped directly and not queued.

There are three labels: keep, don't keep, unsure. Criteria:

- **Keep**: the matched term is normal usage and the whole text is clean.
- **Don't keep**: any gambling, pornography, inserted ads, garbled splicing or keyword stuffing, however much normal content there is.
- **Unsure**: only when genuinely undecidable.

Two labeling tools exist: a static prototype page and an online review service. The online service exports `labels.jsonl`, one `{"id", "batch", "queue", "reviewer", "label", "t"}` per line.

Both kinds of labels are merged with a local merge tool, which also reports inter-annotator agreement. Labels are keyed by document id, so old labels stay valid when the queue is re-exported.

## LCCC chat (lccc-v2)

`scripts/corpus/zh_lccc.py` does the following:

1. Remove spaces left by character-level segmentation, keeping only spaces between Latin letters or digits.
2. Convert Traditional to Simplified at the same threshold as the web text and apply the same term lists: matches are dropped; utterances with exactly one weak term are written separately to `review.jsonl`.
3. Each turn becomes one sequence: BOS + utterance + EOS.
4. In train, utterances of 10 characters or more that also occur in validation or test are removed. Shorter common replies such as 哈哈, 谢谢, 好的 are not leakage and stay in train.
5. An utterance repeated n times keeps ⌈√n⌉ copies, for example 哈哈 41,089→203, 谢谢 37,718→195, 晚安 20,291→143. This suppresses flooding while keeping the frequency order among common replies.

| Split | Utterances | Characters |
| --- | ---: | ---: |
| train | 16,326,323 | 209M |
| validation | 56,620 | 677K |
| test | 28,852 | 347K |

- Train has 14.89M distinct utterances.
- 129,468 removed for overlapping validation or test.
- 125,742 removed for terms, 3,129 sent to review.

## Final export (zh-v13-export-r1)

`scripts/corpus/zh_export.py` exports the corpus as zstd (level 9) Parquet for upload, into `outputs/zh-corpus/export-r1/`:

- `web`: stage-a-v2 kept documents, exact duplicates removed, then every document on spam sites and spam farms removed (same rules as `zh_stage_a.py report`; `spam_allow` domains exempt).
- `review`: documents from the review queue, only those where every vote is keep (rule under [Follow-ups](#follow-ups)). The pass list is produced by the local review tools and is not in the repository.
- `chat`: LCCC v2 train, validation and test. `review.jsonl` is not exported.
- `finish`: writes `SHA256SUMS` and `manifest.json`. The manifest records the hash of every input file and of the script itself.
- 0.05% of web and review documents go to validation, fixed by `sha256('zh-v13-validation\0' + id)`. The split depends only on id, so adding shards later does not change it.
- Each file holds about 4 GB of raw text, 1.6–1.8 GB compressed. A failed download only needs that one file again.
- Columns: web and review documents `id, host, script, han, text`; chat `text, dialogue, turn, script`.

Web and chat were exported on 2026-11-06:

| Part | train | validation | Han characters (train) |
| --- | ---: | ---: | ---: |
| Web (automatically kept) | 29,217,343 documents | 14,681 documents | 30.87B |
| LCCC chat | 16,326,323 utterances | 56,620 utterances (test 28,852) | 209M characters |

- Web input was 30,035,226 documents; removed 7,121 exact duplicates, 561,870 spam-site and 234,211 spam-farm documents.
- 42 GB in all: 39 train files, 22 validation, 1 test. Largest file 1.82 GB.
- The review part and `finish` followed on 2026-11-10 (see below).

Shards added later go into a new version (for example r2): export only the new shards and deduplicate exactly across batches against r1. r1 does not change.

## Follow-ups

- **Review part and `finish`: done 2026-11-10.** A document from the review queue (tier-2 domains included) enters training only if every vote is keep: one don't-keep or unsure vote drops it, and unreviewed documents are dropped too (decided 2026-11-06). The whole queue is only about 3% of the automatically kept documents, so wrongly dropping some costs far less than further rechecking. Human votes and AI-assisted votes (DeepSeek, labeled with a separate reviewer name) count equally. After a first review and a second review of political content, 381,935 documents passed (train 381,744, validation 191). export-r1 then had 104 data files, 45.0 GB, uploaded whole to private storage.
- **Stage B quality scoring: not run.** The frozen corpus did not need it for the 20M or 200M runs.
- **Formal tokenizers and 20M training: done.** The [vocabulary comparison](reports/zh-v1-20261106/tokenizer-vocab.md) (tokenizers only) chose 8K and 12K for 20M (coverage 0.9995, 12K most balanced) and 16K, 24K, 32K for 200M. The formal tokenizers were trained on the frozen train split with `\n` as a user_defined_symbol. The eight-arm 20M experiment is in the [experiment plan](training-zh-v1-plan.md) and the [training report](reports/zh-v1-20261110/training.md). The Chinese benchmark is frozen, see the [Standard Chinese IME benchmark](benchmarks/standard-ime-zh.md); the overlap screen against this corpus found no source dataset included in the corpus.

All outputs are in `outputs/zh-corpus/` and not in Git: `stage-a-v2/`, `review-v2/`, `lccc-v2/`, `export-r1/`.
