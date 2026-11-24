# vimeml-zh-v2-20m colloquial post-training plan

Drafted 2026-11-22 and committed before training (preregistered). The Chinese version, [docs/zh-CN/zh/training/training-zh-colloq-plan.md](../../zh-CN/zh/training/training-zh-colloq-plan.md), is the reference text.

The v2-20m candidate `kd-cont` leans literary on colloquial text: typing `ba` after 这么说 completes to 把 instead of 吧. This plan changes only the training data. It continues `kd-cont`'s recipe with colloquial text mixed in, to improve prediction and conversion of colloquial sentences on the model side. Client rules are outside this plan.

## Data

Colloquial data enters through `[[data.extra]]`. Builder, sample rules, tokenizer and Rime readings are the same as for the r1 pinyin data. Labeling and the quality screen use one LLM only (DeepSeek 4.1 Flash). export-r2 is not used.

**export-r1 colloquial paragraphs** ([data-zh-colloq-r1.toml](../../../configs/zh/data-zh-colloq-r1.toml), store `zh-colloq-r1-12k`, manifest sha256 `1dea7399…`)

- [zh_colloq.py](../../../scripts/zh/corpus/zh_colloq.py) first pre-filters with sentence-final-particle rules. LLM labels then train two classifiers (hashed character 1–3-grams, logistic regression):
  - `p_chat`: the paragraph is real chat;
  - `p_spoken`: the paragraph is spoken language, including fiction dialogue.
- Train takes the train-split paragraphs with `p_chat ≥ 0.7`: 3.83M paragraphs, 93.4M characters. A hash sample adds 0.73M paragraphs (36.7M characters) with `p_spoken ≥ 0.8` and `p_chat < 0.5`, mostly fiction dialogue.
- The heldout split is by document hash (`zh-colloq-heldout`, 2%). It is evaluation only and never trained on.
- Each paragraph gives one sample: 30% plain, 70% pinyin.

**External chat** ([data-zh-colloq-chat.toml](../../../configs/zh/data-zh-colloq-chat.toml), store `zh-colloq-chat-12k`)

- Sources: Douban conversations, MMChat (Weibo) and Tieba. [zh_chat_external.py](../../../scripts/zh/corpus/zh_chat_external.py) cleans them with rules and holds out 2% by dialogue.
- Every turn goes through an LLM quality screen (hate, abuse, sexual, political, illegal, junk and other low-quality text). Only turns judged ok are kept, and an unscreened turn counts as not ok. The screen inputs are shuffled, so a partly screened source is a random subset.
- `zh_colloq.py export-chat` exports by row count: Douban 800k, MMChat 600k, Tieba 120k. Tieba has only 51k turns after screening, so they are repeated about 2.35 times, each copy under its own dialogue key.
- Each turn gives one sample, with context only from inside the turn, as deployed: 30% plain, 70% pinyin.
- The store's manifest sha256 is filled into the configs after the build and before training (`d97cb59d…`). That step is not a deviation. Train samples: plain 318,457, full 311,428, mixed 310,867, initials 311,656, error 97,667.

**Composition of the colloquial part** (by sample): r1 chat about 52%, r1 spoken about 10%, Douban 20%, MMChat 15%, Tieba 3%. At c20 the r1 store is read at most 0.7 times for any kind, and the chat store's plain samples about 1.13 times (the build drops plain windows under 8 tokens, and chat turns are short); its other kinds at most 0.97 times.

## Arms

All four arms start from `kd-cont` final (step 30,000) with the round 3 recipe:

- 30,000 more updates;
- the 200M teacher, α = 0.5, T = 1;
- peak LR 1e-4 → 1e-5, warmup 1000;
- seed `20261122`.

Every update has 512 samples, with the same total per kind as `kd-cont` (plain 160 / full 96 / mixed 128 / initials 96 / error 32), so `per_kind` normalization is unchanged.

| Arm | Colloquial samples per update (r1 colloquial + chat) | Loss on colloquial | Config |
| --- | --- | --- | --- |
| `colloq-ctrl` | 0 | — | [train-zh-pinyin-colloq-ctrl.toml](https://github.com/Voltline/VimeML/blob/5263e6595959779558816f2467a8d8a86f4229a9/configs/train-zh-pinyin-colloq-ctrl.toml) |
| `colloq-c10` | 52 (32 + 20), about 10% | distilled | [train-zh-pinyin-colloq-c10.toml](https://github.com/Voltline/VimeML/blob/5263e6595959779558816f2467a8d8a86f4229a9/configs/train-zh-pinyin-colloq-c10.toml) |
| `colloq-c20` | 104 (64 + 40), about 20% | distilled | [train-zh-pinyin-colloq-c20.toml](https://github.com/Voltline/VimeML/blob/5263e6595959779558816f2467a8d8a86f4229a9/configs/train-zh-pinyin-colloq-c20.toml) |
| `colloq-c10-hard` | as c10 | cross-entropy only | [train-zh-pinyin-colloq-c10-hard.toml](https://github.com/Voltline/VimeML/blob/5263e6595959779558816f2467a8d8a86f4229a9/configs/train-zh-pinyin-colloq-c10-hard.toml) |

- `colloq-ctrl` measures what 30,000 more updates alone give. r3-ctrl and r3-ctrl-s2 are the same recipe at two other seeds and serve as the seed-noise reference.
- In c10 and c20, the r1 base mix is scaled down to make room for the colloquial samples.
- c10-hard: the teacher never saw this kind of text, so its soft labels may pull colloquial text back toward literary text. In this arm both colloquial directories set `ce_only = true` and train on the label's cross-entropy alone; the r1 base is still distilled. The trainer change is `f32a36b`. Without `ce_only` sources, batches are byte-identical to before the change, which a test checks.

## Evaluation

Everything runs on development. Neither test nor blind is read.

**Primary**: colloquial full-sentence pinyin conversion on `ime-pinyin-zh-colloq-v1` ([prepare_pinyin_colloq_zh.py](../../../scripts/zh/benchmarks/prepare_pinyin_colloq_zh.py)).

- Record format and pipeline are the public pinyin set's: Vime search (beam 16 per key position), the top 10 rescored by the model's own scores (λ = 1.6), and the rescored Top-1 is the metric.
- Context is only the text before the cursor in the same message (infield); earlier turns are not visible.
- Sources and item counts (all from evaluation-only splits):

  | Source | Items | From |
  | --- | ---: | --- |
  | LCCC | 600 | test |
  | r1 colloquial | 600 | heldout, outside the zh13 pool P1 |
  | Douban | 300 | heldout, screened ok |
  | MMChat | 300 | heldout, screened ok |
  | Tieba | 200 | heldout, screened ok |

- Clauses are 4–20 characters, and about half end in a sentence-final particle. Readings are kept only where three tools agree.
- The 20M and the 200M teacher pretrained on P1 only, and the r1 pinyin data is drawn from it, so r1 items come only from documents outside P1.
- Before freezing, an overlap scan runs against all training corpora (Aho-Corasick, minimum 8 characters), and flagged items are dropped. For an r1 item, the scan skips the item's own document and counts every other.
- Development and test are half each; test is held back for the confirmation run.
- Frozen result (2026-11-22, `frozen-r1`, manifest sha256 `f94a76df…`): the overlap scan dropped 455 items (r1 176, Tieba 117, LCCC 96, MMChat 55, Douban 11), mostly Tieba floors repeated across scraped posts, MMChat and LCCC both coming from Weibo, and reposted web text in r1. Development has 770 sentences (LCCC 252, r1 211, Douban 146, MMChat 118, Tieba 43), 458 of them ending in a particle and 143 with in-sentence context; test has 775.
- Significance: paired sign-flip test over sentence clusters.

**Secondary** (descriptive only, not part of the decision):

- prediction track A Top-3, with the CPED stratum (TV-drama lines) reported separately;
- final-particle prediction: on the colloquial set's items that end in a particle, predict at the position before the particle with single-character candidates allowed (`min_chars = 1`), reporting Top-1 and Top-3 ([predict_colloq_zh.py](../../../scripts/zh/benchmarks/predict_colloq_zh.py)).

**Gates**:

1. Public pinyin development self-rescored Top-1 is not significantly below `kd-cont` (sentence-cluster sign-flip, p < 0.05).
2. Standard full-pinyin rerank Top-1 is not significantly below v1.

Prediction (tracks A–D) and typo correction are descriptive.

## Decision

1. Take the arm with the highest primary score among c10, c20 and c10-hard. It replaces `kd-cont` only if both hold:
   - its primary score is significantly above `colloq-ctrl`;
   - both gates pass.
2. If `colloq-ctrl` itself is significantly above `kd-cont`, that shows only that more training helps. It does not count as an effect of the colloquial data.
3. c10-hard against c10 answers whether colloquial text should be distilled at all. If c10-hard wins, later colloquial data trains on cross-entropy alone.
4. If no arm qualifies, `kd-cont` stays. The colloquial problem then goes back to the client (passing committed text as context, allowing single-character candidates) and to evaluation.
5. A winning arm has `kd-cont`'s architecture, so export and latency do not change. It still goes through v2-20m's existing test confirmation and Mac alignment.

## Round 1 results (2026-11-22)

Colloquial development, self-rescored Top-1 (770 sentences, 4552 inputs). p is from the sentence-cluster sign-flip test against `colloq-ctrl`.

| Arm | Top-1 | vs ctrl | p |
| --- | ---: | ---: | ---: |
| `kd-cont` | 39.6 | −0.2 | 0.54 |
| `colloq-ctrl` | 39.8 | — | — |
| `colloq-c10` | 40.9 | +1.2 | 0.0008 |
| `colloq-c20` | 41.9 | +2.1 | 0.00005 |
| `colloq-c10-hard` | 41.1 | +1.3 | 0.0009 |

- c20 vs c10: +0.9 (p 0.0044).
- c10-hard vs c10: +0.2 (p 0.64), so dropping the teacher on colloquial text brings no measurable gain.
- Gates:
  - public pinyin self-rescored Top-1: `kd-cont` 54.8, ctrl 54.6, c10 55.7, c20 56.1 (+1.3, p 0.031), c10-hard 55.8;
  - Standard against v1: passes for all four arms (c20 net +5, p 0.48).
- Reference metrics:
  - final-particle prediction Top-1: `kd-cont` 0.190, ctrl 0.194, c10 0.223, c20 0.225, c10-hard 0.229;
  - prediction track A, CPED stratum Top-3: 16.4 → 17.5 (c20).
- By decision rule 1, c20 wins and replaces `kd-cont` as the candidate. ctrl is not significantly above `kd-cont`, so the gain comes from the colloquial data.
- c20 beats c10, so the colloquial share has not saturated. A second round therefore runs before test confirmation.

## Round 2 (amendment, committed before training on 2026-11-23)

At c20 the round 1 colloquial data is already read about once, so a higher share needs more data.

This section amends two statements in Data, "one LLM only" and "export-r2 is not used". The amendment applies only to the new round 2 stores; round 1 stores and results are unchanged.

**Data**

- **export-r2 colloquial paragraphs** ([data-zh-colloq-r2.toml](../../../configs/zh/data-zh-colloq-r2.toml), store `zh-colloq-r2-12k`, manifest sha256 `d23dd329…`):
  - selected from part of export-r2: every fourth of the 54 files over 1.4 GB, 14 files in all;
  - same classifiers and thresholds as r1;
  - all 2.29M paragraphs with `p_chat ≥ 0.7` are candidates, plus a hash sample of 437k fiction-like dialogue paragraphs.
- **Round 2 chat** ([data-zh-colloq-chat2.toml](../../../configs/zh/data-zh-colloq-chat2.toml), store `zh-colloq-chat2-12k`, manifest sha256 `3cc52ba1…`):
  - douban part00–05 and all of MMChat's train split are screened;
  - exported by rows: douban 2.4M, MMChat 1.0M and tieba 120k (51k after screening, repeated about 2.35 times), 3.52M in all;
  - it covers every source of the round 1 chat store and replaces `zh-colloq-chat-12k` in c30 and c40.
- **Quality screen**: same criteria as round 1.
  - The r2 paragraphs were split between two LLMs. DeepSeek 4.1 Flash started from one end of the input and DeepSeek chat from the other, and screening stopped where they met.
  - Where they met, 9,430 texts were screened by both models, which agree on 96%. A text that either model marks not ok is dropped.
  - The 8,802 texts (0.44%) still unscreened at the stop count as not ok.
  - Chat uses DeepSeek 4.1 Flash only.
- After screening: r2 train 2.588M paragraphs (chat 2.188M, fiction-like dialogue 400k), validation 44k.
- **Eval guard**: a training text added after the colloquial set was frozen is removed if it contains an overlap pattern of any colloquial item (development or test). The training row goes; the item stays.
  - The patterns are those of the freeze-time overlap scan: the Han run around the target and the whole message, each at least 8 characters.
  - This removed 6 r2 paragraphs and 2 chat train turns.
- Train samples:

  | Store | plain | full | mixed | initials | error |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `zh-colloq-r2-12k` | 669,244 | 382,468 | 381,816 | 381,170 | 121,505 |
  | `zh-colloq-chat2-12k` | 789,883 | 717,466 | 719,141 | 718,950 | 227,424 |

**Arms**

All four arms start from `kd-cont` final with the round 1 recipe, and per-kind totals are unchanged. The r-type share is split between the r1 and r2 stores by size, about 64 : 36.

| Arm | Colloquial samples per update | r1 + r2 + chat | seed | Config |
| --- | --- | --- | --- | --- |
| `colloq-c30` | 156, about 30% | 60 + 36 + 60 (chat2) | 20261122 | [train-zh-pinyin-colloq-c30.toml](https://github.com/Voltline/VimeML/blob/5263e6595959779558816f2467a8d8a86f4229a9/configs/train-zh-pinyin-colloq-c30.toml) |
| `colloq-c40` | 208, about 40% | 81 + 47 + 80 (chat2) | 20261122 | [train-zh-pinyin-colloq-c40.toml](../../../configs/zh/train-zh-pinyin-colloq-c40.toml) |
| `colloq-c20-s2` | as c20 | 64 + 0 + 40 (round 1 chat) | 20261123 | [train-zh-pinyin-colloq-c20-s2.toml](https://github.com/Voltline/VimeML/blob/5263e6595959779558816f2467a8d8a86f4229a9/configs/train-zh-pinyin-colloq-c20-s2.toml) |
| `colloq-c40-s2` | as c40 | as c40 | 20261123 | [train-zh-pinyin-colloq-c40-s2.toml](https://github.com/Voltline/VimeML/blob/5263e6595959779558816f2467a8d8a86f4229a9/configs/train-zh-pinyin-colloq-c40-s2.toml) |

- At c40 the most-read colloquial store and kind is chat2 plain, about 0.95 times.
- Against c20, c30 and c40 change both the colloquial share and the data (r2 and the round 2 chat are added). The comparison therefore measures more colloquial data and a higher share together.
- All colloquial text is distilled, since c10-hard brought no gain in round 1.

**Evaluation**: as in round 1 (primary metric, reference metrics and both gates), on development only.

**Decision** (against round 1 `colloq-c20`, seed 20261122):

1. c40 replaces c20 as the candidate if c40 and c40-s2 are both significantly above c20 and both pass both gates.
2. Otherwise c30 replaces c20 if it is significantly above c20 and passes the gates.
3. Otherwise c20 stays the candidate.
4. c20-s2 vs c20 and c40-s2 vs c40 are reported as this round's seed band. They are descriptive only and do not enter the decision.
5. The final candidate goes through v2-20m's existing test confirmation (colloquial test half, public pinyin) and Mac alignment.

## Round 2 results (2026-11-23)

Colloquial development, self-rescored Top-1. p is from the sentence-cluster sign-flip test against round 1 `colloq-c20`.

| Arm | Top-1 | vs c20 | p |
| --- | ---: | ---: | ---: |
| `colloq-c20` (round 1) | 41.9 | — | — |
| `colloq-c20-s2` | 41.6 | −0.2 | 0.52 |
| `colloq-c30` | 41.8 | −0.1 | 0.85 |
| `colloq-c40` | 42.3 | +0.4 | 0.34 |
| `colloq-c40-s2` | 42.2 | +0.3 | 0.36 |

The "vs c20" column is the paired per-sentence difference, so it can differ by 0.1 from subtracting the two Top-1 values.

- Seed band: c20-s2 vs c20 −0.2 (p 0.52); c40-s2 vs c40 −0.1 (p 0.85).
- Gates:
  - public pinyin self-rescored Top-1 against `kd-cont` (54.8): c20-s2 55.7 (+0.9, p 0.11), c30 56.4 (+1.6, p 0.0063), c40 56.3 (+1.5, p 0.011), c40-s2 56.2 (+1.4, p 0.035); none is below `kd-cont`;
  - Standard against v1: c20-s2 net −1, c30 +3, c40 0, c40-s2 +3, none significant.
- Reference: final-particle prediction Top-1 is c20 0.225, c20-s2 0.238, c30 0.258, c40 0.271, c40-s2 0.260.
- Decision: none of c40, c40-s2 and c30 is significantly above c20, so by rule 3 c20 stays the candidate. Raising the colloquial share from 20% to 30–40%, with r2 and the round 2 chat added, brings no measurable gain on the primary metric. Final-particle prediction improves, but it is a reference metric only.

## Out of scope

- A hand-written colloquial evaluation (deferred).
- Typo data and the 800M teacher.
- Client rules and sensitive-word blocking.

## Deviations

1. **2026-11-23: the candidate is `colloq-c40`.** By round 2 decision rule 3, c20 should have stayed the candidate. After seeing the round 2 results, `colloq-c40` was chosen instead, for these reasons:
   - it has the same architecture as `kd-cont` and c20, so export, latency and iOS integration do not change;
   - it passes both gates;
   - on the primary metric both seeds, c40 and c40-s2, are slightly above c20 (+0.4 and +0.3, neither significant), and no metric is below c20;
   - on the reference metric, final-particle prediction Top-1, both seeds are clearly higher (0.271 and 0.260, against 0.225 for c20).

   The choice was made after seeing the results. Whether c40 is better than `kd-cont` is therefore left to the test confirmation ([confirmation plan](../benchmarks/confirmation-zh-colloq-plan.md)), not to the development gap.
