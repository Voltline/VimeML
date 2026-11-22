# vimeml-zh-v2-20m colloquial post-training plan

Drafted 2026-11-22 and committed before training (preregistered). The Chinese version, [docs/zh-CN/training-zh-colloq-plan.md](zh-CN/training-zh-colloq-plan.md), is the reference text.

The v2-20m candidate `kd-cont` leans literary on colloquial text: typing `ba` after 这么说 completes to 把 instead of 吧. This plan changes only the training data. It continues `kd-cont`'s recipe with colloquial text mixed in, to improve prediction and conversion of colloquial sentences on the model side. Client rules are outside this plan.

## Data

Colloquial data enters through `[[data.extra]]`. Builder, sample rules, tokenizer and Rime readings are the same as for the r1 pinyin data. Labeling and the quality screen use one LLM only (DeepSeek 4.1 Flash). export-r2 is not used.

**export-r1 colloquial paragraphs** ([data-zh-colloq-r1.toml](../configs/data-zh-colloq-r1.toml), store `zh-colloq-r1-12k`, manifest sha256 `1dea7399…`)

- [zh_colloq.py](../scripts/corpus/zh_colloq.py) first pre-filters with sentence-final-particle rules. LLM labels then train two classifiers (hashed character 1–3-grams, logistic regression):
  - `p_chat`: the paragraph is real chat;
  - `p_spoken`: the paragraph is spoken language, including fiction dialogue.
- Train takes the train-split paragraphs with `p_chat ≥ 0.7`: 3.83M paragraphs, 93.4M characters. A hash sample adds 0.73M paragraphs (36.7M characters) with `p_spoken ≥ 0.8` and `p_chat < 0.5`, mostly fiction dialogue.
- The heldout split is by document hash (`zh-colloq-heldout`, 2%). It is evaluation only and never trained on.
- Each paragraph gives one sample: 30% plain, 70% pinyin.

**External chat** ([data-zh-colloq-chat.toml](../configs/data-zh-colloq-chat.toml), store `zh-colloq-chat-12k`)

- Sources: Douban conversations, MMChat (Weibo) and Tieba. [zh_chat_external.py](../scripts/corpus/zh_chat_external.py) cleans them with rules and holds out 2% by dialogue.
- Every turn goes through an LLM quality screen (hate, abuse, sexual, political, illegal, junk and other low-quality text). Only turns judged ok are kept, and an unscreened turn counts as not ok. The screen inputs are shuffled, so a partly screened source is a random subset.
- `zh_colloq.py export-chat` exports by row count: Douban 800k, MMChat 600k, Tieba 120k. Tieba has only 51k turns after screening, so they are repeated about 2.35 times, each copy under its own dialogue key.
- Each turn gives one sample, with context only from inside the turn, as deployed: 30% plain, 70% pinyin.
- The store's manifest sha256 is filled into the configs after the build and before training. That step is not a deviation.

**Composition of the colloquial part** (by sample): r1 chat about 52%, r1 spoken about 10%, Douban 20%, MMChat 15%, Tieba 3%. At c20 the r1 store is read at most 0.7 times for any kind, and the chat store less than once.

## Arms

All four arms start from `kd-cont` final (step 30,000) with the round 3 recipe:

- 30,000 more updates;
- the 200M teacher, α = 0.5, T = 1;
- peak LR 1e-4 → 1e-5, warmup 1000;
- seed `20261122`.

Every update has 512 samples, with the same total per kind as `kd-cont` (plain 160 / full 96 / mixed 128 / initials 96 / error 32), so `per_kind` normalization is unchanged.

| Arm | Colloquial samples per update (r1 colloquial + chat) | Loss on colloquial | Config |
| --- | --- | --- | --- |
| `colloq-ctrl` | 0 | — | [train-zh-pinyin-colloq-ctrl.toml](../configs/train-zh-pinyin-colloq-ctrl.toml) |
| `colloq-c10` | 52 (32 + 20), about 10% | distilled | [train-zh-pinyin-colloq-c10.toml](../configs/train-zh-pinyin-colloq-c10.toml) |
| `colloq-c20` | 104 (64 + 40), about 20% | distilled | [train-zh-pinyin-colloq-c20.toml](../configs/train-zh-pinyin-colloq-c20.toml) |
| `colloq-c10-hard` | as c10 | cross-entropy only | [train-zh-pinyin-colloq-c10-hard.toml](../configs/train-zh-pinyin-colloq-c10-hard.toml) |

- `colloq-ctrl` measures what 30,000 more updates alone give. r3-ctrl and r3-ctrl-s2 are the same recipe at two other seeds and serve as the seed-noise reference.
- In c10 and c20, the r1 base mix is scaled down to make room for the colloquial samples.
- c10-hard: the teacher never saw this kind of text, so its soft labels may pull colloquial text back toward literary text. In this arm both colloquial directories set `ce_only = true` and train on the label's cross-entropy alone; the r1 base is still distilled. The trainer change is `f32a36b`. Without `ce_only` sources, batches are byte-identical to before the change, which a test checks.

## Evaluation

Everything runs on development. Neither test nor blind is read.

**Primary**: colloquial full-sentence pinyin conversion on `ime-pinyin-zh-colloq-v1` ([prepare_pinyin_colloq_zh.py](../scripts/benchmarks/prepare_pinyin_colloq_zh.py)).

- Record format and pipeline are the public pinyin set's: Vime search (beam 16 per key position), the top 10 rescored by the model's own scores (λ = 1.6), and the rescored Top-1 is the metric.
- Context is only the text before the cursor in the same message (infield); earlier turns are not visible.
- Sources and item counts (all from evaluation-only splits):

  | Source | Items | From |
  | --- | ---: | --- |
  | LCCC | 600 | test |
  | r1 colloquial | 600 | heldout |
  | Douban | 300 | heldout, screened ok |
  | MMChat | 300 | heldout, screened ok |
  | Tieba | 200 | heldout, screened ok |

- Clauses are 4–20 characters, and about half end in a sentence-final particle. Readings are kept only where three tools agree.
- Before freezing, an overlap scan runs against all training corpora (Aho-Corasick, minimum 8 characters), and flagged items are dropped.
- Development and test are half each; test is held back for the confirmation run.
- Significance: paired sign-flip test over sentence clusters.

**Secondary** (descriptive only, not part of the decision):

- prediction track A Top-3, with the CPED stratum (TV-drama lines) reported separately;
- final-particle prediction: on the colloquial set's items that end in a particle, predict at the position before the particle with single-character candidates allowed (`min_chars = 1`), reporting Top-1 and Top-3.

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

## Out of scope

- A hand-written colloquial evaluation (deferred).
- Typo data and the 800M teacher.
- Client rules and sensitive-word blocking.

## Deviations

(None yet.)
