# V3 B joint retraining design — 2026-10-08

Status: four-epoch training, evaluation, selection, and archival complete. B trains the V2 architecture from random initialization on the old V2 training corpus, the selected current eight-shard JpnMix pool, and two real-dialogue corpora. No old weights or optimizer states initialize this run.

## Frozen input scope

Only the already-cleaned sampled pool from eight JpnMix shards participates; the other downloaded shards and unsampled full source text are excluded. WRIME and JMultiWOZ remain benchmark sources. Virtual maps reside in `artifacts/training-data/corpus-v3-b-all8-chat05-c128/`.

The 49,205 pricing/catalogue exclusions are reused. Additional normalized exact/anchor-containment screening excludes 1,062 old sequences, 16,854 new sequences, and 67 exact cross-pool web duplicates. This screening does not establish comprehensive semantic decontamination. Conversation-grouped chat validation/test remains unchanged.

The mixture targets 95% web and 5% chat effective tokens after the deterministic 30% eligible prefix crop. Each epoch contains approximately one billion effective tokens; four epochs total 4,001,730,586, including 200,086,509 chat tokens. Approximately 38.53 chat exposures are reported as repetition. Total pre-crop targets are 4,315,153,613 with 313,423,027 dropped. Fixed `epoch_mean_tokens` scaling avoids disproportionate short-batch weighting.

## Schedule and environment

[train-v3-b.toml](../../../configs/train-v3-b.toml) sets 12.54M V2 architecture, matching 16K tokenizer, context 128, batch 512, BF16 compiled backbone, four epochs/265,841 updates, AdamW, learning rate 1e-3 cosine decay to 1e-4, and 2,000 warmup updates. The reported device is RTX 4080 SUPER with 32,760 MiB, running PyTorch 2.7.0+cu128; it is not recorded as a 4090 D.

Large assets reside in `/root/autodl-tmp/vimeml`; detached screen and optional online tracking preserve run visibility. Public source does not contain access credentials or run URLs. The completed run takes 13,428.8 seconds with 307,681 effective update tokens/s under the report's timing definition.

## Evaluation and selection boundary

Complete new/old/chat domain BPC and original AJIMEE/historical-development evaluations accompany epoch checkpoints. Standard development uses 728 frozen AI-reviewed cases and selects epoch 2 step 132,915 before fixed-reference evaluation. The consumed 725-case set is a registered reference comparison only; the old 500 draft cases remain a separate regression track. Reference and regression results do not alter selection. Historical 1,000-case blind text remains unscored.

The [final report](b-evaluation.md) retains all five evaluated candidate checkpoints, selected-model metrics, paired changes, and archival paths. A/B have different pretraining and mixture budgets; this experiment is not an equal-compute architecture comparison.
