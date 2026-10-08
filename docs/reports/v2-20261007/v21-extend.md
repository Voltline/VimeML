# V2.1 continuation and selection — 2026-10-07

The extension starts from restart1 step 98,426 with weights and 45 AdamW state entries preserved. The frozen corpus, effective batch 512, BF16 compiled backbone, tokenizer, and crop remain unchanged; data offset is 6.

## Schedule and completed run

[train-v21-extend.toml](../../../configs/train-v21-extend.toml) sets a maximum five-epoch/246,065-update budget. Learning rate stays at 3e-5 for 147,639 updates, then follows a two-epoch cosine decay to 1e-5; no warmup is applied. Full-validation baseline is 3.0811788322564775. Two epochs exceeding the historical best by more than 0.01 BPC trigger the recorded degradation stop rule.

The run begins at 20:15 and receives SIGINT at approximately 21:54 Beijing time after 161,095 updates, approximately 3.273 epochs. Update throughput is approximately 280,712 effective tokens/s. The maximum five-epoch budget is not completed. Later complete-epoch validation does not improve the early minimum; last-checkpoint BPC is 3.0920557.

## Selected release

The selected best is extension step 40,000: complete validation BPC 3.0802686, AJIMEE 144/200, historical development 122/137, and expanded draft development 1,487/2,000. It is distinct from the final checkpoint. V1/V2.0/restart1 and extension artifacts remain separately archived.

The draft score gains 34 cases over V1 and 11 over restart1, with exploratory paired p=0.00648 and 0.22155 respectively. These scores are not fresh blind evidence; labels remain drafts and model selection/development exposure limit inference.

Core ML INT8 scores are 144/200, 122/137, and 1,481/2,000. Numerical alignment failure is preserved and analyzed separately. [V2 summary](../v2-summary.md) and [quantization review](../mac-20261008/v21-quantization-review.md) document the deployment decision. Original model, score, and stage records remain local and immutable.
