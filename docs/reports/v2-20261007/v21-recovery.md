# Interrupted V2.1 predecessor — 2026-10-07

The recovered CPU-only instance has no active training process. The archived `continue` model's best/last files are step 0, epoch 0, batch cursor 0, and total tokens 0, with empty optimizer state. All 46 model tensors match frozen V2.0 step 375,000. The log ends at initial validation loss 4.3895957; no saved formal updates are recoverable. Possible unsaved work cannot be established from these files.

The 93,596,818-byte rescue archive contains 18 source files plus a manifest, configurations, environment/plan records, logs, and step-zero checkpoints. It excludes SSH/W&B credentials and preserves sanitized historical logs. Local evidence resides in `outputs/autodl-v21-recovery-20261007/` and `artifacts/models/tiny-ja-v2.1-e16k-d320-l6-continue/`.

The original GPU becomes available again at 18:54 Beijing time. `restart1` begins from frozen V2.0 with batch 512 and a distinct run, rather than treating the step-zero archive as resumed progress. Restart1 and the later extension complete separately. [Restart design](v21-plan.md) and [extension results](v21-extend.md) identify the actual trained models.

Instance images preserve the environment but do not necessarily preserve `/root/autodl-tmp`; data-disk artifacts require separate transport. Historical data/source/rescue archives remain local, with frozen manifests and no credentials.
