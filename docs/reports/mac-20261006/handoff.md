# V1 cross-platform artifact import — 2026-10-06

The original Mac snapshot lacks VimeML Git history, so import uses per-file reconciliation with conflicting versions retained. Later collaboration uses Git branches/pull requests and separate artifact archives, as defined in [exchange conventions](../../reference/artifact-exchange.md).

V1 has 7,386,624 parameters and uses INT8 block-32 storage, FP32 CPU computation, and minimum iOS 18. Historical development 122/137 and AJIMEE 124/200 match FP32 hit counts; strict alignment and ranking/score differences remain documented.

| Material | Local archive |
| --- | --- |
| Source/results/training input | `handoff/mac-20261006-v1/VimeML-{source,results,training-inputs}.zip` |
| Client snapshot | `handoff/mac-20261006-v1/Vime-client.zip`, base `abc3e4d`, patch and resources |
| Original transport records | Same directory, original manifest and checksum record |
| Import/conflict evidence | `outputs/maintenance/mac-merge-20261006/`, `outputs/history/mac-import-20261006/` |
| Deployment comparison | `artifacts/deployment/tiny-ja-v1-conservative-int8-b32-v1/` |

Original failed compression experiments, memory timeout, and traces are preserved. The later Vime client remains in its own repository; historical Swift examples do not replace it. The approximately 152.7-second real-extension record has footprint sample peak 21.25 MiB/private RSS peak 51.922 MiB, with incomplete installed-build identity. [Merge verification](merge-verification.md) and [Core ML measurements](../../reference/coreml-v1.md) retain detailed boundaries.
