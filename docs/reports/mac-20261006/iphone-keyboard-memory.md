# V1 real keyboard-extension memory — 2026-10-06

An iPhone 16 Pro Max manual-input session runs with LM reranking and next-word suggestions enabled. The trace covers approximately 152.7 seconds and contains 127 extension process samples at approximately one-second intervals.

Sampled physical-footprint peak is 21.25 MiB; private resident peak is 51.922 MiB. These metrics have different definitions. Screenshot observations of 6.7 → 29.31 → 45.77 MB are individual readings rather than complete-session maxima. Host-app measurements are not substituted for extension data.

Installed build/model identity is not fully embedded in this record. Coarse sampling can miss short peaks, and late-session growth does not establish a long-term plateau. The observed input experience is acceptable within this limited session, without a universal memory or stability guarantee.

Original Instruments data, exports, and summaries remain in ignored `outputs/deployment/`. [Simulator/historical audit](simulator-memory.md), [import verification](merge-verification.md), and [V1 Core ML](../../reference/coreml-v1.md) retain separate evidence scopes.
