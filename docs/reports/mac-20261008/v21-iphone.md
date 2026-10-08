# V2.1 iPhone and keyboard-extension measurements — 2026-10-08

Scope: signed Release deployment on iPhone 16 Pro Max, iOS 27.2 build 24B5099f, with Xcode 27.1. The real keyboard-extension record covers 201.297 seconds, approximately 10:08:23.641–10:11:44.938 Beijing time, with LM reranking and next-word suggestions enabled. Host-app measurements are separate.

## Extension memory

Instruments supplies 164 process samples with median interval 1.036 seconds. An opt-in internal audit covers 191.902 seconds with 1,920 approximately 100-ms samples.

| Measurement | MiB | Scope |
| --- | ---: | --- |
| Instruments sampled physical-footprint peak | 28.0018 | Extension, approximately one-second sampling |
| Internal sampled physical-footprint peak | 28.2987 | Extension, approximately 100-ms sampling |
| Kernel lifetime physical-footprint peak | 34.7206 | Extension process lifetime |
| Private resident peak | 73.7813 | Distinct from physical footprint |
| Shared resident peak | 40.2813 | Distinct resident metric |
| Total resident peak | 192.7813 | Distinct resident metric |

Median footprint rises from approximately 18.33 MiB at 30–60 seconds to 27.2675 MiB at 180–201 seconds. A long-term plateau, leak mechanism, or cache explanation is not established. Loads take 153 ms at 5.557 seconds and 28 ms at 136.596 seconds; the second load is not evidence of process restart without additional identity records.

## Latency

| Operation | Calls | Mean ms | p95 ms | Maximum ms |
| --- | ---: | ---: | ---: | ---: |
| LM reranking | 127 | 32.93 | 53.92 | 58.91 |
| Next-word computation | 39 | 29.05 | 38.11 | 50.34 |
| Base candidate stage | 256 | 17.86 | 59.37 | 75.82 |
| Touch-to-visual stage | — | 1.01 | 1.86 | 4.42 |
| Result-to-UI publication | — | 42.49 | 8.79 | 1,926.09 |

The publication tail's cause is unresolved. Mean exceeding p95 reflects the long tail rather than an interchangeable latency definition. Subjective input experience in this limited record is acceptable; it does not establish universal response-time or power guarantees.

## Host and integration evidence

Separate host instruments report sampled footprint peak 24.2831 MiB; a separate XCTest reports 59.64 MiB. Host cold-load time is 185.54 ms under the recorded test setup, not a full OS-reboot cold-cache measurement. None is substituted for extension memory.

Recorded fixtures cover 6,755 tokenizer cases, four joint-scoring fixtures, twenty beam fixtures, padding/causality, fallback, cancellation, and next-word/session behavior. Initial functional/performance tests pass; a later incremental build fails linking OrderedCollections, followed by a successful clean rebuild. Failure evidence is retained. App-group audit controls distinguish an initially incorrect container control from the actual app audit.

Original logs, traces, fixtures, and result archive remain local. Current client implementation resides in the separate Vime repository. This report does not require repeating device tests during documentation maintenance.
