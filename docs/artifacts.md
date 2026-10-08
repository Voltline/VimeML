# Artifact organization

Git contains source code, configurations, review annotations, tests, documentation, and licenses. Large or private experiment artifacts remain in ignored directories. The same division applies to release preparation and cross-platform transfers.

| Local root | Contents |
| --- | --- |
| `datasets/` | Downloaded Parquet/TSV/dialogue sources and source notices |
| `outputs/` | Cleaned corpus, original reports, candidate scores, traces, and maintenance records |
| `artifacts/tokenizers/` | SentencePiece models, vocabulary, configuration, and manifests |
| `artifacts/token-data/` | Encoded token stores and source offsets |
| `artifacts/training-data/` | Window indices and frozen virtual mixture maps |
| `artifacts/models/` | Checkpoints and associated run/validation summaries |
| `artifacts/benchmarks/` | Frozen inputs, labels, and actual candidate exports |
| `artifacts/deployment/` | Core ML packages and packaged runtime resources |
| `artifacts/tracking/`, `runs/` | Local tracking metadata and TensorBoard logs |
| `handoff/` | Manifested source/artifact exchange archives |

## Frozen model paths

| Experiment | Model directory / selected checkpoint |
| --- | --- |
| V1 | `artifacts/models/tiny-ja-v1/best.pt` |
| V2.0 | `artifacts/models/tiny-ja-v2.0-e16k-d320-l6/best.pt`, step 375,000 |
| V2.1 restart1 | `artifacts/models/tiny-ja-v2.1-e16k-d320-l6-restart1/best.pt`, step 98,426 |
| V2.1 release | `artifacts/models/tiny-ja-v2.1-e16k-d320-l6-extend5/best.pt`, step 40,000 |
| V3 A | `artifacts/models/tiny-ja-v3-a-chat05-d320-l6/`, selected step 25,146 |
| V3 B | `artifacts/models/tiny-ja-v3-b-all8-chat05-d320-l6/epoch-2.pt`, step 132,915 |

V3 B also retains `best.pt` step 250,000 and `best-epoch.pt` step 265,841. These have different selection meanings. The interrupted V2.1 `continue` directory contains step-zero artifacts, not completed continuation training.

Incomplete transfers retain `.part` names and are not loadable model assets. Archive records identify relative paths, sizes, steps, provenance, and frozen identities. Existing manifests are reused rather than repeatedly recomputed. Source edits do not alter historical model fingerprints.

Credentials, account identifiers, and tracking run URLs remain outside Git and transfer archives. Restricted benchmark text and raw per-case outputs are excluded from public releases. [Exchange protocol](reference/artifact-exchange.md) and [licensing notice](../NOTICE.md) define the remaining boundaries.
