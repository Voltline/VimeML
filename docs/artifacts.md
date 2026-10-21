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
| `handoff/` | Transfer manifests, small source snapshots, and still-referenced archives; local `handoff/README.md` indexes them |

## Frozen model paths

| Experiment | Model directory / selected checkpoint |
| --- | --- |
| V1 | `artifacts/models/tiny-ja-v1/best.pt` |
| V2.0 | `artifacts/models/tiny-ja-v2.0-e16k-d320-l6/best.pt`, step 375,000 |
| V2.1 restart1 | `artifacts/models/tiny-ja-v2.1-e16k-d320-l6-restart1/best.pt`, step 98,426 |
| V2.1 release | `artifacts/models/tiny-ja-v2.1-e16k-d320-l6-extend5/best.pt`, step 40,000 |
| V3 A | `artifacts/models/tiny-ja-v3-a-chat05-d320-l6/`, selected step 25,146 |
| V3 B | `artifacts/models/tiny-ja-v3-b-all8-chat05-d320-l6/epoch-2.pt`, step 132,915 |
| V4-B | `artifacts/models/tiny-ja-v4-b12cosine-e8k-d384-l7/best.pt`, step 584,000 |
| V4.1 | `artifacts/models/tiny-ja-v41-e8k-d384-l7/best-ordinary.pt` (epoch 13) and `last.pt` (epoch 16) |
| V4.2 | `outputs/training-v42-20261020/remote/checkpoints/best-lm.pt`, step 1,081,942 (epoch 20); inference tensors `artifacts/models/tiny-ja-v42-e8k-d384-l7/deployment.pt` |
| V4.3 | `outputs/training-v43-20261020/remote/checkpoints/epoch-{27,30}.pt`; controlled run `remote/controlled-10ep/checkpoints/epoch-10.pt` |
| V5 | `outputs/training-v5-20261021/remote/checkpoints/epoch-{29,30}.pt` (epoch 30 primary); controlled run `remote/controlled-10ep/checkpoints/epoch-10.pt` |

Frozen IME benchmark releases live in `artifacts/benchmarks/`: `ime-standard-ja-v1-ai-expert-r1` (Standard v1), `ime-standard-ja-v2-blind-ai-expert`, `ime-homophone-ja-v1-ai-expert` and `ime-incremental-ja-proto1`; the matching `-draft` and `-review` directories hold their construction and review records. `ime-candidates-v2-handoff{,.zip}` is the pending Mac candidate package.

V4.2 Core ML packages are `artifacts/deployment/tiny-ja-v4.2-epoch20-{fp32,int8-b32,kv-*,kvchunk-*}-v1` with bundle `-bundle-v1` and client resources `-client-resources-v1` (single-token KV) and `-v2` (chunked KV). The `tiny-ja-v2.1-extend5-kv-*-mac-v42-measure-v1` packages are measurement-only rebuilds of the V2.1 KV export.

V3 B also retains `best.pt` step 250,000 and `best-epoch.pt` step 265,841. These have different selection meanings. The interrupted V2.1 `continue` directory contains step-zero artifacts, not completed continuation training.

Incomplete transfers retain `.part` names and are not loadable model assets. Once an incoming archive is imported and verified, the transport ZIP is removed and only its manifest is kept; deletion records and per-member SHA256 lists live under `outputs/maintenance/handoff-cleanup-*/`. Archive records identify relative paths, sizes, steps, provenance, and frozen identities. Existing manifests are reused rather than repeatedly recomputed. Source edits do not alter historical model fingerprints.

Credentials, account identifiers, and tracking run URLs remain outside Git and transfer archives. Restricted benchmark text and raw per-case outputs are excluded from public releases. [Exchange protocol](reference/artifact-exchange.md) and [licensing notice](../NOTICE.md) define the remaining boundaries.
