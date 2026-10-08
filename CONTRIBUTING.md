# Contribution conventions

## Scope and source organization

Implementations reside in `src/vimeml/`; thin command wrappers reside in matching `scripts/` subdirectories. Configurations belong in `configs/`. The Vime client is maintained separately. Cross-repository changes use independent branches and pull requests rather than source-directory replacement.

Primary documentation is English, with objective technical narration and dated experiment records. Chinese equivalents reside in `docs/zh-CN/`. Reports distinguish measured results, interpretation, limitations, and artifact status. Architecture labels, checkpoint steps, tokenizer versions, and benchmark roles remain explicit.

## Artifact and experiment integrity

Frozen checkpoints, labels, candidate pools, configurations, and original reports remain unchanged. New experiments use distinct artifact and output directories. Model selection is recorded before reference-set scoring. Previously consumed test sets are treated as fixed references, not fresh blind evaluations.

Datasets, weights, per-case scores, traces, transfer packages, credentials, account identifiers, and tracking run URLs remain outside Git. Public model releases contain appropriately licensed assets and aggregate metrics. Upstream legal notices remain verbatim.

## Proportionate validation

Validation follows the changed behavior. Documentation-only changes require local-link and metadata checks; they do not require model inference, full-file checksums, training runs, or the complete regression suite. Existing frozen score caches are reused when checkpoint and candidate identity match. New functional changes require only relevant checks sufficient to establish their behavior; failures and measurement boundaries remain in the experiment record.

Runtime/dependency changes require an explicit environment record. Core ML and device claims require Apple-runtime evidence. Build metadata and platform dependency pins describe source installation separately from the standalone Hugging Face inference bundles.
