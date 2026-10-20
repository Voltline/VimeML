# Cross-platform source and artifact exchange

Source changes use Git branches and pull requests. Model assets, scores, compiled resources, and traces use manifest-bearing archives. VimeML owns model/data/evaluation tools; Vime owns the client. Each repository is merged independently rather than overwritten with a client source snapshot.

## Inference-input archive

`scripts/deployment/prepare_mac.py` packages a clean committed worktree as a Git bundle plus inference inputs. The completed 2026-10-08 input summary is `handoff/mac-20261008-v21/vimeml-v21-mac-20261008-input.json`. The V4.2 Core ML input is created by `prepare_mac_v42.py`; its task brief is [V4.2 Core ML handoff](mac-v42-coreml.md) and its summary is `handoff/mac-20261020-v42/vimeml-v42-mac-20261020-input.json`. The V4.3 input (epochs 30 and 27, with V4.2/V2.1 baseline packages for same-session measurement) is created by `prepare_mac_v43.py`; its task brief is [V4.3 Core ML handoff](mac-v43-coreml.md). Consumed input ZIPs were removed on 2026-10-20 after their payloads were verified locally. `setup_mac.py` creates a distinct checkout, binds origin, and restores ignored assets without overwriting an existing directory.

| Input | Scope |
| --- | --- |
| `repository.bundle` | Committed source, documentation, and reachable history |
| `deployment.pt` | Matching FP32 tensors without optimizer/RNG state |
| Tokenizer and manifest | Exact vocabulary and artifact provenance |
| Development candidates and FP32 scores | AJIMEE 200, historical 137, and draft 2,000 cases |
| Logits/text fixtures | Lengths 1/16/128, padding/causality, and fixed examples |
| V1 INT8 assets | Deployment comparison and fallback |

Blind text, training corpora, token stores, optimizers, virtual environments, and credentials are excluded. V1/V2 tokenizer incompatibility requires coordinated model/tokenizer/fixture identity.

## Repository synchronization

Each repository records its base/head commits and merges through a pull request to `main`. The completed V2.1 deployment changes are merged. Other worktrees synchronize with `git pull --ff-only origin main`; artifact ZIPs restore only ignored assets. Historical Git bundles preserve their original snapshot rather than acting as a current branch replacement.

## Result archive

`package_mac_results.py` packages new model resources and reports with `handoff.json`: format, environment, repository identities, relative paths/sizes, and both passed and failed checks. Training inputs and client source trees are excluded.

The returned V2.1 archive restored 306 artifacts to their original ignored paths; import metadata resides in `outputs/maintenance/mac-return-20261008/import-report.json`. Git ancestry, path/size/count, and ZIP CRC checks establish transport/source identity. Conflicting same-name assets retain separate versions; original manifests remain unchanged.

The returned V4.2 archive (`vimeml-v42-mac-results-v4.zip`, PR #3) restored 250 files to `artifacts/deployment/tiny-ja-v4.2-*` and `outputs/deployment/v42-*` without conflicts; import metadata resides in `outputs/maintenance/mac-return-20261020/`. Imported per-case scores were checked against the four frozen candidate pools.

## Handoff directory hygiene

After a verified import, `handoff/` keeps manifests and small source snapshots only. A transport archive is removed once every member is byte-identical to a local file, present as a Git blob, or explicitly preserved; members missing locally are first restored to their original ignored paths. The 2026-10-20 cleanup removed nine such archives (about 3.48 GB); per-member SHA256 manifests and preserved files are in `outputs/maintenance/handoff-cleanup-20261020/`, and `handoff/README.md` indexes what remains.

Dependencies are rebuilt from platform requirement files. Frozen identity records are reused; credentials and tracking URLs remain local. [Artifact layout](../artifacts.md) and [V1 import history](../reports/mac-20261006/handoff.md) describe earlier transfers.
