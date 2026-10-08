# Cross-platform source and artifact exchange

Source changes use Git branches and pull requests. Model assets, scores, compiled resources, and traces use manifest-bearing archives. VimeML owns model/data/evaluation tools; Vime owns the client. Each repository is merged independently rather than overwritten with a client source snapshot.

## Inference-input archive

`scripts/deployment/prepare_mac.py` packages a clean committed worktree as a Git bundle plus inference inputs. The completed 2026-10-08 input is `handoff/vimeml-v21-mac-20261008.zip`; its base commit is recorded in `handoff-manifest.json`. `setup_mac.py` creates a distinct checkout, binds origin, and restores ignored assets without overwriting an existing directory.

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

The returned V2.1 archive is `handoff/mac-20261008-v21/vimeml-v21-mac-results.zip`. It restores 306 artifacts to their original ignored paths; import metadata resides in `outputs/maintenance/mac-return-20261008/import-report.json`. Git ancestry, path/size/count, and ZIP CRC checks establish transport/source identity. Conflicting same-name assets retain separate versions; original manifests remain unchanged.

Dependencies are rebuilt from platform requirement files. Frozen identity records are reused; credentials and tracking URLs remain local. [Artifact layout](../artifacts.md) and [V1 import history](../reports/mac-20261006/handoff.md) describe earlier transfers.
