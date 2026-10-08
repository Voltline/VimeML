# V1 import verification — 2026-10-06

The historical reconciliation preserves frozen checkpoint/tokenizer/token-manifest identities, original Core ML manifests, six compression experiment packages, score caches, failure evidence, and client traces. Conflicting source versions are retained separately. The imported snapshot is not used to overwrite the independent Vime client repository.

Original transport/model checks and detailed per-file decisions reside in `outputs/maintenance/mac-merge-20261006/merge-report.json` and `final-preservation-verification.json`. Existing frozen identities remain authoritative; later tool changes do not regenerate original manifests.

## Score-cache reconciliation

Metrics reconstructed from original score caches match original reports. Windows does not execute Core ML prediction during this reconciliation.

| INT8 comparison | Complete ranking changes | First-choice changes | Maximum suffix-sum difference |
| --- | ---: | ---: | ---: |
| Historical development vs conservative Core ML | 118 | 0 | 1.994759 |
| Historical development vs Windows FP32 | 119 | 0 | 1.985330 |
| AJIMEE vs Windows FP32 | 163 | 2 | 2.692832 |

Development Top-1/Top-5 is 122/134 out of 137; AJIMEE is 124/151 out of 200. Two AJIMEE first-choice differences are accepted spellings. The 16/20 completion first-choice comparison uses conservative Core ML, not Windows FP32. Strict INT8 logits failure remains preserved.

Original simulator JSON and extension trace summaries reconcile with archived outputs. The extension has 127 samples, footprint peak 21.25 MiB, and private RSS peak 51.922 MiB; sampling/build-identity limitations remain. Historical host-app memory is presented separately.

At the original source merge, the recorded Python regression suite reports 123 passes and one Mac-specific skip out of 124. Those checks belong to this dated import and are not repeated by subsequent documentation maintenance. `verification.json` retains the machine-readable historical summary. Vendored sources/legal text remain verbatim.
