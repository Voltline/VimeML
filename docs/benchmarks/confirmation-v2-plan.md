# Confirmation scoring plan v2 (preregistered)

Registered 2026-10-21. This plan was committed before [Standard v2 blind](standard-ime-v2-blind.md) and the [homophone challenge](homophone-ime.md) were opened. The machine-readable version is [`configs/ime-confirmation-v2.json`](../../configs/ime-confirmation-v2.json) and the entry point is [`confirm_ime_v2.py`](../../scripts/benchmarks/confirm_ime_v2.py). Metrics and statistics follow [metrics v2](metrics-v2.md). At commit time no model had been run on either set and no coverage had been computed. Results: [confirmation v2 results](confirmation-v2-results-20261021.md).

> **Correction (after scoring)**: at registration this plan described V2.1 as the current deployment and V4.2 as an optional variant; in fact V4.2 had already been adopted for deployment. This affects only the model-role descriptions and the deployment wording after the H1 decision, not the hypotheses, models, metrics, clusters or decision rules. The frozen JSON is unchanged, and the text below keeps the registered wording.

## Question

Since V2.1 was deployed, V4.2 gets 18 more exact Top-1 items on development, but not significantly by author cluster (p = 0.081). V4.3 (adds web data) and V5 (larger model) brought no visible improvement on development. All of these conclusions come from a development set used many times. This plan gives one confirmation on two unopened sets.

## Models

All FP32 weights, scored as in the frozen comparison (full-vocabulary suffix log-probability sum, joint tokenization of left context and candidate, TF32 off, ties keep AzooKey order). Snapshots are pinned by SHA-256 and step.

| Key | Checkpoint | Step | Role |
| --- | --- | ---: | --- |
| `v21` | V2.1 extend5 `best.pt` | 40,000 | Current deployment (FP32 weights of the INT8 release) |
| `v42-e20` | V4.2 `best-lm.pt` (epoch 20) | 1,081,942 | Optional replacement, device measurements done; selected by LM BPC |
| `v43-e30` | V4.3 epoch 30 | 3,225,888 | V4.2 recipe plus web data; best BPC |
| `v5-e30` | V5 epoch 30 | 3,225,888 | Same data and budget as V4.3, d512/L8; best BPC |

AzooKey's first candidate (no language model) is reported as a reference and is not tested. All checkpoints were selected by LM BPC before this plan; selection used neither set.

## Hypotheses and decision

**Primary hypothesis H1**: Standard v2 blind, n-best 20, `v21 → v42-e20`, exact Top-1.

- Confirmed if: net > 0 with two-sided author-cluster sign-flip p < 0.05, and the same pair shows no notation-lenient Top-1 drop with p < 0.05 on either Standard v2 blind or the homophone set (n20).
- Confirmed: V4.2 is a confirmed exact Top-1 improvement over the deployed V2.1.
- Not confirmed: no difference can be claimed; V2.1 stays deployed and V4.2 remains an optional variant.

**Secondary hypotheses**: exact Top-1, n20, four tests with Holm correction (family α = 0.05), two-sided, direction from the sign of the net; the same notation-lenient guard as H1.

| ID | Contrast | Track | Question |
| --- | --- | --- | --- |
| S1 | `v21 → v42-e20` | Homophones | Does V4.2 differ on written-text homophones |
| S2 | `v43-e30 → v5-e30` | Standard v2 blind | Model size only, conversational text |
| S3 | `v43-e30 → v5-e30` | Homophones | Model size only, written-text semantics |
| S4 | `v42-e20 → v43-e30` | Standard v2 blind | Adding web data, conversational text |

H1 is separate from the secondary family: it is the only deployment question and does not share α with the secondary tests.

**Power**: a cluster bootstrap of the development author-cluster difference for `v21 → v42-e20` shows that if the true effect equals the development one (+18/728), H1 is confirmed at α = 0.05 with probability about **40%**. "Not confirmed" is therefore a likely outcome and must not be read as "V4.2 is not better"; the result reports the net and its cluster bootstrap 95% interval. The current set sizes cannot raise the power; this is stated in advance to prevent post hoc readings.

## Descriptive reports (no decisions)

- Exact, notation-lenient and pool-conditional Top-1 and error classes for every model on every track;
- coverage (n20, n50, recovered and lost), opened together with scoring;
- every metric and pairwise comparison on the n-best 50 pool;
- all pairwise model comparisons on both tracks (uncorrected);
- sensitivity to template clusters and union clusters;
- Standard split by source (WRIME, JMultiWOZ);
- homophones: exact Top-1 when the reference is the more versus the less frequent word of its group; among groups with both items kept, the number with both correct.

Cluster units: authors for Standard (WRIME users, JMultiWOZ users); homophone groups for the homophone set.

## Execution

1. Commit the plan and scripts first. `run` checks that both are committed and unmodified, that code and dependency hashes match, that checkpoint and candidate-pool file hashes match, and that no completed consumption record exists.
2. **Preflight**: every model first rescores the exposed v1 development n20 pool and must reproduce its recorded ranking on all 728 items, otherwise blind is not opened.
3. Write the consumption record to `artifacts/benchmarks/ime-candidates-v2-imported/consumption.json`, then score the n20 and n50 pools of both tracks model by model, and finally write `summary.json`.
4. Run once. If a technical failure interrupts before `summary.json` is written, the run may be repeated under the same plan in a new directory and recorded in the consumption record; never rerun to change a result.
5. Any change after the first run (code defect, metric correction) is reported as a deviation, and the original output is kept.

Public content holds only summary numbers and item IDs, never item text, candidates or the homophone group list.

## Limitations

- Labels on both sets are AI expert review, not native-speaker gold.
- v2 blind authors come from the same blind role as the consumed v1 blind; the homophone set comes from a single Wikipedia shard.
- Once opened, both sets count as used; later model selection needs a new blind version.
