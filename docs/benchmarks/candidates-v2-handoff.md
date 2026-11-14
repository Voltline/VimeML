# Candidate export v2 (Mac handoff)

Prepared on 2026-10-21 and exported on the Mac the same day (Swift 6.4, converter and dictionary versions as required). All 9 (track, n-best) pairs were imported; the returned archive's sha256 is `aef93ac5…`. In the delivered package `tracks.tsv` had CRLF line endings, so the Mac CLI read `n_best="20\r"` and exited with an error. The operator converted only the copy of `tracks.tsv` in the package to LF and reran; the input files checked byte for byte on import were unchanged. The packaging script now writes that file with LF, and the export script strips a trailing `\r`.

The package is `artifacts/benchmarks/ime-candidates-v2-handoff.zip`, built by [`build_candidate_handoff_v2.py`](../../scripts/benchmarks/build_candidate_handoff_v2.py) `package`. On the Mac, `bash run_candidates_mac.sh <converter checkout>` produces `ime-candidates-v2-azookey-results.zip`, which is loaded with `import --archive`. The export runs no language model. Converter, dictionary and typo/Zenzai/stable settings are exactly as before; only `--config_n_best` changes.

| Track | Items | n-best | Purpose |
| --- | ---: | --- | --- |
| [Standard v2 blind](standard-ime-v2-blind.md) | 729 | 20, 50 | New confirmation set; coverage opens together with scoring |
| [Homophone challenge](homophone-ime.md) | 619 | 20, 50 | Written-text semantics |
| Standard v1 development | 728 | 20, 50 | How many pool misses n50 recovers; n20 is compared with the original export to check determinism (inputs, left contexts and references match the original item by item) |
| v1 development, empty left context | 728 | 20 | Whether the pool depends on the left context (the premise of free-running [segmented input](incremental-ime.md)) |
| [Segmented-input prototype](incremental-ime.md) | 2,000 | 20, 50 | Segment-level protocol prototype |

On import, each (track, n-best) is checked: the returned inputs and mappings must match the package files byte for byte, and the converter, dictionary version, flags and completion marker are verified before loading in the standard export format. The loader reads n-best from the manifest (the original export defaults to 20 and is unaffected).

How many Standard pool misses n-best 50 recovers was measured first on v1 development, where reading the references is allowed. The primary pool for v2 blind and the homophone set stays n20 with n50 reported alongside; the rule is in [metrics v2](metrics-v2.md).

## Development results (2026-10-21)

Only v1 development and the segmented prototype (derived from development) were read; v2 blind and the homophone set were not read and their coverage was not computed. Script [`evaluate_candidate_pools.py`](../../scripts/benchmarks/evaluate_candidate_pools.py); output in `outputs/candidates-v2-dev-20261021-r2/` (local).

**Candidate pools**

- Determinism: the n20 re-export matches the original on 728/728 inputs, with identical candidate lists.
- Left context: for the 407 items that have a left context, clearing it leaves the candidate list identical on 407/407. With Zenzai off, AzooKey's pool does not depend on the left context, so free-running segmented input needs only one export per reading.
- n50: items in pool 630 → 652 (+22: WRIME 16, JMultiWOZ 6), none dropped out; mean pool size 28.8 (many readings have fewer than 50). On 725/728 items the n20 pool is a subset of the n50 pool, but on 33 items AzooKey's first candidate changes and new candidates enter the top 20.
- Segmented prototype: segments in pool 1,913/2,000 at n20, 1,941/2,000 at n50.

**Reranking (FP32, the same scoring as the frozen comparison)**

The n20 column is the recorded scoring; n50 rescores the n50 pool with the same checkpoint. Scores of shared candidates differ only by a per-item constant (largest within-item spread 6.7e-5; the relative order of shared candidates is unchanged on 728/728). Scores accumulate after the common prefix of all candidates, so on the few items where a new n50 candidate shortens that prefix (V2.1 5 items, V4.2 3 items) all scores shift together and ranking is unaffected.

| Model | Exact n20 → n50 | Net (gain/loss) | Cluster sign-flip p | Cluster bootstrap 95% | Lenient n20 → n50 | Net (gain/loss) | Cluster sign-flip p | Cluster bootstrap 95% |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| V2.1 | 505 → 499 | −6 (11/17) | 0.37 | [−17, 4] | 578 → 588 | +10 (14/4) | 0.051 | [1, 19] |
| V4.2 epoch 20 | 523 → 524 | +1 (12/11) | 1.00 | [−8, 9] | 594 → 603 | +9 (14/5) | 0.033 | [2, 17] |

Error-class transitions (n20 → n50):

| Transition | V2.1 | V4.2 |
| --- | ---: | ---: |
| Pool miss → correct | 11 | 12 |
| Pool miss → notation / notation (case) | 4 | 6 |
| Pool miss → semantic | 7 | 4 |
| Correct → notation / notation (case) | 15 | 9 |
| Correct → semantic | 2 | 2 |

Conclusion: about half of the 22 pool misses that n50 recovers are answered correctly; most of the rest are notation differences. At the same time, notation variants among the new candidates take away items that were correct before. Exact Top-1 gains nothing net; lenient Top-1 gains about +9 to +10, with p between 0.03 and 0.06, still a development observation. A larger n-best mainly widens notation competition, not semantic distinctions. Until models reliably pick the right notation, increasing n-best will not raise the exact primary metric.
