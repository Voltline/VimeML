# Training-free rescoring: PMI, length and engine-score fusion

2026-10-21. The [error decomposition](ime-errors.md) showed that the remaining errors come mostly from pool misses and notation differences. This experiment tests whether Top-1 can be raised without training, by changing only the scoring formula. Conclusion: PMI and length normalization do nothing; fusing the AzooKey engine score helps a lot on Standard but hurts on AJIMEE, and a gated version still fails the held-out checks. **Deployment keeps plain LM scoring.**

## Protocol

Entry point [`rescore_ime.py`](../../../scripts/benchmarks/rescore_ime.py). Only per-candidate values stored in completed `scores.jsonl` files are used; no model is run and neither the fixed reference nor any blind set is read. Notation: c is the in-context suffix log-probability sum, f the same candidate's context-free log-probability sum, n the candidate's token count, z AzooKey's own candidate score.

| Plan | Method and grid | Registration |
| --- | --- | --- |
| 1 | PMI `c − λf` (λ 0.05–1.0, 11 values); length `c + αn` (α −1–2, 8 values); engine `c + βz` (β 0.02–2.0, 10 values); PMI + engine (110 pairs) | `outputs/rescoring-20261021/plan.json`, SHA256 `91b65166…` |
| 2 | Gated engine: use `c + βz` only when the LM's top-two gap `c(top1) − c(top2) < τ` (10 β values × τ 0.5–8, 6 values) | `outputs/rescoring-gated-20261021/plan.json`, SHA256 `66e3df27…`; registered after plan 1's primary candidate failed; the gate was designed from Standard only |

Both plans were written before any result was computed, and hashes are checked at run time. Each method picks its parameters by Standard development exact Top-1 only (ties go to the parameter closest to baseline). The best method on Standard is the primary candidate; the rest are descriptive. Legacy, AJIMEE and historical dev are held-out check sets; Standard also gets a two-fold cross-validation by source (WRIME 483 / JMultiWOZ 245). Every result reports exact Top-1, notation-lenient Top-1 and semantic error count, with a paired exact McNemar test against baseline. Ties sort stably by the frozen AzooKey order; the recomputed baseline matches the original evaluation's Top-1 item for item.

## Results

V4.2 epoch 20. Cells are "exact / lenient / semantic errors", with the exact Top-1 net against baseline in parentheses:

| Method (chosen parameters) | Standard /728 | Legacy /500 | AJIMEE /200 | Historical dev /137 |
| --- | --- | --- | --- | --- |
| Baseline | 523 / 594 / 17 | 370 / 427 / 4 | 139 / 142 / 16 | 123 / 131 / 1 |
| PMI (λ=0.05) | 522 / 593 / 17 (−1) | 370 / 427 / 4 (0) | 139 / 142 / 16 (0) | 123 / 131 / 1 (0) |
| Length (α=−0.25) | 523 / 595 / 16 (0) | 370 / 427 / 5 (0) | 138 / 141 / 17 (−1) | 124 / 131 / 1 (+1) |
| **Engine (β=0.5), plan 1 primary** | 539 / 602 / 13 (**+16**, p=0.004) | 370 / 424 / 5 (0) | 135 / 138 / 20 (**−4**) | 123 / 132 / 0 (0) |
| PMI + engine (0.05, 0.5) | 539 / 602 / 13 (+16) | 371 / 424 / 5 (+1) | 135 / 138 / 20 (−4) | 123 / 132 / 0 (0) |
| **Gated engine (β=0.5, τ=2), plan 2** | 539 / 602 / 13 (**+16**) | 371 / 425 / 4 (+1) | 137 / 140 / 18 (**−2**) | 121 / 131 / 1 (−2) |

Primary candidates for other models (exact Top-1 net; lenient net):

| Model | Method | Standard | Legacy | AJIMEE | Historical dev |
| --- | --- | --- | --- | --- | --- |
| V5 epoch 30 | Engine β=0.5 | +26; +12 | +3; −3 | −3; −6 | +4; +1 |
| V5 epoch 30 | Gated β=0.5, τ=2 | +27; +12 | +3; −3 | −2; −5 | +4; 0 |
| V2.1 | Engine β=1.0 | +24; +14 | −4; −3 | −10 (p=0.05); −11 (p=0.03) | 0; −1 |
| V2.1 | Gated β=1.0, τ=3 | +25; +15 | −4; −3 | −8; −9 | +1; 0 |

Source cross-validation within Standard (V4.2 engine fusion): tuned on JMultiWOZ and tested on WRIME, +4 (lenient 0); tuned on WRIME and tested on JMultiWOZ, +10 (lenient +8). The Standard gain transfers between the two conversational sources, but not to the other sets.

## Interpretation

1. **PMI and length normalization have no effect.** The chosen parameters all collapse to near baseline. A context-free prior is not where the remaining errors come from.
2. **Engine fusion helps only on conversational data, and about half of that is notation.** On Standard V4.2 gains +16 exact but only +8 lenient: the rest comes from pulling spellings toward AzooKey's forms, and Standard's reference spellings resemble AzooKey's. Semantic errors fall from 17 to 13, a real but small gain.
3. **The engine score brings AzooKey's semantic errors back.** AzooKey's original order has 57 semantic errors on AJIMEE, which the LM brings down to 16; with engine fusion they rise back to 20, and for V2.1 from 14 to 25. Gating only softens this.
4. **No fixed fusion weight is better in every domain.** The right β depends on the kind of text and the engine's quality in that domain, and at run time there is no way to know which domain the current input belongs to.

## Decisions

- Deployment keeps plain LM scoring, with no engine score or other normalization.
- If the client later wants a style "closer to the engine's spellings", β can become an explicit user setting, but not a default, and Standard results cannot justify its benefit.
- Ranking post-training (step 3 of the original plan) is not pursued for now: only a dozen or so in-pool semantic errors remain on the conversational sets, so ranking training would mainly learn the references' notation style; and there is no candidate generator on Windows to build training candidates from the evaluation's distribution.
- The direction with the highest remaining ceiling is candidate pool coverage (about 66 Standard and 35 AJIMEE items have no correct conversion among 20 candidates), which needs a larger AzooKey N-best on the Mac and a newly frozen benchmark version. Next come semantic errors in encyclopedic domains such as AJIMEE, a domain-data question.

## Limitations

The parameter grid was tuned on Standard, which has been used in many experiments; the held-out sets are development/regression sets, not blind. Plan 2 was registered after seeing plan 1's held-out results, so it is a second attempt, and conclusions should be read as two tests with multiple comparisons. The notation-lenient treatment uses the conservative rule, which misses about 15% of notation differences.
