# Confirmation scoring v2 results (2026-10-21)

Run once under the [confirmation plan v2](confirmation-v2-plan.md) (commit `858f7f1`, pushed before scoring). All 4 models reproduced their recorded rankings on v1 development (728/728) before the two sets were opened; one run, no deviations. Output is local in `outputs/confirmation-v2-20261021/`; entry point [`confirm_ime_v2.py`](../../scripts/benchmarks/confirm_ime_v2.py). Only summary numbers are published.

## Decisions

| Hypothesis | Contrast | Track | Exact gain/loss (net) | Author-cluster p | Holm p | Cluster bootstrap 95% | Decision |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| **H1** | V2.1 → V4.2 | Standard v2 blind | 60/38 (+22) | 0.0508 | — | [2, 43] | **Not confirmed** |
| S1 | V2.1 → V4.2 | Homophones | 22/34 (−12) | 0.17 | 0.33 | [−28, 4] | None |
| S2 | V4.3 → V5 | Standard v2 blind | 29/48 (−19) | 0.048 | 0.14 | [−37, −2] | None |
| S3 | V4.3 → V5 | Homophones | 28/10 (+18) | 0.0062 | **0.025** | [6, 30] | **Improvement** |
| S4 | V4.2 → V4.3 | Standard v2 blind | 34/42 (−8) | 0.42 | 0.42 | [−25, 9] | None |

No contrast triggered the notation-lenient Top-1 guard.

**H1 is not confirmed.** V4.2 gets 22 more items right, matching the development +18 in direction and size. Per-item McNemar p = 0.033, template-cluster and union-cluster sensitivity p = 0.036 and 0.041, and the cluster bootstrap interval excludes 0. But the preregistered primary test is the author-cluster sign-flip, p = 0.0508 (20,000 fixed-seed draws, Monte Carlo standard error about 0.0015), which misses 0.05. This is exactly the case the plan described in advance: power about 40%, and "not confirmed" does not mean "not better". No extra draws or reruns will be made to change the conclusion.

V4.2 is already the deployed version; the plan wrongly recorded V2.1 as the current deployment at registration (see the correction in the plan). H1 therefore answers whether the adopted V4.2's advantage over the previous deployment V2.1 meets the preregistered standard of evidence. It narrowly does not. The point estimate is +22 with interval [2, 43], and the notation-lenient treatment shows no drop, so there is no evidence that adopting V4.2 was a regression. The area to watch is written-text homophones, where V4.2 gets 12 fewer items than V2.1 (S1, Holm p = 0.33, not significant).

**S3 is confirmed**: enlarging only the model (V4.3 → V5, same data and budget) raises exact Top-1 on written-text homophones by 18 items.

## Per model

Standard v2 blind (729 items, 62 author clusters; 677 items in the n20 pool):

| Model | Exact | Notation-lenient | Pool-conditional | WRIME /486 | JMultiWOZ /243 | Semantic errors | Notation errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AzooKey first | 516 | 567 | 0.762 | 309 | 207 | 65 | 96 |
| V2.1 | 576 | 629 | 0.851 | 353 | 223 | 23 | 78 |
| V4.2 epoch 20 | **598** | **639** | **0.883** | **372** | **226** | 18 | 61 |
| V4.3 epoch 30 | 590 | 637 | 0.871 | 364 | **226** | 18 | 69 |
| V5 epoch 30 | 571 | 633 | 0.843 | 354 | 217 | 20 | 86 |

Notation errors are notation plus notation_upper; 52 items are pool misses for every model.

Homophone challenge (619 items, 320 homophone groups, 299 with both items kept; 526 items in the n20 pool):

| Model | Exact | Notation-lenient | Pool-conditional | More frequent /314 | Less frequent /305 | Both correct /299 groups | Semantic errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AzooKey first | 305 | 319 | 0.580 | 194 | 111 | 58 | 184 |
| V2.1 | 450 | 463 | 0.856 | **252** | 198 | 161 | 53 |
| V4.2 epoch 20 | 438 | 452 | 0.833 | 248 | 190 | 160 | 65 |
| V4.3 epoch 30 | 434 | 452 | 0.825 | 248 | 186 | 153 | 63 |
| V5 epoch 30 | **452** | **465** | **0.859** | **252** | **200** | **167** | 52 |

## Descriptive results

- **The two tracks rank the models in opposite orders.** Conversational text: V4.2 > V4.3 > V2.1 > V5. Written-text homophones: V5 ≈ V2.1 > V4.2 > V4.3. The V4 family (8K vocabulary, chat data added) leads on conversation and trails on written text by 12–16 items (uncorrected p 0.07–0.17). V5 is best on written text but 27 items behind V4.2 on conversation (uncorrected p = 0.013).
- **V5 loses on notation in conversation.** V5's semantic errors are on par with V4.2 (20 vs 18), but it has 25 more notation errors; under the notation-lenient treatment V4.2 → V5 is only −6 (p = 0.41). Scale improves semantic choice in written text but moves conversational spellings away from the references.
- **Frequency bias on homophones.** Every model is less accurate on the less frequent word than on the more frequent one (V2.1 80% vs 65%); AzooKey's first candidate even more so (62% vs 36%).
- **Coverage and n-best 50.** n50 raises in-pool items from 677 to 687 on Standard and from 526 to 555 on homophones, with no losses. Exact Top-1 of every language model drops 2–7 items on Standard and rises 2–4 on homophones, matching the development observation: on conversational text, extra candidates mostly add notation competition.

## Conclusions

1. Deployment: V4.2 stays deployed. It beats V2.1 by about 20 items on both conversational sets, but neither time reaches author-cluster significance; it trails by 12 on written-text homophones, not significantly.
2. Vocabulary, data and scale changes each trade one kind of text against the other; none is better on both tracks. Simply scaling up pretraining further cannot improve both.
3. The remaining conversational gap is mostly notation, the written-text gap mostly semantics. The former suits pool-based ranking training or personalization; the latter benefits from model capacity.

Both sets are now used. Later model selection may use development only; a new confirmation needs a new blind version.
