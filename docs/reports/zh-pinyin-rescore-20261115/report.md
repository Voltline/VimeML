# Inference-time rescoring for pinyin generation (2026-11-15)

**Background.** The public recheck in the [round-2 evaluation](../zh-pinyin-r2-20261114/evaluation.md) found that generation by the three-task model `r2-cont` is significantly below the production logic on full pinyin (−4.0, p 0.02). The advice then was to route by input kind.

**What this report tests.** A fix that needs no retraining:

1. Take the generated top 10.
2. Score each candidate with a plain language model.
3. Add that score to the generation score and rerank.

All results are development results; the public set's test split was not read.

The registered text is the Chinese version, [docs/zh-CN/reports/zh-pinyin-rescore-20261115/report.md](../../zh-CN/reports/zh-pinyin-rescore-20261115/report.md).

## Conclusions

- **Rescoring lifts generation a lot and removes the full-pinyin regression. Routing by input kind is no longer needed.**
- **Overall gains.** With vimeml-tiny-zh-v1 as the prior and λ = 1.6, `r2-cont` goes:
  - from 29.9% to 36.4% on the private benchmark (+6.4, p 5e-5);
  - from 39.5% to 51.3% on the public set (+11.8, p 5e-5).
- **Full pinyin moves from below the production logic to significantly above it:**
  - private benchmark: 72.7% vs 67.2% (+5.6, p 5e-5);
  - public set: 85.5% vs 81.3% (+4.2, p 0.001).
- **The public set is a clean check.** λ was tuned on the private benchmark only, and the public set uses the same fixed λ. The gain there is larger than on the private benchmark.
- **Giving v1 the input-field context (the production rule) adds more.** Public set overall 51.3% → 53.0% (p 0.0001), +4.1 on inputs with a non-empty context; see [Prior with context](#prior-with-context).
- **Production cost.** It is one more v1 scoring pass over the top 10 candidates. v1 already scores Rime's 16 sentences in production; it would score the 10 generated candidates instead.

## Method

**Candidates.** The three-task model's top 10 from pinyin-constrained beam search (beam 32), as in the round-2 evaluation.

**New score for each candidate:**

> score = genScore + λ × lmScore

- **genScore** is the conditional score from generation, log P(text + EOS ∣ keys).
- **lmScore** is the prior model's absolute log probability of the text: from BOS, with no keys and no left context. This makes scores comparable across candidates.

**Two priors were tried:**

| Prior | Description |
| --- | --- |
| v1 | production vimeml-tiny-zh-v1 (20M, 12K vocabulary, i.e. `12k-fresh-s1`), the same model #70 reranks with and the starting point of the three-task model |
| self | the generator itself, scored as a plain LM (no keys) |

**λ grid:** 0 to 2.0 in steps of 0.1.

**Paired tests:**

- Clustered by sentence, with the cluster sign-flip p and the cluster bootstrap 95% interval in items.
- On the public set, inputs with identical keys in the same sentence are merged, which leaves 3,428 inputs.

## Private benchmark (author's real input, not public; 774 sentences, 4,612 inputs)

`r2-cont` Top-1 (%) as λ varies, v1 prior:

| λ | all | full | mixed | initials |
| ---: | ---: | ---: | ---: | ---: |
| 0 (no rescoring) | 29.9 | 65.9 | 23.9 | 10.4 |
| 0.5 | 35.2 | 71.7 | 29.3 | 12.7 |
| 1.0 | 36.1 | 72.7 | 30.4 | 12.7 |
| 1.6 | 36.4 | 72.7 | 30.6 | 13.0 |
| 2.0 | 36.3 | 72.6 | 30.6 | 13.0 |

The curve is nearly flat from 1.0 to 2.0. The best value for all inputs is 1.6, which is used from here on.

Paired comparisons at λ = 1.6, v1 prior:

| Comparison | Group | Base | Rescored | Difference | Gains/losses | Cluster p | 95% (items) |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| vs no rescoring | all | 29.9 | 36.4 | +6.4 | +419/−122 | 5e-5 | [+238, +355] |
| vs no rescoring | full | 65.9 | 72.7 | +6.8 | +85/−32 | 5e-5 | [+32, +74] |
| vs no rescoring | mixed | 23.9 | 30.6 | +6.7 | +322/−87 | 5e-5 | [+187, +283] |
| vs no rescoring | initials | 10.4 | 13.0 | +2.6 | +12/−3 | 0.035 | [+2, +17] |
| vs production | all | 26.0 | 36.4 | +10.4 | +605/−126 | 5e-5 | [+407, +551] |
| vs production | full | 67.2 | 72.7 | +5.6 | +71/−28 | 5e-5 | [+24, +62] |
| vs production | mixed | 18.6 | 30.6 | +12.0 | +511/−93 | 5e-5 | [+355, +481] |
| vs production | initials | 7.8 | 13.0 | +5.2 | +23/−5 | 0.001 | [+8, +28] |

**Self prior.** Its best λ is 1.3: 34.9% overall (+4.9 vs no rescoring, p 5e-5) and 70.4% on full pinyin. That is about 1.5 points below the v1 prior.

## Public recheck (ime-pinyin-zh-v1 development)

λ is fixed at 1.6 and was not tuned on the public set.

| Comparison | Group | Base | Rescored | Difference | Gains/losses | Cluster p | 95% (items) |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| vs no rescoring | all | 39.5 | 51.3 | +11.8 | +468/−64 | 5e-5 | [+346, +464] |
| vs no rescoring | full | 77.3 | 85.5 | +8.1 | +59/−12 | 5e-5 | [+32, +63] |
| vs no rescoring | mixed | 35.9 | 49.3 | +13.5 | +351/−45 | 5e-5 | [+261, +353] |
| vs no rescoring | initials | 15.9 | 24.7 | +8.8 | +58/−7 | 5e-5 | [+36, +67] |
| vs production | all | 31.6 | 51.3 | +19.7 | +753/−77 | 5e-5 | [+601, +751] |
| vs production | full | 81.3 | 85.5 | +4.2 | +38/−14 | 0.001 | [+10, +38] |
| vs production | mixed | 25.0 | 49.3 | +24.4 | +608/−54 | 5e-5 | [+492, +615] |
| vs production | initials | 7.8 | 24.7 | +17.0 | +107/−9 | 5e-5 | [+79, +117] |

- **The λ curve is flat here too.** From 1.0 to 2.0, all inputs stay between 51.1% and 51.4%. In hindsight the best value is 1.4 (51.4%), only 0.1 points above 1.6.
- **The self prior is again below the v1 prior.** At λ = 1.3 it gives 49.5% overall and 85.3% on full pinyin.

## Prior with context

Production Vime gives v1 the text before the cursor in the input field: the last 64 characters, cut after the last full stop, question mark, exclamation mark or newline (`VimeLanguageModel.sentenceContext`). Here the context is rebuilt by the same rule, and v1 scores the 10 candidates after it, all in one call as Vime's `scores(context:candidates:)` does. Generation itself still has no context.

**Two contexts:**

| Variant | Private benchmark | Public set |
| --- | --- | --- |
| in-field (Vime rule) | earlier fragments of the same message, joined by commas | the text before the clause in the same turn |
| with previous (history) | last 64 characters of the previous message + in-field | previous turn + in-field |

- The in-field context is non-empty for 318 of 4,612 private inputs and 1,463 of 3,428 public inputs.
- The production logic (Rime 16 + v1) is recomputed with the same context, so both sides of each comparison use the same context.
- λ = 1.6.

Top-1 (%), all inputs:

| Set | Context | Production | Generation + rescoring | Difference (cluster p) |
| --- | --- | ---: | ---: | --- |
| private | none | 26.0 | 36.4 | +10.4 (5e-5) |
| private | in-field | 26.2 | 36.7 | +10.5 (5e-5) |
| private | with previous | 26.0 | 37.7 | +11.8 (5e-5) |
| public | none | 31.6 | 51.3 | +19.7 (5e-5) |
| public | in-field | 32.3 | 53.0 | +20.7 (5e-5) |
| public | with previous | 32.1 | 52.9 | +20.7 (5e-5) |

Rescoring with context, paired against rescoring without:

| Set | Subset | In-field | With previous |
| --- | --- | --- | --- |
| private | all 4,612 | 36.4 → 36.7 (+23/−7, p 0.07) | 36.4 → 37.7 (+144/−81, p 0.005) |
| private | in-field non-empty 318 | 44.0 → 49.1 (p 0.07) | 44.0 → 48.4 (p 0.16) |
| public | all 3,428 | 51.3 → 53.0 (+81/−21, p 0.0001) | 51.3 → 52.9 (+151/−97, p 0.02) |
| public | in-field non-empty 1,463 | 52.6 → 56.7 (p 0.0001) | 52.6 → 56.5 (p 0.001) |

- **The in-field context helps and costs nothing.** With an empty context the scores are unchanged. With a context, the public set gains 4.1 points (p 0.0001). The private benchmark moves the same way (+5.1), but with only 318 inputs it is not significant.
- **Context helps generation + rescoring more than the production logic.** On the public set the production logic gains 0.7 and generation + rescoring 1.8. The generated top 10 contains the target more often, so context has more to pick from.
- **The previous message helps too, but a keyboard cannot see it.** The private benchmark gains 1.4 (p 0.005), since the author's messages run on from each other. Vime only reads the input field, so production uses the in-field context only.
- **λ stays.** With context, λ is still flat from 1.0 to 2.0; the best value is between 1.3 and 1.9, and 1.6 is within 0.2 points of it.

## Arm 3 gains too

The same rescoring (v1 prior, λ = 1.6) applied to [arm 3 (joint pretraining from scratch)](../zh-pinyin-pt-20261115/record.md), Top-1 (%):

| Set | Group | Arm 3: no rescoring → rescored | `r2-cont` rescored |
| --- | --- | --- | ---: |
| private | all | 27.7 → 35.3 (+7.6, p 5e-5) | 36.4 |
| private | full | 61.5 → 71.7 (+10.2, p 5e-5) | 72.7 |
| public | all | 37.7 → 50.1 (+12.5, p 5e-5) | 51.3 |
| public | full | 74.9 → 86.7 (+11.8, p 5e-5) | 85.5 |

- **Arm 3 gains more on full pinyin than `r2-cont`.** The gap between the two arms narrows from 2–4 points to about 1 point.
- **On public full pinyin, arm 3 is slightly ahead.**
- **No winner is declared.** The two rescored arms were not paired against each other.

## Reading

- **The problem is the scoring, not the candidate pool.**
  - The round-2 public recheck already showed this. On full pinyin, the generated top 10 contains the target 96.0% of the time, yet Top-1 is only 77.3%.
  - The three-task model's conditional score favors candidates that fit the keys but read unnaturally.
  - Adding a plain LM's absolute probability restores fluency, so the answer gets picked out of the top 10.
- **The v1 prior beats self.** v1 is the three-task model before pinyin post-training (same 12K vocabulary, same architecture). Post-training eroded its plain LM ability (Standard rerank −24 in round 2), so scoring with v1 brings back the language model from before post-training.
- **The language part of the generation score is weak.** Adding an LM score on top of the conditional score gives the language prior more weight. The best λ is above 1, which says the generation score alone underweights language.

## Limitations

- **Generation itself had no context,** and rescoring a pool that also includes Rime's 16 sentences is still untested.
- **One public check.** λ was chosen on the private benchmark. The public set is a check, but it has only one set of synthetic keys.
- **Latency was not measured.** The extra cost is one 20M forward over 10 short candidates, which can be batched.

## Next steps

1. **Vime client Chinese path:** generated top 10 → v1 scores after the input-field context → rerank by genScore + 1.6 × lmScore. No routing by input kind.
2. **Test generation with context,** and a pool that also includes the 16 sentences.
3. **Evaluate the 12K joint-pinyin 200M the same way** when it arrives, generation and rescoring both. Decide on round 3 after that.
