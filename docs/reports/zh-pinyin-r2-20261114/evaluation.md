# zh three-task round 2: evaluation (2026-11-14)

This report evaluates the two arms of the [round-2 training record](training.md):

- `r2-cont`: continued from round-1 `s1-task` for 60,000 steps.
- `r2-base`: restarted from the 12K base for 90,000 steps.

Round-1 `s1-task` and `s1-tok` are listed for reference; round 1 has no report of its own, so the numbers here are the reference. All results are exploratory development results; blind was not read.

## Conclusions

- **Pinyin generation: round 2 beats round 1, and continue vs restart cannot be separated.**
  - On the private benchmark, `r2-cont` generation Top-1 is 29.9%. That is +2.3 points over `s1-task` (cluster p 5e-5).
  - `r2-base` scores 29.2%, −0.7 against `r2-cont` (p 0.23), which is not significant.
- **Generation now beats production's Rime 16 sentences + v1 rerank (Vime #70 logic).**
  - Top-1 on all inputs goes from 26.0% to 29.9% (+3.9, p 5e-5). All of the gain comes from abbreviated input.
  - Full pinyin is 67.2% vs 65.9% (−1.3, p 0.47), which is not significant.
  - The [public recheck](#public-recheck-ime-pinyin-zh-v1-development) confirms the lead on abbreviated input (+8.0 overall), but generation is significantly worse on full pinyin (−4.0, p 0.02). Production should route by input kind.
- **The other two tasks pay for it.** Against the references:
  - Standard rerank: −24 against the 12K base (p 0.003).
  - Prediction track A Top-3: −2.8 points against vimeml-tiny-zh-v1 (p 5e-5).
  - Track B key savings: −1.47 points.

  Round 1 already showed the same direction, and round 2 did not improve it.
- **KV-cached generation gives exactly the same results as the full forward.** On a desktop CPU with beam 16, the median latency is 92 ms (p95 182 ms), about a quarter of the uncached time.

**Overall:**

- Pinyin generation is usable.
- Continued training is enough: it saves a third of the steps.
- A single 20M model doing all three tasks still loses measurably on rerank and prediction. The three-task model cannot replace vimeml-tiny-zh-v1 yet.

## Benchmarks

| Benchmark | Content | Used for |
| --- | --- | --- |
| Private pinyin benchmark | The author's real input, not published; 774 sentences and 4,612 inputs (full pinyin 774, mixed abbreviation 3,492, all initials 346) | Pinyin generation; Rime 16 sentences vs generation |
| [Standard Chinese IME benchmark](../../benchmarks/standard-ime-zh.md) | Development full pinyin, 1,456 items | Rerank, with no keys, scored the production way |
| [Prediction benchmark](../../benchmarks/predict-ime-zh.md) | Development, track A (Top-3) and track B (key savings) | Prediction |

Only statistics from the private benchmark are reported; its items and answers do not enter the repository. Its overlap with the training corpus was checked and is limited to short common phrases.

All paired tests cluster by sentence (private benchmark) or dialogue (prediction). They report the cluster sign-flip p and the cluster bootstrap 95% interval. See [Evaluation](../../evaluation.md) for the statistics.

## Pinyin generation (private benchmark)

**Setup.** Generation is a pinyin-constrained beam search (beam 32, top 10) in task mode, which conditions on the keys.

**Reference rows:**

- Rime's first candidate.
- "Swap": word swapping plus the production model's rerank.
- The base model's LM-mode generation (`base-lm-p4`, initial penalty 4).

Top-1:

| Model | All | Full | Mixed | Initials |
| --- | ---: | ---: | ---: | ---: |
| Rime first | 16.4 | 50.1 | 10.2 | 3.5 |
| Swap | 24.5 | 61.6 | 18.0 | 7.2 |
| `base-lm-p4` generation | 12.9 | 45.3 | 7.0 | 0.6 |
| `s1-tok` | 26.4 | 59.8 | 20.8 | 8.1 |
| `s1-task` | 27.6 | 63.2 | 21.6 | 8.7 |
| `r2-cont` | **29.9** | **65.9** | **23.9** | **10.4** |
| `r2-base` | 29.2 | 64.2 | 23.4 | 9.0 |

**Top-3 / Top-10, all inputs:**

| Model | Top-3 | Top-10 |
| --- | ---: | ---: |
| `s1-task` | 39.5 | 49.3 |
| `r2-cont` | 40.6 | 50.8 |
| `r2-base` | 40.4 | 51.1 |

**Character error rate** (edit distance between the first candidate and the target, divided by target length):

| | Rime | Swap | `r2-cont` | `r2-base` |
| --- | ---: | ---: | ---: | ---: |
| Character error rate | 38.4% | 35.4% | 29.1% | 29.4% |

Paired comparisons (Top-1, all 4,612 inputs):

| Comparison | Difference (points) | Gains/losses | Cluster sign-flip p | Cluster bootstrap 95% |
| --- | ---: | --- | ---: | --- |
| `r2-cont` vs `s1-task` | +2.3 | +272/−165 | 5e-5 | [+1.3, +3.4] |
| `r2-cont` vs `s1-tok` | +3.5 | +362/−200 | 5e-5 | [+2.3, +4.8] |
| `r2-base` vs `r2-cont` | −0.7 | +223/−257 | 0.23 | [−1.9, +0.4] |

**`r2-cont` vs `s1-task` by input kind:**

| Kind | Difference (points) | p | Significant |
| --- | ---: | ---: | --- |
| Full pinyin | +2.7 | 0.03 | Yes |
| Mixed abbreviation | +2.3 | 0.0004 | Yes |
| All initials | +1.7 | 0.21 | No |

The Top-3 gain is smaller: +1.1 (p 0.045).

## Rime 16 sentences vs generation

**Production logic.** Vime's production Chinese logic (#70) takes Rime's first 16 whole-sentence candidates. The v1 model reranks them as a plain LM, without keys.

**Setup.** This comparison runs on all 4,612 private inputs, with the same Rime candidates for every variant:

| Variant | Meaning |
| --- | --- |
| v1-prod | Production logic: v1 scores the 16 sentences, #70 ordering |
| task | `r2-cont` scores the same slots by the conditional score log P(text+EOS ∣ keys), #70 ordering |
| gen | `r2-cont`'s own generation |
| union | 16 sentences plus the generated top 10, all ranked by `r2-cont`'s conditional score |

Top-1 (%):

| Group | Inputs | v1-prod | task | gen | union |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 4,612 | 26.0 | 25.7 | 29.9 | 29.8 |
| Full pinyin | 774 | 67.2 | 65.5 | 65.9 | 65.6 |
| Mixed | 3,492 | 18.6 | 18.6 | 23.9 | 23.8 |
| Initials | 346 | 7.8 | 8.1 | 10.4 | 9.8 |

Paired against v1-prod (all inputs):

| Variant | Difference (points) | Gains/losses | Cluster sign-flip p | Cluster bootstrap 95% |
| --- | ---: | --- | ---: | --- |
| task | −0.3 | +165/−178 | 0.59 | [−1.2, +0.7] |
| gen | +3.9 | +483/−301 | 5e-5 | [+2.4, +5.5] |
| union | +3.8 | +479/−304 | 1e-4 | [+2.2, +5.4] |

**gen vs v1-prod by input kind:**

| Kind | Difference (points) | p | Significant |
| --- | ---: | ---: | --- |
| Mixed | +5.2 | 5e-5 | Yes |
| Full pinyin | −1.3 | 0.47 | No |
| Initials | +2.6 | 0.12 | No |

**How often the target is among the candidates (%):**

| Group | Rime's 16 sentences | Generated top 10 | Either |
| --- | ---: | ---: | ---: |
| All | 33.0 | 50.8 | 53.5 |
| Full pinyin | 80.0 | 88.4 | 91.3 |
| Mixed | 24.9 | 45.6 | 48.3 |
| Initials | 9.5 | 19.9 | 21.4 |

Reading:

- **The bottleneck is the candidate pool, not the scoring.** Only a third of the 16 sentences contain the target. A stronger conditional score (task) merely ties.
- **Generation raises target coverage to half,** and that is where the Top-1 gain comes from.
- **Adding the 16 sentences to the generated pool (union) adds nothing** (−0.2 vs gen, p 0.10). Generation already covers nearly every target the 16 sentences can supply.
- **On full pinyin, Rime is already strong and generation has no significant edge.** If production switches to generation, no regression on full pinyin is a condition that needs separate confirmation.

## Rerank (Standard development, full pinyin)

The new checkpoints are scored the production way (plain LM, no keys) and paired with the 12K base `12k-fresh-s1` (1,363 / 1,456):

| Model | Top-1 | Difference | Gains/losses | McNemar p | Cluster sign-flip p | Cluster bootstrap 95% |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `s1-tok` | 1,355 | −8 | +16/−24 | 0.27 | 0.26 | [−20, +4] |
| `s1-task` | 1,343 | −20 | +15/−35 | 0.007 | 0.006 | [−34, −7] |
| `r2-base` | 1,349 | −14 | +15/−29 | 0.049 | 0.050 | [−27, −1] |
| `r2-cont` | 1,339 | −24 | +18/−42 | 0.003 | 0.003 | [−39, −9] |

**Private-benchmark check.** The same kind of check on the private benchmark puts the new checkpoint in place of the production model in the swap pipeline. The direction agrees and the size is smaller:

| Model | Top-1 | Saved/broken |
| --- | ---: | --- |
| Original | 24.5% | — |
| `s1-task` | 24.0% | 88/113 |
| `r2-cont` | 23.5% | 83/128 |
| `r2-base` | 23.4% | 85/138 |

**Reading:**

- More pinyin training on the continued arm makes rerank worse: `s1-task` −20, then `r2-cont` −24.
- The restarted `r2-base` regresses less (−14) but is still at the edge of significance.

## Prediction (development)

| Model | Track A Top-3 | Track B key savings |
| --- | ---: | ---: |
| vimeml-tiny-zh-v1 | 27.60 | 16.74 |
| `s1-tok` | 26.81 | 16.44 |
| `s1-task` | 25.36 | 15.54 |
| `r2-cont` | 24.78 | 15.27 |

`r2-base` was not run on the prediction benchmark.

Paired comparisons for `r2-cont`:

| Reference | Track A difference (points) | Gains/losses | Cluster sign-flip p | Cluster bootstrap 95% | Track B difference [95%] |
| --- | ---: | --- | ---: | --- | --- |
| vimeml-tiny-zh-v1 | −2.8 | +258/−461 | 5e-5 | [−3.5, −2.1] | −1.47 [−1.67, −1.27] |
| `s1-task` | −0.6 | +235/−277 | 0.075 | [−1.2, +0.0] | −0.26 [−0.43, −0.09] |

## Generation latency (KV cache)

**What already existed:** `TinyGPTZh13.forward_cached` and the chunked KV path in the Core ML export.

**What this round added:** a KV version of the local generator.

- The prompt runs through the model once.
- Each step feeds one new token per hypothesis.
- Cache rows are reordered to follow the parent hypotheses, copying only the filled positions.
- The search logic is line for line the same as the full-forward version.

**Setup.** Every 10th private input (462 inputs), `r2-cont`, FP32, desktop CPU with 4 threads, beam 16, top 10.

**Identity with the full forward:**

- Top-1 identical on 462/462 inputs.
- Top-10 lists identical on 462/462 inputs.
- Largest Top-1 score difference: 1.05e-5.

**Search length:** median 8 beam steps (p95 13).

| Group | Inputs | Median key length | Full forward p50 / p95 ms | KV p50 / p95 ms | Speed-up (p50) |
| --- | ---: | ---: | --- | --- | ---: |
| All | 462 | 18 | 358 / 1,041 | 92 / 182 | 3.9× |
| Full pinyin | 156 | 22 | 432 / 1,236 | 95 / 190 | 4.5× |
| Mixed | 283 | 16 | 351 / 963 | 92 / 172 | 3.8× |
| Initials | 23 | 6 | 198 / 319 | 65 / 135 | 3.0× |

A 24-input single-thread run gave KV p50 / p95 of 205 / 283 ms.

These are PyTorch numbers on a desktop CPU and do not represent Core ML latency on a phone.

## Public recheck (ime-pinyin-zh-v1 development)

The same comparison was rerun on the development split of the [public pinyin-generation test set](../../benchmarks/pinyin-ime-zh.md):

- 578 sentences and 3,444 inputs: 578 full pinyin, 2,288 mixed, 578 all initials.
- Same settings as above: beam 32, top 10, task mode; Rime uses Vime #70's 16-sentence configuration.
- Test was not read.

Generation (%):

| Model | Top-1 all | Full | Mixed | Initials | Top-3 all | Top-10 all | Character error rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s1-task` | 40.1 | 76.0 | 37.4 | 14.9 | 53.3 | 63.6 | 23.3% |
| `r2-cont` | 39.6 | 77.3 | 36.1 | 15.9 | 54.9 | 65.4 | 23.5% |
| `r2-base` | 40.9 | 79.4 | 37.2 | 17.0 | 55.1 | 65.4 | 23.1% |

**Paired comparisons.** These cluster by sentence. Inputs with identical keys within a sentence are merged, leaving 3,428 inputs.

| Comparison | Measure | Difference (points) | Gains/losses | p |
| --- | --- | ---: | --- | ---: |
| `r2-cont` vs `s1-task` | Top-1 | −0.5 | +198/−216 | 0.54 |
| `r2-cont` vs `s1-task` | Top-3 | +1.6 | — | 0.026 |
| `r2-base` vs `r2-cont` | Top-1 | +1.3 | +218/−174 | 0.09 |

**What this means:**

- Round 2's Top-1 gain over round 1 on the private benchmark (+2.3) does not replicate here. Only a small Top-3 gain remains.
- Continued and restarted training still cannot be told apart.

Rime 16 sentences vs generation (`r2-cont`, Top-1, %):

| Group | Inputs | v1-prod | task | gen | union | gen − v1-prod (cluster p, 95%) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| All | 3,444 | 31.6 | 30.6 | 39.6 | 39.3 | +8.0 (5e-5, [+5.8, +10.2]) |
| Full pinyin | 578 | 81.3 | 77.7 | 77.3 | 77.5 | **−4.0 (0.020, [−7.3, −0.9])** |
| Mixed | 2,288 | 25.0 | 24.5 | 36.1 | 35.5 | +11.0 (5e-5, [+8.3, +13.7]) |
| Initials | 578 | 7.8 | 7.6 | 15.9 | 15.9 | +8.1 (5e-5, [+5.2, +11.1]) |

**How often the target is among the candidates (%):**

| Group | Rime's 16 sentences | Generated top 10 | Either |
| --- | ---: | ---: | ---: |
| All | 36.7 | 65.4 | 67.8 |
| Full pinyin | 88.1 | 96.0 | 97.6 |

Reading:

- **Abbreviated input: generation's lead is larger on the public set.** Mixed is +11.0 and initials +8.1, both significant.
- **Full pinyin: generation is significantly below the production logic** (−4.0, p 0.02).
  - The private benchmark showed the same direction (−1.3) without significance; the public set confirms it.
  - Reranking the 16 sentences with `r2-cont`'s conditional score (task) is worse in the same way (−3.6, p 0.004).
  - Scored as a plain LM (lm), `r2-cont` reaches 81.7%, level with v1.
  - So the problem lies in the three-task model's conditional scoring, not in the Rime pool: on full pinyin, the 16 sentences contain the target 88.1% of the time.
- **If production adopts generation, it should route by input kind:**
  - full pinyin keeps Rime 16 sentences + v1 rerank;
  - inputs with abbreviations use generation.

  Projected from the per-kind results (an extrapolation, not a paired test):

  | Set | Routed | Generation alone |
  | --- | ---: | ---: |
  | Private benchmark | about 30.2% | 29.9% |
  | Public set | about 40.3% | 39.6% |

## Limitations

- **Development only.** The private benchmark is one author's input and leans towards mixed abbreviation; the public set's keys are synthetic.
- **Incomplete coverage.**
  - `r2-base` was not run on prediction.
  - The third arm has not been evaluated yet. It will be filled in with the same scripts.
- **Desktop CPU only for latency.**

## Next steps

1. When the third arm arrives, evaluate it the same way and pair it with `r2-cont`.
2. Address the rerank and prediction regressions of the three-task model before deciding whether production switches to generation. Options include:
   - more plain data in the mix;
   - splitting generation into its own head or model.
3. The public recheck is done (above); its test split is kept for the final confirmation. The next three-task round has to address the conditional-scoring regression on full pinyin.
