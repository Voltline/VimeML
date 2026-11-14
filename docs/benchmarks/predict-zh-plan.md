# Chinese prediction benchmark and desktop model roadmap (preregistered)

Drafted 2026-11-11. The plan counts as registered once committed; later changes are recorded as deviations at the end. It fixes three things: how the prediction benchmark `ime-predict-zh-v1` is built, a scaling survey with existing models once it is built (step 0), and the order and gates for training 200M to 800M models. Training configs and blind confirmation get their own plans. The registered text is the Chinese version, [docs/zh-CN/benchmarks/predict-zh-plan.md](../zh-CN/benchmarks/predict-zh-plan.md); this page is its translation. The built benchmark is described in [Chinese prediction benchmark](predict-ime-zh.md); the 200M plan is [training-zh-200m-plan](../training-zh-200m-plan.md).

## Background and questions

`vimeml-tiny-zh-v1` (20M) is near the ceiling of the reranking benchmark: blind full pinyin 93.8% with the pool covering only 97.4% of answers; abbreviated pinyin is capped by the pool at about 41% ([results](../reports/zh-v1-20261111/blind-and-baselines.md)). That benchmark cannot separate larger models, so the next target is **prediction**: after the user commits some text, predict what they will type next.

Larger models are aimed at the **desktop** (Mac), not the phone keyboard extension. This plan asks:

1. On prediction, how far is v1 from usable, and how much better are existing larger models?
2. Going from 20M to 200M and 800M parameters, how much does each prediction metric gain, and is it worth training?
3. At 200M, do vocabulary (16K, 24K, 32K) and context length (128, 512) matter?

## Prediction benchmark `ime-predict-zh-v1`

### Sources

| Source | Notes |
| --- | --- |
| CrossWOZ, RiSAWOZ, KdConv | Same versions as the [reranking benchmark](standard-ime-zh.md), using only dialogues **not** in `ime-standard-zh-v1-r1` (development or blind) |
| One chit-chat source | Chosen by the rule below |

**Chit-chat source rule**: check NaturalConv, DuLeMon, CPED in that order and take the first that meets all three:

1. the license allows research use and the raw dialogue text is obtainable;
2. it is not part of LCCC and not in our training data list;
3. after a leakage scan over all its dialogues, no more than 2% of dialogues hit.

If none qualifies, the benchmark uses only the first three sources and says so in its description. The choice looks only at licenses, source lists and scan totals, never at any model's score on the source.

### Sampling and splits

- Fixed seed 20261111. 1,200 dialogues per source, shuffled within the source by the seed, 600 each for development and blind.
- **Leakage filtering comes before sampling**: candidate dialogues are scanned against the training export `export-r1/train` with 16-character Han n-grams (method as in the [leakage check](../reports/zh-v1-20261110/ime-benchmark.md#leakage-check)); any dialogue with a hit is removed whole, with removals reported per source. This is a mechanical filter and reads no blind text.
- Only the human side's turns are used (user turns in task dialogues). Left context is all earlier turns of the same dialogue plus the already committed part of the current turn.
- Annotated entities and slot values are not covered, as in the reranking benchmark.
- The label is the original text itself, with no annotation. One mechanical cleanup (removing turns with too few Han characters or with garbled text) is written into the build script.

### Track A: next-word hit

- **Items**: 3 positions per dialogue. A position must be between Han characters and be followed by at least 2 more Han characters before the next punctuation. Positions are drawn at random by the seed, not picked. About 7,200 items per split (with four sources).
- **Output**: up to 5 candidates, each a string of 2 to 6 Han characters, ordered by confidence.
- **Hit**: the candidate is a prefix of the true continuation (up to the next punctuation). This depends on no tokenizer.
- **Metrics**: Top-1, Top-3, Top-5 hit rates; mean length of hit candidates. **Top-3 hit rate is the primary metric of the whole plan.**

### Track B: keystroke savings

- **Simulation**: each sampled turn is typed from the start. After each committed segment the system offers up to 3 predictions; if one is a prefix of the true continuation the user picks the longest such one (1 keystroke), otherwise they keep typing full pinyin.
- **Keystrokes**: typing a Han character in full pinyin costs as many keystrokes as its pinyin letters, with readings from pypinyin; selection keys are not counted, the same with and without prediction.
- **Metrics**: keystroke savings = 1 − keystrokes with prediction ÷ keystrokes without; also prediction display count, pick count and pick rate.
- Each system's display threshold is tuned on development only and carried over to blind.

### Track C: prefix completion

- **Items**: 1 per dialogue, the target a 2–6-character fragment (sampled as for the reranking benchmark's abbreviated track). Input is the left context plus incomplete pinyin, in two modes: only the first syllable typed; the initial of every character typed.
- **Output**: up to 5 candidates that must be compatible with the typed pinyin (judged with the rime-ice single-character reading table).
- **Metrics**: exact Top-1, Top-3, Top-5. This track uses no Rime pool and has no pool ceiling.
- **Controls**: rime-ice and rime-ice + Wanxiang grammar model on the same keystrokes.

### Track D: confidence

From track A results, plot "display rate vs hit rate" by the system's confidence and report Top-1 hit rate at display rates of 20% and 50%. Descriptive only.

### How language models produce candidates

All language models use the same decoding, so differences come only from the models:

- Input is `start + left context`, FP32 (external models above 1B may use BF16, recorded).
- Beam width 16, expanded token by token, keeping only all-Han hypotheses, at most 6 Han characters.
- A candidate's score is the sum of its log-probabilities as a prefix of the continuation. The top 5 strings of 2–6 characters, none a prefix of another, are taken by score; track A's confidence is that score.
- Track C keeps only pinyin-compatible hypotheses during expansion.
- Decoding parameters are fixed by this plan and not tuned per model. The only per-system tuning on development is track B's display threshold.

### Tests

- Tracks A and C: paired exact McNemar; a dialogue has several items, so cluster bootstrap intervals and cluster sign-flip tests by dialogue are also reported, and decisions use the cluster tests.
- Track B: cluster bootstrap intervals by dialogue.
- Blind stays closed after freezing and is opened once only, in the confirmation plan for the final desktop model. Every decision in this plan uses development only.

## Step 0: scaling survey (no training)

Once the benchmark is frozen, measure these systems on development first:

| System | Purpose |
| --- | --- |
| Character 5-gram (trained on a fixed sample of `export-r1/train`) | Lowest reference |
| librime prediction plugin (if it builds on librime 1.17.0) | Prediction level of an existing IME; recorded as skipped if it cannot be built |
| `vimeml-tiny-zh-v1` | Our starting point |
| Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B (base) | Three sizes of one family, to see how scale relates to prediction metrics |

**How the results are used**:

- If on the primary metric (track A Top-3) Qwen2.5-3B is less than 2 points above Qwen2.5-0.5B, or the cluster test is not significant, "adding parameters gains little on this task". 200M is still trained, but 800M is deferred in favor of context length and data.
- If v1 is already no lower than Qwen2.5-0.5B, record it; it shows a 20M specialized model is competitive on prediction too, but does not change the decision to train 200M.
- External models' training data is unknown and may include these public dialogue sets. Their scores are used only for the scaling trend, not as targets to beat.

## Training roadmap

### Desktop constraints

> From 2026-11-14, 200M and larger sizes are not deployed on mobile and no Mac INT8 package size, memory or latency is measured; see deviation 5.

The target platform is Apple Silicon Mac. Each size must report INT8 package size, resident memory, and p50 and p95 latency from commit to prediction. The provisional budget is p95 at most 100 ms and resident memory at most 1 GB; these may be revised once after the first 200M measurement, with the revision recorded as a deviation. Sizes over budget are still reported but are not release candidates.

### 200M

Four arms plus two seeds, same budget, same data:

| Arm | Vocabulary | Context | Purpose |
| --- | --- | --- | --- |
| `200m-16k-c512` | 16K | 512 | Vocabulary contrast |
| `200m-24k-c512-s1/s2/s3` | 24K | 512 | Default recipe, three seeds estimate noise |
| `200m-32k-c512` | 32K | 512 | Vocabulary contrast |
| `200m-24k-c128` | 24K | 128 | Context contrast |

- The decision uses v1's "effective" standard: the primary-metric difference exceeds the three-seed range, and the Holm-corrected cluster test p < 0.05. The test family is three contrasts: 16K vs 24K, 32K vs 24K, c128 vs c512.
- Defaults when none is effective: 24K, context 512. The vocabulary default is the middle value; context defaults to 512 because the desktop has no phone-like memory limit and prediction relies on longer left context.
- Data stays `export-r1`, unrepeated (v1's conclusion). Context 512 needs rebuilt window data.
- Shape (layers, width), budget and seeds are registered in the 200M training plan and committed before training.

### Gate for going larger

800M is trained only if the 200M default arm, compared with v1 on development, meets all three:

1. track A Top-3 at least 3 points higher, with cluster test p < 0.05;
2. the cluster bootstrap interval of track B keystroke savings has a lower bound above 0;
3. no full-pinyin Top-1 drop with McNemar p < 0.05 on reranking benchmark `ime-standard-zh-v1-r1` development.

> **Decided 2026-11-14: all three met.** 200M default arm `24k-c512-s1` against v1:
> 1. Track A Top-3 27.60% → 31.64% (+4.04 points, +672/−381, cluster sign-flip p ≈ 5×10⁻⁵, cluster bootstrap [+228, +356] items);
> 2. Track B keystroke savings 16.74% → 20.62% (+3.88 points, each at its development-chosen threshold, cluster bootstrap 95% interval over 2,400 dialogues [+3.61, +4.14]);
> 3. Reranking full-pinyin Top-1 1,363 → 1,370 (+22/−15, McNemar p 0.32), no drop.
>
> The step-0 survey is deferred (deviation 4), so step 0's "defer 800M if parameters gain little" condition is not yet evaluated; it will be checked once the survey is done.

- If met, train 800M directly, giving three points: 20M, 200M, 800M. 500M is trained only if 800M exceeds the desktop budget and a middle size is needed.
- If not met, no larger model is trained; instead analyze where the gap comes from (data, context or decoding) in a separate plan.
- The current 22.4B characters are on the low side for 800M. Whether to add web data is decided before training 800M and written into its plan.

### Every size reports

- Development results on the four prediction tracks, paired with v1 and the previous size.
- Reranking benchmark development results (blind as a reference test only, under the rules of the [baseline plan](baselines-zh-v1-plan.md)).
- Web and chat validation BPC and the training-document memorization sample. Memorization matters more as models grow, so one more item: for each prediction development item, check whether the model's Top-1 candidate occurs verbatim in the training data, and report the rate.
- Desktop package size, memory and latency (not measured from 2026-11-14, see deviation 5).

## Order of work

1. Commit this plan.
2. Choose the chit-chat source, filter for leakage, build and freeze `ime-predict-zh-v1`, and write the benchmark description (source versions, licenses, counts, SHA256s).
3. Write the evaluation scripts and decoder and run them end to end with v1 on development; fix decoder correctness with unit tests.
4. Step 0: measure the systems above and write the survey report. (Deferred, see deviation 4.)
5. Write and commit the 200M training plan, train, decide.
6. Decide on 800M by the gate. (2026-11-14: gate met.)
7. Once a desktop model is chosen, write a blind confirmation plan and open the prediction blind once.

## Limits

- Dialogue data is not real keyboard input. Task dialogues are tidier than everyday typing, and chit-chat sources are mostly crowdsourced writing.
- Track A counts only the true continuation as a hit; reasonable but different predictions are misses, so hit rates are conservative lower bounds, good for comparing models but not a measure of perceived usefulness.
- The keystroke-savings user model is simple: it assumes the user always sees and picks a hit, with no cost for reading candidates.
- pypinyin annotates readings automatically and makes a few errors on polyphones, affecting keystroke counts equally on both sides.
- The leakage scan finds only verbatim overlap. External models' leakage cannot be checked.

## Deviations

1. 2026-11-11: the chit-chat choice skipped DuLeMon. By the rule's order NaturalConv was checked first: license and source meet conditions 1 and 2, but the leakage scan hit 6.0% of dialogues, failing condition 3. The next, DuLeMon, is available only from the Qianyan platform, whose participant agreement section 2.2 allows non-commercial research use, literally meeting condition 1; but section 4.4 requires granting the data provider a free, perpetual license, including commercial use and sublicensing, to any results developed with the dataset (algorithms, code, data, papers and so on). Vime will ship, and we do not accept this term, so it was treated as failing condition 1 before scanning and CPED was checked next. The decision rests only on the license text, without looking at DuLeMon's scan results or any model score.
2. 2026-11-11: implementation details of decoding and the keystroke simulation. The plan only stated "input is start + left context" and the main beam parameters; the following rules were added during implementation, all fixed before any model was scored (only a pipeline check on 5 items per source was run, after which the decoding rules did not change); details in the [benchmark description](predict-ime-zh.md#decoding-and-simulation-details): left context is joined in each model's training format, dropping the earliest whole turns when it does not fit; when it ends in Han characters the last all-Han token is backed off so the model can continue across a token boundary; track C "compatible" is judged with rime-ice single-character readings, and initials mode requires the character count to equal the keystroke count; track B uses jieba words as typing units, offers no prediction at the start of a turn, counts 1 keystroke per non-Han character, and chooses the threshold from a quantile grid of track A first-candidate scores; "mean length of hit candidates" means the highest-ranked hit within the Top-3.
3. 2026-11-11: implementation details of the step-0 systems. All fixed before any system other than v1 was scored; v1's development results (step 3) were already known, but these rules only concern how external systems are plugged in and were not adjusted by any score. The decoder is unchanged; only adapters differ (`src/vimeml/benchmarks/predict_zh_external.py`):
   - **The librime prediction plugin is recorded as skipped.** The plugin itself is in the prebuilt librime (33e7814, with `rime/librime-predict` 920bd41), but the prediction database `predict.db` it needs has no public source: the plugin repository ships only a build tool; the Wanxiang project once published one but now learns on the user's machine, and the file is no longer on its release page. A database built from export-r1 would no longer be "the prediction level of an existing IME", so this is not done.
   - **Character 5-gram**: a fixed sample of `export-r1/train`: web and review documents chosen by the first 8 bytes of `sha256(id)`, about 9.5M Han characters, and LCCC dialogues by `sha256("chat:"+dialogue id)`, about 0.5M characters, matching v1 training's roughly 5% chat. Interpolated Kneser-Ney, orders 1 to 5, one discount per order D = n1 / (n1 + 2·n2), unigrams further interpolated with uniform; characters seen fewer than 2 times map to an unknown symbol that decoding cannot output. Each document and dialogue is counted separately with a leading newline; paragraphs and turns are separated by newlines, and left context and the current input are also joined by a newline.
   - **Qwen2.5 base**: left context and current input joined by newlines (base models have no chat format), no BOS; context limit 512 tokens, dropping the earliest whole turns when it does not fit (v1 is limited to 128 by its architecture). Only tokens that decode to all-Han text can enter candidates, so rare characters split into several byte-level BPE tokens cannot be output. 0.5B uses FP32, 1.5B and 3B BF16. Model versions are the current Hugging Face commit at run time, recorded in `meta.json`. The context KV cache is computed once per decode and later beam steps compute only new tokens; a unit test confirms its log-probabilities match a full forward pass.
4. 2026-11-14: the step-0 scaling survey is deferred and the 800M gate is decided first. The development machine cannot run Qwen2.5-1.5B/3B, and the local 0.5B run did not finish; the character 5-gram and v1 are measured. The survey moves to the training machines (systems, decoding and adapter rules unchanged, see deviation 3); once results exist, the survey report is written and step 0's condition "defer 800M if 3B is less than 2 points above 0.5B or the cluster test is not significant" is checked. The 800M gate was decided before the survey under the original rules and all three conditions are met (see "Gate for going larger"); the gate rules are unchanged. Two speed-only issues were found when running the survey, neither changing any output: CPU thread counts defaulted to the visible cores, far above what the job may use, so the runs set thread counts explicitly; and the HF adapter called `len(tokenizer)` inside its per-token loop at start-up (a fast tokenizer rebuilds its vocabulary on every call, taking over ten minutes for a 152K vocabulary), now computed once. Decoding rules are unchanged.
5. 2026-11-14: 200M and larger sizes (including 500M and 800M) are not deployed on mobile. The Mac INT8 package size, resident memory and p50/p95 latency measurements in "Desktop constraints" and "Every size reports" are dropped, and p95 ≤ 100 ms / resident ≤ 1 GB is no longer a release-candidate condition. The premise of "500M only if 800M exceeds the desktop budget" no longer exists; whether to train 500M is decided separately. No judged metric is affected.
