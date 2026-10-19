# V4.1 joint pretraining

V4.1 completed sixteen epochs and 865,557 updates on 2026-10-16. It trained the fixed 8K, 15,070,848-parameter V4 architecture from random weights with seed **1219827974**, generated once from the operating system random source and then frozen. V2.1 remains the deployed model. [Detailed training record (Chinese)](zh-CN/training-v41.md) · [Full evaluation (Chinese)](zh-CN/reports/v41-20261016/evaluation.md).

## Data and objective

| Pool | Frozen population | Role |
| --- | --- | --- |
| Original V2.1 FineWeb/Tatoeba, encoded with V4 8K | 25,185,368 original training sentences; exclude 97 known validation overlaps | Traverse retained LM windows once per epoch |
| V3 RealPersonaChat/MRMP | 477,334 training turns; 4,986 validation turns | Approximately 5% of actual cropped LM target tokens, with repeated sampling |
| C1r2 conversion | 982,853 retained training examples; 18,675 validation examples | One conversion pass per LM epoch |

Original split assignments and reference labels remain unchanged. Chat uses independent turns with dialogue-level splits. Conversion uses dual-dictionary silver readings derived from the original LM text. Preparation excludes known exact cross-pool and validation overlaps without reading test/blind data. V3 JpnMix web additions are absent from V4.1.

The objective is `0.8 × LM + 0.2 × conversion`, each normalized by its own supervised target count per update. LM batch is 512; exact conversion quotas distribute 18 or 19 examples per update. Total exposure is 9,683,477,884 LM targets and 157,132,320 conversion targets. Sixteen LM epochs therefore do not mean equal-sized batches or equal token exposure for the two tasks.

BF16, compiled backbone, fused AdamW, context 128, and 30% prefix cropping follow the frozen [configuration](../configs/train-v41.toml). The schedule spans all sixteen epochs from 6e-4 to 6e-5 after 1,000 warmup updates. There is no plateau or LM-retention early stop in this experiment. Checkpoints retain AdamW, RNG, committed data cursors, and validation history.

## Results

All local IME comparisons use frozen pools, primitive FP32, disabled TF32, full-vocabulary suffix log-probability sums, stable ties, and no truncation or final EOS. Reading-conditioned scores supervise and score a distinct structured input path. Existing V2.1/V4-B cached results use the same candidate and reference snapshots.

| Metric | V2.1 | V4-B | V4.1 epoch 13, ordinary LM | V4.1 epoch 16, ordinary LM | V4.1 epoch 16, CTX/READ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard development /728 | 505 | 498 | **509** | 505 | 461 |
| Legacy regression /500 | 374 | **377** | 370 | 376 | 356 |
| AJIMEE /200 | **144** | **144** | 132 | 137 | 98 |
| Historical development /137 | 122 | 120 | **124** | **124** | 118 |

Epoch 13 was selected on Standard development before the local regression comparisons. Its +4 versus V2.1 comprises 45 improvements and 41 regressions; descriptive paired p=0.74653. Epoch 16 is more balanced on the regression sets but does not establish a consistent replacement advantage. These exposed, correlated development/regression sets are not a new independent benchmark.

Final complete LM BPC is **3.121224**, versus V2.1 **3.080191** and V4-B **3.027777**, on the same original validation text with FP32 evaluation. Best chat BPC is **3.153930** at epoch 15; final chat BPC is **3.172254**. Final conversion validation NLL is **0.207814**.

On the same epoch-16 weights, clearing READ improves structured Standard scores from **461 to 502**, near ordinary LM's **505**. READ removes 48 errors but introduces 89, a net loss of 41. This input ablation changes content, lengths, and positions; it does not isolate the underlying training cause. Empty READ is also a training view. The reading path is unsuitable for deployment based on these results.

## Source and artifacts

| Component | Path |
| --- | --- |
| Data preparation | `scripts/corpus/prepare_v41.py`, `src/vimeml/training/v41_data.py` |
| Formal trainer | `scripts/training/train_v41.py`, `src/vimeml/training/joint_v41.py` |
| AutoDL handoff | `scripts/training/prepare_v41_handoff.py` |
| Local comparison | `scripts/benchmarks/evaluate_v41.py`; requires a separate empty output directory |
| Weights | `artifacts/models/tiny-ja-v41-e8k-d384-l7/` |
| Download receipt and actual training source | `outputs/training-v41-20261016/` |
| Frozen local evaluation, original script snapshot, and acceptance | `outputs/evaluation-v41-20261016/` |

`best-ordinary.pt` is epoch 13, step703259; `last.pt` is epoch 16, step865557. `best.pt` and `best-rerank.pt` select conditional reranking, not ordinary-LM quality. Six role checkpoints plus epochs 14 and 15 were downloaded with SHA256 verification, along with every epoch's evaluation records and the exact training source. The completed run's source fingerprints remain bound to that archived source; subsequent repository changes do not rewrite experiment provenance.

The repository comparison entry preserves the scoring logic of the archived local script and adds an explicit output argument; the completed evaluation retains its original script hash. It does not automatically run another evaluation or training job. Datasets, weights, transfer utilities, and credentials remain outside Git.

Next independent testing should register the two ordinary-LM candidates in advance and report domain regressions alongside aggregate gains. V4.1 changes data, tasks, seed, and budget together, so it does not establish a causal benefit for any one component. No deployment replacement, additional training, or Core ML/device validation follows automatically from this experiment.
