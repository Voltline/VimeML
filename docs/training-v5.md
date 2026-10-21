# V5 scale-up pretraining

V5 trains the V4.3 recipe unchanged at d512/L8 to test whether the 15M model is capacity-bound. Data, seed, objective, optimizer and token budget match V4.3; only the model size differs. Both the controlled ten-epoch and the thirty-epoch runs are complete and evaluated locally. The larger model lowers LM BPC by about 0.10 but does not improve IME Top-1, so V5 is not converted to Core ML or measured on device. V2.1 remains deployed and V4.2 the opt-in variant. [Detailed record (Chinese)](zh-CN/training-v5.md).

## Motivation: V4.3 saturation

V4.3 adds nine FineWeb2-Edu shards to the V4.2 pool (cross-pool deduplicated, roughly doubling web windows per epoch). On the same original validation text:

| Run | Supervised tokens | LM BPC |
| --- | ---: | ---: |
| V4.2, 20 epochs | ~12B | 3.045152 |
| V4.3 controlled, 10 epochs | ~12B | 3.048984 |
| V4.3, 30 epochs | ~36B | 3.043831 |

At the same token budget, twice the new data gives no BPC gain, and tripling tokens lowers BPC by only 0.0013. Local FP32 Top-1 (Standard /728, Legacy /500, AJIMEE /200, historical development /137) is 523/370/139/123 for V4.2 epoch 20 and 519/367/143/123 for V4.3 epoch 30, with no significant difference. V4.3 source and configurations are in the repository (`joint_v43.py`, `v43_data.py`, `train-v43.toml`, `train-v43-10ep.toml`) and match the run's `code_sha256`; [`evaluate_v43.py`](../scripts/benchmarks/evaluate_v43.py) is the local evaluation entry.

## Architecture

| Setting | V4.2/V4.3 | V5 |
| --- | --- | --- |
| d_model / layers / heads | 384 / 7 / 6 | 512 / 8 / 8 |
| SwiGLU width | 960 | 1280 |
| Parameters | 15,070,848 | 28,385,792 |
| KV cache shape | (7,1,6,128,64) | (8,1,8,128,64) |

Head width stays 64 and FFN width 2.5×d. Everything else follows V4: RMSNorm, SwiGLU, no biases, tied head, learned positions, context 128, and the V4 8K Unigram tokenizer (SHA256 `296a2bdf…81b5`). V5 is a separate architecture identity (`tiny_gpt_v5`); `TinyGPTV4` refuses V5 checkpoints and vice versa.

## Training settings

| Configuration | Control | Budget |
| --- | --- | --- |
| [`train-v5.toml`](../configs/train-v5.toml) (controlled-10ep) | `train-v43-10ep.toml` | 10 epochs, ~12B supervised tokens |
| [`train-v5-30ep.toml`](../configs/train-v5-30ep.toml) | `train-v43.toml` | 30 epochs, ~36B supervised tokens |

| Setting | Value |
| --- | --- |
| Data | Frozen V4.3 mixture `corpus-v43-mixed-c128` (manifest `81d7e610…`), first 10 or all 30 epoch maps |
| Seed | 1219827974 |
| Objective | Ordinary unmarked next-token loss, normalized per update, 5% chat |
| Optimizer | Fused AdamW, lr 6e-4 → 6e-5 cosine, 1,000 warmup, weight decay 0.1, clip 1.0, batch 512 |
| Precision | BF16 AMP, compiled backbone; complete FP32 validation every epoch |
| Cost | One L20C: 3.83 h (10 epochs) and 11.83 h (30 epochs) of update time; 12.7 GiB peak allocation |

Epoch maps, crops and sampling equal V4.3's, so differences from V4.3 at the same epoch count come from model size alone (initialization necessarily differs with shape). Entry points are `scripts/training/train_v5.py` (`--dry-run` validates configuration, code and data identity) and `scripts/training/cluster_v5.sh` (resumes from `last.pt`; set `VIMEML_PYTHON`). Both runs' `code_sha256` matches the repository byte for byte; the download receipt is in `outputs/training-v5-20261021/`.

## Results

Complete LM BPC is best at the final epoch in both runs:

| Run | LM BPC | Expanded web BPC | Chat BPC |
| --- | ---: | ---: | ---: |
| V4.3 controlled, 10 epochs | 3.048984 | 3.055103 | 3.110230 |
| **V5 controlled, 10 epochs** | **2.948747** | **2.952379** | **3.084419** |
| V4.3, 30 epochs | 3.043831 | 3.049611 | 3.128712 |
| **V5, 30 epochs** | **2.942180** | **2.945641** | **3.108036** |

V5 after 10 epochs already beats V4.3 after 30. The gap to V4.3 widens from 0.077 at epoch 1 to 0.102 at epoch 30.

Local FP32 Top-1 uses [`evaluate_v5.py`](../scripts/benchmarks/evaluate_v5.py) with the V4.2 scoring semantics; the fixed reference and blind sets were not read.

| Model | Standard /728 | Legacy /500 | AJIMEE /200 | Historical dev /137 |
| --- | ---: | ---: | ---: | ---: |
| V2.1 | 505 | 374 | 144 | 122 |
| V4.2 epoch 20 | 523 | 370 | 139 | 123 |
| V4.3 controlled epoch 10 | 514 | 378 | 142 | 125 |
| V4.3 epoch 30 | 519 | 367 | 143 | 123 |
| V5 controlled epoch 10 | 504 | 372 | 145 | 123 |
| **V5 epoch 30 (primary)** | 514 | 365 | 145 | 124 |

Paired exact McNemar, net (improved/regressed, p):

| Comparison | Standard | Legacy | AJIMEE | Historical dev |
| --- | --- | --- | --- | --- |
| V5-10ep vs V4.3-10ep (size only) | −10 (33/43, 0.30) | −6 (14/20, 0.39) | +3 (9/6, 0.61) | −2 (2/4, 0.69) |
| V5 vs V4.3, epoch 30 (size only) | −5 (30/35, 0.62) | −2 (13/15, 0.85) | +2 (8/6, 0.79) | +1 (3/2, 1.00) |
| V5 epoch 30 vs V4.2 epoch 20 | −9 (31/40, 0.34) | −5 (17/22, 0.52) | +6 (13/7, 0.26) | +1 (6/5, 1.00) |

No size-only comparison is significant, and directions disagree across tracks. Epoch 30 is the primary candidate under the preregistered best-BPC rule. Epoch 29 (Standard 522) was suggested on the cluster as "most balanced", but its BPC is worse than epochs 27, 28 and 30, so choosing it would be selection on development IME scores; epochs 29 and 30 differ by 62 Standard flips, net +8 (p=0.37), which is noise.

## Conclusions

1. Scale lowers perplexity but does not turn into IME accuracy. More data (V4.3) and a larger model (V5) both improve BPC while Top-1 stays at Standard 504–523 and AJIMEE 138–145, within paired noise. Under fixed AzooKey pools and suffix log-probability scoring, LM quality is no longer the bottleneck; the [error decomposition (Chinese)](zh-CN/reports/analysis-20261021/ime-errors.md) attributes the remaining Standard errors mainly to candidate-pool misses and notation variants.
2. V5 is not deployed. At 1.9× V4.2's parameters (about 32 MB INT8), linear extrapolation from V4.2 measurements puts iPhone stateless rerank p95 near 230 ms against a 250 ms soft budget, with no accuracy gain. No Core ML conversion or device measurement was made.
3. No blind set was consumed, because V5 gives no grounds to replace V2.1 or V4.2.

The size-only control answers whether a larger model is better on the same data and budget: yes for LM, no for IME reranking. It says nothing about other scoring methods, post-training or data mixtures. All IME numbers come from nonblind development and regression sets and support descriptive and negative conclusions only.
