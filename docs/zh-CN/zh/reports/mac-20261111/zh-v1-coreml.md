# vimeml-tiny-zh-v1 Core ML 转换、INT8 量化与量化调查 —— 2026-11-11

源模型为 `artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/deployment.pt`，第 844,730 步（源 SHA256 `44c423a1…`，部署张量 `7959fea4…`），`TinyGPTZh13`：词表 12,288，d384，9 层，6 头，SwiGLU 960，上下文 128，共享输出头，20,036,736 参数。Tokenizer `zh-unigram-12288`（`68bd37c4…`）。仓库之后被重建；工作在重建后 `main`（`9d3a687`）的全新克隆中继续，被忽略的产物和输出拷贝过去，并核对了代码完全相同。按[交接任务说明](../../mac/mac-zh-v1-coreml.md)执行。只读取了开发集。没有重新训练、选择 checkpoint、改变候选池或参考答案，也没有改变打分。V4.2、V4.3 和 V2.1 的包与记录未动。

环境：Apple M3，16 GiB，macOS 27.3（26C5035r），Xcode 27.2（27B48），Python 3.11.17（uv venv），PyTorch 2.7.1，coremltools 9.0，NumPy 2.3.5，scikit-learn 1.5.1，sentencepiece 0.2.1。Core ML 目标：mlprogram，iOS 18，FP32 计算，`CPU_ONLY`。

开发集分母：全拼赛道 1,456 条，简拼赛道 990 条。下文只给条目 id，不引用基准文本。

## 摘要

| 步骤 | 结果 |
| --- | --- |
| 1. Mac PyTorch CPU FP32 基线 | 通过：fixture 最大 3.7e-5，`forward_cached` 逐步最大 1.5e-5；Top-1 1363/337 与 Windows 参照相同；两个赛道 0 个排序变化 |
| 2. 通用化 Core ML 代码 | 新增 `vimeml.deployment.zh_v1`（12,288 / 9 层，缓存 `(9,1,6,128,64)`）；V4 模块和格式不变 |
| 3. FP32 Core ML | 无状态最大 3.9e-5，chunk-KV 最大 4.6e-5（22 项检查）；两者都是 1363/337，对 Mac PyTorch +0/−0、0 个排序变化 |
| 4. INT8 block32 | 严格 logits 门槛如预期失败（最大 0.392；V4.2 为 0.477）。对 FP32，Top-1 全拼 1364（+2/−1），简拼 335（+0/−2）。chunk-KV INT8 与无状态 INT8 相差 4.2e-5 以内，所有排序相同 |
| 5. 量化调查 | 只看大小与准确率，不选配方：INT4 b32、调色板 6 bit 和 4 bit（分组通道 32）、INT8 b32 + FP16 embedding |
| 6. 与 V4.2 的 Mac 测量对比 | 用提供的 V4.2 包完成（未重建）。INT8 大小是 V4.2 的 1.33 倍；无状态 `predict` p50 为 1.28–1.37 倍（T=128：8.49 ms 对 6.43 ms）；模型峰值内存增量 18.4 对 13.4 MiB |
| 7. Vime iPhone 测试宿主 | 资源已准备。Vime 端的 `.zhV1` 变体留给 Vime 自己的维护者；在模拟器上做了一次性本地检查并通过（4/4），之后已撤回。基准测试和 iPhone 运行**未做** |

## 1. Mac PyTorch 基线

`outputs/deployment/zh-v1-mac-pytorch-baseline-20261111`。`TinyGPTZh13` 在 CPU FP32（4 线程）上加载 `deployment.pt`。全部五个 fixture 用例（`bos`、`short`、`long`、`right_pad`、`changed_future` 的有效前缀）都在 Windows logits 的 atol=rtol=3e-4 以内；最大误差 3.72e-5。在 `long` 上逐 token 的 `forward_cached` 与 `long_kv_step_logits` 一致，最大误差 1.5e-5。

| 赛道 | Windows CUDA FP32 | Mac PyTorch | 改善 / 退步 | 排序变化条目 | 最大候选分数差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全拼 / 1,456 | 1363 | 1363 | +0/−0 | 0 | 5.7e-5 |
| 简拼 / 990 | 337 | 337 | +0/−0 | 0 | 5.3e-5 |

分数文件哈希为 `cde45ed7…`（全拼）和 `746b9697…`（简拼）。打分沿用 `zh13_lm_scores.score_item`，未改动：BOS + 上下文与候选联合分词，公共前缀之后的后缀对数概率求和，不加 EOS；结果与对比由 `score_standard_ime_zh.py score` 和 `compare` 生成。

## 2. 通用化 Core ML 代码

V4 代码（`v4_coreml`、`v4_chunk`）在约定常量和格式中写死了 8,192 / 7 层。没有改它，而是在旁边新增一个薄模块 `src/vimeml/deployment/zh_v1.py`，复用共享部分：

- 约定：`VOCAB=12288`、`LAYERS=9`、`CACHE_SHAPE=(9,1,6,128,64)`，容差（3e-4）和权重阈值（2048）相同。
- 新格式：`vimeml_inference_bundle_zh_v1`、`vimeml_coreml_zh_v1`、`vimeml_coreml_zh_v1_kv_chunk`、`vimeml_alignment_zh_v1`、`vimeml_kv_chunk_alignment_zh_v1`。
- 无状态转换复用 `graph.trace_graph`；chunk-KV 转换复用 `v4_chunk.KVChunkGraphV4`（它对形状是通用的）。
- 为未改动的打分器提供运行时适配器：`BundleLMZh`、`CoreMLLMZh`，以及 `CachedChunkLMZh`——它为 `score_item` 服务，公共前缀一次 chunk 调用，每个候选后缀再一次调用。`zh13_lm_scores.run` 新增可选参数 `loaded=(model, processor, meta)`；默认路径不变。
- `scripts/zh/deployment/coreml_zh_v1.py` 提供各阶段（`baseline`、`export`、`convert`、`convert-chunk`、`align`、`align-chunk`、`compress`、`evaluate`），每个阶段输出到新目录。
- `tests/test_deployment_zh_v1.py`（6 个测试）：约定常量、chunk 图 trace 对完整前向、缓存打分对无状态 `score_item`、稳定并列的名次差异、变化条目 id、推理包一致性。用 `python -m unittest` 运行完整单元测试通过：39 个 OK，5 个跳过（改动前为 22 OK / 5 跳过）。

## 3. FP32 转换与对齐

| 包 | 接口 | 逻辑字节 |
| --- | --- | ---: |
| `tiny-zh-v1-fp32-v1` | `input_ids int32[1,T]`，1≤T≤128 → `logits float32[1,T,12288]` | 80,367,198 |
| `tiny-zh-v1-kvchunk-fp32-v1` | `input_ids int32[1,T]`、`position int32[1]`、`key_cache`/`value_cache float32(9,1,6,128,64)` → `logits`、新缓存 | 80,404,135 |

| 检查 | 最大绝对误差 | 结果 |
| --- | ---: | --- |
| PyTorch 推理包，五个 fixture（`zh-v1-align-pytorch-bundle`） | 3.72e-5 | 通过 |
| 无状态 Core ML，五个 fixture（`zh-v1-align-fp32`） | 3.91e-5 | 通过；填充 / 未来 token 不变性完全成立 |
| Chunk-KV Core ML 对 fixture：一次性、分块、逐 token、共享前缀分支、回滚、快照不可变 —— 22 项检查（`zh-v1-align-kvchunk-fp32`） | 4.63e-5 | 通过 |

| 赛道 | Mac PyTorch | Core ML FP32 无状态 | Core ML FP32 chunk-KV |
| --- | ---: | ---: | ---: |
| 全拼 | 1363 | 1363（+0/−0，0 个排序变化） | 1363（对无状态 +0/−0，0 个排序变化，最大差 3.4e-5） |
| 简拼 | 337 | 337（+0/−0，0 个排序变化） | 337（对无状态 +0/−0，0 个排序变化，最大差 3.1e-5） |

## 4. INT8 block32

配方同 V4.2：`linear_symmetric`，INT8，`per_block` 块大小 32，权重阈值 2048，参数选择用 `v2.parameter_selection`。量化 47 个矩阵（20,029,440 个元素）：token embedding `[12288,384]`（gather 与共享输出头共用）、位置 embedding `[128,384]`，以及每层 × 9 的融合 QKV `[1152,384]`、注意力投影 `[384,384]`、FFN gate 和 up `[960,384]`、FFN down `[384,960]`。保持不变：`causal_mask` `[128,128]` 和零输出头偏置 `linear_45_bias_0` `[12288]`；norm 不是矩阵。FP32 计算，不量化激活。

| 包 | 逻辑字节 |
| --- | ---: |
| `tiny-zh-v1-int8-b32-v1` | 22,792,958 |
| `tiny-zh-v1-kvchunk-int8-b32-v1` | 22,830,199 |

严格 logits 门槛（`zh-v1-align-int8-b32`）**如预期失败**；它作为证据保留，没有为绕过它做调整：

| 用例 | 最大 | 均值 | RMSE | p50 | p90 | p99 | p99.9 | 容差外 | Argmax 相同 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bos | 0.128 | 0.0267 | 0.0327 | 0.0238 | 0.0522 | 0.0805 | 0.1089 | 95.2% | 1/1 |
| short | 0.310 | 0.0408 | 0.0525 | 0.0330 | 0.0859 | 0.1468 | 0.2015 | 95.2% | 15/16 |
| long | 0.392 | 0.0463 | 0.0598 | 0.0374 | 0.0972 | 0.1700 | 0.2349 | 95.5% | 127/128 |
| right_pad、changed_future | 同 short | | | | | | | | |

填充和未来 token 不变性完全为 0。

Chunk-KV INT8 对 FP32 fixture 失败（最大 1.294，来自共享前缀分支检查；一次性 `long` 为 0.392）——这是预期中 INT8 误差经缓存前缀累积的结果。对**同一权重的无状态 INT8** 包则通过：21 项检查，最大 4.15e-5（`zh-v1-align-kvchunk-int8-b32-vs-stateless-int8`）。iOS 资源使用的就是这个门槛。

开发集排序，参照 = Core ML FP32 无状态：

| 赛道 | FP32 | INT8 无状态 | 改善 / 退步 | McNemar p | 排序变化条目 | Top-1 变化条目 | 最大分数差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全拼 | 1363 | 1364 | +2/−1 | 1 | 878 | 4 | 0.661 |
| 简拼 | 337 | 335 | +0/−2 | 0.5 | 619 | 17 | 1.465 |

"排序变化条目"统计完整候选顺序随任何分数变动而改变的条目（稳定并列）；"Top-1 变化"统计首位候选改变的条目，无论对错。正确性发生变化的全部条目：

- 全拼改善：`zh-std-crosswoz-ea1daad81c`、`zh-std-risawoz-e8c74bc9fe`
- 全拼退步：`zh-std-crosswoz-0c2c5afa15`
- 简拼改善：无
- 简拼退步：`zh-std-kdconv-422abcb556`、`zh-std-risawoz-f9b102f070`

Chunk-KV INT8（`zh-v1-eval-kvchunk-int8-b32`）在两个赛道上与无状态 INT8 的 Top-1 相同、变化 id 相同、0 个排序变化（最大分数差全拼 3.1e-5，简拼 2.3e-5）。

## 5. 量化调查（只看大小与准确率）

只做无状态包，参数选择相同，FP32 计算。调查只记录数字，不选配方；部署配方仍是 INT8 b32。

| 配方 | 包 | 字节 | 严格最大（long 均值） | long argmax | 全拼 Top-1（对 FP32） | 简拼 Top-1（对 FP32） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | `tiny-zh-v1-fp32-v1` | 80,367,198 | 3.9e-5 | 128/128 | 1363 | 337 |
| INT8 b32（已部署） | `tiny-zh-v1-int8-b32-v1` | 22,792,958 | 0.392（0.046） | 127/128 | 1364（+2/−1） | 335（+0/−2） |
| INT8 b32 + FP16 embedding | `tiny-zh-v1-survey-int8-b32-embfp16-v2` | 26,921,756 | 0.200（0.032） | 128/128 | 1365（+2/−0） | 337（+0/−0） |
| 调色板 6 bit，分组通道 32 | `tiny-zh-v1-survey-palette6-gc32-v1` | 15,658,010 | 2.04（0.240） | 117/128 | 1364（+7/−6） | 339（+4/−2） |
| INT4 b32 | `tiny-zh-v1-survey-int4-b32-v1` | 12,778,238 | 6.99（0.822） | 95/128 | 1349（+10/−24） | 332（+8/−13） |
| 调色板 4 bit，分组通道 32 | `tiny-zh-v1-survey-palette4-gc32-v1` | 10,368,794 | 9.49（1.328） | 76/128 | 1320（+16/−59） | 316（+17/−38） |

对 FP32 的排序变化 / Top-1 变化条目（全拼；简拼）：FP16 embedding 634/3；462/10。调色板 6 bit 1424/15；983/62。INT4 1451/50；990/190。调色板 4 bit 1450/104；990/289。每个调查包的变化 id 在各自的 `zh-v1-eval-survey-*/judgement.json` 中（`improved_ids`、`regressed_ids`）。

说明：

- FP16 embedding：INT8 压缩后，把 token embedding 常量换成 FP16 常量加 `constexpr_cast` 到 FP32；embedding 的 FP16 舍入最多 1.22e-4。第一次构建（`…-embfp16-v1`）从 MIL 程序重新转换时丢失了无状态输入范围（上界 2 而不是 128）；它被删除，重建为 `-v2`，重新声明了输入、输出和元数据，并对照源包核对了输入规格。只评估了 v2。
- 调色板和 INT4 配方改变的候选顺序比 INT8 多得多；6 bit 调色板尽管严格最大误差为 2.0，在这些赛道上的 Top-1 与 FP32 在噪声范围内相同，所以只看 Top-1 会低估它的误差。

## 6. 与 V4.2 的 Mac 测量对比

2026-10-20 验收的四个 V4.2 epoch 20 包（无状态 FP32/INT8、chunk-KV FP32/INT8）以 ZIP 提供（SHA256 `4728fc85…`），原样解压到 `artifacts/deployment/`，并核对了其 SHA256 列表；包大小与 V4.2 报告一致。没有重建。

测量工具：`measure_v42.py run --set zh-v1` → `outputs/deployment/zh-v1-mac-measure-v1`。每个包先用 `coremlcompiler` 编译，然后每个模型每轮在一个全新的 Python worker 中测量，八个模型交替五轮，`CPU_ONLY`，5 次预热和 50 次计时调用。延迟取五个进程各自 p50 / p95 的中位数。条件：接交流电（充电 35→39%），低电量模式关闭，`pmset -g therm` 在六个检查点都没有热或性能警告。1 分钟平均负载开始时为 14.3（准备全新检出遗留的），结束时降到 3.9。两个系列因交替测量而处于相同条件，但绝对数字可能略偏高。这些是 Mac 宿主进程的数字，不能替代 iPhone 或键盘扩展上的测量。

无状态：

| 指标 | zh-v1 FP32 | zh-v1 INT8 | V4.2 FP32 | V4.2 INT8 | zh-v1 / V4.2 INT8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 包字节 | 80,367,198 | 22,792,958 | 60,467,999 | 17,164,271 | 1.33× |
| 加载，首个进程（ms） | 261.4 | 197.5 | 196.2 | 137.3 | 1.44× |
| 加载，各进程 p50（ms） | 30.6 | 28.4 | 27.6 | 26.2 | 1.08× |
| 进程内首次 `predict`，T=32，各进程 p50（ms） | 107.2 | 102.4 | 89.3 | 13.7 | — |
| T=1 p50 / p95（ms） | 1.66 / 2.06 | 1.65 / 2.11 | 1.34 / 1.52 | 1.29 / 1.71 | 1.28× |
| T=8 p50 / p95 | 2.45 / 3.06 | 2.43 / 2.88 | 1.76 / 2.19 | 1.78 / 2.35 | 1.37× |
| T=32 p50 / p95 | 3.98 / 4.53 | 3.85 / 4.28 | 2.98 / 3.32 | 2.95 / 3.35 | 1.31× |
| T=64 p50 / p95 | 5.46 / 5.88 | 5.52 / 5.88 | 4.21 / 4.85 | 4.12 / 4.55 | 1.34× |
| T=128 p50 / p95 | 8.40 / 9.28 | 8.49 / 8.88 | 6.31 / 6.84 | 6.43 / 6.71 | 1.32× |
| 加载后内存占用（MiB） | 309.7 | 310.1 | 309.8 | 310.2 | — |
| 加载增量 / 峰值增量（MiB） | 1.1 / 18.5 | 1.1 / 18.4 | 1.0 / 12.3 | 1.0 / 13.4 | 峰值 1.37× |
| 生命周期峰值内存占用（MiB） | 327.1 | 327.5 | 320.8 | 322.1 | +5.4 MiB |

比值列比较的是 INT8 的 p50。每个进程的首次 `predict` 是加载后的第一次调用，T=32，包含 Core ML 处理第一个形状的工作。它在进程之间差异很大（V4.2 INT8 13.7 ms，其他约 90–107 ms），只能作为参考指标。其他长度下首个进程的冷调用，zh-v1 为 1.8–9.4 ms，V4.2 为 1.5–7.0 ms。约 309 MiB 的内存占用基线是 Python、PyTorch 和 coremltools 宿主进程；加载增量和峰值增量才是模型自己的部分。

Chunk-KV（前缀 P 一次调用预填充，然后逐 token 解码）：

| 指标（ms） | zh-v1 FP32 | zh-v1 INT8 | V4.2 FP32 | V4.2 INT8 |
| --- | ---: | ---: | ---: | ---: |
| 包字节 | 80,404,135 | 22,830,199 | 60,496,928 | 17,193,424 |
| 加载，首个进程 / p50 | 255.8 / 27.2 | 202.2 / 26.0 | 200.3 / 28.7 | 167.1 / 28.2 |
| 首次预填充 T=128，各进程 p50 | 10.2 | 10.0 | 8.0 | 7.9 |
| 预填充 P=7 / 31 / 63 / 119，p50 | 3.60 / 5.25 / 6.67 / 9.37 | 3.60 / 5.14 / 6.74 / 9.33 | 2.90 / 3.98 / 4.99 / 7.13 | 2.77 / 3.78 / 4.90 / 7.18 |
| 预填充 P=119 p95 | 9.69 | 9.57 | 7.33 | 7.29 |
| P=7 / 119 之后解码，p50 | 2.61 / 2.70 | 2.63 / 2.71 | 2.10 / 2.05 | 2.21 / 2.05 |
| P=119 之后解码，p95 | 3.05 | 3.05 | 2.54 | 2.49 |
| 峰值增量 / T=128 chunk 峰值增量（MiB） | 31.3 / 18.4 | 29.8 / 18.4 | 20.8 / 13.0 | 20.8 / 12.4 |

在 Mac 上，zh-v1 在大小、无状态延迟、chunk 预填充和解码上约为 V4.2 的 1.3 倍，模型峰值内存增量约为 1.4 倍。这与任务说明根据 12,288 词表和九层所做的预期一致。与 V4.2 一样，INT8 减小了大小但没有降低 CPU 延迟，因为计算仍是 FP32。

## 7. iOS 资源与 Vime 测试宿主

`scripts/zh/deployment/package_zh_v1_ios.py` → `artifacts/deployment/tiny-zh-v1-ios-resources-v1`：

- `Resources/TinyChineseV1INT8.mlmodelc`、`TinyChineseV1KVChunkINT8.mlmodelc`（各约 22 MB）、`VimeChineseTokenizerV1.model`、`VimeLMManifestZhV1.json`（格式 `vime_ios_lm_zh_v1`，编译文件 SHA256，`strict_fp32_logits_alignment_passed: false`）。
- `VimeLMFixturesZhV1.json`：在专门编写的文本上的分词、打分和 chunk-KV 逐步 fixture，没有一条来自基准。Chunk-KV 后缀和与无状态和相差 6.2e-6 以内。

打包要求同一权重的 chunk/无状态 INT8 门槛已通过。

Vime 源码的改动由 Vime 自己的维护者完成，所以本会话没有在那里留下任何长期改动。为了检查资源能在客户端运行时中工作，使用了一个临时的本地 Vime worktree，之后删除：

- 它有一个在 `runtimePreference` 之外、需要显式启用的 `.zhV1` case。
- 它只从 UIKit 测试 bundle 加载资源，所以 app 和键盘 bundle 不变。
- 它有四个功能测试。

结果：

- 模拟器（iPhone 18 Pro Max，iOS 27.2，Debug）通过了四个 zh-v1 测试：
  - 不是运行时选项，也不在 app bundle 中。
  - 标识与 logits 形状 `[1,T,12288]`。
  - 原生 SentencePiece 与所有分词 fixture 一致。
  - 联合后缀分数与 Mac Core ML INT8 相差 0.002 以内，有无转换缓存都是如此。
- 现有 V4.2 测试（6 个）和模型测试（24 个）在同一次运行中也通过，共 34/34。
- 之后删除了 worktree、分支、安装的资源和构建产物；没有向 Vime 提交或推送任何内容。
- diff 作为 Vime 端参考保留在 `outputs/deployment/zh-v1-vime-test-host-reference-v1/`（在结果 ZIP 中，不在 Git 中）。

**未做：** 模拟器和 iPhone 16 Pro Max 上的 `VimeLanguageModelBenchmarkTests` 运行（与 V4.2 交替四轮，每个进程一个变体）。这需要 Vime 端的变体，等它合入后可用现有的 `Scripts/benchmark_lm_variants.py --variants zhV1,v42` 运行。客户端只检查了无状态 INT8：Vime 对 V4.2 也没有 KV 运行时，所以 chunk-KV 资源只由 Mac fixture 门槛覆盖。

## 测试

这项工作针对性的单元测试（zh-v1、V4 和 V2 部署、KV 缓存、Core ML 压缩、部署工具、zh-v1 确认、zh13 tokenizer）通过：44 OK，5 跳过，切换到全新克隆前后都是如此。完整的 `unittest discover` 在这台 Mac 上有错误。它们来自本检出中没有的日文资源（V4 tokenizer、V4 token 数据、后训练数据、一个 V4 checkpoint），以及未安装的 `fugashi` 和 `opencc` 包。没有本次改动时同样出现这些错误；没有为它们安装依赖。

## 输出

包在 `artifacts/deployment/tiny-zh-v1-*` 下；记录在 `outputs/deployment/zh-v1-*` 下（对齐、评估分数、结果、`judgement.json`、`model.json`、Mac 测量 `zh-v1-mac-measure-v1`）。大文件通过 `scripts/zh/deployment/package_zh_v1_mac_results.py` 回传。
