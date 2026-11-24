# V4.3 Core ML、量化与真机交接

这是 macOS 会话的任务说明。内容包括：把 V4.3 epoch 30 和 27 转换为 Core ML，用已部署 V2.1 的配方量化，并在 Mac 和 iPhone 16 Pro Max 测试宿主上与 V4.2 epoch 20 和 V2.1 对比测量。它不授权替换已部署模型、重新训练或改动基准输入。V4.2 的运行是模板：[V4.2 任务说明](mac-v42-coreml.md)、[V4.2 Core ML 报告](../reports/mac-20261020/v42-coreml.md)。

## 模型

V4.3 的约定与 V4.2 相同：8K tokenizer（SHA256 `296a2bdf…81b5`），d384，七层，六头，SwiGLU 960，上下文 128，共享输出头，15,070,848 参数，普通 LM，无转换标记。它从随机权重开始，在 V4.2 数据池加九个去重后的 FineWeb2-Edu 分片上训练了 30 个 epoch（每个 epoch 网页数据约翻倍）。

| 变体 | Checkpoint | 步数 | 选择依据 |
| --- | --- | ---: | --- |
| `v43-epoch30` | `epoch-30.pt` | 3,225,888 | 完整 LM BPC 最优（3.043831） |
| `v43-epoch27` | `epoch-27.pt` | 2,903,298 | 指定的对比 epoch |
| `v42-epoch20` | V4.2 `best-lm.pt` | 1,081,942 | 基线，包已构建 |

由于图和权重形状与 V4.2 相同，包大小、加载时间、内存和延迟应在测量噪声内与 V4.2 一致。超出噪声的差异是需要调查的发现，而不是预期中的提升。各变体的差别在质量上（Top-1、INT8 漂移）。

## 环境准备

```bash
unzip vimeml-v43-mac-<date>.zip -d vimeml-v43-mac
python3 vimeml-v43-mac/setup_mac.py            # default ~/Documents/Sources/VimeML-v43-coreml
cd ~/Documents/Sources/VimeML-v43-coreml
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e .
```

`setup_mac.py` 克隆 `repository.bundle`，核对记录的提交，设置 `origin`，创建 `claude/mac-v43-coreml`，并在大小和 SHA256 检查后恢复载荷。使用 V4.2 的固定版本（Python 3.11、PyTorch 2.7.1、coremltools 9.0、NumPy 2.3.5、scikit-learn 1.5.1），记录实际版本、macOS 构建号和硬件。

## 载荷

| 路径 | 内容 |
| --- | --- |
| `artifacts/models/tiny-ja-v43-e8k-d384-l7/epoch{30,27}-deployment.pt` | 仅 FP32 张量；内部记录了源 checkpoint 的 SHA256 |
| `outputs/deployment/tiny-ja-v43-epoch{30,27}-fp32-input/` | Windows PyTorch CPU FP32 fixture（`bos`、`short`、`long`、`right_pad`、`changed_future`、`long_kv_step_logits`） |
| `outputs/evaluation-v43-20261020/epoch{30,27}/` | V4.3 FP32 CUDA 分数和配对比较：任务级参照 |
| `outputs/evaluation-v42-20261020/`、`outputs/evaluation-v4-20261015/{inputs,v21}/` | V4.2 和 V2.1 的逐例分数；四个冻结的候选池 |
| `artifacts/tokenizers/v4-a/ja-unigram-8192/`、`artifacts/token-data/corpus-v4-8k/manifest.json` | 匹配的 tokenizer 和标识 manifest |
| `outputs/training-v43-20261020/` | 下载回执和训练记录，仅用于标识 |
| `artifacts/deployment/tiny-ja-v4.2-epoch20-{bundle,fp32,int8-b32,kvchunk-fp32,kvchunk-int8-b32,client-resources-v1}-v1/` | 已验收的 V4.2 包；不要重建 |
| `artifacts/deployment/tiny-ja-v2.1-extend5-{bundle,fp32,int8-b32}-v1/`、`…-kv-{fp32,int8-b32}-mac-v42-measure-v1/` | V2.1 已部署包和仅用于测量的 KV 包 |

不包括 blind、test 和固定参照数据、训练语料、优化器状态和凭据。

## 参照数字（FP32 CUDA，冻结候选池）

| 模型 | Standard 开发集 /728 | Legacy /500 | AJIMEE /200 | Historical 开发集 /137 |
| --- | ---: | ---: | ---: | ---: |
| V2.1 | 505 | 374 | 144 | 122 |
| V4.2 epoch 20 | 523 | 370 | 139 | 123 |
| V4.3 epoch 30 | 519 | 367 | 143 | 123 |
| V4.3 epoch 27 | 521 | 372 | 138 | 124 |

V4.2 Core ML INT8 为 522/371/139/123。

## 任务

所有 `coreml_v4.py` 阶段都接受 `--variant v43-epoch30` 或 `--variant v43-epoch27`；默认仍是 V4.2。包目录按 `tiny-ja-v4.3-epoch{30,27}-<kind>-v1` 命名，报告放在 `outputs/deployment/v43-e{30,27}-*` 下。每一步对两个 epoch 都要运行。

1. **Mac PyTorch 基线。** 运行 `coreml_v4.py --variant … baseline`。Fixture 必须在 3e-4 以内。四个赛道的 Top-1 必须与 CUDA 参照相同；报告逐例排序差异。
2. **推理包与 FP32 Core ML。** 运行 `export`、`convert` 和 `convert-kvchunk`。单 token KV 图已被 chunk 图取代，除非时间允许否则跳过。
3. **FP32 门槛。** 对 fixture 运行 `align`（无状态）和 `align-kvchunk`。用 `evaluate` 和 `evaluate-kvchunk` 评估四个赛道。Top-1 必须与 PyTorch 基线相同。
4. **INT8。** 通过 `compress` 和 `compress-kvchunk` 使用不变的配方：linear_symmetric int8，per_block 32，weight_threshold 2048，FP32 计算，CPU_ONLY，iOS 18。严格 logits 对齐预期会失败。报告误差分布并与 V4.2 对比（最大绝对误差、分位数）。检查 kvchunk INT8 对同一权重的无状态 INT8。对 FP32 Core ML 评估四个赛道，列出每个退步用例的 ID。
5. **Mac 测量。** 运行 `measure_v42.py run --set v43 --output outputs/deployment/v43-mac-measurements-v1`。它在全新进程中交替运行 V4.3 e30 FP32/INT8、e27 INT8、V4.2 FP32/INT8 和 V2.1 FP32/INT8 无状态包，以及各自的 chunk-KV 包，全部在同一会话中。按 V4.2 的方式记录热和电源条件。先报告 V4.3 对 V4.2 epoch 20，再报告对 V2.1。
6. **iOS 资源。** 用 INT8 无状态包和 INT8 chunk-KV 包，以及与 V4.2 相同的 Vime fixture 模板，运行 `package_v4_ios.py --variant v43-epoch{30,27} --chunk`。资源命名为 `TinyJapaneseV43E30INT8.mlmodelc`、`VimeLMManifestV43E30.json` 等，所以 V4.2 资源不受影响。
7. **iPhone 测试宿主。** 在用于 V4.2 的 Vime worktree（`claude/v42-lm-measure`，或从它新建的分支）中，在 `.v42` 旁边为 V4.3 e30 和 e27 添加需显式启用的变体。所有生产调用点保持 `.v21` 为默认。在模拟器和 iPhone 16 Pro Max 上运行模型功能测试。然后用 V2.1、V4.2 和两个 V4.3 变体交替四轮运行 `VimeLanguageModelBenchmarkTests`，每个进程一个变体（`Scripts/benchmark_lm_variants.py`）。在每次运行开始和结束时记录热状态和低电量模式。如果手机不是 `nominal`，要说明，并尽可能在冷却后重跑一次。报告加载和重新加载时间、首次 predict、`predict` T=1/8/32/64/128 的 p50 和 p95、重排、beam 和下一词的 p50 和 p95，以及加载前、加载后和峰值的内存占用。键盘扩展的测量仍不在范围内，除非添加起来很容易。

## 规则

- 不要读取 blind、test 或固定参照数据。不要根据这些结果重新训练、选择其他 checkpoint、改变候选池或参考答案，或调整打分。
- 每次运行写到新的输出目录。绝不覆盖 V4.2 或 V2.1 的包或记录。
- 失败、超出容差和跳过的步骤按实际记录。
- 已提交的文档中不要引用基准文本（WRIME、JMultiWOZ、AJIMEE 句子）；只给用例 ID 和汇总。
- 源码改动提交到 `claude/mac-v43-coreml`，不加 Claude co-author 尾注。在 `docs/ja/reports/mac-<date>/v43-coreml.md` 写一份注明日期的报告，并加入索引。
- 大文件按 `package_v42_mac_results.py` 的风格以结果 ZIP 返回，附路径、大小、环境和通过/失败检查。源码改动通过 Git 和 pull request 提交。
