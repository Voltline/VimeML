# vimeml-tiny-zh-v1 Core ML、量化与真机交接

这是 macOS 会话的任务说明。内容包括：把第一个中文拼音模型 `vimeml-tiny-zh-v1` 转换为 Core ML，用已部署 V4.2 的配方量化，并在 Mac 和 iPhone 16 Pro Max 测试宿主上测量。它不授权在 Vime 中发布该模型、重新训练、选择其他 checkpoint 或改动基准输入。V4.2 和 V4.3 的运行是模板：[V4.2 任务说明](../../ja/mac/mac-v42-coreml.md)、[V4.3 任务说明](../../ja/mac/mac-v43-coreml.md)、[V4.2 Core ML 报告](../../ja/reports/mac-20261020/v42-coreml.md)。模型背景：[盲测确认与基线](../reports/zh-v1-20261111/blind-and-baselines.md)。

## 模型

| 项 | 值 |
| --- | --- |
| 发布名 | `vimeml-tiny-zh-v1`（内部代号 `zh13`，arm `12k-fresh-s1`） |
| 源 checkpoint | `artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/final.pt`，第 844,730 步，SHA256 `44c423a1…7d4`（不在包中） |
| 打包的权重 | 同目录下的 `deployment.pt`：仅 FP32 张量，内部记录了源 SHA256 |
| Tokenizer | `artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model`，SentencePiece Unigram，SHA256 `68bd37c4…7e5a`；pad 0，unk 1，bos 2，eos 3 |
| 结构 | 词表 12,288，d384，**九**层，六头，SwiGLU 960，RMSNorm，上下文 128，可学习位置编码，共享输出头，无偏置 |
| 参数 | 20,036,736 |
| 格式 / 类 | `vimeml_tiny_gpt_zh13`，`vimeml.training.model_zh13.TinyGPTZh13`（使用 V4 块族的 `TinyGPTV2` 子类）；用 `vimeml.benchmarks.zh13_lm_scores.load` 加载 |
| KV 缓存 | key 和 value 各 `(9, 1, 6, 128, 64)` float32 |

块结构与 V4.2 相同，但词表是 12,288 而不是 8,192，层数是九层而不是七层。预计大小约为 V4.2 的 1.33 倍：FP32 约 80 MB，INT8 block-32 约 23 MB（V4.2 INT8 为 17.16 MB）。延迟和内存应以相近的倍数上升。比 V4.2 大是预期之内的；大小是否可接受是报告要回答的问题，而不是改配方的理由。

## 环境准备

```bash
unzip vimeml-zh-v1-mac-<date>.zip -d vimeml-zh-v1-mac
python3 vimeml-zh-v1-mac/setup_mac.py            # default ~/Documents/Sources/VimeML-zh-v1-coreml
cd ~/Documents/Sources/VimeML-zh-v1-coreml
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e .
```

`setup_mac.py` 克隆 `repository.bundle`（已推送的 `main`），核对记录的提交，设置 `origin`，停留在 `main`，并在大小和 SHA256 检查后恢复载荷。使用 V4.2 的固定版本（Python 3.11、PyTorch 2.7.1、coremltools 9.0、NumPy 2.3.5、scikit-learn 1.5.1），记录实际版本、macOS 构建号和硬件。

## 载荷

| 路径 | 内容 |
| --- | --- |
| `artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/` | `deployment.pt`，以及训练时的 `config.json`、`summary.json`、`environment.json`（仅用于标识） |
| `artifacts/tokenizers/zh-v13/zh-unigram-12288/` | `tokenizer.model`、`tokenizer.vocab` |
| `outputs/deployment/vimeml-tiny-zh-v1-fp32-input/` | Windows PyTorch CPU FP32 fixture：`bos`、`short`、`long`、`right_pad`、`changed_future` 的 logits 和 `long_kv_step_logits` |
| `outputs/benchmarks/ime-standard-zh-v1/frozen-r1/` | **仅开发集**（全拼 1,456 条，简拼 990 条）：条目、冻结的 20 候选池、Rime 输出、重叠列表 |
| `outputs/benchmarks/zh13-20m/runs/12k-fresh-s1/` | v1 参照的逐候选分数（`full.jsonl`、`abbrev.jsonl`）及 meta |
| `outputs/benchmarks/zh13-20m/judgement/scores/{12k-fresh-s1,rime}-{full,abbrev}/` | v1 和 Rime 候选池顺序的打分结果（`items.jsonl`、`summary.json`） |

不包括盲测条目、训练数据、带优化器状态的完整 checkpoint 和凭据。也不包括 V4.2 和 V2.1 的包；从现有 V4.3 检出目录（`~/Documents/Sources/VimeML-v43-coreml/artifacts/deployment/`）拷贝已验收的包复用，绝不重建。

## 参照数字（开发集，精确 Top-1）

| 系统 | 全拼 / 1,456 | 简拼 / 990 |
| --- | ---: | ---: |
| v1，FP32（Windows CUDA 参照分数） | 1,363（93.6%） | 337（34.0%） |
| Rime 候选池顺序（librime 1.17.0 + rime-ice） | 1,001（68.8%） | 142（14.3%） |

打分：BOS + 左侧上下文 + 候选联合分词；分数是从任一候选与单独上下文出现差异的第一个 token 起，FP32 全词表对数概率之和；不加 EOS；并列保持候选池顺序。实现：`vimeml.benchmarks.zh13_lm_scores.score_item`。分数文件输入 `scripts/zh/benchmarks/score_standard_ime_zh.py score … --scores`，`… compare` 对两个结果做配对比较。

## 任务

1. **Mac PyTorch 基线。**
   - 加载 `deployment.pt`，检查 fixture 保持在 3e-4 以内（无状态和 `forward_cached` 逐步）。
   - 在 Mac CPU 上用 `scripts/zh/benchmarks/score_zh13_lm.py` 对两个赛道重新打分开发集，再运行 `score_standard_ime_zh.py score`。
   - Top-1 必须是 1,363 和 337。与 `judgement/scores/12k-fresh-s1-*` 的 `compare` 应没有改善或退步；按条目 id 报告任何名次差异。
2. **zh 部署代码。** V4 Core ML 代码写死了日文约定：`v4_coreml.VOCAB = 8192`、`JapaneseLMV4` 及其 token 数据检查、`TinyGPTV4`，以及 `v4_coreml`、`v4_chunk`、`package_v4_ios.py` 和 `measure_v42.py` 中的七层缓存。
   - 把它们通用化，让词表、层数和加载器来自推理包，或在旁边新增一个薄的 zh 模块。
   - 不要改变 V4.2 的输出：现有 V4 测试和 V4.2 包的 manifest 必须保持有效。
   - 包命名为 `artifacts/deployment/` 下的 `tiny-zh-v1-<kind>-v1`，报告放在 `outputs/deployment/zh-v1-*` 下。
3. **推理包与 FP32 Core ML。** 构建推理包、无状态 FP32 包（int32 `[1,T]`，1≤T≤128 → float32 `[1,T,12288]`）和 chunk-KV FP32 包（缓存 `(9,1,6,128,64)`）。
   - 以 3e-4 对 fixture 运行无状态和 chunk-KV 对齐。
   - 通过 Core ML 用同样的打分评估两个赛道的开发集。Top-1 必须等于 Mac PyTorch 基线。
4. **INT8。** 无状态和 chunk-KV 都使用不变的配方：`linear_symmetric` int8，`per_block` 32，`weight_threshold` 2048，FP32 计算，CPU_ONLY，iOS 18。
   - 严格 logits 对齐预期会失败。报告误差分布（最大绝对误差、分位数），并与 V4.2 并列。
   - 检查 chunk-KV INT8 对同一权重的无状态 INT8。
   - 对 FP32 Core ML 评估两个开发赛道，列出每个退步或改善的条目 id。
5. **量化调查（探索性）。** 在两个赛道上比较以下配方的包大小和开发集 Top-1：
   - 4 bit `linear_symmetric` per-block 32；
   - 6 bit 和 4 bit k-means 调色板化（coremltools 9 支持时按分组通道，否则按张量）；
   - INT8，共享 embedding/输出头保持 FP16。

   计算设置同任务 4，只做无状态。这是供之后决策用的大小/质量图。不要选出胜者，也不要根据这些结果改变任务 4 的配方。
6. **Mac 测量。** 给 `measure_v42.py` 增加 `zh-v1` 模型集。
   - 在同一会话的全新进程中，把 v1 FP32/INT8（无状态和 chunk-KV）与 V4.2 epoch 20 FP32/INT8 交替运行。
   - 报告包大小、加载时间、首次 predict、T=1/8/32/64/128 的 p50 和 p95，以及内存。
   - 按 V4.2 的方式记录热和电源条件。
7. **iOS 资源与 iPhone 测试宿主。**
   - 把 INT8 无状态和 INT8 chunk-KV 模型连同 tokenizer 打包成名为 `TinyChineseV1INT8.mlmodelc`、`VimeLMManifestZhV1.json` 等的资源，使日文资源不受影响。
   - 在用于 V4.2/V4.3 的 Vime worktree 中，为模型功能测试和 `VimeLanguageModelBenchmarkTests` 添加需显式启用的 `.zhV1` 变体。
   - 所有生产调用点保持不变；客户端的中文输入和 Rime 集成不在范围内。
   - 在模拟器和 iPhone 16 Pro Max 上运行，与 V4.2 交替四轮，每个进程一个变体。
   - 报告加载和重新加载时间、`predict` T=1/8/32/64/128 的 p50 和 p95，以及加载前、加载后和峰值的内存占用，并附开始和结束时的热状态和低电量模式。

## 规则

- 包中只有开发集数据；不要尝试获取盲测数据。不要根据这些结果重新训练、选择其他 checkpoint、改变候选池或参考答案，或调整打分。
- 每次运行写到新的输出目录。绝不覆盖 V4.2、V4.3 或 V2.1 的包或记录。
- 失败、超出容差和跳过的步骤按实际记录。
- 已提交的文档中不要引用基准文本（CrossWOZ、RiSAWOZ、KdConv 的对话轮、目标或候选）；只给条目 id 和汇总。
- 源码改动直接提交到 `main` 并推送到 `origin/main`；不用功能分支或 pull request。不加 Claude co-author 尾注。
- 在 `docs/zh/reports/mac-<date>/zh-v1-coreml.md` 写一份注明日期的报告，并加入索引。
- 大文件（包、测量记录、退步 id 列表）按 `package_v42_mac_results.py` 的风格以结果 ZIP 返回，附路径、大小、SHA256、环境和通过/失败检查。
