# vimeml-tiny-zh-v1 批量 KV 解码：Mac 交接

这是 macOS 会话的任务说明。Vime 现在运行拼音约束的 beam 搜索，beam 按按键位置保留。它的运行时使用现有的 batch-1 chunk-KV INT8 包，所以一个 16 个假设的桶要做 16 次 `predict` 调用。在 iPhone 13 Pro 上每次约 3.8 ms，每次生成 p50 为 492–724 ms，大约慢了 10 倍。

本任务构建一个一次调用推进整个桶的包，对它做门槛检查和测量，并准备客户端资源和 fixture。它不改变 checkpoint、tokenizer、INT8 配方或搜索规则。

模板：

- [zh-v1 Core ML 任务说明](mac-zh-v1-coreml.md)
- [zh-v1 Core ML 报告](../reports/mac-20261111/zh-v1-coreml.md)

## 目标（来自 Vime）

在 iPhone 13 Pro 上，`CPU_ONLY`，同时加载无状态包：

- **延迟：** 一个桶（B=16）在一次批量单 token 步中最多 8 ms。B=8 最多 5 ms。
- **内存：** 报告在今天 43 MB 基础上键盘扩展增加的内存占用。

Vime 会在按键之间复用桶，并异步显示生成结果，所以关键是一个桶每步的开销。

## 接口

`vimeml.deployment.zh_v1_batch.KVBatchGraphZh`：

| 名称 | 类型 | 含义 |
| --- | --- | --- |
| `input_ids` | int32 `[B,1]` | 每行一个新 token |
| `position` | int32 `[B]` | 该 token 之前的行缓存长度；`position[b] < S` |
| `key_cache`、`value_cache` | float32（或 float16）`(9,B,6,S,64)` | 第 b 行 `>= position[b]` 的槽位被掩码；可以是任意有限值，但 NaN/Inf 会传到输出（新缓冲区要清零） |
| → `logits` | float32 `[B,12288]` | 只有新 token；不做 softmax |
| → `new_key`、`new_value` | `(9,B,6,64)` | 新 token 的 key/value；调用方把它们写入第 b 行的槽位 `position[b]` |

行之间从不互相注意。填充行（位置 0 的 token 0）不影响真实行。

**与 Vime 要求的两处有意差异。** 在报告中说明。

1. **`S` 可变（1..128）。** 调用方只传入 `max(position)+1` 个槽位。
   - 若固定 128 个槽位，一个 B=16 缓存为 28 MiB，一次调用要搬运约 56 MiB 输入。
   - 真实的行只是一个短 prompt 加最多约 20 个 token。
2. **只返回新槽位**，不返回完整缓存。每次调用写出两份完整缓存的开销比这一步本身还大。
   - 用调用方持有的缓冲区，除新槽位外不拷贝任何东西。
   - Vime 仍可以用自己的缓冲区承载输入。

**B 变体：**

| 变体 | B | 原因 |
| --- | --- | --- |
| flex | 1..16（`RangeDim`） | Vime 要求的默认 |
| 固定 | 16 | 若 `RangeDim` 很慢，看填充开销 |
| 固定 | 8 | Vime 要求的 B=8 延迟 |

**Prompt 仍用现有的 batch-1 chunk 包。** 运行一次，然后把它的缓存行拷进批量缓冲区。

`merge` 阶段还会构建一个多函数包（`prefill` = chunk，`decode` = batch）。其权重只存一份，所以加载两个函数应只花一份权重的内存而不是两份。检查这在 Mac 上和内存占用上是否成立。

**只返回允许 token 的 logits（Vime 的问题 7）。** 只返回允许 token 的 logits 无法在不改变分数的情况下跳过输出头矩阵乘。搜索需要对全部 12,288 个 token 做 log-softmax，归一化需要每个 logit。

- 输出头约占每 token 计算量的四分之一（约 20M 次乘加中的 4.7M）。
- 能省下的只有 `[B,12288]` 输出拷贝和客户端 softmax。
- 如果测量表明输出拷贝很重要，一个选项是返回 `logsumexp` 加 gather 后 logits 的变体。目前尚未构建。

**MLState。** 不使用。

- 一个桶聚集的假设，其父节点分布在之前不同的桶中，所以每一步都要从任意父节点重建行。
- 用调用方持有的显式缓冲区配合 `outputBackings`，可以零额外拷贝地做到这一点。

## 环境准备

```bash
unzip vimeml-zh-kvbatch-mac-<date>.zip -d vimeml-zh-kvbatch-mac
python3 vimeml-zh-kvbatch-mac/setup_mac.py            # default ~/Documents/Sources/VimeML-zh-kvbatch
cd ~/Documents/Sources/VimeML-zh-kvbatch
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e . && python -m pip install coremltools==9.0 psutil
```

使用 zh-v1 的固定版本（Python 3.11、PyTorch 2.7.1、coremltools 9.0、NumPy 2.3.5）。记录实际版本、macOS 构建号和硬件。

## 载荷

| 路径 | 内容 |
| --- | --- |
| `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/inference/` | FP32 推理包（权重、配置、tokenizer、manifest） |
| `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32/` | 已发布的 INT8 无状态包（对齐参照） |
| `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvchunk/` | 已发布的 INT8 chunk-KV 包（prompt / `prefill`） |
| `outputs/zh-readings/dict/vime_pinyin.dict.yaml` | 搜索使用的读音词典 |
| `outputs/vime-fixtures/zh-generation-v1-lm-r2/` | Vime 的生成 fixture：125 例，beam 16，top 10，FP32 PyTorch 参照 |

不包括基准条目、训练数据、完整 checkpoint 或凭据。生成 fixture 来自公开集的开发集加五个手写按键序列。

## 任务

所有命令都是 `python scripts/zh/deployment/coreml_zh_v1_batch.py <stage> …`。每个阶段在 `outputs/deployment/zh-v1-kvbatch/` 下写新的输出；绝不覆盖输出。

1. **PyTorch 检查。**
   - 运行 `check out/check.json`。必须通过，与 Windows 上一样（最大 3.96e-5）。
   - 它推进 16 行，每行 3–25 个 token。各行从不同的步开始，所以每次调用中位置都不同。每一步都必须与该行自己的完整前向一致。
2. **FP32 flex。**
   - `convert out/fp32-flex`。
   - `align out/fp32-flex out/fp32-flex-align.json`：对 PyTorch FP32，容差 3e-4。
   - 如果转换在动态槽位数或 one-hot 写入上失败，在 `zh_v1_batch.py` 中修复，重跑 `check`，并记录改动。
3. **INT8 flex。**
   - `compress out/fp32-flex out/fp32-flex-align.json out/int8-flex`。
   - `align out/int8-flex out/int8-flex-align.json --stateless artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32`。
   - 这是 Vime 的门槛：每一步都在 INT8 无状态包的 atol = rtol = 3e-4 以内。之前的 chunk-KV INT8 包在 4.2e-5 以内。
4. **固定 B=16 和 B=8。** 用 `--batch 16` 和 `--batch 8` 重复第 2–3 步。B=8 的对齐跳过 16 行的用例；B=16 的对齐还检查填充行。
5. **FP16 缓存（探索性）。**
   - `convert --cache-dtype float16`，然后对齐和压缩。
   - 严格门槛可能失败。如果失败，报告误差分布和 argmax 一致率，并把包仅作为证据保留。
   - 用 `--compute-units CPU_ONLY`、`CPU_AND_NE` 和 `ALL` 测量。Vime 在 chunk 包上看到 ANE 没有提升，GPU 慢 2 倍。
6. **合并包。**
   - `merge out/int8-flex out/int8-merged`（chunk = 已发布的 INT8 chunk 包）。
   - 确认两个函数都能加载（`function_name="prefill"` / `"decode"`）。
   - 与分别加载两个包相比，比较包字节数和 RSS。
7. **Mac 延迟。**
   - 对 INT8 flex、固定 16、固定 8、合并包和 FP16 变体运行 `measure <pkg> out/<pkg>-latency.json`。
   - 网格：B ∈ {1, 4, 8, 16}，S ∈ {16, 32, 64, 128}，20 次预热后 200 次重复，同时加载无状态 INT8 包。
   - 报告每次调用的 p50/p95，固定包还要报告填充开销（B=16 调用中的 B=1 行）。
8. **Vime fixture。**
   - `fixture out/int8-flex out/fixture`。
   - 它包含 16 个不同长度的句子前缀。每个先用 chunk 包预填充，然后在一次解码调用中全部推进。
   - 每行 top-5 分别来自包、INT8 无状态包和 PyTorch FP32。
9. **搜索检查。**
   - `search out/int8-flex outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl out/search-r2`。
   - fixture 的 `meta.json` 期望 INT8 的文本集合相同。2026-11-15 的运行表明并非如此：INT8 本身（批量或无状态）在 86/125 例中保持相同文本集合，Top-1 为 125/125，掉出的大多在第 9–10 名。拿 INT8 包与无状态 INT8 搜索比较，而不是与 FP32 fixture 比较。
   - 还要用 FP32 flex 包并把 `--chunk` 设为 FP32 chunk 包运行一次。此时排序应完全一致，分数在 1e-3 以内。
   - 报告每次生成的平均调用数/行数和 Mac 上每次生成的 p50。
10. **客户端资源。** 根据测量选出要发布的 INT8 变体；除非 flex 明显慢于固定 16，否则默认 flex。然后运行：

    ```bash
    python scripts/zh/deployment/package_zh_v1_kvbatch_ios.py --batch out/int8-flex --alignment out/int8-flex-align.json \
        --search out/search-r2 --fixture out/fixture --output out/ios-kvbatch
    ```

    它写出 `Resources/TinyChineseV1KVBatchINT8.mlmodelc` 和一个 `kv_batch` manifest 段，包含现有 kv 条目的字段加上批大小范围。现有 zh-v1 资源保持不动。
11. **iPhone（如果有测试宿主）。**
    - iPhone 13 Pro 和 iPhone 16 Pro Max 上 B = 1、4、8、16 的每次 predict 延迟，S 取真实桶会用的值（16 和 32）。
    - 在已加载无状态包的情况下增加的内存占用。
    - 客户端代码归 Vime 所有。不要从本会话提交 Vime 端的改动。
12. **发布副本。** 把选定的 INT8 包拷贝到 `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch/`。不要编辑 `RELEASE.json` 或 `SHA256SUMS.txt`；由 Windows 端更新。

## 报告

按 zh-v1 Core ML 报告的格式写 `docs/zh/reports/mac-<date>/zh-v1-kvbatch.md`。内容应包括：

- 环境；
- 每个阶段的结果，包括作为证据保留的失败；
- 接口和两处有意差异；
- 延迟表（Mac，若运行了还有 iPhone）；
- 内存和内存占用；
- 合并包的结果；
- FP16/ANE 的结果；
- 搜索检查；
- 客户端资源使用哪个包，以及原因。

报告中不出现基准文本。

带回：

- `outputs/deployment/zh-v1-kvbatch/`（空间不够时可不带 FP32 包）；
- 选定的发布副本；
- 报告；
- 资源目录。

用 `scripts/zh/deployment/package_zh_v1_mac_results.py` 或带 SHA256SUMS 的普通 zip 打包。
