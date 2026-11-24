# vimeml-tiny-zh-v1 神经网络引擎探测（第 0 步）：Mac 任务说明

本任务判断 Apple 神经网络引擎（ANE）是否值得作为 20M zh 模型的部署路径。由于架构与 v2-20m 相同，这里使用 zh-v1 的权重。zh-v1 发布及其包保持不变。

## 原因

ja-v1 期间 ANE 被搁置了。那次测试使用的是无状态整序列包（FP16，枚举长度 16/32/64/128，逐通道 INT8，完整 `[1,T,V]` logits）。它比 CPU 慢 3–4 倍，原因一直没有查明。此后所有 zh-v1 包都是 FP32 并以 `CPU_ONLY` 运行，所以都用不上 ANE。

ANE 需要 FP16 计算和固定形状。它还偏好通道在前的张量、用 1×1 卷积代替 Linear 层，以及按头拆分的注意力。有两种负载值得关注：

- **整句打分**（`score`，共享前缀上的 B16×T16）：每次调用约 10 GFLOP。这是 ANE 最可能占优的地方。
- **Beam 解码**（`scored`，B16，T=1）：矩阵很小。这里 CPU 可能仍然更快，那样结果就是混合方案。

## 测量内容：`vimeml.deployment.zh_ane`

同一套数学的两种布局。两者都接收 v3 的输入（共享前缀加每行缓存），形状固定。

| 布局 | 说明 |
| --- | --- |
| `bt` | v3 的数学和缓存布局。加性掩码为 -1e4，用 `where` 构建（v3 的 -1e9 在 FP16 中变成 -inf，而 0·inf 得到 NaN）。 |
| `ane` | 隐藏状态 `[B,384,1,T]`，每个投影都用 1×1 `conv2d`，注意力按头拆分。Key 为 (9,B,S,384)，value 为 (9,B,384,S)。RMSNorm 输入缩放 1/16，使 x² 之和保持在 FP16 范围内（真实模型的隐藏值可达约 53）。 |

函数网格，每种布局和权重格式一个多函数包：

| 函数 | 形状 |
| --- | --- |
| `score_b16_t8_p32`、`score_b16_t16_p32`、`score_b16_t32_p32`、`score_b16_t16_p64` | B，T 个续写 token，P 个前缀槽位；输出 `target_logprobs` [B,T] |
| `step_b16_p32_s16`、`step_b16_p64_s32` | T = 1，S 个行缓存槽位，256 个 `allowed` id；输出 `logsumexp`、`allowed_logits`、`new_key`、`new_value` |

权重：`fp16`、`int8c`（逐通道 INT8）和 `int8b32`（逐块 32 INT8）。计算精度始终为 FP16。输入输出默认 FP16（可用 `--io fp32`）。

## Windows 检查

| 检查 | 结果 |
| --- | --- |
| 两种布局在真实权重上、在每个网格函数上与 v3 的 FP32 相同（`check`） | 最大 4.1e-5 |
| 单元测试（`tests/test_deployment_zh_ane.py`）：FP32 相等（含新 key/value）、traced = eager、布局转换 | 3/3 |
| CPU 上的 PyTorch FP16，`bt`：133 个排序用例 | top-1 133/133，排序完全一致 122/133，11 对翻转，最大求和差 0.034 |
| CPU 上的 PyTorch FP16，`ane`：133 个排序用例 | top-1 133/133，排序完全一致 110/133，26 对翻转，最大求和差 0.33 |
| PyTorch FP16，16 个 beam 行（step） | `bt` 和 `ane` 都是 top-1 16/16，top-5 顺序 15/16，最大对数概率差 0.030 / 0.037 |

133 个排序用例是 125 个搜索 fixture（上下文加不同的前几名候选）和 8 个整句打分 fixture。CPU 上的 PyTorch 半精度不是 ANE 的算术，所以以 Mac 上的 Core ML 运行为准。

`coremltools` 没有在 Windows 上运行。

## Mac 步骤

在仓库根目录用 v3 任务的 venv 运行。使用 `OUT=outputs/deployment/zh-v1-ane-probe`。每个阶段都拒绝覆盖已有输出。`check` 和 `accuracy` 读取 `outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl`，即 v3 搜索阶段用过的同一文件。

1. **检查。**
   ```
   python scripts/zh/deployment/ane_probe_zh_v1.py check $OUT/check.json
   python -m unittest tests.test_deployment_zh_ane
   ```
2. **转换。** 六个包：
   ```
   for L in bt ane; do for W in fp16 int8c int8b32; do
     python scripts/zh/deployment/ane_probe_zh_v1.py convert $OUT/$L-$W --layout $L --weights $W
   done; done
   ```
   如果某个转换失败，记录错误并继续其他的。如果 FP16 I/O 失败，对那个包试 `--io fp32` 并说明。
3. **计划。** 对每个包：
   ```
   python scripts/zh/deployment/ane_probe_zh_v1.py plan $OUT/<pkg> $OUT/plan-<pkg>.json --units CPU_AND_NE ALL
   ```
   它按函数报告有多少 op、多少估计开销分到 CPU、GPU 和 ANE，以及哪些 op 类型留在 ANE 之外。把 `CPU_AND_NE` 的 op 计数放进报告。
4. **准确率**，在 `CPU_AND_NE` 上，对每个转换成功的包：
   ```
   python scripts/zh/deployment/ane_probe_zh_v1.py accuracy $OUT/<pkg> $OUT/accuracy-<pkg>.json --unit CPU_AND_NE
   ```
   在同样的用例上对 `CPU_ONLY` 的 v3 INT8 包打分（`v3_int8_cpu`）。两者都与 PyTorch FP32 比较。对最好的包，也运行 `--unit CPU_ONLY`。
5. **延迟。** 所有包在 `CPU_ONLY` 和 `CPU_AND_NE` 上，与相同 B/T/P/S 的 `CPU_ONLY` v3 INT8 交替运行：
   ```
   python scripts/zh/deployment/ane_probe_zh_v1.py measure $OUT/latency.json \
     bt-fp16=$OUT/bt-fp16 bt-int8c=$OUT/bt-int8c bt-int8b32=$OUT/bt-int8b32 \
     ane-fp16=$OUT/ane-fp16 ane-int8c=$OUT/ane-int8c ane-int8b32=$OUT/ane-int8b32
   ```
   如果耗时太长，去掉第 3 步中放到 ANE 上的工作很少的包，并说明是哪些。然后对最好的一两个包也用 `--units CPU_AND_GPU ALL` 运行。
6. **内存。** 始终包括 `CPU_ONLY` 上的 v3 INT8。加上最好的包在 `CPU_AND_NE` 和 `CPU_ONLY` 上：
   ```
   python scripts/zh/deployment/ane_probe_zh_v1.py memory $OUT/memory.json \
     ane-fp16=$OUT/ane-fp16:CPU_AND_NE ane-fp16=$OUT/ane-fp16:CPU_ONLY
   ```
   每轮负载为 16 次 step 调用和一次 score 调用。每种配置在一个不计入测量的预热进程之后，在全新进程中运行。ANE 的缓冲区部分位于系统守护进程中，所以 `phys_footprint` 可能少算。在数字旁边注明这一点。
7. **功耗**，如果有 `sudo powermetrics`。在一个终端中：
   ```
   sudo powermetrics --samplers cpu_power,ane_power -i 1000 -n 25 > $OUT/power-<label>.txt
   ```
   同时在另一个终端中：
   ```
   python scripts/zh/deployment/ane_probe_zh_v1.py loop <pkg or v3> score_b16_t16_p32 --unit CPU_AND_NE --seconds 30
   ```
   对 v3（`loop v3 score_b16_t16_p32`，CPU）、`CPU_AND_NE` 上最好的包，以及两者的 `step_b16_p32_s16` 都这样做。报告平均 CPU 和 ANE 功耗以及每秒调用次数。每次调用的能耗是功耗除以每秒调用次数。

## 继续 / 停止

如果在 `CPU_AND_NE` 上某个包和函数满足以下全部条件，就**继续**为 v2-20m 做 ANE 部署设计：

- 在 B16×T16 和 B16×T32 上，score 函数的热 p50 至少比 `CPU_ONLY` 上的 v3 INT8 快 1.5 倍；
- 同样负载下峰值内存不高于 `CPU_ONLY` 上的 v3 INT8；
- 在每个排序用例上 top-1 与 PyTorch FP32 相同，翻转对数与 CPU 上的 v3 INT8 接近。

Step 函数要报告，但不作为门槛。如果只有打分占优，结果就是混合方案：解码在 CPU 上，打分在 ANE 上。

否则就**停止**：报告数字，这条线到此结束。两种情况下都要报告冷加载和首次调用时间，因为首次加载时的 ANE 编译是已知开销。

## 报告

写 `docs/zh/reports/mac-<date>/zh-v1-ane-probe.md`，包含：

- 环境（芯片、macOS、coremltools、Python）；
- 每个阶段一张表：包大小、ANE op 占比、准确率（top-1、排序完全一致、翻转对数、对 PyTorch FP32 和对 v3 的最大求和差）、每个函数和计算单元的热 p50/p95 并与 v3 并列、加载和首次调用时间、内存和功耗；
- 按上述规则给出继续 / 停止的结论；
- 与这些步骤的每一处偏差。

回传报告和 JSON 输出。不要回传 `.mlpackage` 目录。
