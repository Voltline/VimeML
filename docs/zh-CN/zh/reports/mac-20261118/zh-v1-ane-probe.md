# vimeml-tiny-zh-v1 Neural Engine 探查（第 0 步）Mac 结果 —— 2026-11-18

按 [ANE 探查任务说明](../../mac/mac-zh-v1-ane-probe.md)第 1–7 步执行。运行目录为 `~/Documents/Sources/VimeML-zh-kvbatch`，fast-forward 拉到 `main` `519784d`。

**拉取之前。** Mac 上未跟踪的 v3 报告副本（`docs/zh/reports/mac-20261118/zh-v1-kvbatch-v3.md`）与上游文件冲突。已移到 `outputs/deployment/mac-20261118-kvbatch-v3-local-backup/`（与上游相差 16 行编辑性改动）。拉取后 `git status` 干净。

**未改动：**

- 没有编辑任何跟踪文件；
- 没有动 `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/` 下的任何内容（包括 `RELEASE.json` / `SHA256SUMS.txt`）；
- 没有提交或推送任何内容。

**环境**（`logs/00-environment.log`）：

- 硬件与系统：Apple M3（16 GiB），macOS 27.3（26C5047e），Xcode 27.3（27C5035i）。
- Python：3.11.17（uv venv），coremltools 9.0，PyTorch 2.7.1，NumPy 2.3.5。
- 电源与负载：接交流电，平均负载约 2–3。

**输出。** 全部在 `outputs/deployment/zh-v1-ane-probe/`（`$OUT`）下：各阶段日志在 `logs/`，辅助脚本在 `tools/`。没有覆盖任何输出，失败的运行也保留了。

**排序用例。** 排序用例是 `outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl`，与 v3 搜索阶段同一个文件（133 例）。

## 结论：按任务说明的规则为 **no-go**，但打分速度提升非常大

| 规则（在 `CPU_AND_NE` 上） | 最佳包 `bt-fp16-iofp32` | 结果 |
| --- | --- | --- |
| 打分比 v3 INT8 CPU 快 ≥ 1.5 倍，热 p50，B16×T16 | 2.18 ms 对 16.11 ms，**7.4 倍** | 通过 |
| 打分比 v3 INT8 CPU 快 ≥ 1.5 倍，热 p50，B16×T32 | 4.10 ms 对 28.33 ms，**6.9 倍** | 通过 |
| 同一负载下峰值内存 ≤ v3 INT8 CPU | **149.7 MiB** 对 15.8 MiB | **失败** |
| 每个排序用例的 Top-1 都与 PyTorch FP32 相同 | **132/133** | **失败** |
| 翻转对数与 v3 INT8 CPU 接近 | 41 对 92（更少） | 通过 |

其他包：

- 凡是把计算放到 ANE 上的包都在同样两条规则上失败。它们都在同一个用例上错 top-1，`bt-int8c-iofp32` 也有同样约 150 MiB 的峰值。
- 唯一 top-1 为 133/133 的包是 `ane-int8b32`。它在 CPU 上运行，不在 ANE 上（第 3 节），在 CPU_AND_NE 上比 v3 慢。

所以没有任何包满足全部三条规则，按任务说明的定义这条线到此结束。其中两项失败的差距很小，值得在彻底关闭这条线之前了解：

- **Top-1。** 只错了一个用例 `zh-pin-crosswoz-94d91a6a27/initials`（10 个候选）。每个 FP16 包都错这一例，`bt-fp16-iofp32` 在 `CPU_ONLY` 上也错。所以原因是 FP16 运算，不是 ANE 本身。Windows 上的 PyTorch FP16 为 133/133。在其他各项指标上，FP16 包的排序都比 v3 INT8 更接近 FP32：翻转对 41–47 对 92，完全一致排序 95–100 对 71。
- **内存。** 150 MiB 是模型加载期间的瞬时峰值，即使 ANE 程序已经被预热进程缓存也是如此。20 轮负载后保留 17 MiB（v3：2.3 MiB）。系统守护进程持有的 ANE 缓冲区不计入，所以真实开销更高。

step 函数不设门槛：

- `bt-fp16-iofp32` 在 ANE 上解码比 v3 快（P32 S16 时 1.59 对 3.11 ms），在 `CPU_ONLY` 上也略快。
- `ane` 布局的 step 不比 v3 快。

## 与任务说明的偏离

1. **单元测试。** `python -m unittest tests.test_deployment_zh_ane` 按原样运行（3 个测试 OK）。v3 任务中的 `PYTHONPATH` 问题这里没有出现。
2. **第 3 步 `plan`。** 任务说明中的这个阶段只为包的默认函数做计划。coremltools 9.0 的 `MLComputePlan.load_from_path` 没有 `function_name` 参数，所以另外五个函数的每个算子都不返回设备分配。因此 `plan-<pkg>.json` 只有 `score_b16_t8_p32` 的计数，其余为空。这些文件按运行结果保留。
   - `tools/plan_per_function.py`（新增，不属于仓库代码）逐个处理函数：复制包，把该函数设为 `defaultFunctionName`，编译后做计划。
   - 结果在 `plan-perfn-*.json`。第 3 节用的是这些。
3. **使用 FP16 I/O 的 `bt-fp16` 和 `bt-int8c`。** 它们的 score 函数能在 `CPU_AND_NE` 上加载运行，两个 step 函数不能：
   - 逐函数计划报错 "doesn't support the multi-function description syntax"；
   - 加载报错 "`.functionName` must be nil unless the model type is ML Program"（`logs/03b`、`03c`、`03d`、`04-accuracy-bt-*.log`、`04b-load-probe-bt.log`）；
   - 因此任务说明中的 `accuracy` 在 step 检查时崩溃，没有写出 `accuracy-bt-fp16.json` / `accuracy-bt-int8c.json`。日志保留。

   按任务说明允许的方式修复：
   - 两者都用 `--io fp32` 重新转换为 `bt-fp16-iofp32` 和 `bt-int8c-iofp32`。此后每个函数都能在 `CPU_ONLY`、`CPU_AND_NE` 和 `ALL` 上加载运行（`tools/load_probe.py`）。
   - `bt` FP16 / 按通道 INT8 包的精度、延迟和内存都用这两个 `-iofp32` 包。
   - `ane` 布局和 `bt-int8b32` 保留 FP16 I/O，对它们可用。
4. **第 5 步。** 没有剔除任何包：六个包的运行用了 394 s。
   - `bt-fp16-iofp32` 和 `bt-int8c-iofp32` 替代了 FP16 I/O 版本，后者会在 `measure` 加载 step 时崩溃。
   - 最好的两个包 `bt-fp16-iofp32` 和 `bt-int8c-iofp32` 还在 `CPU_AND_GPU` 和 `ALL` 上跑了一遍（`latency-gpu-all.json`）。
5. **第 6 步。** 任务说明的示例用 `ane-fp16`。我改用最佳包 `bt-fp16-iofp32`，在 `CPU_AND_NE` 和 `CPU_ONLY` 上测，另加 `bt-int8c-iofp32` 在 `CPU_AND_NE` 上测。
6. **第 7 步功耗：跳过。** `sudo -n true` 提示需要密码。这台 Mac 没有免密 sudo，我也不输入密码。所以没有测功耗，配合功耗采样的 `loop` 运行也没有做。

## 1. 检查（`check.json`、`logs/01-check.log`、`logs/01-unittest.log`）

- 两种布局在每个网格函数上 FP32 都与 v3 相同：最大 **6.0e-5**（Windows：4.1e-5）。
- Mac CPU 上的 PyTorch FP16，133 个排序用例：

  | 布局 | top-1 | 完全一致排序 | 翻转对 | 最大和差 |
  | --- | ---: | ---: | ---: | ---: |
  | `bt` | 133 | 122 | 11 | 0.040 |
  | `ane` | 133 | 112 | 25 | 0.354 |

- step（16 个 beam 行）：
  - `bt`：top-1 16/16，top-5 排序 16/16；
  - `ane`：top-1 16/16，top-5 排序 15/16。
- 单元测试：3/3 OK。

## 2. 转换（`logs/02-*.log`）

六个包一次全部转换成功，另有两个 `--io fp32` 重建。

| 包 | 字节 | 对 v3 INT8（23,139,825） |
| --- | ---: | ---: |
| `bt-fp16` | 40,975,536 | 1.77× |
| `bt-fp16-iofp32` | 40,985,026 | 1.77× |
| `bt-int8c` | 21,088,408 | 0.91× |
| `bt-int8c-iofp32` | 21,097,882 | 0.91× |
| `bt-int8b32` | 22,246,296 | 0.96× |
| `ane-fp16` | 52,996,963 | 2.29× |
| `ane-int8c` | 28,419,139 | 1.23× |
| `ane-int8b32` | 29,847,363 | 1.29× |

同样的权重，`ane` 布局比 `bt` 大约 12 MB（fp16）/ 约 7 MB（int8）。它的卷积布局投影和按头拆分似乎没有与共享的 embedding/输出头常量共用。未进一步调查。

## 3. 计划：`CPU_AND_NE` 上的 ANE 分配（`plan-perfn-*.json`）

ANE 上的算子 / 全部算子，以及 ANE 占估计开销的比例（c）：

| 包 | score t8 p32 | score t16 p32 | score t32 p32 | score t16 p64 | step p32 s16 | step p64 s32 |
| --- | --- | --- | --- | --- | --- | --- |
| `bt-fp16` | 391/406 c.84 | 391/406 c.90 | 391/406 c.94 | 391/406 c.91 | 计划报错 | 计划报错 |
| `bt-fp16-iofp32` | 394/409 c.84 | 394/409 c.90 | 394/409 c.94 | 394/409 c.91 | 420/450 c.86 | 420/450 c.89 |
| `bt-int8c` | 391/410 c.63 | 391/410 c.76 | 391/410 c.85 | 391/410 c.77 | 计划报错 | 计划报错 |
| `bt-int8c-iofp32` | 394/413 c.63 | 394/413 c.76 | 394/413 c.85 | 394/413 c.77 | 420/458 c.68 | 420/458 c.74 |
| `bt-int8b32` | 11/410 c.02 | 191/410 c.04 | 191/410 c.05 | 191/410 c.05 | 0/450 | 0/450 |
| `ane-fp16` | 1186/1203 c.83 | 1186/1203 c.88 | 1186/1203 c.92 | 1186/1203 c.89 | 1524/1551 c.74 | 1524/1551 c.84 |
| `ane-int8c` | 1186/1207 c.58 | 1186/1207 c.71 | 1186/1207 c.82 | 1186/1207 c.72 | 1524/1559 c.49 | 1524/1559 c.63 |
| `ane-int8b32` | 14/1207 c.03 | 14/1207 c.03 | 968/1207 c.12 | 14/1207 c.03 | 0/1559 | 0/1559 |

说明：

- **FP16 / 按通道包中不在 ANE 上的部分。** 每个函数约 15–21 个算子留在 CPU 上：输入 `cast`、掩码（`greater_equal`、`select`、`tile`）、目标 `gather` 和少量 `add`/`expand_dims`。INT8 按通道包里反量化又多出几个。这些包中剩余的开销比例没有归到任何设备。
- **按 32 分块的 INT8。** 它让整个图离开 ANE：带分块 32 权重的 `linear` 被标为 "ANE unsupported"（46 个算子），其余算子也跟着去了 CPU。在 `ALL` 下，分块 32 的包去了 GPU（`plan-*.json` 的 ALL 行）。
- **FP16 I/O 的 `bt-fp16` / `bt-int8c` step。** 在 `CPU_AND_NE` 上不产生计划（偏离 3）。在 `ALL` 下计划把它们全部放在 GPU 上（`plan-perfn-v2-bt-*.json`）。

## 4. 精度（`accuracy-*.json`、`logs/04*-accuracy-*.log`）

所有行都是在 133 个排序用例上与 PyTorch FP32 对比。v3 INT8 包在 `CPU_ONLY` 上、由同一阶段在同样用例上打分，作为对照行。该阶段输出的是 probe 对 FP32 和 v3 对 FP32，不直接输出 probe 对 v3。

| 包 / 单元 | top-1 | 完全一致排序 | 翻转对 | 最大和差 | 最大差中位数 | 错 top-1 的用例 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **v3 INT8 / CPU_ONLY** | **133** | 71 | 92 | 0.309 | 0.119 | – |
| `bt-fp16-iofp32` / NE | 132 | **100** | **41** | 0.287 | 0.068 | crosswoz-94d91a6a27/initials |
| `bt-fp16-iofp32` / CPU | 132 | 97 | 46 | 0.148 | 0.052 | crosswoz-94d91a6a27/initials |
| `bt-int8c-iofp32` / NE | 132 | 40 | 176 | 0.543 | 0.247 | crosswoz-94d91a6a27/initials |
| `bt-int8b32` / NE（在 CPU 上运行） | 132 | 70 | 103 | 0.307 | 0.127 | risawoz-88ae3e71d0/full |
| `ane-fp16` / NE | 132 | 95 | 47 | 0.319 | 0.075 | crosswoz-94d91a6a27/initials |
| `ane-int8c` / NE | 132 | 43 | 172 | 0.622 | 0.248 | crosswoz-94d91a6a27/initials |
| `ane-int8b32` / NE（在 CPU 上运行） | **133** | 63 | 104 | 0.533 | 0.150 | – |

step 检查（`step_b16_p32_s16`，16 个 beam 行，对 PyTorch FP32）：

| 包 / 单元 | top-1 | top-5 排序 | 最大 log-prob 差 |
| --- | ---: | ---: | ---: |
| `bt-fp16-iofp32` / NE | 16/16 | 14/16 | 0.077 |
| `bt-fp16-iofp32` / CPU | 16/16 | 14/16 | 0.195 |
| `bt-int8c-iofp32` / NE | 16/16 | 13/16 | 0.468 |
| `bt-int8b32` / NE | 16/16 | 13/16 | 0.234 |
| `ane-fp16` / NE | 16/16 | 14/16 | 0.075 |
| `ane-int8c` / NE | 16/16 | 13/16 | 0.463 |
| `ane-int8b32` / NE | 16/16 | 12/16 | 0.283 |

说明：

- ANE 上的 FP16 权重比 v3 INT8 排序更接近 FP32：翻转对减半，完全一致排序更多。它们恰好错了一个用例的 top-1。
- 在 FP16 计算上叠加按通道 INT8 明显比 v3 的分块 32 INT8 差：翻转对约为 v3 的 2 倍。分块 32 不能在 ANE 上运行。

## 5. 延迟（`latency.json`、`latency-gpu-all.json`、`logs/05*-latency*.log`）

设置：

- Mac Python `predict()` 墙钟时间，含输入转换。
- 20 次预热调用后，3 次交替运行 × 100 次调用的中位数。
- v3 = 已发布 INT8 包（`ios18-int8-block32-kvbatch-v3`），`CPU_ONLY`，B/T/P/S 相同。

### 热 p50 / p95，ms，`CPU_AND_NE`（以及 v3）

| 函数 | v3 CPU | `bt-fp16-iofp32` | `bt-int8c-iofp32` | `bt-int8b32` | `ane-fp16` | `ane-int8c` | `ane-int8b32` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| score B16 T8 P32 | 9.29 / 9.50 | **1.59** / 1.61 | 1.34 / 1.37 | 4.11 / 4.25 | 5.27 / 5.33 | 5.27 / 5.34 | 11.62 / 12.01 |
| score B16 T16 P32 | 16.11 / 16.32 | **2.18** / 2.27 | 2.11 / 2.20 | 7.03 / 7.15 | 4.71 / 4.83 | 4.71 / 4.80 | 11.19 / 12.02 |
| score B16 T32 P32 | 28.33 / 28.66 | **4.10** / 4.21 | 4.10 / 4.19 | 11.44 / 11.69 | 5.30 / 5.39 | 5.27 / 5.34 | 11.90 / 12.07 |
| score B16 T16 P64 | 16.82 / 16.95 | **2.36** / 2.47 | 2.31 / 2.40 | 7.09 / 7.14 | 4.83 / 4.90 | 4.82 / 4.89 | 11.91 / 12.53 |
| step B16 P32 S16 | 3.11 / 3.15 | **1.59** / 1.62 | 1.30 / 1.34 | 2.19 / 2.23 | 3.01 / 3.10 | 2.88 / 2.95 | 3.41 / 3.49 |
| step B16 P64 S32 | 3.33 / 3.39 | **2.42** / 2.45 | 2.12 / 2.14 | 3.03 / 3.06 | 3.64 / 3.68 | 3.48 / 3.52 | 4.91 / 4.98 |

`CPU_AND_NE` 上打分相对 v3 的加速：

| 包 | T8 | T16 | T32 | T16 P64 |
| --- | ---: | ---: | ---: | ---: |
| `bt-fp16-iofp32` | 5.9× | **7.4×** | **6.9×** | 7.1× |
| `bt-int8c-iofp32` | 6.9× | 7.7× | 6.9× | 7.3× |
| `ane-fp16` | 1.8× | 3.4× | 5.4× | 3.5× |

### 同样的包在 `CPU_ONLY` 上（热 p50，ms）

| 函数 | v3 | `bt-fp16-iofp32` | `bt-int8c-iofp32` | `bt-int8b32` | `ane-*`（三个） |
| --- | ---: | ---: | ---: | ---: | ---: |
| score T8 P32 | 9.29 | 3.71 | 3.74 | 3.78 | 9.90 |
| score T16 P32 | 16.11 | 6.17 | 6.17 | 6.22 | 9.77 |
| score T32 P32 | 28.33 | 10.15 | 10.16 | 10.22 | 11.45 |
| score T16 P64 | 16.82 | 6.55 | 6.54 | 6.64 | 10.55 |
| step P32 S16 | 3.11 | 1.78 | 1.78 | 2.19 | 3.00–3.29 |
| step P64 S32 | 3.33 | 2.33 | 2.32 | 3.02 | 4.51 |

单是用 v3 的 `bt` 计算方式改为 FP16 计算，CPU 路径就比 v3 的 FP32 计算 INT8 包快 2.5–2.8 倍。这是一个附带结果：不需要 ANE。代价是 FP16 精度（同样错一个 top-1，第 4 节）和 `CPU_ONLY` 上 124 MiB 的常驻内存（第 6 节）。

### `CPU_AND_GPU` 和 `ALL`，最好的两个包（热 p50，ms；`latency-gpu-all.json`，v3 重测：T16 16.22，T32 29.11）

| 函数 | `bt-fp16-iofp32` GPU | `bt-fp16-iofp32` ALL | `bt-int8c-iofp32` GPU | `bt-int8c-iofp32` ALL |
| --- | ---: | ---: | ---: | ---: |
| score T8 P32 | 4.03 | 1.59 | 6.20 | 1.35 |
| score T16 P32 | 6.04 | 2.15 | 6.36 | 2.08 |
| score T32 P32 | 10.01 | 4.09 | 10.74 | 4.09 |
| score T16 P64 | 6.46 | 2.36 | 6.82 | 2.30 |
| step P32 S16 | 3.29 | 3.41 | 4.27 | 4.26 |
| step P64 S32 | 4.19 | 4.19 | 5.11 | 5.11 |

- 在 `ALL` 下，打分与 `CPU_AND_NE` 相同（去了 ANE），但 step 函数去了 GPU，比 v3 慢。
- 单用 GPU 打分比 ANE 慢。

### 冷加载与首次调用（延迟 JSON 中的 `loads`）

`load_ms` 是在一个进程中加载某个包全部六个函数的时间；首次调用列是加载后每个函数的第一次 `predict`。在 ANE 单元上，这些是本会话中每个包的首次加载，因此包含 ANE 编译。

| 包 / 单元 | 加载（6 个函数） | 首次调用，score | 首次调用，step |
| --- | ---: | ---: | ---: |
| `bt-fp16-iofp32` / CPU_ONLY | 3.4 s | 10–26 ms | 16–28 ms |
| `bt-fp16-iofp32` / CPU_AND_NE | **17.7 s** | 6–9 ms | 13–24 ms |
| `bt-int8c-iofp32` / CPU_AND_NE | **21.4 s** | 5–8 ms | 12–22 ms |
| `bt-int8b32` / CPU_AND_NE | 13.8 s | 12–21 ms | 14–27 ms |
| `ane-fp16` / CPU_AND_NE | **39.6 s** | 10–12 ms | 17–29 ms |
| `ane-int8c` / CPU_AND_NE | **44.0 s** | 9–11 ms | 15–27 ms |
| `ane-fp16` / CPU_ONLY | 12.2 s | 13–17 ms | 17–31 ms |
| `bt-fp16-iofp32` / CPU_AND_GPU | 3.7 s | 44–953 ms | 60–74 ms |
| `bt-fp16-iofp32` / ALL | 12.8 s | 5–8 ms | 50–60 ms |

- 在 ANE 上，一个包的首次加载要 14–44 s（六个函数）。这是 ANE 编译，任务说明已把它列为已知开销。
- 加载完成后，首次调用就已经是热的。
- 内存阶段在一个预热进程之后再加载每种配置，预热进程的作用是让 ANE 先编译并缓存程序。那里的 `after_load` ≤ 2 MiB，但没有记录加载时间。iOS 是否保留这份缓存、保留多久，从这次 Mac 运行无法得知。

## 6. 内存（`memory.json`、`logs/06-memory.log`）

设置：

- 每轮：16 次 `step_b16_p32_s16` 和 1 次 `score_b16_t16_p32`；v3 用同样形状的 `scored` / `score`。
- 20 轮，每种配置在一个不计入的预热进程之后用 5 个全新进程，取中位数。
- `phys_footprint` 相对 `start`（已导入 coremltools、已构建输入）；模型以预编译形式（`.mlmodelc`）加载。

| 配置 | 加载后 | 第一轮后 | 20 轮后 | 峰值 | 最小峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **v3 INT8 / CPU_ONLY** | 1.2 | 2.0 | 2.3 | **15.8** | 15.7 |
| `bt-fp16-iofp32` / CPU_AND_NE | 0.8 | 17.0 | 17.1 | **149.7** | 149.2 |
| `bt-int8c-iofp32` / CPU_AND_NE | 1.9 | 18.1 | 18.1 | **149.8** | 149.3 |
| `bt-fp16-iofp32` / CPU_ONLY | 124.1 | 124.9 | 125.3 | **132.0** | 132.0 |

- **在 ANE 单元上，峰值出现在加载期间。** `after_load` 快照已经带着约 150 MiB 的生命周期峰值，而它当前的内存占用不到 2 MiB。每个快照的峰值在 JSON 的各次运行中。
  - 这发生在 ANE 程序已被预热进程缓存的情况下。
  - INT8 按通道包峰值相同，所以与权重大小无关。
  - 负载后保留约 17 MiB，v3 为 2.3 MiB。
- **ANE 缓冲区被低估。** ANE 程序及其缓冲区部分位于系统守护进程（`aned`）中，本进程的 `phys_footprint` 不计入。上面的 ANE 数字是下限。
- **`CPU_ONLY` 上 FP16 权重常驻内存。** 该包从加载起就常驻 124 MiB，约为其 41 MB 权重的 3 倍。v3 则从文件映射其 INT8 权重。

## 7. 功耗

跳过：这台 Mac 没有免密 `sudo`（偏离 6）。没有 `power-*.txt`，也没有 `loop` 运行。

## 文件

| 路径 | 内容 |
| --- | --- |
| `check.json` | 第 1 步 |
| `bt-*/`、`ane-*/`（不在 ZIP 中） | 各个包，含 `bt-fp16-iofp32/`、`bt-int8c-iofp32/` |
| `plan-*.json` | 任务说明中的 `plan` 阶段运行结果（只有默认函数） |
| `plan-perfn-*.json`、`plan-perfn-retry-*.json`、`plan-perfn-v2-*.json` | 逐函数计划（`tools/plan_per_function.py`）；retry/v2 是 `bt` FP16 I/O 的重跑 |
| `accuracy-*.json` | 第 4 步（`-cpu` = 最佳包的 `CPU_ONLY` 运行） |
| `latency.json`、`latency-gpu-all.json` | 第 5 步 |
| `memory.json` | 第 6 步 |
| `tools/` | `plan_per_function.py`、`load_probe.py`、`run_accuracy.sh`、`run_measure.sh`、`run_rest.sh` |
| `logs/` | 各阶段日志，含失败的运行 |

## Windows 端后续（2026-11-18）

**两个 top-1 错误在 FP32 下都是近似平局。** 前两名候选的 PyTorch FP32 分数和：

| 用例 | FP32 差距，top-1 对 top-2（nats） |
| --- | ---: |
| `zh-pin-crosswoz-94d91a6a27/initials`（每个 FP16 包都错） | 0.033 |
| `zh-pin-risawoz-88ae3e71d0/full`（`bt-int8b32` 错） | 0.044 |

两个差距都小于这里每个包的单和误差，包括 CPU 上的 v3 INT8（中位数 0.119，最大 0.309）。v3 保住这两例是运气。就排序质量而言，ANE 上的 FP16 比 v3 更接近 FP32（翻转对 41 对 92）。精度规则失败在平局上，不在排序质量上。

**真正的障碍是加载时的内存。**
- 在 `CPU_AND_NE` 上，即使 ANE 程序已缓存，模型加载期间进程峰值也达到约 150 MiB。按通道 INT8 权重峰值相同。
- 键盘扩展受峰值内存限制，这么大的瞬时峰值使这条路径按实测不可行。
- `CPU_ONLY` 上 FP16 权重常驻 124 MiB，出于同样原因排除 FP16 CPU 路径。

**结论。** 按任务说明的定义为 no-go。zh-v1 的 ANE 路线到此结束。以下结果留给后续工作：
- ANE 上打分：在 Mac 上比 v3 快 6.9–7.4 倍，用的是普通的 `bt` 布局。卷积/按头拆分的 `ane` 布局更慢、更大；
- 按块 INT8 不能在 ANE 上运行；
- ANE 首次编译每个包需要 14–44 s。

**本次运行发现的工具缺口**（未修复；探查已关闭）：
- `plan` 只覆盖默认函数。coremltools 9.0 每个编译模型只为一个函数做计划；Mac 端的 `plan_per_function.py` 为每个函数分别设置 `defaultFunctionName`。
- `bt` 布局中多函数 FP16 I/O 的 step 函数加载失败。用 `--io fp32` 构建的包可以加载。
