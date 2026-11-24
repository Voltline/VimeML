# zh 桌面端延迟：Windows 上的 PyTorch（2026-11-18/19）

测量完整拼音流程的桌面端延迟（Vime 分桶搜索，beam 16，取前 10，再对这 10 个候选做自重打分），模型为 `zh200m-12k-pinyin-pt`（v2-200m 候选）、`kd-cont`（v2-20m 候选）和 vimeml-tiny-zh-v1（lm 模式）。测量环境为一台 Windows 笔记本（RTX 5060 Laptop GPU；Intel CPU，torch 8 线程），Python 3.13，PyTorch。输入：ime-pinyin-zh-v1 development 每 6 条取 1 条记录，取其全部输入（580 条 = 全拼 97、混拼 386、首字母 97），带字段内上下文；前 5 条输入为预热，不计入。

偏离计划：发布计划原定在 Mac 上测桌面端延迟。桌面客户端如果要做，会是 Windows 客户端，也不会用 Swift 写，所以桌面端延迟改为在 Windows 上用 PyTorch 测。

脚本：`scripts/zh/benchmarks/latency_zh.py`（搜索 + 重打分）、`scripts/zh/benchmarks/latency_rime_zh.py`（Rime 候选 + 重排）。CUDA graph 后端：`src/vimeml/deployment/zh_torch_graph.py`。

## 结果（2026-11-19，ms，p50/p95）

完整流程（搜索加重打分），已含下文的搜索改动：

| 模型 | Graph FP16 | Graph FP32 | Eager FP32，CUDA | CPU，8 线程 |
| --- | --- | --- | --- | --- |
| 200M | **83/205** | 170/419 | 172/407 | 795/1900 |
| kd-cont | **34/68** | 42/96 | 94/210 | 143/336 |
| v1 (lm) | **32/63** | 37/75 | 105/255 | 125/294 |

按输入类型，Graph FP16：

| 模型 | 全拼 | 混拼 | 首字母 |
| --- | --- | --- | --- |
| 200M | 100/264 | 84/194 | 61/113 |
| kd-cont | 41/87 | 34/65 | 27/42 |
| v1 | 38/77 | 33/58 | 25/41 |

总时间的构成（p50）：200M Graph FP16 模型 62、Python 11、重打分 10；kd-cont Graph FP16 模型 20、Python 8、重打分 6。Graph 捕获在加载时进行一次，耗时 1.2–2.7 s，不计入。

- **20M 模型：** 主要收益来自 CUDA graph（kd-cont 从 eager 94 降到 graph 42 ms），FP16 加成不大。
- **200M：** 每一步都要读一遍权重，所以单靠 graph 没有变化（172 → 170 ms），FP16 让时间减半（83 ms）。
- FP16 与 FP32 的搜索输出对比：200M top-1 579/580，kd-cont 578/580。

## 改动

1. `zh_search.log_softmax` 返回主机端数组。原来搜索每走一步就从设备张量读一个 log-prob，每次都要设备同步（profile 中 36 s 里占 29 s）；改后在 CUDA 上约快 4 倍。
2. 搜索循环缓存每个按键位置的走法，一组走法用一次数组读取打分，只有进入 beam 时才构建假设（与之前一样按稳定降序）。kd-cont（task）和 v1（lm）在全部 580 条输入上的前 10 文本和分数与旧搜索完全相同。graph 后端上 60 条 kd-cont 输入的 profile：8.19 → 3.26 s。
3. `zh_torch_graph.GraphBackend` 从 CUDA graph 重放解码步（固定批次 16 行，缓存宽度 32/64/128），token、位置和行索引一次拷贝到设备。FP32 搜索输出在全部 580 条输入上与 eager 完全相同（最大分数差 6e-6）。

## 搜索改动之前（2026-11-18）

### Eager PyTorch，FP32，完整流程，只含 log_softmax 修复（ms，p50/p95）

| 模型 | CUDA | CPU，8 线程 |
| --- | --- | --- |
| 200M | 190/427（模型 130、Python 39、重打分 11） | 810/1984（模型 616） |
| kd-cont | 123/280 | 181/438 |
| v1 (lm) | 114/259 | 165/382 |

按输入类型：

| 模型、设备 | 全拼 | 混拼 | 首字母 |
| --- | --- | --- | --- |
| 200M CUDA | 243/506 | 190/419 | 141/292 |
| 200M CPU | 1022/2536 | 813/1794 | 546/1184 |
| kd-cont CUDA | 147/358 | 123/270 | 102/184 |
| kd-cont CPU | 235/543 | 185/410 | 142/296 |
| v1 CUDA | 134/318 | 114/254 | 93/179 |
| v1 CPU | 202/462 | 168/381 | 131/262 |

### CUDA graph，仅搜索，循环改动之前（ms，p50/p95）

Eager 解码受 kernel 启动次数限制；200M 每一步还要读约 800 MB 的 FP32 权重（单是 graph 重放，缓存宽度 32/64/128 时为 7.1/8.0/9.9 ms）。graph 后端为每个宽度捕获固定 16 行的批次。

| 模型 | 后端 | p50/p95 | 与 eager FP32 对比 |
| --- | --- | --- | --- |
| 200M | eager FP32 | 206/469 | — |
| 200M | graph FP32 | 169/442 | 全部 580 条前 10 完全相同，最大分数差 7.5e-6 |
| 200M | graph FP16 | 116/295 | top-1 579/580，前 10 列表有 519 条相同 |
| 200M | graph BF16 | 126/324 | top-1 574/580 |
| kd-cont | eager FP32 | 131/322 | — |
| kd-cont | graph FP32 | 76/212 | 完全相同 |
| kd-cont | graph FP16 | 61/164 | top-1 578/580 |

## Rime

librime 1.17 配 rime-ice（Standard 基准用的引擎）约 7 ms 返回候选，但几乎总是只有一个整句候选，重排没有可选的（60 条输入上 v1 重排 top-1 18.3）。在公开集上，Rime-16 + v1 重打分的 Top-1 为 31.6，200M 用自己的重打分为 58.8。Rime 不是准确率上的替代方案，继续作为基线。
