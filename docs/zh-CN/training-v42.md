# V4.2 LM／Chat 预训练

V4.2 从随机权重开始，沿用 V4.1 的原 LM 与聊天数据池，移除 C1r2 转换目标与结构化 READ 输入，将预算从 16 轮延长到 20 轮。实验检验在去掉转换任务后，同一固定 15M 模型的普通 LM 重排是否更强。V4.1 结果见[训练设置](training-v41.md)与[完整评估](reports/v41-20261016/evaluation.md)。

2026-10-20 正常完成 20 轮、1,081,942 次更新。按完整 LM BPC 预先选出的第 20 轮在 Standard development 达到 523/728；旧网页与 AJIMEE 相对 V2.1 仍有回退。完整结果与配对比较见[评估报告](reports/v42-20261020/evaluation.md)。

## 架构与随机性

模型保留 V4 的 8,192 词表、context 128、d384、七层、六头、SwiGLU 960，共 15,070,848 参数。tokenizer SHA256 为 `296a2bdf…81b5`，与 V4/V4.1 相同。

seed 沿用 V4.1 的 **1219827974**，用于权重初始化、训练顺序、prefix crop 和聊天采样。没有加载任何既有 checkpoint，AdamW 状态从零开始。

## 数据组成

| 数据池 | 用法 |
| --- | --- |
| V2.1 FineWeb/Tatoeba，V4 8K 编码 | 沿用 V4.1 排除掩码，每个 epoch 完整遍历保留窗口 |
| V3 RealPersonaChat / MRMP | 沿用 V4.1 聊天 bank 与对话级划分，占实际 LM 目标 token 约 5% |
| C1r2 转换训练集 | 移除 |

[`prepare_v42.py`](../../scripts/corpus/prepare_v42.py) 以冻结的 V4.1 数据请求为基础，仅删除转换相关字段并将 epochs 由 16 改为 20；合并 bank、排除掩码、seed、30% prefix crop（至少保留八个目标）和 5% 聊天配额不变。脚本要求重建的第 0–15 轮映射与 V4.1 逐字节一致，否则拒绝写入。没有读取 test 或 blind。

每轮约 2,770 万个窗口、54,095–54,098 次更新；最后不足整批的窗口保留。20 轮共 553,949,647 个窗口、12,104,389,994 个监督目标（含混合聊天）。

## 目标函数与训练设置

目标为普通无标记 next-token loss，覆盖全部有效目标，每次更新按总监督 token 数归一化。没有 `<CTX>`、`<READ>`、`<OUT>` 输入，也没有任务权重。

| 项目 | 设置 |
| --- | --- |
| 配置 | [train-v42.toml](../../configs/train-v42.toml) |
| 环境 | 单张 NVIDIA L20C；PyTorch 2.7.0+cu128 |
| 精度与运行时 | BF16 AMP；动态形状编译主干；有效目标词表头与 CE 保持 eager |
| 优化器 | fused AdamW，betas (0.9, 0.95)、weight decay 0.1、clip 1.0 |
| Batch | 512，单个 microbatch |
| 学习率 | 1,000 次更新 warmup；完整 20 轮 cosine，6e-4 至 6e-5 |
| 验证 | 每 2,000 次更新子集验证；每轮完整 FP32 LM 与 Chat BPC |
| 保存 | 每 5,000 次更新、每轮、`best-lm.pt` 与 `last.pt` |

训练中不做 IME 评分；各轮 checkpoint 保留用于事后候选池评估。远端 `training_seconds=11564.97`（约 3.21 小时）只累计更新阶段，不含验证和保存，不是作业墙钟时间。

## 产物与恢复

| 路径 | 内容 |
| --- | --- |
| `artifacts/training-data/corpus-v42-mixed-c128/` | 派生数据请求、排除掩码及二十轮映射 |
| `artifacts/models/tiny-ja-v42-e8k-d384-l7/` | 训练端权重、optimizer/RNG/游标与验证记录 |
| `outputs/training-v42-20261020/` | 固定 revision 下载凭据、训练记录、实际源码与 `best-lm.pt` |
| `outputs/evaluation-v42-20261020/` | 本地四组 FP32 重排、配对与验收 |
| [Voltline/vimeml-tiny-ja-v4.2](https://huggingface.co/Voltline/vimeml-tiny-ja-v4.2) | 公开发布：FP32 推理包、INT8 stateless 与分段 KV Core ML、源码快照与聚合评测；Hub commit `c83c6179f6ebac1af61088ee7e7f833dd9e03d58` |

正式入口在读取数据前校验配置、源码、tokenizer、数据与全部映射身份；恢复保留游标、AdamW、RNG 与验证历史。`cluster_v42.sh` 在存在 `last.pt` 时自动续训，训练解释器通过 `VIMEML_PYTHON` 指定。

仓库中的训练器、入口与配置与归档的实际运行源码逐字节一致，并与运行记录的 `code_sha256` 相符；仅启动脚本中站点专用的解释器默认路径改为 `VIMEML_PYTHON`。[`evaluate_v42.py`](../../scripts/benchmarks/evaluate_v42.py) 保留归档本地评估脚本的评分逻辑，仅增加显式输出目录参数；已完成评估保留原脚本哈希。数据和权重不纳入 Git。

## 结论边界

相对 V4.1，本轮同时移除转换目标并延长学习率预算；相对 V4-B，还改变了聊天混合、裁剪、seed 和曝光。因此结果支持这组完整训练方案，不能单独归因于移除 READ。第 20 轮已冻结。预注册的固定 reference 评估为 541/725，较 V2.1 仅 +6 且不显著，翻转以书写形式差异为主，见[reference 与退化分析](reports/v42-20261020/reference-regressions.md)；现有证据不支持替换 V2.1。Mac Core ML FP32/INT8 转换、分段 KV 图以及模拟器和 iPhone 16 Pro Max 测试宿主测量见 [V4.2 Core ML 报告（英文）](../reports/mac-20261020/v42-coreml.md)：INT8 包 17.16 MB，四组 Top-1 为 522/371/139/123，设备上单次调用略慢、峰值内存略低；键盘扩展尚未验证。
