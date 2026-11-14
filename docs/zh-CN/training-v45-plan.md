# V4.5 集成蒸馏（预注册）

> **状态：已注册，未执行。** 没有训练任何 student；工作转向中文模型后，日文线暂停。部署的日文模型仍是 V4.2 第 20 轮。计划按原文保留，以备日后恢复。

2026-11-05 注册，在任何 student 开始训练之前提交。V4.5 要回答一个问题：把四个 V4.2 配方 seed 的集成蒸馏进一个 15M 模型，部署成本与 V4.2 相同，能保留多少集成的 IME 收益。

## 为什么做

- [V4.4 消融](training-v44.md)显示，改预训练配方在这一规模上已经没有收益。Standard development 上的 seed 噪声带是 501–523（SD 9.5）。四个同配方 seed 的 logP 相加集成达到 535。
- [集成复核](benchmarks/ensemble-replication-v2-results-20261105.md)在没有参与选择的 Standard v2 blind 上复现了这个效应：集成 619，V4.2 598，净 +21，作者簇 p = 0.0012。书面语同音词上 +7，不显著。
- 集成需要四倍的参数和计算量，不能放进键盘扩展。蒸馏是把这部分收益变成可部署模型的直接办法。

## 设计

**Teacher**：四个 V4.2 配方 seed 的第 20 轮，冻结不更新。

| teacher | 来源（私有 HF，固定 revision） | step | SHA-256 |
| --- | --- | ---: | --- |
| `v42` | `Voltline/vimeml-v42-training-output` `99d91488…` `checkpoints/best-lm.pt` | 1,081,942 | `0e83ea74…` |
| `s1` | `Voltline/vimeml-v44-training-output` `a77fd281…` `s1/checkpoints/epoch-20.pt` | 1,081,944 | `31cc8120…` |
| `s2` | 同上，`s2/…` | 1,081,940 | `dd8a33ac…` |
| `s3` | 同上，`s3/…` | 1,081,937 | `bac6f0fe…` |

**Student**：八个，即两种合并方式 × 四个孪生 seed。

- 每个 student 是一个 CE teacher 的**精确孪生**：配置、seed、初始化与冻结的 20 轮混合映射都与对应的 teacher 相同（`v42` 用 V4.2 的混合，`s1`–`s3` 用 V4.4 各自的混合）。唯一的变化是训练目标。因此每个 student 都可以和自己的孪生做逐题配对比较。
- 训练器在构建计划时检查：除 student 字段（stage、arm、输出目录、`[distillation]`、`[tracking]`、`lm_microbatch`）外，配置必须与孪生配置逐项相等，且混合清单 SHA 一致。
- teacher 在 fork 出的 RNG 下构建，不影响 student 的初始化和数据流。

| 臂 | 合并方式 | 孪生 | 配置 |
| --- | --- | --- | --- |
| `geo-v42`、`geo-s1`、`geo-s2`、`geo-s3` | 归一化几何平均 | V4.2 / `s1` / `s2` / `s3` | [`train-v45-geo-*.toml`](../../configs/train-v45-geo-v42.toml) |
| `mix-v42`、`mix-s1`、`mix-s2`、`mix-s3` | 算术平均 | 同上 | [`train-v45-mix-*.toml`](../../configs/train-v45-mix-v42.toml) |

- **合并方式。** `geo`：q ∝ exp(平均 log p_k)，是"logP 相加"排序集成在 token 层面的对应形式。`mix`：q = 平均 p_k，是常见的集成蒸馏目标。两者都在 FP32 中合并。
- **损失。** 每个监督 token 上为 0.5 × CE(标签) + 0.5 × CE(q)，温度 1。和 V4.2 一样按每次更新的监督 token 总数归一化。
- **其余与 V4.2 完全相同。** 20 轮、cosine 6e-4 → 6e-5、batch 512、BF16 + compile、5% 聊天、prefix crop 0.30。唯一的数值改动是 `lm_microbatch` 改为 256：损失是求和后再归一化，所以只影响显存，不改变目标。
- **代价。** 每次更新多四次 teacher 前向。本地未编译测量时，一次更新约为 CE 的 3.4 倍，所以预计每个 student 约 10 小时（V4.2 为 3.2 小时）。正式训练前，在集群上用 `--smoke 300` 测量显存与速度。

入口为 [`joint_v45.py`](../../src/vimeml/training/joint_v45.py)、[`train_v45.py`](../../scripts/training/train_v45.py)、[`prepare_v45_teachers.py`](../../scripts/training/prepare_v45_teachers.py)、[`cluster_v45.sh`](../../scripts/training/cluster_v45.sh) 与 [`launch_v45_all.sh`](../../scripts/training/launch_v45_all.sh)。W&B 的隔离方式与 V4.4 相同，group 为 `v45-distillation`。

## 评估（训练结束前固定）

- **选点**：每个 student 只评第 20 轮，不按 BPC 或 IME 结果挑选。
- **打分**：与 V4.4 完全相同：本地 FP32、TF32 关闭，完整词表后缀 logP 和，n-best 20 冻结池，单模型规则（左文过长时退回 AzooKey 顺序）。轨道为 Standard /728、Legacy /500、AJIMEE /200、历史 development /137，不读 blind 或测试集。统计口径按[指标 v2](benchmarks/metrics-v2.md)。
- **主比较**：Standard /728 精确 Top-1，两种合并方式分别判定。对每个孪生 seed s，d_s = student_s − twin_s。孪生的成绩是 523 / 501 / 518 / 511，均值 513.25。某种合并方式要四条同时满足才算"有效"：
  1. 四个 d_s 的均值 ≥ +10（约为一个 development seed SD）；
  2. 四个 d_s 中至少三个 > 0；
  3. 合并检验：每题取四个 student 的平均正确率减去四个孪生的平均正确率，做作者簇 sign-flip，双侧，两种合并方式之间 Holm 校正，族 α = 0.05；
  4. 保护条件：用同样的合并检验比较书写宽松 Top-1，Standard、Legacy、AJIMEE 上都没有 p < 0.05 的退化。
- **冒烟**：历史 development /137 上，student 均值比孪生均值低 3 题以上时标记，不参与判定。
- **描述性报告**（不参与判定）：
  - 每个轨道上 student、孪生与四 teacher 集成（Standard 535）的均值与范围；
  - 收回比例 =（student 均值 − 513.25）/（535 − 513.25）；
  - `geo` 与 `mix` 的逐 seed 配对差；
  - LM 与聊天验证 BPC 的逐轮曲线，与孪生对照；
  - 训练中的 KL 与 teacher CE；
  - 错误分类（语义 / 表记）；
  - 四个 student 自身的集成。

## 结论如何使用

- **有效时**：若两种合并方式都有效，取 d_s 均值较大的一种。部署候选预先固定为该方式下孪生为 V4.2 的 student（`geo-v42` 或 `mix-v42`），不在四个 seed 里按 IME 成绩挑。它和 V4.2 结构完全相同，只需重新导出 Core ML，不需要重新测设备延迟。部署前还要在一套新的 blind 上预注册确认它相对 V4.2 的优势；两套 v2 集合已经用过两次，不再用于确认。
- **无效时**：报告收回比例。若收回比例低于一半，说明 15M 容量或这一训练预算装不下集成的偏好，配方工作在这一规模收尾，重心转向候选池覆盖与表记。若在一半以上但没有达到判定标准，下一步再考虑只用软目标（权重 1.0）或更长训练，那需要另行注册。
- 无论结果如何，V4.2 第 20 轮继续部署，直到新的确认完成。
