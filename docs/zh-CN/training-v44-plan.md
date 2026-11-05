# V4.4 单因素消融（预注册）

2026-11-04 注册，在任何一臂开始训练之前提交。V4.4 不是一个新配方，而是 V4.2 配方上的六个并行运行：三个只换 seed，三个只改一个数据变量。目的有两个：第一次测出"同一配方换 seed"时 IME 重排的波动有多大；在这个噪声带上检验聊天比例与 prefix crop 是否值得继续改。结果见 [V4.4 消融结果](training-v44.md)（注册后补充的链接，正文保持注册时原文）。

## 为什么做

- V4.2 第 20 轮是部署版本，但它对 V4.3、V5、WSD 的领先都在噪声内（例如 WSD V4.2-50ep 在 Standard 上 515 vs 523）。所有这些比较都只有一个 seed，从来没有测过 run 间波动，所以"噪声内"本身没有尺度。
- [错误分解](reports/analysis-20261021/ime-errors.md)显示 V4.2 在 Standard 上的 205 个错误里，池外 98 个、书写形式 71–90 个、池内语义只有 17 个；继续降低 BPC 已七次没有带来重排收益。剩下预训练可能影响的主要是书写形式，它取决于训练文本的文体（聊天比例）和上下文长度分布（prefix crop）。
- WSD 已经做过且没有 IME 收益（[WSD 实验](training-wsd.md)），不再跑。

## 六臂

共同配方与 V4.2 完全相同：tiny_gpt_v4 d384/L7（15,070,848 参数）、8K 词表、20 轮、cosine 6e-4 → 6e-5（1,000 warmup）、batch 512、BF16 + compile、fused AdamW（wd 0.1、clip 1.0）、V4.2 数据池与排除掩码。每臂只改下表中的一个变量，并有自己的 20 轮混合映射（[`prepare_v44.py`](../../scripts/corpus/prepare_v44.py)）。

| 臂 | seed | 聊天比例 | prefix crop | 配置 | 混合清单 SHA-256 |
| --- | ---: | ---: | ---: | --- | --- |
| V4.2（已有，不重训） | 1219827974 | 0.05 | 0.30 | [`train-v42.toml`](../../configs/train-v42.toml) | `b1db1c1f…` |
| `s1` | 137218780 | 0.05 | 0.30 | [`train-v44-s1.toml`](../../configs/train-v44-s1.toml) | `d375bd20…` |
| `s2` | 299822247 | 0.05 | 0.30 | [`train-v44-s2.toml`](../../configs/train-v44-s2.toml) | `6a713a9b…` |
| `s3` | 2097573425 | 0.05 | 0.30 | [`train-v44-s3.toml`](../../configs/train-v44-s3.toml) | `38f8d943…` |
| `chat0` | 1219827974 | **0** | 0.30 | [`train-v44-chat0.toml`](../../configs/train-v44-chat0.toml) | `3bb2e433…` |
| `chat10` | 1219827974 | **0.10** | 0.30 | [`train-v44-chat10.toml`](../../configs/train-v44-chat10.toml) | `b08df1a9…` |
| `crop60` | 1219827974 | 0.05 | **0.60** | [`train-v44-crop60.toml`](../../configs/train-v44-crop60.toml) | `26dada4c…` |

- **seed 臂换的是完整 seed**：初始化、crop 决策、聊天抽样与数据顺序都变，测到的是真实 run 间噪声。三个处理臂因为改了 crop 或聊天配额，数据顺序同样会变，所以和 seed 臂可以直接比较。三个 seed 用 `secrets.randbelow(2**31)` 一次抽出，彼此相距远，`seed + epoch` 的聊天抽样流不会重叠。
- **聊天语料只有约 580 万 token**：5% 时每轮已重复 5.2 遍；`chat10` 每轮约 11 遍、20 轮共约 220 遍。15% 会到约 350 遍，测到的主要是过拟合，所以不用。
- **更新步数不完全相同**：聊天窗口在网页窗口之外追加，且比网页窗口短，所以按窗口计的每轮更新数变化比 token 比例大：第 1 轮窗口数 V4.2 为 27,697,799，`chat0` 25,208,203（约 −9%），`chat10` 30,463,978（约 +10%），`crop60` 27,579,336。不做补偿，报告中如实列出。
- **混合构建核对**：`control` 用 V4.2 的三个值重建映射，前 16 轮与冻结的 V4.1 映射逐字节一致（2026-11-04 本地核对）；集群上若存在 V4.2 混合目录，还要求 20 轮全部一致。六臂的清单在本地构建，SHA 写入配置；集群重建必须逐字节复现，否则不训练。为让 SHA 与操作系统无关，`request.json` 与 `manifest.json` 统一用 LF 换行。

训练入口为 [`train_v44.py`](../../scripts/training/train_v44.py)，集群启动为 [`cluster_v44.sh`](../../scripts/training/cluster_v44.sh)（单臂）与 [`launch_v44_all.sh`](../../scripts/training/launch_v44_all.sh)（六臂各占一张卡）。W&B 不使用机器上的登录：凭据只来自私有文件中的 `WANDB_API_KEY`，每臂的 W&B 配置、缓存与 netrc 指向自己的目录；没有凭据或在线初始化失败时改为 offline 记录，之后 `wandb sync`。六个 run 在同一 group `v44-ablation` 中。

## 评估（训练结束前固定）

- **选点**：每臂第 20 轮，不按 BPC 或任何 IME 结果挑选。上传 HF 的也只有第 20 轮（去掉优化器状态）与逐轮记录。
- **打分**：与 V4.2 评估完全相同——本地 FP32、TF32 关闭，完整词表后缀 logP 和，左文与候选联合分词，并列保持 AzooKey 顺序，n-best 20 冻结池。四个非 blind 轨道：Standard /728、Legacy /500、AJIMEE /200、历史 development /137。不读 blind 或测试集。统计口径按[指标 v2](benchmarks/metrics-v2.md)。
- **噪声带**：V4.2 第 20 轮与 `s1`–`s3` 共四个点，每个轨道报告精确 Top-1 与书写宽松 Top-1 的均值、极差与标准差（n − 1）；六对 seed 之间报告作者簇 sign-flip p 与逐题 McNemar（描述性：相同配方的两次运行有多常"显著"）。
- **处理臂判定**（`chat0`、`chat10`、`crop60`，主轨道 Standard /728 精确 Top-1，双侧）：三条同时满足才算"探索性有效"：
  1. Top-1 落在四个 seed 点的极差之外；
  2. 相对 V4.2 第 20 轮的作者簇 sign-flip 检验，三个处理臂之间 Holm 校正，族 α = 0.05；
  3. 按指标 v2 的保护条件，Standard、Legacy、AJIMEE 上书写宽松 Top-1 都没有 p < 0.05 的退化（作者簇检验）。
- **冒烟**：历史 development /137 上精确 Top-1 比 V4.2 少 3 条以上时标记，不参与比较。
- **描述性报告**：Legacy 与 AJIMEE 的精确 Top-1、池条件 Top-1、逐题 McNemar（与 v1 报告对照）、LM 与聊天验证 BPC（逐轮曲线），以及 `chat0` → V4.2 → `chat10` 的剂量-反应方向。这些不参与判定。

## 结论如何使用

- 两套 blind 已在[确认评分 v2](benchmarks/confirmation-v2-results-20261021.md)中用掉，本轮 development 结论只能是探索性的。某处理臂判定有效时，下一步是先制作新的 blind 集，在其上预注册确认，确认后才可能以该臂部署为 V4.4。
- 没有处理臂有效时，V4.2 第 20 轮继续部署，"改预训练配方"在这一规模与预算下收尾，工作重心转向候选池覆盖（Standard 错误的 48%）。
- 无论结果如何，seed 噪声带都会回写到此前单 seed 比较的解读上（V4.3、V5、WSD 相对 V4.2 的差异是否在带内）。
