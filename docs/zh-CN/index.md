# 文档索引

英文主要文档见[Documentation index](../index.md)（英文版与本目录路径一一对应）；以中文注册的计划以中文版为准。中文概览见[README.zh-CN.md](../../README.zh-CN.md)与[模型卡](model-card.md)。

**目录布局**：每条语言线的文档、脚本和配置各放在自己的目录下：中文是 `docs/zh/`、`scripts/zh/`、`configs/zh/`，日文是 `docs/ja/`、`scripts/ja/`、`configs/ja/`。共用工具（审核流水线、GPU 队列、Mac 环境配置）放在 `scripts/common/`；`src/` 下的 `vimeml` 包两条线共用。每条线内部的文档再分为 `training/`（计划与训练记录）、`benchmarks/`、`reports/<批次>/`、`mac/`（Mac 任务说明）和 `reference/`。已被取代的脚本、配置和文档在 2026-11-24 删除，指向它们的链接改为提交 `5263e659` 的固定链接。

**当前状态（2026-11-24）**

- **中文**：`vimeml-zh-v2-20m`（`colloq-c40`）已发布在 [Voltline/vimeml-zh-v2](https://huggingface.co/Voltline/vimeml-zh-v2)：test 精确 Top-1 公开拼音 51.4%、口语拼音 42.3%，线上 v1 路径为 31.0%。Vime 默认用它做重排和联想，整句生成放在开发开关后面；vimeml-tiny-zh-v1 继续发布，作为回退。
  - `vimeml-zh-v2-200m`（桌面）2026-11-23 test 确认通过：公开拼音 54.7%，Standard 重排 95.2%；同日发布到 v2 仓库的 `200m/`。
  - `vimeml-zh-v2-800m`（桌面）：第二版架构（arch2）在开发集上取代第一版（重打分 Top-1 65.9% 对 65.3%，Standard 持平，联想 Top-3 0.360 对 0.339）。2026-11-24 test 确认通过：63.3%，200M 为 54.7%（p 5e-5），全拼 88.7%，Standard 重排 95.3%（[报告](zh/reports/zh-v2-800m-confirmation-20261124/report.md)）；同日发布到 v2 仓库的 `800m/`。
- **日文**：当前部署 V4.2 第 20 轮 / INT8 block32，替换了 V2.1 extend5 step40000，V1 保留作基线。V4.4 测出 seed 噪声带；V4.5 蒸馏已撤下。日文线按[假名 → 文字生成路线](ja/training/training-ja-gen-plan.md)恢复：一个 20M 模型同时生成、重排和预测，Vime 把它的候选与 AzooKey 的合并。总体取舍见[V2系列总结](ja/reports/v2-summary.md)。

## 中文线

### 发布与数据

| 文档 | 内容 |
| --- | --- |
| [vimeml-zh-v2 发布范围](zh/release-zh-v2-plan.md) | `vimeml-zh-v2-20m`（手机，2026-11-23 已发布）、`vimeml-zh-v2-200m`、`vimeml-zh-v2-800m`（桌面），都用 12K、都能拼音生成；200M 六臂与 20M 联合预训练只作研发记录；发布门槛；数据合规只针对 20M |
| [vimeml-tiny-zh-v1语料](zh/data-zh-v1.md) | 已冻结为 export-r1（2026-11-10）：FineWeb2 cmn_Hani 21 分片与 LCCC，繁转简、规则过滤、垃圾站剔除、95.5 万条人工审核队列 |
| [中文 tokenizer 词表对比](zh/reports/zh-v1-20261106/tokenizer-vocab.md) | 只训 tokenizer：8K–32K 与覆盖率 0.9995/0.9999；20M 只实验 8K 和 12K，200M 再开 16K–32K |

### 基准与确认

| 文档 | 内容 |
| --- | --- |
| [Standard 中文 IME 基准](zh/benchmarks/standard-ime-zh.md) | CrossWOZ/RiSAWOZ/KdConv 2,901 条，全拼与简拼两轨，librime + rime-ice 候选池；AI 专家审核加人工抽查，blind 未开启 |
| [公开拼音生成测试集 ime-pinyin-zh-v1](zh/benchmarks/pinyin-ime-zh.md) | 2026-11-14 冻结：CrossWOZ、RiSAWOZ、KdConv、CPED；578/576 句，全拼、混合简拼、全首字母三类按键；已与 export-r1 查重；test 暂不读取 |
| [中文联想基准 ime-predict-zh-v1](zh/benchmarks/predict-ime-zh.md) | 2026-11-11 冻结：CrossWOZ、RiSAWOZ、KdConv、CPED 各 1,200 段对话；每个 split 下一词 7,200 题、省键 6,000 条左右发言、前缀补全 2,400 题；blind 关闭 |
| [公开中文键盘错字集](zh/benchmarks/typo-ime-zh.md) | `ime-typo-zh-v1`：578 个公开开发集句子，干净、一处错字、两处错字三种全拼输入；候选池由 v1 冻结生成；LM 与 task mode 显示规则、@2 与干净输入误显示率、校准 margin |
| [vimeml-tiny-zh-v1 blind 确认计划](zh/benchmarks/confirmation-zh-v1-plan.md) | 预注册：选定 `12k-fresh-s1`，在 blind 全拼上对比 Rime，只运行一次；2026-11-11 已执行，通过 |
| [vimeml-tiny-zh-v1 对照基线计划](zh/benchmarks/baselines-zh-v1-plan.md) | 预注册：雾凇、雾凇 + 万象语法模型、朙月拼音、Qwen2.5-0.5B、Qwen3-0.6B-Base，development 与 blind（参考测试） |
| [vimeml-tiny-zh-v1 blind 确认与对照基线](zh/reports/zh-v1-20261111/blind-and-baselines.md) | blind 全拼 Top-1 93.8%（Rime 72.3%），确认通过；胜过 Qwen2.5-0.5B（91.8%）、Qwen3-0.6B（89.2%）、雾凇 + 万象（82.2%）、朙月（68.9%） |
| [vimeml-zh-v2 test 确认计划](zh/benchmarks/confirmation-zh-v2-plan.md) | 预注册：v2-20m（`kd-cont`）在公开拼音 test 上对比线上逻辑（Rime 16 句 + v1 重排），带全拼和 Standard 守卫；v2-200m 对比 v2-20m；各运行一次。v2-20m 2026-11-20 通过（[报告](zh/reports/zh-v2-confirmation-20261120/report.md)）；之后 `colloq-c40` 替换 `kd-cont` 成为 v2-20m；v2-200m 2026-11-23 通过：49.1 → 54.7（+343/−150，p 5e-5），Standard 对 v1 净 +21（p 0.002）（[报告](zh/reports/zh-v2-200m-confirmation-20261123/report.md)）；v2-800m（补登）2026-11-24 通过：对 v2-200m 54.7 → 63.3（+429/−134，p 5e-5），Standard 对 v1 净 +22（p 0.002）（[报告](zh/reports/zh-v2-800m-confirmation-20261124/report.md)） |
| [口语候选 test 确认计划](zh/benchmarks/confirmation-zh-colloq-plan.md) / [结果](zh/reports/zh-colloq-confirmation-20261123/report.md) | 预注册：colloq-c40 在口语拼音 test 上对比 `kd-cont`，带公开拼音（第二次使用）、全拼和 Standard 守卫。2026-11-23 通过：口语 39.0 → 42.3（+217/−69，p 5e-5），公开 49.1 → 51.4；c40 成为 v2-20m |

### 20M 路线：v1 到 v2-20m

| 文档 | 内容 |
| --- | --- |
| [vimeml-tiny-zh-v1 实验计划](zh/training/training-zh-v1-plan.md) | 预注册：八臂，同为 224 亿字，新数据/5 轮/20 轮 × 8K/12K，12K 新数据三个 seed 测噪声 |
| [vimeml-tiny-zh-v1 八臂训练记录](zh/reports/zh-v1-20261110/training.md) | 2026-11-10 八臂全部完成，无抢占无失败；第 20 测点 BPC 与记忆抽样 |
| [vimeml-tiny-zh-v1 八臂 IME 打分与判定](zh/reports/zh-v1-20261110/ime-benchmark.md) | 全拼 development 八臂 93.3%–94.4%，Rime 68.8%；五个对比都不有效，按计划默认新数据和 12K |
| [zh 三任务第二轮训练记录](zh/reports/zh-pinyin-r2-20261114/training.md) / [评测](zh/reports/zh-pinyin-r2-20261114/evaluation.md) | 2026-11-14 两组：从第一轮续训 60,000 步、从 12K 基座重训 90,000 步，v2 拼音数据。私有基准生成 Top-1 29.9%，超过 Rime 16 句 + v1 重排（26.0%，p 5e-5）；Standard 重排 −24、联想 Top-3 对 v1 −2.8；KV 生成 CPU p50 92 ms。公开集复核：全部 +8.0，但全拼 −4.0（p 0.02），应按输入类型分流 |
| [zh 三任务第三组：从零联合预训练](zh/reports/zh-pinyin-pt-20261115/record.md) | 2026-11-15 20M 从零把拼音样本混进基座预训练：重排只差 −4（不显著），联想比 r2-cont 高 1.1，但生成私有基准 −2.2、公开集 37.8% 对 39.6%；20M 不适合联合预训练，保留续训路线 |
| [拼音生成的推理期重打分](zh/reports/zh-pinyin-rescore-20261115/report.md) | 生成前 10 按 genScore + 1.6 × v1 LM 分数重排：r2-cont 私有基准 29.9% → 36.4%，公开集 39.5% → 51.3%；全拼显著高于线上逻辑（公开 +4.2），不再需要按输入类型分流 |
| [Vime 拼音搜索的束宽扫描](zh/reports/zh-pinyin-beam-20261115/report.md) | v1 LM 模式带上下文、按键位分桶：束宽 16/12/8/4 公开集 Top-1 52.5/51.6/49.1/44.1，私有基准 33.6/32.9/32.0/29.2；每档都显著下降，损失在简拼；默认保持 16；该模式下 v1 重打分无收益 |
| [v2-20m 蒸馏计划](zh/training/training-zh-kd-plan.md) | 12K 联合拼音 200M 当老师，在线蒸馏 20M（α 0.5，T 1）；kd-base 与 kd-cont 两臂；判定看自身重打分 Top-1 是否显著高于 v1 |
| [蒸馏两臂的开发集评测](zh/reports/zh-kd-eval-20261117/report.md) | 2026-11-17 公开集自身重打分 Top-1：kd-cont 54.8（比 v1 高 2.1，p = 0.041，通过），kd-base 53.9（+1.2，不显著）；Standard 与 v1 打平；联想仍低于 v1（A Top-3 25.6 对 27.6）；私有集结论一致，全拼比线上 v1 高 4.4；kd-cont 定为 v2-20m 候选 |
| [第三轮蒸馏四臂的开发集评测](zh/reports/zh-kd-r3-eval-20261117/report.md) | 2026-11-17 公开集自身重打分 Top-1：r3-plain 54.8、r3-data 54.7、r3-ctrl 54.6，与 kd-cont（54.8）持平；r3-alpha（α 0.9）53.5，显著更低；联想 A Top-3 25.8–26.5，低于 v1 的 27.6；Standard 与 v1 打平；kd-cont 仍是 v2-20m 候选，第四轮换 800M 当老师 |
| [键盘错字计划](zh/training/training-zh-typo-plan.md) | 新增 `typo` 拼音任务（公开文本上的 26 键编辑）、按实际按键的 task mode 打分、导出附带校准参数、公开 `ime-typo-zh-v1` 评测集；20M 加/不加两臂；2026-11-19 结果：typo 臂未通过，继续用 LM 规则（[报告](zh/reports/zh-typo-eval-20261119/report.md)） |
| [v2-20m 口语后训练计划](zh/training/training-zh-colloq-plan.md) | 在 `kd-cont` 上通过 `[[data.extra]]` 混入经过筛查的口语文本（export-r1 聊天段落、豆瓣、MMChat、贴吧）；ctrl / c10 / c20 / c10-hard（口语只用交叉熵）四臂；判定看口语拼音集 `ime-pinyin-zh-colloq-v1` 是否显著高于 ctrl，公开拼音集和 Standard 守门 |

### 桌面模型：200M 与 800M

| 文档 | 内容 |
| --- | --- |
| [中文联想基准与桌面端模型路线](zh/benchmarks/predict-zh-plan.md) | 预注册：联想基准 `ime-predict-zh-v1`（下一词命中、省键率、前缀补全、置信度），先用现成模型摸底规模收益，再按门槛训 200M、800M 桌面端模型 |
| [联想基准第 0 步规模摸底](zh/reports/zh-predict-step0-20261115/survey.md) | Qwen2.5 0.5B/1.5B/3B 四条轨道随规模单调上升，3B 对 0.5B 轨道 A Top-3 +7.18；v1、zh-200m 都显著低于 0.5B；800M 不推迟 |
| [zh-200m 六臂训练计划](zh/training/training-zh-200m-plan.md) | 预注册：d1024，16K/L15、24K/L14、32K/L14 × 上下文 512，另加 24K 上下文 128 对照；每更新 512 窗口（偏离记录第 2 条），同 v1 数据与配方，三个 seed |
| [zh-200m 六臂训练记录](zh/reports/zh-200m-20261113/training.md) | 2026-11-13 六臂全部完成；M20 网页 BPC 3.8251–3.9153（v1 为 4.3533）；含窗口索引重建、上下文上限、批次策略三处问题的如实记录 |
| [zh-200m 六臂联想与重排判定](zh/reports/zh-200m-20261114/judgement.md) | 轨道 A Top-3 0.2999–0.3164（seed 极差 0.0165）；三个对比都不有效，按计划默认 24K、上下文 512；重排 Top-1 1,370–1,383，守门通过；Top-1 逐字出现在训练数据的比例不高于 20M |
| [12K 联合拼音 200M 的开发集评测](zh/reports/zh-200m-pinyin-eval-20261116/report.md) | 2026-11-16 公开集拼音生成 Top-1 自身重打分 58.8%（v1 52.7，+6.2；用 v1 先验 57.3）；Standard 全拼 Top-1 95.2 对 93.6（p = 0.0002）；联想四轨全面领先；200M 定联合预训练、自身打分；r2-cont 自身打分只比 v1 高 0.8（不显著），v2-20m 等蒸馏 |
| [桌面延迟：Windows 上的 PyTorch](zh/reports/zh-desktop-latency-20261118/report.md) | 2026-11-18/19：完整流程（搜索 + 重打分）CUDA graph + FP16 p50/p95：200M 83/205 ms、kd-cont 34/68、v1 32/63；200M 靠 FP16 减半，20M 主要靠 CUDA graph |
| [zh-800m 训练计划](zh/training/training-zh-800m-plan.md) | 预注册：d1536/L27、7.84 亿参数、12K 分词器、上下文 512；export-r1 全部 + export-r2 一遍，LCCC 2%，每更新 2048 窗口 |
| [zh-800m arch1 评测](zh/reports/zh-800m-eval-20261120/report.md) | 2026-11-20，只用开发集：联想轨道 A Top-3 0.3394（200M 0.3164），拼音重打分 Top-1 65.3%（200M 57.3%，v1 LM 模式 52.6%），Standard 对 v1 净 +26；没有长片段逐字背诵 |
| [zh-800m arch2 评测与判定](zh/reports/zh-800m-arch2-eval-20261123/report.md) | 2026-11-23，只用开发集：拼音重打分 Top-1 65.9%，arch1 65.3%（p 0.49）；Standard 94.99% 对 95.40%（净 −6，p 0.29）；arch2 取代 arch1；联想轨道 A Top-3 0.3604 对 0.3394（p 5e-5）；20 个测点 BPC 全部更低 |
| [vimeml-zh-v2-800m test 确认](zh/reports/zh-v2-800m-confirmation-20261124/report.md) | 2026-11-24 通过：自重打分 Top-1 63.3%，v2-200m 为 54.7%（+429/−134，p 5e-5），全拼 88.7%，全首字母 38.0%；Standard 重排 95.3%，对 v1 净 +22 |
| [zh-800m 第二版架构（arch2）](zh/training/training-zh-800m-arch2-plan.md) | 正式训练 k = 4，学习率同 arch1（3e-4 → 3e-5，前置检查对照轮定）：d1792/L23、Q 28 × 64 + KV 7 × 64（GQA）、RoPE、QK-norm，7.77 亿参数；多个段落窗口打包进 2048 长的行，段内因果 mask，数据和顺序与 arch1 相同，批量加大到 4 倍（200M 前置检查定，8 倍未试），不用 FP8；先定位瓶颈（通信/主机/计算）再动手；目标一遍 1.5 天以内，质量不低于 arch1 |

### Core ML 与部署

| 文档 | 内容 |
| --- | --- |
| [vimeml-tiny-zh-v1 Mac 结果核对与 iOS 发布](zh/reports/zh-v1-20261111/ios-release.md) / [Core ML 报告](zh/reports/mac-20261111/zh-v1-coreml.md) | 交接包核对通过；INT8 b32 对 FP32 全拼 +2/−1、简拼 +0/−2；Mac 上约为 V4.2 的 1.3 倍；iPhone 未测；发布到 Hugging Face |
| [zh-v1 批量 KV 解码 Core ML 包](zh/reports/mac-20261115/zh-v1-kvbatch.md) | 2026-11-15：一次调用推进整个束桶；INT8 flex（B 1..16）与无状态 INT8 包相差 ≤4.8e-5；M3 Mac p50 B=16 3.35–3.76 ms、B=8 2.69–2.92 ms；束搜索结果与无状态 INT8 逐条一致（125/125）；已加入发布副本；iPhone 未测 |
| [zh-v1 批量 KV 解码 v2 Core ML 包](zh/reports/mac-20261115/zh-v1-kvbatch-v2.md) | 2026-11-15：只读缓存、多 token 预填充、打分输出头；选定 `unified-scored`：提示与解码一个包（不再需要 chunk 包），Core ML 占用约 40 MiB 与 v1 相同，T=1 图 ≥200 MiB；M3 Mac p50 B=16 ≤4.4 ms，搜索 p50 79 ms；搜索结果与 v1 逐条一致（125/125）；已加入 Hugging Face 发布副本；iPhone 未测 |
| [zh-v1 整句打分 Core ML 包（序列输出头）](zh/reports/mac-20261116/zh-v1-kvsequence.md) | 2026-11-16：每个新位置输出给定 token 的对数概率，最多 16 条已知续写一次调用打完；对齐最大误差 3.34e-5；M3 Mac B=16 T=16 单次 17.5–18.4 ms，16 次解码调用 54.5–62.2 ms（快 3.1–3.4 倍）；内存随 T 增长（T ≤ 8 与 `unified-scored` 相同，T=16 +9 MiB，T=64 +63 MiB），每次调用 T 上限建议 8 或 16；fixture 排序与 FP32 一致 8/8；作为附加资源，搜索仍用 `unified-scored`；已加入 Hugging Face 发布；iPhone 未测 |
| [zh-v1 批量 KV v3 Core ML 包（共享前缀、单包）](zh/reports/mac-20261118/zh-v1-kvbatch-v3.md) | 2026-11-18：一个 multifunction INT8 包（`logits`、`scored`、`score`，23.1 MB，权重只存一份），提示只作为共享前缀传一次，打分只输出目标 log-prob；INT8 对齐最大误差 4.15e-5；搜索与 v2 一致 125/125；M3 Mac B=16 解码 2.9–3.3 ms（v2 3.0–3.9），T=16 打分 16.1–17.1 ms（序列包 17.2–17.8）；内存峰值 T 16/32/64 为 18.5/31.4/56.8 MiB，无状态包+v2+序列包为 40.5/70.2/150.0，常驻约 4.5 MiB（原 16–57）；词表分 4 块不采用（峰值更高、包大 5.3 MB）；设备确认后可去掉无状态包；已加入 Hugging Face 发布；iPhone 未测 |
| [zh-v1 神经引擎探针（第 0 步）](zh/reports/mac-20261118/zh-v1-ane-probe.md) | 2026-11-18：不通过。M3 Mac，在 v3 输入上用 FP16 固定形状的图。ANE 打分比 CPU 上的 v3 INT8 快 6.9–7.4 倍（B16×T16 为 2.18 ms，v3 为 16.11 ms），解码快 1.4–2.0 倍。卡在内存：加载模型时进程峰值约 150 MiB（v3 为 15.8 MiB）。第一名 132/133，唯一的不一致在 FP32 下就是近似并列（差 0.033 nats）。候选对颠倒 41 对，v3 为 92 对。分块 INT8 不能在 ANE 上运行。首次 ANE 编译 14–44 s。CPU 上跑 FP16 常驻 124 MiB |
| [zh 批量 KV v4 Core ML 包（`prefix` 函数、FP16 KV）](zh/reports/mac-20261120/zh-kvbatch-v4.md) | 2026-11-20：v3 加一个不带输出头的 `prefix` 函数；选定 `fp32kv`；iPhone 16 Pro Max 上 `score>prefix` 常驻 +7.5 MB，`score>scored` +34.8 MB（约 27 MB 来自第二个带输出头的函数，加载 `scored` 后会回来，所以 `scored` 改为按需加载）；48 token 前缀 4.60 → 3.60 ms，单 token 1.93 → 1.25 ms；搜索与 v3 一致 125/125；FP16 KV 不采用（峰值不降、单步更慢），FP16 计算对齐未过；已加入 Hugging Face 发布 |
| [vimeml-zh-v2-20m 的 Core ML 构建与 Vime 接入](zh/reports/mac-20261121/zh-v2-20m.md) | 2026-11-21，`kd-cont`：无状态与 KV v4 INT8 包、task mode 搜索门槛、Vime 接入；iPhone 13 Pro / 16 Pro Max 上 Rime 重排 p50 4.1 / 3.0 ms，一次性整句生成 91 / 59 ms，峰值内存 75 / 63 MB |

Mac 任务说明（交给 macOS 会话的任务书，上面每份报告对应其中一份）：

| 任务书 | 内容 |
| --- | --- |
| [zh-v1 Core ML](zh/mac/mac-zh-v1-coreml.md) | 用 V4.2 配方转换并量化 vimeml-tiny-zh-v1；Mac 与 iPhone 测量 |
| [zh-v1 批量 KV](zh/mac/mac-zh-v1-kvbatch.md)、[v2](zh/mac/mac-zh-v1-kvbatch-v2.md)、[序列输出头](zh/mac/mac-zh-v1-kvsequence.md)、[v3](zh/mac/mac-zh-v1-kvbatch-v3.md) | 为分桶束搜索和整句打分逐步改进的 KV cache 包 |
| [zh 批量 KV v4](zh/mac/mac-zh-kvbatch-v4.md) | 按 iPhone 内存剖析加 `prefix` 函数、试 FP16 KV |
| [zh-v1 神经引擎探针](zh/mac/mac-zh-v1-ane-probe.md) | 第 0 步：ANE 是否值得作为 20M 模型的部署路径 |
| [vimeml-zh-v2-20m](zh/mac/mac-zh-v2-20m.md) | `kd-cont` 的 Core ML 构建与 Vime task mode 接入 |
| [vimeml-zh-v2-20m colloq-c40](zh/mac/mac-zh-v2-20m-colloq.md) | 换 c40 权重重建并在 Vime 替换；含 2026-11-23 Mac 结果与搜索门槛偏离（122/125） |

## 日文线

| 文档 | 内容 |
| --- | --- |
| [数据](ja/data.md) | 冻结语料、来源、清洗和split |
| [评测](ja/evaluation.md) | 评分语义、同池结果、统计与标签限制 |
| [V1训练](ja/training/training.md) | 冻结基线复现 |
| [V2实验设计](ja/training/plan_v2.md) / [V2训练](ja/training/training-v2.md) | 模型、tokenizer、裁剪、配置和运行方式 |
| [V3数据实验](ja/training/plan_v3.md) / [A评测](ja/reports/v3-20261008/a-evaluation.md) / [B联合重训](ja/reports/v3-20261008/b-plan.md) / [B评测](ja/reports/v3-20261008/b-evaluation.md) | A与B训练、评测、归档完成；B未形成稳定替换收益，保留V2.1部署 |
| [V4 基础训练](ja/training/training-v4.md) | 固定 8K 架构、已完成 A/B 调度、结果与恢复 |
| [V4 后训练](ja/training/posttraining-v4.md) / [转换数据](ja/training/data-v4-c1.md) | 已完成 C1/C1r2、冻结数据、条件评分与后训练设置 |
| [Standard IME基准](ja/benchmarks/standard-ime.md) / [首次结果](ja/benchmarks/standard-ime-results-20261008.md) | WRIME/JMultiWOZ独立来源、旧网页回归、审核与版本协议；V2.1/V3 A固定比较 |
| [IME 指标 v2](ja/benchmarks/metrics-v2.md) | 书写宽松与池条件 Top-1、作者簇检验、判定规则与轨道角色 |
| [Standard v2 blind](ja/benchmarks/standard-ime-v2-blind.md) / [同音词挑战](ja/benchmarks/homophone-ime.md) / [分段输入原型](ja/benchmarks/incremental-ime.md) | 新确认集合、书面语同音词集合与分段协议原型；前两个已按确认计划使用 |
| [确认评分计划 v2](ja/benchmarks/confirmation-v2-plan.md) | Standard v2 blind 与同音词集合首次评分的预注册假设、模型、判定规则与功效 |
| [确认评分 v2 结果](ja/benchmarks/confirmation-v2-results-20261021.md) | H1（V2.1 → V4.2）+22、作者簇 p = 0.0508，未证实；V4.3 → V5 在同音词上 +18 |
| [seed 集成复核结果](ja/benchmarks/ensemble-replication-v2-results-20261105.md) | v2 blind 第二次使用：四 seed 集成 619 对 V4.2 598（+21，作者簇 p = 0.0012），复现；同音词 +7，不显著 |
| [候选导出 v2](ja/benchmarks/candidates-v2-handoff.md) | n-best 20/50、确定性与左文依赖对照的 Mac 交接 |

## 通用

| 文档 | 内容 |
| --- | --- |
| [Core ML](ja/coreml.md) | 导出、量化、客户端接口与设备范围 |
| [本地产物](artifacts.md) / [跨平台管理](artifact-exchange.md) | 路径、归档、Git与ZIP协作 |
| [iOS 参考与来源](examples/ios/README.md) | `examples/ios/` 的 Core ML 探针、SentencePiece 桥接与历史接入代码说明 |
| [Standard IME Mac 交接模板](templates/benchmarks/standard-ime-mac.md) | 日文 Standard 候选导出在 Mac 上的交接模板 |

## V4 实验记录

| 阶段 | 报告 |
| --- | --- |
| 基础 LM | [A 续训实验](ja/reports/v4-20261015/evaluation.md)、[B 完整十二轮对照](ja/reports/v4-20261015/b12cosine-evaluation.md) |
| 转换后训练 | [C1r2 数据修订、完整结果与 Reading 消融](ja/reports/v4-20261016/c1r2.md) |
| V4.1 混合预训练 | [训练设置](ja/training/training-v41.md)、[十六轮完整评估](ja/reports/v41-20261016/evaluation.md) |
| V4.2 LM／Chat 预训练 | [训练设置](ja/training/training-v42.md)、[二十轮完整评估](ja/reports/v42-20261020/evaluation.md)、[固定 reference 与退化分析](ja/reports/v42-20261020/reference-regressions.md) |
| V5 规模扩展预训练 | [设计、V4.3 饱和证据、结果与结论](ja/training/training-v5.md)：BPC −0.10，IME Top-1 无提升，不部署 |
| WSD 调度实验 | [V4.2-50ep 与 V5-50ep 的设计、曲线与结论](ja/training/training-wsd.md)：BPC 3.0328 / 2.9317（各规模最佳），IME Top-1 无提升，不部署 |
| V4.4 消融（预注册） | [V4.2 配方上的六臂单因素实验](ja/training/training-v44-plan.md)：三个 seed 重复测 run 间噪声，聊天 0 / 10%，prefix crop 0.60；只评第 20 轮 |
| V4.4 消融结果 | [seed 噪声带与处理臂判定](ja/training/training-v44.md)：处理臂均无效（去掉聊天变差）；Standard seed 带 501–523，V4.2 在顶端，此前单 seed 差异都在带内；探索性四 seed 集成 535 |
| V4.5 集成蒸馏（预注册） | [四 seed 集成蒸馏进 15M student](ja/training/training-v45-plan.md)（2026-11-25 撤下，从未执行；由生成路线中的 200M 老师取代） |
| 生成路线（ja-gen） | [计划](ja/training/training-ja-gen-plan.md)：2026-11-25 注册第 0–1 阶段：新公开集 `ime-kana-ja-v1`（test 封存）、候选池覆盖率、Sudachi/UniDic 一致读音、12K 词表；之后 20M 后训练、200M 老师与蒸馏、口语后训练、确认、Vime 合并 |
| 生成路线第 0 阶段 | [覆盖率与基线](ja/reports/ja-gen-stage0-20261125/report.md)：`ime-kana-ja-v1` 冻结（600 + 600）；Standard 开发集池外 98 条中 N-best 100 补回 40 条，58 条只能靠生成；V4.2 在新开发集宽松 Top-1 85.2 |
| 生成路线 2.1 | [20M 读音后训练](ja/reports/ja-gen-stage2-20m-20261127/report.md)：读音配方保留，`ime-kana-ja-v1` 开发集 4-seed 宽松 Top-1 87.17（V4.2 85.2，LM 对照 85.79）；Standard 重排 504.5（LM 对照 515.25） |
| IME 错误分解 | [瓶颈不在 LM 容量](ja/reports/analysis-20261021/ime-errors.md)：池外 48%、书写形式 35–44%、语义 8% |
| 免训练重打分 | [PMI、长度与引擎分融合](ja/reports/analysis-20261021/rescoring.md)：均未通过独立检验，部署保持纯 LM 打分 |
| V4.2 Apple 部署 | [Core ML、INT8、分段 KV、Mac 与 iPhone 测量](ja/reports/mac-20261020/v42-coreml.md)；Mac 任务说明：[V4.2](ja/mac/mac-v42-coreml.md)、[V4.3](ja/mac/mac-v43-coreml.md) |
| 外部日语模型 | [TinySwallow](ja/reports/v4-20261016/external-1p5b.md)、[RakutenAI](ja/reports/v4-20261016/external-rakuten-1p5b.md)、[Sarashina](ja/reports/v4-20261016/external-sarashina-1p4b.md) |

V4-a 本地工作：[Tokenizer 初始化、训练与词表对照](ja/reports/v4-20261014/tokenizer.md)、[8K token 数据与窗口](ja/reports/v4-20261014/token-data.md)。
模型与训练入口见 [固定架构、本地验收和启动配置](ja/reports/v4-20261014/model.md)。

## V2 实验记录

| 阶段 | 报告 |
| --- | --- |
| 数据与模型设计 | [Tokenizer](ja/reports/v2-20261007/tokenizer.md)、[模型](ja/reports/v2-20261007/model.md)、[Token store](ja/reports/v2-20261007/token-data.md)、[Prefix crop](ja/reports/v2-20261007/prefix-crop.md) |
| V2.0 | [运行环境](ja/reports/v2-20261007/environment.md)、[完整评测](ja/reports/v2-20261007/evaluation.md) |
| V2.1 restart1 | [配置](ja/reports/v2-20261007/v21-plan.md)、[性能对照](ja/reports/v2-20261007/v21-performance.md)、[中断恢复](ja/reports/v2-20261007/v21-recovery.md)、[评测](ja/reports/v2-20261007/v21-evaluation.md) |
| V2.1 extend5 | [追加训练与模型选择](ja/reports/v2-20261007/v21-extend.md) |
| 扩大IME数据集 | [候选构建](ja/reports/v2-20261007/ime-3000-handoff.md)、[初评](ja/reports/v2-20261007/expanded-ime-evaluation.md)、[标签与错误分析](ja/reports/v2-20261007/label-error-audit.md) |
| Core ML与iPhone | [转换和量化](ja/reports/mac-20261008/v21-coreml.md)、[误差分析](ja/reports/mac-20261008/v21-quantization-review.md)、[宿主与真实扩展](ja/reports/mac-20261008/v21-iphone.md)、[KV cache](ja/reports/mac-20261009/v21-kv-cache.md) |

## V1参考与历史

[FP32结果](ja/reference/results.md)、[Core ML实测](ja/reference/coreml-v1.md)、[模拟器内存](ja/reports/mac-20261006/simulator-memory.md)、[真实扩展](ja/reports/mac-20261006/iphone-keyboard-memory.md)、[快照归档](ja/reports/mac-20261006/handoff.md)、[导入核对](ja/reports/mac-20261006/merge-verification.md)。

详细方法：[语料流水线](ja/reference/data-pipeline.md)、[AJIMEE导出](ja/reference/ajimee-benchmark.md)、[开发集生成](ja/reference/development-generation.md)、[组合排序](ja/reference/hybrid-ranking.md)、[联想](ja/reference/phrase-demo.md)、[中文联想测试页](zh/reference/phrase-demo-zh.md)。

`history/`保存V1实测前的[部署设计](ja/history/deployment.md)、[命令参考](ja/history/coreml-guide.md)和[阶段记录](ja/history/workflow.md)，仅描述当时状态。各报告注明日期、版本与测量范围；原始JSON、逐候选分数和trace保存在`outputs/`。
