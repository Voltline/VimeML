# VimeML

为 Vime 输入法训练的小型因果语言模型。

- **中文**：v2 模型直接从拼音按键生成整句，同时给 librime（雾凇）候选重排、预测下一个词，三件事由同一个模型完成。
- **日文**：利用左文给 AzooKey 的假名转汉字候选重排。

这些是输入法专用的自定义 PyTorch 模型，不是指令微调助手，也不接入 Transformers `AutoModel`。

[English README](README.md) · [中文文档索引](docs/zh-CN/index.md) · [中文模型卡](docs/zh-CN/model-card.md) · [许可与归属](NOTICE.md)

## 已发布模型

| 模型 | 语言 | 参数量 | 用途 | Hugging Face |
| --- | --- | ---: | --- | --- |
| **vimeml-zh-v2-20m** | 中文 | 20,036,736 | iPhone：拼音生成、Rime 重排、联想（Core ML INT8） | [Voltline/vimeml-zh-v2](https://huggingface.co/Voltline/vimeml-zh-v2) `20m/` |
| **vimeml-zh-v2-200m** | 中文 | 193,625,088 | 桌面（PyTorch、GPU）：同样三项任务 | [Voltline/vimeml-zh-v2](https://huggingface.co/Voltline/vimeml-zh-v2) `200m/` |
| vimeml-tiny-zh-v1 | 中文 | 20,036,736 | iPhone：Rime 重排与联想；v2 的回退 | [Voltline/vimeml-tiny-zh-v1](https://huggingface.co/Voltline/vimeml-tiny-zh-v1) |
| **V4.2** | 日文 | 15,070,848 | iPhone：AzooKey 重排（当前部署） | [Voltline/vimeml-tiny-ja-v4.2](https://huggingface.co/Voltline/vimeml-tiny-ja-v4.2) |
| V2.1 | 日文 | 12,537,920 | 上一部署版本 | [Voltline/vimeml-tiny-ja-v2.1](https://huggingface.co/Voltline/vimeml-tiny-ja-v2.1) |
| V1 | 日文 | 7,386,624 | 基线 | [Voltline/vimeml-tiny-ja-v1](https://huggingface.co/Voltline/vimeml-tiny-ja-v1) |

每个发布包含推理权重（不含优化器状态）、分词器、Core ML 包、冻结的源码副本和聚合评测结果。训练 checkpoint、语料和私有基准文本不公开。

## 中文

### v2：拼音生成整句

| 模型 | 平台 | 状态 |
| --- | --- | --- |
| `vimeml-zh-v2-20m` | iPhone | 2026-11-23 发布（`colloq-c40`） |
| `vimeml-zh-v2-200m` | 桌面 | 2026-11-23 发布（`200m/`；公开拼音 54.7%） |
| `vimeml-zh-v2-800m` | 桌面 | 2026-11-24 test 确认通过 |

v2 都沿用 v1 的 12,288 词分词器。提示由光标前的文本（最多 64 字）和按键组成，按键可以是全拼、首字母或混合简拼。束搜索按键位分桶，每个位置保留 16 条，一个字只有在读音或声母与后续按键匹配时才能加入。前 10 句再由同一个模型在语言模型模式下重打分。

**vimeml-zh-v2-20m** 从 v1 出发，依次经过拼音任务训练、200M 蒸馏和口语后训练（[训练过程与方法](https://huggingface.co/Voltline/vimeml-zh-v2)）。下表所有数字都来自按预注册计划只打开一次的 test，指标是显示的第一候选精确命中（Top-1）。

| 测试集 | 线上 v1 路径（Rime + v1 重排） | `kd-cont` | **v2-20m** |
| --- | ---: | ---: | ---: |
| `ime-pinyin-zh-v1-r1` test，3,436 个输入 | 31.0% | 49.1% | **51.4%** |
| ……全拼 / 混合简拼 / 全首字母 | 81.6 / 23.7 / 9.2 | 82.6 / 47.4 / 22.2 | **84.4 / 49.8 / 24.7** |
| `ime-pinyin-zh-colloq-v1-r1` test，4,560 个口语输入 | — | 39.0% | **42.3%** |

Standard blind 全拼 Rime 候选重排：v2-20m 94.3%，v1 93.8%（净 +7，p 0.31）。报告：[v2-20m 确认](docs/zh-CN/reports/zh-v2-confirmation-20261120/report.md)、[口语确认](docs/zh-CN/reports/zh-colloq-confirmation-20261123/report.md)、[发布范围](docs/zh-CN/release-zh-v2-plan.md)。

手机上 v2-20m 是一个批量 KV 的 Core ML INT8 包（23.3 MB）。Vime 默认用它做 Rime 重排和联想；整句生成放在开发开关后面，等逐键生成在旧机型上够快再打开。用同形状、同格式的 `kd-cont` 实测：Rime 重排 p50 在 iPhone 13 Pro 上 4.1 ms、iPhone 16 Pro Max 上 3.0 ms；一次性整句生成 91 ms 和 59 ms（[iPhone 报告](docs/reports/mac-20261121/zh-v2-20m.md)（英文）、[c40 Mac 结果](docs/reference/mac-zh-v2-20m-colloq.md)（英文））。

桌面模型用 PyTorch 运行。笔记本 GPU 上用 CUDA graph + FP16，200M 每个输入（含搜索和重打分）p50/p95 为 83/205 ms（[桌面延迟](docs/reports/zh-desktop-latency-20261118/report.md)（英文））。它在公开拼音 test 上为 54.7%（全拼 86.5%），Standard 重排 95.2%，v1 为 93.8%（[200M 确认](docs/zh-CN/reports/zh-v2-200m-confirmation-20261123/report.md)）。800M（第二版架构，开发集上取代第一版：[arch2 评测](docs/zh-CN/reports/zh-800m-arch2-eval-20261123/report.md)）在同一 test 上为 63.3%（全拼 88.7%），比 200M 高 8.6 个点，Standard 重排 95.3%（[800M 确认](docs/zh-CN/reports/zh-v2-800m-confirmation-20261124/report.md)）。

### v1：候选重排

`vimeml-tiny-zh-v1` 利用左文重排 librime + 雾凇的拼音候选。模型随机初始化，在 224 亿字清洗过的 FineWeb2 `cmn_Hani` 网页文本和 LCCC 对话上训练；代码和产物路径用内部代号 `zh13`。Hugging Face 发布包含无状态 Core ML 包和批量 KV 包（直到 v4：共享前缀、单个 multifunction 包）。

`ime-standard-zh-v1-r1` blind 精确 Top-1（CrossWOZ、RiSAWOZ、KdConv 对话；标签经 AI 审核）。模型在冻结的雾凇 20 候选池上重排：

| 系统 | 全拼 /1,445 | 简拼 /964 |
| --- | ---: | ---: |
| **vimeml-tiny-zh-v1** | **93.8%** | **35.0%** |
| Qwen2.5-0.5B，同一候选池和打分 | 91.8% | 31.0% |
| Qwen3-0.6B-Base，同一候选池和打分 | 89.2% | 30.6% |
| 雾凇 + 万象语法模型 | 82.2% | 15.7% |
| 雾凇（候选池来源） | 72.2% | 14.6% |

预注册的对 Rime 确认通过（+329/−19，p = 3.6e-74）。报告：[实验计划](docs/zh-CN/training-zh-v1-plan.md)、[训练记录](docs/zh-CN/reports/zh-v1-20261110/training.md)、[blind 与对照基线](docs/zh-CN/reports/zh-v1-20261111/blind-and-baselines.md)、[iOS 发布](docs/zh-CN/reports/zh-v1-20261111/ios-release.md)。

## 日文

当前部署 V4.2 第 20 轮：INT8 block32 权重、FP32 计算、Core ML CPU_ONLY、iOS 18，包体 17.16 MB，替换了 V2.1 extend5 step40000。

固定 AzooKey 候选池上的精确 Top-1（Standard 标签经 AI 审核）：

| 模型 | Standard dev /728 | 固定 reference /725 | Legacy /500 | AJIMEE /200 |
| --- | ---: | ---: | ---: | ---: |
| V2.1 FP32 | 505 | 535 | 374 | 144 |
| V4.2 FP32 | **523** | 541 | 370 | 139 |
| V4.2 INT8（部署） | 522 | — | 371 | 139 |
| V5 第 30 轮 | 514 | — | 365 | **145** |

V4.2 与 V2.1 差异不显著：预注册的 Standard v2 blind 上 598/729 对 576（作者簇 p = 0.0508，差一点达到阈值）。V3、V4-B、V4.1、V4.3、V5 和 WSD 实验均已完成，没有替换 V4.2；V5 验证 BPC 降 0.10，IME Top-1 不变。V4.4 测出 seed 噪声带（Standard 501–523），四 seed 集成在 blind 上达到 619。

日文线为中文工作暂停，接下来恢复。详见[评测](docs/zh-CN/evaluation.md)、[V4.2 训练](docs/zh-CN/training-v42.md)、[V4.2 Core ML](docs/reports/mac-20261020/v42-coreml.md)（英文）、[V4.4 消融](docs/zh-CN/training-v44.md)、[确认评分结果](docs/zh-CN/benchmarks/confirmation-v2-results-20261021.md)。

## 安装

Python ≥ 3.11。

```bash
python -m venv .venv
source .venv/bin/activate          # Windows：.venv\Scripts\Activate.ps1
python -m pip install -e .
```

`requirements.txt` 固定各平台版本；`requirements/` 下另有 Linux CUDA 训练（`cuda.txt`）、评测准备（`evaluation.txt`）和外部模型对照（`external-eval.txt`）的依赖。

运行已发布模型，直接用 Hugging Face 上的发布包即可：每个包自带源码和 `infer.py`（v2：`python infer.py --keys woxiangchifan --context 今天中午`）。源码仓库从本地 checkpoint 及其 manifest 运行。Core ML 转换和运行时核对需要 macOS。iOS 客户端在独立的 Vime 仓库维护。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/vimeml/` | 语料处理、分词、训练、评测和部署 |
| `scripts/` | 按子系统分类的命令入口 |
| `configs/` | 带版本的实验与数据配置 |
| `requirements/` | 附加依赖 |
| `docs/` | 英文文档与按日期的实验记录 |
| `docs/zh-CN/` | 中文文档；以中文注册的计划以中文版为准 |
| `examples/ios/` | Core ML 探针、SentencePiece 桥接、历史接入代码 |
| `templates/` | 产物交接模板 |
| `tests/` | 针对性测试 |
| `datasets/`、`artifacts/`、`outputs/`、`runs/`、`handoff/` | 本地、Git 忽略的数据、模型、结果和交接 |

## 许可

代码和发布权重采用 GPL-3.0-or-later，并依 GPLv3 第 7 条附加允许经 App Store 等应用商店分发（[LICENSE-EXCEPTION](LICENSE-EXCEPTION)）；2026-10-22 之前的发布采用 GPL-2.0-only。语料和第三方组件保留各自条款；打包模型的应用需要展示的归属见 [NOTICE.md](NOTICE.md)。[贡献约定](CONTRIBUTING.md)。
