# VimeML

用于日语假名转汉字候选重排的小型因果语言模型，兼有实验性的短文本续写和下一词联想功能。AzooKey负责词典候选检索，语言模型利用左文重排固定候选池。模型未进行指令微调，不定位为通用聊天助手。

[English README](README.md)为主要入口；[英文模型卡](MODEL_CARD.md)描述模型与评测边界，[中文模型卡](docs/zh-CN/model-card.md)提供对应概览。

## 模型与状态

| 模型 | 参数量 | 分词器 | Context | 状态 |
| --- | ---: | --- | ---: | --- |
| V1 | 7,386,624 | 16K SentencePiece Unigram，V1词表 | 128 | 公开基线与回退：[Voltline/vimeml-tiny-ja-v1](https://huggingface.co/Voltline/vimeml-tiny-ja-v1) |
| V2.1 | 12,537,920 | 16K SentencePiece Unigram，V2词表 | 128 | 上一部署版本，公开：[Voltline/vimeml-tiny-ja-v2.1](https://huggingface.co/Voltline/vimeml-tiny-ja-v2.1) |
| V4.2 | 15,070,848 | 8K SentencePiece Unigram，V4词表 | 128 | **当前部署**，公开：[Voltline/vimeml-tiny-ja-v4.2](https://huggingface.co/Voltline/vimeml-tiny-ja-v4.2) |

当前部署版本为V4.2第20轮：INT8 block32权重压缩、FP32计算、Core ML CPU_ONLY、最低iOS18，包体17.16 MB。它替换了V2.1 extend5 step40000（同一配方，包体14.33 MB）。V3 A/B、V4-B、V4.1均已完成，未形成稳定替换收益。

V4.2（2026-10-20）是V4系列最强的模型：8K词表、随机初始化、网页文本加5%对话训练20轮。它在Standard开发集领先，但在固定reference上只比V2.1多6条（p=0.56），AJIMEE落后，差异主要是假名／汉字书写选择。INT8 Core ML包17.16 MB；iPhone 16 Pro Max测试宿主中峰值内存低5.6 MB、重排长尾更快，单次调用慢0.16–0.35 ms。在预注册的Standard v2 blind上为598/729，V2.1为576（+22，作者簇p=0.0508，差一点未达到注册阈值）；书面语同音词挑战上为438/619，V2.1为450（不显著）（[确认评分结果](docs/zh-CN/benchmarks/confirmation-v2-results-20261021.md)）。键盘扩展内存与打字延迟尚未测量。

V4.3（增加网页数据）与V5（d512/L8，2,840万参数，数据与预算同V4.3）为已完成的否定性实验：V5验证BPC降低0.10，但IME Top-1在噪声范围内不变，二者均不转换、不部署（[V5训练与结论](docs/zh-CN/training-v5.md)）。

公开发布包含推理权重、匹配分词器、Core ML资源和聚合评测报告。训练checkpoint、语料、凭据及私有基准文本独立保存。自定义模型格式不直接兼容Transformers AutoModel；Hugging Face推理包的运行说明以各自模型卡为准。

## 固定评测结果

IME各列为固定AzooKey候选池上的精确Top-1命中数，评分为联合分词后缀logP和。Standard IME标签为冻结的AI专家审核版，不是母语人工gold。

| 模型 | Standard dev /728 | 固定reference /725 | Legacy /500 | AJIMEE /200 | 历史dev /137 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2.1 FP32 | 505 | 535 | 374 | 144 | 122 |
| V2.1 INT8（上一部署） | 506 | — | **375** | 144 | 122 |
| V3 A | 508 | **546** | 368 | 136 | 123 |
| V4-B | 498 | — | 377 | 144 | 120 |
| V4.1 第16轮 | 505 | — | 376 | 137 | **124** |
| V4.2 FP32 | **523** | 541 | 370 | 139 | 123 |
| V4.2 INT8（部署） | 522 | — | 371 | 139 | 123 |
| V4.3 第30轮 | 519 | — | 367 | 143 | 123 |
| V5 第30轮 | 514 | — | 365 | **145** | **124** |

725条reference已在最初的V2.1/V3 A比较中消耗，之后的模型须预注册后只评一次（V3 B：533；V4.2：541）。Standard开发集被多次实验重复使用；配对p值仅作探索性描述。INT8行来自Mac Core ML同一测试环境；两种INT8模型的严格logits对齐均失败，任务命中数最多变化1条。

原有V1/V2历史集合：

| 模型 | 同一旧验证集BPC ↓ | AJIMEE /200 | 历史dev /137 | 草稿dev /2000 |
| --- | ---: | ---: | ---: | ---: |
| V1 FP32 | 3.4736559 | 124 | 122 | 1453 |
| V2.0 FP32 | 3.2598221 | 125 | 124 | 1463 |
| V2.1发布版FP32 | 3.0802686 | 144 | 122 | 1487 |
| V2.1发布版INT8 | 未测量 | 144 | 122 | 1481 |

同一验证文本上，V4-B、V4.2、V4.3与V5（8K词表）的BPC分别为3.027777、3.045152、3.043831和2.942180。完整方法与限制见[英文评测文档](docs/evaluation.md)；V4.2见[训练设置](docs/zh-CN/training-v42.md)、[完整评估](docs/zh-CN/reports/v42-20261020/evaluation.md)、[reference与退化分析](docs/zh-CN/reports/v42-20261020/reference-regressions.md)和[Core ML报告](docs/reports/mac-20261020/v42-coreml.md)。

## 安装与目录

Python≥3.11。`pyproject.toml`提供源代码editable安装元数据，依赖读取`requirements.txt`；Linux CUDA 训练和评测的附加依赖分别记录在对应requirements文件。源仓库推理需要本地匹配checkpoint、tokenizer及manifest；默认CLI仍选择V1，V4 checkpoint通过`vimeml.training.infer_v4.JapaneseLMV4`加载。Core ML转换与运行时评测需要macOS。

`src/vimeml/`保存实现，`scripts/`保存分类命令入口，`configs/`保存配置，`docs/`保存英文主要文档，`docs/zh-CN/`保存中文文档与详细实验报告。`datasets/`、`artifacts/`、`outputs/`、`runs/`、`handoff/`为忽略目录。当前iOS客户端在独立Vime仓库维护；`examples/ios/`为历史接入参考。

[英文文档索引](docs/index.md) · [中文文档索引](docs/zh-CN/index.md) · [贡献约定](CONTRIBUTING.md) · [许可与归属](NOTICE.md)

原创代码与发布权重采用GPL-3.0-or-later，并依GPLv3第7条附加允许经App Store等应用商店分发的额外许可（见[LICENSE-EXCEPTION](LICENSE-EXCEPTION)）；2026-10-22之前的发布采用GPL-2.0-only。第三方组件、语料和基准保留各自许可。公开仓库和模型发布不包含受限原始基准文本。
