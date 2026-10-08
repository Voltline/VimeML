# VimeML

用于日语假名转汉字候选重排的小型因果语言模型，兼有实验性的短文本续写和下一词联想功能。AzooKey负责词典候选检索，语言模型利用左文重排固定候选池。模型未进行指令微调，不定位为通用聊天助手。

[English README](README.md)为主要入口；[英文模型卡](MODEL_CARD.md)描述模型与评测边界，[中文模型卡](docs/zh-CN/model-card.md)提供对应概览。

## 模型发布

| 版本 | 参数量 | 分词器 | Context | Hugging Face |
| --- | ---: | --- | ---: | --- |
| V1 | 7,386,624 | 16K SentencePiece Unigram，V1词表 | 128 | [Voltline/vimeml-tiny-ja-v1](https://huggingface.co/Voltline/vimeml-tiny-ja-v1) |
| V2.1 | 12,537,920 | 16K SentencePiece Unigram，V2词表 | 128 | [Voltline/vimeml-tiny-ja-v2.1](https://huggingface.co/Voltline/vimeml-tiny-ja-v2.1) |

当前部署版本为V2.1 extend5 step40000：INT8 block32权重压缩、FP32计算、Core ML CPU_ONLY、最低iOS18，包体14.33 MB。V1保留为基线和回退。V3 A/B训练、评测与归档已完成，尚未形成足以替换V2.1的稳定任务收益。

公开发布包含推理权重、匹配分词器、Core ML资源和聚合评测报告。训练checkpoint、语料、凭据及私有基准文本独立保存。自定义模型格式不直接兼容Transformers AutoModel；Hugging Face推理包的运行说明以各自模型卡为准。

## 固定评测结果

| 模型 | 同一旧验证集BPC ↓ | AJIMEE /200 | 历史dev /137 | 草稿dev /2000 |
| --- | ---: | ---: | ---: | ---: |
| V1 FP32 | 3.4736559 | 124 | 122 | 1453 |
| V2.0 FP32 | 3.2598221 | 125 | 124 | 1463 |
| V2.1 restart1 FP32 | 3.0811788 | 137 | 121 | 1476 |
| V2.1发布版FP32 | 3.0802686 | 144 | 122 | 1487 |
| V2.1发布版INT8 | 未测量 | 144 | 122 | 1481 |

量化后两套小集命中数不变，草稿集下降0.3个百分点。严格logits对齐仍失败，任务指标支持已记录条件下的部署取舍，不构成数值等效证明。Standard IME采用冻结AI专家审核标签，并非母语人工gold；已开启的725条集合为固定参考，不能再称新盲测。完整方法与限制见[英文评测文档](docs/evaluation.md)。

## 安装与目录

Python≥3.11。`pyproject.toml`提供源代码editable安装元数据，依赖读取`requirements.txt`；AutoDL和评测附加依赖分别记录在对应requirements文件。源仓库推理需要本地匹配checkpoint、tokenizer及manifest；默认CLI仍选择V1。Core ML转换与运行时评测需要macOS。

`src/vimeml/`保存实现，`scripts/`保存分类命令入口，`configs/`保存配置，`docs/`保存英文主要文档，`docs/zh-CN/`保存中文参考资料。`datasets/`、`artifacts/`、`outputs/`、`runs/`、`handoff/`为忽略目录。当前iOS客户端在独立Vime仓库维护；`examples/ios/`为历史接入参考。

[英文文档索引](docs/index.md) · [中文文档索引](docs/zh-CN/index.md) · [贡献约定](CONTRIBUTING.md) · [许可与归属](NOTICE.md)

原创代码与发布权重采用GPL-2.0；第三方组件、语料和基准保留各自许可。公开仓库和模型发布不包含受限原始基准文本。
