# VimeML

Vime日语输入法的小型语言模型。AzooKey根据假名检索候选，LM利用句内左文重排，并提供短语和下一词联想。

## 实验状态

V2系列实验于2026-10-08完成。当前部署模型为 **V2.1 extend5 best / step40000，INT8 block32**：12.54M参数、16K词表、context128，Core ML包14.33MB，CPU_ONLY、FP32计算、最低iOS18。训练、同池评测和iPhone真实键盘验证见[V2实验总结](docs/reports/v2-summary.md)。V1保留作基线和回退。

| 模型 | 完整validation BPC ↓ | AJIMEE /200 | 原dev /137 | 扩大草稿 /2000 |
| --- | ---: | ---: | ---: | ---: |
| V1 FP32 | 3.4736559 | 124 | 122 | 1453 |
| V2.0 FP32 | 3.2598221 | 125 | 124 | 1463 |
| V2.1 restart1 FP32 | 3.0811788 | 137 | 121 | 1476 |
| V2.1 extend5 best FP32 | 3.0802686 | 144 | 122 | 1487 |
| V2.1 extend5 INT8 | — | 144 | 122 | 1481 |

量化后原两集命中数不变，扩大草稿下降0.3个百分点。严格logits对齐失败与UI发布长尾保留为已接受的实验差异，当前输入体验可接受。扩大标签尚未正式审核，1000条blind未计分；完整结论与统计范围见[评测](docs/evaluation.md)和[量化分析](docs/reports/mac-20261008/v21-quantization-review.md)。

## 使用

Python≥3.11。依赖见`requirements.txt`、`requirements-autodl.txt`和`requirements-evaluation.txt`。

```powershell
.venv\Scripts\python.exe -X utf8 scripts/training/infer.py
.venv\Scripts\python.exe -X utf8 -u scripts/tools/phrase_demo.py
```

Python推理入口默认使用V1，其他checkpoint/tokenizer由命令行指定；联想网页默认地址为`http://127.0.0.1:8765/`。iOS客户端在独立Vime仓库维护，`examples/ios/`为历史接入参考。

## 文档与目录

| 内容 | 入口 |
| --- | --- |
| 文档导航与报告 | [索引](docs/index.md)、[V2总结](docs/reports/v2-summary.md) |
| 模型与训练 | [V2实验设计](docs/plan_v2.md)、[V2训练](docs/training-v2.md)、[V1训练](docs/training.md) |
| 数据与评测 | [语料](docs/data.md)、[评分与数据集](docs/evaluation.md) |
| 部署与产物 | [Core ML](docs/coreml.md)、[本地产物](docs/artifacts.md)、[跨平台管理](docs/reference/artifact-exchange.md) |

`src/vimeml/`保存实现，`scripts/`保存命令入口，`configs/`保存版本配置。Git跟踪源码、配置、测试、文档及许可证；模型、语料、评分、日志和迁移包保存在忽略目录。W&B账号、run链接及凭据不提交。
