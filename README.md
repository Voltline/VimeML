# VimeML

Vime 日语输入法的小型语言模型。AzooKey 根据假名生成候选，LM 按句内左文重排候选，并提供短语／下一词联想。

## 当前状态

| 版本 | 训练与评测 | 状态 |
| --- | --- | --- |
| V1，7.39M 参数 | 1 epoch；validation BPC 3.4736559；AJIMEE 124/200 | 冻结基线，已有 iOS INT8 接入记录 |
| V2.0，12.54M 参数 | 4 epochs；best step 375000；BPC 3.2598221；AJIMEE 125/200 | 完成并归档，尚未验证 Core ML 部署 |
| V2.1 restart1 | 额外2 epochs；best step98426；BPC 3.0811788；AJIMEE 137/200 | 完成评测并下载，旧step0已归档 |
| V2.1 extend5 | 追加3.27轮后停止；保留best step40000；BPC3.0802686；AJIMEE144/200 | 已完成评测与下载，[结果](docs/reports/v2-20261007/v21-extend.md) |

新2,000条development草稿标签下，V1/V2.0/restart1/extend5 best Top-1为72.65% / 73.15% / 73.80% / 74.35%；extend5相对restart1配对p=.22155，相对V1为.00648（探索性、未校正多重比较）。标签待正式审核，1,000条blind未LM计分。[追加结果](docs/reports/v2-20261007/v21-extend.md)。

## 使用与文档

Python ≥3.11。本机依赖见 `requirements.txt`，AutoDL 镜像依赖见 `requirements-autodl.txt`，读音准备依赖见 `requirements-evaluation.txt`。

```powershell
.venv\Scripts\python.exe -X utf8 scripts/training/infer.py
.venv\Scripts\python.exe -X utf8 -u scripts/tools/phrase_demo.py
```

推理默认使用 V1；联想网页地址为 `http://127.0.0.1:8765/`。

| 入口 | 内容 |
| --- | --- |
| [文档索引](docs/index.md) | 指南、实验报告与参考资料 |
| [Mac准备与双仓库协作](docs/mac-v21-preparation.md) | 2026-10-08 Core ML任务、单一ZIP与Git分支／PR流程 |
| [V2 训练](docs/training-v2.md) | 配置、运行方式和当前状态 |
| [V2 实验约定](docs/plan_v2.md) | 结构、评分语义、评测与部署目标 |
| [数据](docs/data.md) / [V1 训练](docs/training.md) | 语料处理与基线复现 |
| [评测](docs/evaluation.md) / [Core ML](docs/coreml.md) | 排序规则、转换与设备结果 |
| [产物与迁移](docs/artifacts.md) | 本地数据、模型及迁移范围 |

## 目录

```text
src/vimeml/       数据、tokenizer、训练、评测与部署核心
scripts/          对应模块的命令入口和实验工具
configs/          版本配置
tests/            回归测试
docs/             指南、reference、reports、history
examples/ios/     Core ML 与 Swift 集成参考
annotations/      语料审核依据
```

Git 保存代码、配置、测试和文档。语料、模型、评分、日志、交接包和密钥保存在忽略目录 `datasets/`、`artifacts/`、`outputs/`、`runs/`、`handoff/`，随实例镜像迁移或单独复制。

W&B账号和run链接只保存在本地监控记录，不提交仓库。

Mac准备包为`handoff/vimeml-v21-mac-20261008.zip`；解压后`python3 setup_mac.py`建立带Git历史的独立工作区。Core ML适配与量化留在Mac执行，代码以分支／PR合并，产物单独回传ZIP。
