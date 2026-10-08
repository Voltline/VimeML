# 本地产物与目录

Git保存源码、配置、测试、文档和许可证。`datasets/`、`artifacts/`、`outputs/`、`runs/`、`handoff/`为忽略目录；模型和原始实验资料保留原始字节，可重建缓存与临时测试目录独立清理。

## 模型、数据与报告

| 路径 | 内容 |
| --- | --- |
| `artifacts/models/tiny-ja-v2.1-e16k-d320-l6-extend5/` | best step40000、last step161095和三轮评测；两个本地checkpoint约301MB，epoch权重在远端归档 |
| `artifacts/models/tiny-ja-v2.1-e16k-d320-l6-restart1/` | best/last step98426、epoch1和两轮报告 |
| `artifacts/models/tiny-ja-v2.0-e16k-d320-l6/`、`tiny-ja-v1/` | V2.0与V1基线 |
| `artifacts/tokenizers/ja-unigram-16k-v2/` | V2词表，token ID与V1不同 |
| `artifacts/token-data/corpus-v2-16k/`、`artifacts/training-data/corpus-v2-c128/` | Token store与窗口索引 |
| `artifacts/benchmarks/ime-expanded-v21-candidates-v1/` | Development2000和blind1000；草稿标签，blind未计分 |
| `artifacts/deployment/tiny-ja-v2.1-extend5-{bundle,fp32,int8-b32,client-resources}-v1/` | 推理bundle、Core ML包和客户端资源 |
| `outputs/ime-eval/`、`outputs/model-checks/`、`outputs/deployment/` | 分数、数值参考、推理、设备结果与trace |
| `outputs/history/` | 已退役的脚本、快照、日志和冲突版本 |
| `outputs/maintenance/` | 导入、目录整理及文档改版记录 |

语料原文在`outputs/corpus-fast-v1/`（约50GB），原输入在`datasets/`（约2.44GB）。`rows.bin`依赖JSONL原始字节位置；部署只需要推理权重、tokenizer、manifest与参考结果，不依赖训练原文、token store或优化器。

## 迁移与归档

| 归档 | 内容 |
| --- | --- |
| `handoff/vimeml-v21-mac-20261008.zip` | V2推理输入、Git bundle、冻结评分和V1对照 |
| `handoff/mac-20261008-v21/vimeml-v21-mac-results.zip` | Mac结果包与原`handoff.json`；306份产物已恢复 |
| `handoff/mac-20261006-v1/` | V1源码、客户端与结果快照 |
| `handoff/vimeml-v2-*.tar.gz`、`vimeml-v21-*.tar.gz` | AutoDL代码、输入、恢复、模型和评测快照 |

源码通过两个仓库各自的Git分支／PR合并，产物通过ZIP恢复。客户端当前实现位于独立Vime仓库；`examples/ios/`及旧ZIP仅为历史参考。流程见[跨平台管理](reference/artifact-exchange.md)，V2导入记录位于`outputs/maintenance/mac-return-20261008/import-report.json`。

## 来源与环境

Manifest中的路径、代码指纹和失败项描述原运行环境，目录迁移不改写原始记录。模型／tokenizer变更时建立新身份，后续阶段复用冻结manifest；传输检查清单、大小与容器完整性。

AutoDL数据目录为`/root/autodl-tmp/vimeml`，不随系统镜像保存。依赖由`requirements.txt`与`requirements-autodl.txt`重建；虚拟环境和凭据不进入迁移包。Mac环境记录在`outputs/deployment/environment/`，W&B账号和run链接只保存在本地忽略目录。
