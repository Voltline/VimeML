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
| `artifacts/benchmarks/ime-standard-ja-v1-ai-expert-r1/`、`ime-standard-ja-v2-blind-ai-expert/`、`ime-homophone-ja-v1-ai-expert/`、`ime-incremental-ja-proto1/` | Standard v1、v2 blind、同音词挑战与分段原型的冻结版；同名`-draft`、`-review`为构建与审核记录 |
| `artifacts/benchmarks/ime-candidates-v2-handoff{,.zip}` | 候选导出v2的Mac交接包 |
| `artifacts/benchmarks/ime-candidates-v2-imported/<轨道>-n<k>/` | 导入的候选导出v2（回传包在 `handoff/benchmarks/`） |
| `outputs/training-v43-20261020/remote/`、`outputs/training-v5-20261021/remote/` | V4.3（第27、30轮）与V5（第29、30轮）checkpoint、记录和源码；`controlled-10ep/`为10轮对照 |
| `artifacts/deployment/tiny-ja-v2.1-extend5-{bundle,fp32,int8-b32,client-resources}-v1/` | 推理bundle、Core ML包和客户端资源 |
| `outputs/ime-eval/`、`outputs/model-checks/`、`outputs/deployment/` | 分数、数值参考、推理、设备结果与trace |
| `outputs/history/` | 已退役的脚本、快照、日志和冲突版本 |
| `outputs/maintenance/` | 导入、目录整理及文档改版记录 |

语料原文在`outputs/corpus-fast-v1/`（约50GB），原输入在`datasets/`（约2.44GB）。`rows.bin`依赖JSONL原始字节位置；部署只需要推理权重、tokenizer、manifest与参考结果，不依赖训练原文、token store或优化器。

## 迁移与归档

| 归档 | 内容 |
| --- | --- |
| `handoff/mac-20261020-v42/` | V4.2 Core ML输入摘要与回传`handoff.json`；250份产物已恢复 |
| `handoff/mac-20261008-v21/` | V2.1输入摘要与回传`handoff.json`；306份产物已恢复 |
| `handoff/mac-20261006-v1/` | V1清单、源码与客户端快照 |
| `handoff/remote/` | 云 GPU 训练的代码、报告快照与V4传输清单 |
| `handoff/benchmarks/` | Standard IME审核、Mac交接与候选回传包 |

已验证可复现的大型传输包于2026-10-20删除，记录见`outputs/maintenance/handoff-cleanup-20261020/`与`handoff/README.md`。

源码通过两个仓库各自的Git分支／PR合并，产物通过ZIP恢复。客户端当前实现位于独立Vime仓库；`examples/ios/`及旧ZIP仅为历史参考。流程见[跨平台管理](reference/artifact-exchange.md)，V2导入记录位于`outputs/maintenance/mac-return-20261008/import-report.json`。

## 来源与环境

Manifest中的路径、代码指纹和失败项描述原运行环境，目录迁移不改写原始记录。模型／tokenizer变更时建立新身份，后续阶段复用冻结manifest；传输检查清单、大小与容器完整性。

云 GPU 实例的大文件放在数据盘上，不随系统镜像保存。依赖由`requirements.txt`与`requirements-cuda.txt`重建；虚拟环境和凭据不进入迁移包。Mac环境记录在`outputs/deployment/environment/`，W&B账号和run链接只保存在本地忽略目录。
