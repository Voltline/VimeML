# 本地产物与目录

Git保存代码、配置、测试和文档；`datasets/artifacts/outputs/runs/handoff`是忽略目录。正式模型、语料和manifest保持原始字节；清理只删除可重建缓存与临时测试目录。

## 当前产物

| 路径 | 内容 |
| --- | --- |
| `artifacts/models/tiny-ja-v2.1-e16k-d320-l6-extend5/` | best step40000、last step161095及三轮评测；两个本地checkpoint约301MB，epoch权重保留远端 |
| `artifacts/models/tiny-ja-v2.1-e16k-d320-l6-restart1/` | 冻结best/last step98426、epoch1及原两轮报告 |
| `artifacts/models/tiny-ja-v2.0-e16k-d320-l6/`、`tiny-ja-v1/` | V2.0与V1冻结基线 |
| `artifacts/tokenizers/ja-unigram-16k-v2/` | V2词表；不能替换成V1词表 |
| `artifacts/token-data/corpus-v2-16k/`、`artifacts/training-data/corpus-v2-c128/` | 冻结token store与窗口索引；仅训练需要 |
| `artifacts/benchmarks/ime-expanded-v21-candidates-v1/` | development2000与blind1000；草稿标签，blind未LM计分 |
| `outputs/ime-eval/`、`outputs/model-checks/`、`outputs/deployment/` | 逐候选分数、推理／数值参考和端侧结果 |
| `outputs/history/v2-20261007-session/` | 已结束会话的辅助脚本、运行快照与日志；不是当前命令入口 |
| `outputs/maintenance/workspace-cleanup-20261007/` | 整理前文档、移动清单与清理记录 |

训练语料原文在`outputs/corpus-fast-v1/`（约50GB），原输入在`datasets/`（约2.44GB）；`rows.bin`依赖JSONL原字节位置。转换无需复制原文、token store、训练索引或优化器。

## Mac迁移

当前唯一准备包为`handoff/vimeml-v21-mac-20261008.zip`：完整Git历史、独立V2权重、tokenizer/原token manifest、development候选与FP32参考、V1 INT8对照。使用方式和Git约定见[Mac准备](mac-v21-preparation.md)。包内`deployment.pt`模型tensor与原best一致，移除优化器；原checkpoint没有重写。

历史包保留：`handoff/mac-20261006-v1/`是昨日Mac源码／Vime客户端快照，`handoff/vimeml-v2-*.tar.gz`及`vimeml-v21-*.tar.gz`是原AutoDL启动、训练输入、恢复和评测记录。无需把这些全量包一起传给Mac；它们不是当前源码合并方式。

Mac仅回传新增V2部署资源与报告ZIP；代码通过两个仓库各自的分支／PR合并。Vime现有客户端以Mac仓库为准，`examples/ios/`和昨日ZIP为历史参考，不能覆盖客户端最新实现。

## 来源与环境

原始manifest里的绝对路径、代码指纹和失败结果仍代表当时运行，不改成当前路径或伪装成新验证。后续复用已绑定identity，只在模型／tokenizer资源变化时记录新的身份；迁移按清单、大小和Git commit检查，不逐文件重复SHA256。

AutoDL使用`/root/autodl-tmp/vimeml`数据盘，该目录不随系统镜像保存。Python环境分别由`requirements.txt`和`requirements-autodl.txt`重建，不迁移venv或凭据。Mac旧环境记录在`outputs/deployment/environment/`。

2026-10-06归档见[Mac合并记录](reports/mac-20261006/merge-verification.md)，当前训练与选择见[追加结果](reports/v2-20261007/v21-extend.md)。
