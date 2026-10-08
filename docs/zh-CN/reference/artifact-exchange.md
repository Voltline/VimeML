# 跨平台源码与产物管理

源码通过Git分支和PR合并；模型、分数、编译资源和trace通过带清单的ZIP迁移。VimeML负责训练、评测和模型部署工具，Vime仓库负责客户端实现，两者分别提交，不用客户端源码快照覆盖另一仓库。

## 推理输入包

`scripts/deployment/prepare_mac.py`从干净且已提交的工作区生成Git bundle与推理输入。2026-10-08输入包为`handoff/vimeml-v21-mac-20261008.zip`，基点由包内`handoff-manifest.json`记录。

```bash
python3 setup_mac.py
```

包内入口建立独立Git工作区，绑定origin并复制忽略目录中的资料。默认目标为`~/Documents/Sources/VimeML-v21-coreml`，已有目录不覆盖。原包对应当时的Git快照；后续代码以主分支为准。

| 材料 | 内容 |
| --- | --- |
| `repository.bundle` | 已提交源码、文档与可达Git历史 |
| `deployment.pt` | best的相同FP32模型tensor；去除optimizer/RNG，不用于恢复训练 |
| Tokenizer／token manifest | 模型对应的精确token ID与来源 |
| 三套development候选与FP32分数 | AJIMEE200、原dev137、扩大草稿2000，同池量化对照 |
| Logits与文本fixture | 长度1/16/128、PAD/causal及12条固定推理 |
| V1 INT8资源 | 部署对照与回退参考 |

输入包不含blind、训练语料、token store、优化器、虚拟环境或凭据。V2词表与V1不同，模型、tokenizer、manifest和客户端fixture成套更新。模型接口与评分定义见[Core ML](../coreml.md)和[评测](../evaluation.md)。

## 源码协作

| 仓库 | 提交范围 |
| --- | --- |
| VimeML | 转换／量化工具、评测实现、模型实验报告 |
| Vime | 模型资源接入、分词／评分、排序／联想与设备测量 |

每个仓库从明确基点建立分支，记录base/head commit，通过PR回到`main`。2026-10-08分支为`codex/mac-v21-coreml`与`codex/v21-coreml-client`，部署代码已合入。其他工作区以`git pull --ff-only origin main`同步代码；ZIP只恢复忽略目录中的产物。

## 结果包

`scripts/deployment/package_mac_results.py`打包新增模型、报告和必要compiled资源。顶层`handoff.json`记录格式版本、环境、两个仓库的base/head、PR、相对路径／大小及通过和失败项。历史模型、训练输入和客户端源码不纳入结果包。

2026-10-08结果包归档于`handoff/mac-20261008-v21/vimeml-v21-mac-results.zip`，306份产物恢复到原`artifacts/`与`outputs/`路径；导入记录为`outputs/maintenance/mac-return-20261008/import-report.json`。来源核对使用Git ancestry、路径、数量、大小与ZIP CRC。同名产物冲突另存版本，原模型manifest和报告保持原始字节。

依赖由`requirements.txt`和`requirements-autodl.txt`重建。身份记录在首次导出或资源变化时建立，后续复用冻结manifest；W&B账号、run链接和凭据只保存在忽略目录。

相关记录：[V2总结](../reports/v2-summary.md)、[量化](../reports/mac-20261008/v21-coreml.md)、[本地产物](../artifacts.md)、[V1快照导入](../reports/mac-20261006/handoff.md)。
