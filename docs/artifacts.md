# 本地产物

`datasets/ outputs/ artifacts/ runs/ handoff/ venv/` 不进 Git。clone 之后只有代码和文档，模型、数据和评测结果需要另外复制。正式版本的 manifest 和哈希都是原始字节，整理目录时不要改动。

## Windows（完整）

| 路径 | 内容 | 大小 |
| --- | --- | ---: |
| `datasets/` | 9 个 FineWeb parquet 与 Tatoeba TSV | 2.44 GB |
| `outputs/corpus-fast-v1/` | 正式语料、来源、片段、审计、SQLite、构建记录 | 50.24 GB |
| `outputs/corpus-review-v1/` | 语料审核的 API 回复与依据 | 33 MB |
| `outputs/history/` | 早期批准依据与旧试验统计 | 数 MB |
| `artifacts/tokenizers/ja-unigram-16k-v1/` | 词表与全量统计 | < 1 MB |
| `artifacts/token-data/corpus-v1-16k/` | 三个 split 的 token / offset / row / source | 1.74 GB |
| `artifacts/training-data/corpus-v1-c128/` | 窗口索引与数据检查 | < 1 MB |
| `artifacts/models/tiny-ja-v1/` | best / last checkpoint、配置、指标、指纹 | 178 MB |
| `artifacts/benchmarks/`、`artifacts/ranking-policies/` | 冻结候选、开发集、λ 策略 | 数 MB |
| `outputs/ime-eval/`、`outputs/phrase-demo/` | 评测分数与联想原始输出 | 约 30 MB |
| `runs/`、`artifacts/tracking/` | TensorBoard / W&B | 数 MB |

语料主要占用在 `train.jsonl`（约 20.44 GB）和 `provenance.jsonl`（约 15.96 GB）。`rows.bin` 指向 JSONL 的字节位置，改写或删除 JSONL 会让来源定位和完整性检查失效。

## Mac（Core ML 所需子集）

| 路径 | 内容 | 大小 |
| --- | --- | ---: |
| `artifacts/deployment/tiny-ja-v1-inference-v1/` | 推理包 | 28 MB |
| `artifacts/deployment/tiny-ja-v1-{fp16-v2,conservative-v3,conservative-palette4-g16-v4,conservative-int8-b32-v1}/` | 各 Core ML 版本 | 41 MB |
| `artifacts/deployment/tiny-ja-v1-ios18-int8-release-v1/` | iOS 发布包 | 19 MB |
| `outputs/deployment/` | reference、对齐、评测、联想、耗时、分布对比 | 44 MB |
| `artifacts/benchmarks/ajimee-jwtd-v2-v1/`、`ime-dev-v2/`、`ime-dev-reviewed-v2/` | 评测候选 | 6 MB |
| `outputs/ime-eval/tiny-ja-v1-{ajimee,dev-v2-scores}/` | FP32 基线分数 | 10 MB |
| `outputs/phrase-demo/tiny-ja-v1{,-sample}/` | FP32 联想基线 | < 1 MB |
| `artifacts/models/tiny-ja-v1/`、`artifacts/tokenizers/`、`artifacts/token-data/` | 模型与 token（Mac 本地推理用） | 1.8 GB |
| `venv/coreml/` | Core ML Python 环境 | 865 MB |

## 迁移清单

### AutoDL V2 训练

| 路径 | 内容 |
| --- | --- |
| `handoff/vimeml-v2-data.tar.gz` | V2 token store、索引、tokenizer、原候选；1.017 GB |
| `artifacts/models/tiny-ja-v2.0-e16k-d320-l6/best.pt` | 冻结 step375000，V2.1 初始化权重 |
| `artifacts/benchmarks/ime-expanded-v21-candidates-v1/` | 新 3,000 条真实候选及草稿标签 |
| `artifacts/benchmarks/ime-dev-label-reviewed-v3/` | 135 条复核标签及原候选 |
| `handoff/vimeml-v21-rescue-20261007.tar.gz` | 中断于 step0 的 V2.1 checkpoint 与日志；93.6 MB |

实例镜像用于复用系统盘上的环境和代码，`/root/autodl-tmp` 数据盘不随镜像保存。数据通过数据盘迁移或本地归档恢复；训练不需要原 corpus JSONL。旧 V1/V2.0 指纹与原代码快照保持冻结，当前 Git 源码整理不改写它们。详见 [V2.1 迁出记录](reports/v2-20261007/v21-recovery.md)。

### V1 推理与 Core ML

只做 PyTorch 推理：

```text
artifacts/models/tiny-ja-v1/best.pt
artifacts/tokenizers/ja-unigram-16k-v1/tokenizer.model
artifacts/token-data/corpus-v1-16k/manifest.json     # 加载 checkpoint 时用来核对词表，必须逐字节一致
```

做 Core ML 转换与评测：上面「Mac」表中的推理包、reference、benchmarks、FP32 基线分数和联想基线。不需要原始语料、`tokens.bin`、训练 checkpoint 或 API key。

## 约定

- 每个实验使用新目录：`artifacts/deployment/<版本>/` 存模型，`outputs/deployment/<实验>/` 存报告。转换、压缩、打包和本次修订的报告入口拒绝覆盖已有输出；历史脚本是否支持覆盖需查看各自帮助。
- 每个 Core ML 包的 `manifest.json` 记录压缩方法、目标系统、环境、源包哈希、每个文件的 SHA256 和 op 统计。
- 2026-10-06 的清理记录在 `outputs/maintenance/workspace-cleanup-20261006/cleanup-report.json`；旧版文档存档在 `outputs/maintenance/docs-before-rewrite-20261006/`。


## 2026-10-06 Mac 合并

[合并校验记录](reports/mac-20261006/merge-verification.md) 是当前迁移记录。完整 Mac 源码与 Vime 客户端快照、实验结果、训练输入 ZIP 保存在 `handoff/mac-20261006-v1/`；每个ZIP和每个归档条目都已核对SHA256。这个目录不进入Git。

原 Mac README/docs 保存在 `outputs/maintenance/mac-merge-20261006/mac-source/`，Windows 整理前的对应文档在同目录 `windows-before-merge/`。`merge-report.json` 列出每个导入文件的保存位置与哈希。新模型包、Core ML 失败实验、模拟器失败测试和真机trace都保留。冲突产物与退役的 `corpus-pilot-16k` 进入 `outputs/history/mac-import-20261006/`，不恢复为正式基线。

原始 manifest 中的绝对路径与转换脚本哈希是当时的来源信息，不能改成当前路径或脚本哈希。转换脚本后来有增补，当前脚本不应被当作旧包的逐字节来源。
