# 本地产物与迁移

这些目录不提交 Git。Git clone 得到代码与文档，模型、数据和评测文件需要另行复制。正式版本的 manifest / hash 保持原字节，不随目录整理改写。

## 保存范围

| 路径 | 用途 | 大小约数 |
| --- | --- | ---: |
| `datasets/` | 9 个 FineWeb parquet 与 Tatoeba 原始 TSV | 2.44GB |
| `outputs/corpus-fast-v1/` | 正式语料、来源、片段、审计、SQLite、构建记录 | 50.24GB |
| `outputs/corpus-review-v1/` | 全量存疑与分层抽检的 API / 审核依据 | 33MB |
| `outputs/history/` | 早期批准依据及精简的旧试验统计 | 数 MB |
| `artifacts/tokenizers/ja-unigram-16k-v1/` | 正式词表与全量统计 | <1MB |
| `artifacts/token-data/corpus-v1-16k/` | 三个 split 的 token / offset / row / source | 1.74GB |
| `artifacts/training-data/corpus-v1-c128/` | 长句窗口索引和数据检查 | <1MB |
| `artifacts/models/tiny-ja-v1/` | best / last、配置、训练计数、指纹、完整验证 | 约 178MB |
| `artifacts/benchmarks/` / `artifacts/ranking-policies/` | 冻结候选、开发标签、生成依据与策略 | 数 MB |
| `outputs/ime-eval/` / `outputs/phrase-demo/` | 固定评测、原始生成和复核 | 约 30MB |
| `runs/` / `artifacts/tracking/` | TensorBoard / W&B 事件与运行记录 | 数 MB |

表中采用十进制 GB。当前大头是正式语料而非模型或缓存：train.jsonl 约 20.44GB，provenance.jsonl 约 15.96GB。JSONL 保存追踪元数据，rows.bin 指向其中字节位置；任意重写或删除会使来源定位或完整性检查失效。不能仅因已经生成 tokens 就把它们当临时文件。

## 当前推理工具的最小迁移

把以下文件按相同相对目录复制到 Mac 仓库；不需要全部 corpus 或 tokens.bin：

```text
artifacts/models/tiny-ja-v1/best.pt
artifacts/tokenizers/ja-unigram-16k-v1/tokenizer.model
artifacts/token-data/corpus-v1-16k/manifest.json
```

现有 JapaneseLM 加载 checkpoint 时用 token manifest 验证词表绑定，manifest 必须逐字节一致。配置和 Python 实现在 Git 中。额外复制模型的 config/manifest/summary 与 tokenizer 的完整小目录，便于追踪；CPU 环境安装见 [训练指南](training.md)。

## Core ML 效果验证的迁移

除以上模型资料，还需：

- `artifacts/benchmarks/ajimee-jwtd-v2-v1/`：原始公开数据、映射、NOTICE、完整 Mac 候选和版本。
- `artifacts/benchmarks/ime-dev-v2/` 与 `ime-dev-reviewed-v2/`：独立开发候选和审核依据。
- `artifacts/ranking-policies/tiny-ja-v1-hybrid-dev-v2/`：FP32 策略参考。
- `outputs/ime-eval/tiny-ja-v1-ajimee/` 与 `tiny-ja-v1-dev-v2-scores/`：FP32 分数参考。
- `outputs/phrase-demo/tiny-ja-v1/` 与 `tiny-ja-v1-sample/`：生成与自然度基线。

转换本身不需要 API key。实际转换工具实现后将导出独立推理包，不要求应用加载训练 checkpoint 或 token manifest。

## 2026-10-06 清理

移除早期 `legacy/` 脚本和配置（可从此前 Git 提交恢复）、旧 pilot / integration 完整语料、pilot tokenizer、smoke 权重、安装依赖缓存、遗留测试临时目录、重复的根目录 `evaluation_items.json` 和旧手写 pilot 分词 snippet。旧试验的小份配置/统计/manifest 保存在 `outputs/history/retired-experiments/`，不当成仍可恢复的完整产物。

所有正式模型、tokenizer、数据、评分策略、候选、审核与来源文件保留。具体删除清单、字节数和关键 hash 在 `outputs/maintenance/workspace-cleanup-20261006/cleanup-report.json`。不对正在运行的服务或 `.venv/` 做清理；新实验生成缓存按版本隔离，避免混入正式基线。
