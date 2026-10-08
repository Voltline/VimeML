# 本地语料处理

正式语料位于 `outputs/corpus-fast-v1/`，2,289,346 条来源记录、25,713,003 条唯一句子。以下命令用于复现或新版本实验，当前无需重跑。

## 数据和规则

本地输入为 9 个 FineWeb2 Edu Japanese `small_tokens_cleaned` parquet（约 2.43GB compressed）及 Tatoeba 普通日语 Sentences TSV。具体路径在 `configs/corpus-parallel.toml`；全量构建不调用 API。

```text
parquet / TSV → Unicode 与确定性基础清理 → 噪声 keep/drop/review
              → 保守恢复句内换行 → 分句与片段隔离
              → 文档分组、精确去重、split 去泄漏 → TXT / JSONL / 来源审计
```

清洗保留原文与修改原因，明确空白、装饰或独立 URL 可 drop；可疑导航尾缀进入 review。换行恢复采用保守接续规则，不因没有句号就盲目连接标题、数量等相邻行。已有批准记录在 `annotations/approved-v1.jsonl`，校准文档 ID 在 `annotations/calibration-documents-v1.json`。

同一文档与重复文本按组处理，避免相同句子跨 split；种子 42，目标比例 98% / 1% / 1%。精确去重已完成，语义近重复和与公开评测的训练重叠尚未全面审计。仅做句内语料，不生成 reading augmentation。

## 单机并行 / 跨机分区

从仓库根目录手动开始一个新版本：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/corpus/build.py --config configs/corpus-parallel.toml --output outputs/corpus-new-v2 --workers 8
```

worker 各自拥有 SQLite，清洗完成后自动用哈希桶并行合并。`--resume` 复用指纹一致且校验通过的完成分区；中断输出与工作目录分开保存。不要把已经删除的旧 worker 当成可恢复输入。

跨机使用相同代码、配置、数据和批准记录，分区编号不同：

```powershell
# Windows
.\.venv\Scripts\python.exe -X utf8 -u scripts/corpus/preprocess.py --config configs/corpus-sharded.toml --part 0/2 --output outputs/corpus-part-0
```

```bash
# Mac，已有 Python >=3.11
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -u scripts/corpus/preprocess.py --config configs/corpus-sharded.toml --part 1/2 --output outputs/corpus-part-1
```

两个完成分区集中到同一机器后，验证 manifest 后合并；全局去重和 split 只在此处统一完成：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/corpus/merge.py --config configs/corpus-sharded.toml --parts outputs/corpus-part-0 outputs/corpus-part-1 --output outputs/corpus-new-v2 --workers 8
```

## 审核和抽检

首版审核已结束，638 条隔离块继续排除，不因为 API 建议 keep 就恢复。首轮规则问题保留为基线限制，改规则应建立新语料版本。

新版本审核的手动入口：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/review/prepare.py --workers 8
.\.venv\Scripts\python.exe -X utf8 scripts/review/run.py --dry-run
# 在本地设置 SJTU_API_KEY 后才运行
.\.venv\Scripts\python.exe -X utf8 -u scripts/review/run.py --workers 5 --batch-size 10
```

默认输入 / 输出分别为 `outputs/corpus-fast-v1/` 与 `outputs/corpus-review-v1/`。新版本使用 `--help` 查看路径参数并指定独立材料目录。prepare 并行重放规则，包含 review 目标及保留句子、拼接边界、碎片、drop、特殊字符的分层抽检；诊断样本取 train，抽检不是全库均匀随机样本。

API 使用 DeepSeek 提供审查建议，不自动润色、补写或批准语料；默认共享 8 RPM / 80K TPM，重试也计入。dry-run 不读 key、不联网、不写文件。同一命令续跑复用成功缓存，原始回复与失败记录保留。核对 issue/uncertain 并抽查 ok；`approve.py` 仅记录明确选中的边界或片段批准，不批量采用模型 keep。

两个独立账号可以运行 `scripts/review/run_multi_key.py`，默认读取 `SJTU_API_KEY` / `SJTU_API_KEY_2`，每账号 3 并发、8 RPM / 80K TPM。先停止旧进程，避免两个入口同时操作同一输出；更多账号使用 `--key-env ENV_NAME ...`。401/403 禁用相应账号，429 只暂停该账号，全部失效后保存 pending。

## 正式产物与检查

| 文件 | 用途 |
| --- | --- |
| `train/validation/test.txt` | 每行一个完整句子，供 SentencePiece |
| `train/validation/test.jsonl` | 同顺序句子及来源，供编码追踪 |
| `provenance.jsonl` | 重复来源与 span 的完整追踪 |
| `documents.jsonl` / `index.sqlite` | 文档与审核索引 |
| `fragments/dropped_blocks/review_blocks.jsonl` | 隔离与规则审计 |
| `manifest.json` / `stats.json` | 版本指纹与计数 |
| `integrity-check.json` / `build-records/` | 完整性与构建依据 |

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/tools/check_corpus.py
# 只有需要重新读几十GB并计算导出哈希时加 --full
```

原 merge manifest 仍保留构建时的 `stage=merged_corpus_staging` 和 `ready_for_lm_training=false`；后续审核结束与作为 v1 基线使用的决定单独留档，不回写原 manifest，以保持已有指纹。该标志不是已完成训练的状态，也不能用完整性检查替代质量审核。

快速检查核对 manifest、核心代码、统计和文件大小，不等价于重新计算全库 SHA256。`src/vimeml/data/` 和合并核心保持路径及内容，确保已有指纹继续有效。磁盘用途见 [本地产物](../artifacts.md)。
