# 数据

正式语料 `outputs/corpus-fast-v1/`：2,289,346 条来源记录 → 25,713,003 条唯一句子。tiny-ja-v1 用的就是这一版，以下命令用于复现或构建新版本。

## 来源

| 来源 | 文件 | 说明 |
| --- | --- | --- |
| FineWeb2-Edu Japanese | `datasets/fineweb-2-edu-japanese/train-000{10,30,…,170}-of-00283.parquet`，9 个分片，约 2.43 GB 压缩 | 教育类网页文本 |
| Tatoeba | `datasets/Tatoeba/jpn_sentences.tsv` | 普通日语例句 |

路径写在 `configs/corpus-parallel.toml`。构建全程离线，不调用 API。

## 处理流程

```text
parquet / TSV
  → Unicode 与确定性基础清理
  → 噪声判定：keep / drop / review
  → 保守恢复句内换行 → 分句，隔离碎片
  → 按文档分组、精确去重、切分 split（防止同句跨 split）
  → TXT / JSONL / 来源审计
```

- 清洗保留原文和修改原因。纯空白、装饰行、独立 URL 直接 drop；可疑的导航尾缀进入 review。
- 换行恢复只接续明显断开的句子，不会把标题、数量等相邻行盲目拼接。
- split 比例 98% / 1% / 1%，seed 42。相同文本按组分配，保证不跨 split。
- 只做了精确去重。语义近重复、与公开评测集的重叠没有做全面审计。
- 只生成句内语料，没有生成读音增强数据。

## 构建

单机并行（每个 worker 有自己的 SQLite，清洗完自动按哈希桶并行合并）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/corpus/build.py --config configs/corpus-parallel.toml --output outputs/corpus-new-v2 --workers 8
```

`--resume` 会复用指纹一致且校验通过的分区。

跨机分区：两台机器使用相同的代码、配置、数据和批准记录，只是分区编号不同，最后在同一台机器上合并。全局去重和 split 只在合并时做。

```powershell
# Windows
.\.venv\Scripts\python.exe -X utf8 -u scripts/corpus/preprocess.py --config configs/corpus-sharded.toml --part 0/2 --output outputs/corpus-part-0
```

```bash
# Mac
.venv/bin/python -u scripts/corpus/preprocess.py --config configs/corpus-sharded.toml --part 1/2 --output outputs/corpus-part-1
```

```powershell
# 合并
.\.venv\Scripts\python.exe -X utf8 -u scripts/corpus/merge.py --config configs/corpus-sharded.toml --parts outputs/corpus-part-0 outputs/corpus-part-1 --output outputs/corpus-new-v2 --workers 8
```

## 审核

v1 的审核已完成：638 个隔离块继续排除，批准记录在 `annotations/approved-v1.jsonl`，校准文档 ID 在 `annotations/calibration-documents-v1.json`。修改清洗规则需要建立新的语料版本。

审核流程：

1. `prepare.py`：并行重放规则，抽出 review 目标，以及按保留句、拼接边界、碎片、drop、特殊字符分层的抽检样本（样本取自 train，不是全库均匀抽样）。
2. `run.py`：调用 DeepSeek（SJTU 接口）给出审查建议。只给建议，不润色、不补写、不自动批准。默认 8 RPM / 80K TPM，重试也计入；同一命令可续跑，原始回复和失败记录都会保留。
3. 人工核对 issue / uncertain 并抽查 ok，再用 `approve.py` 只记录明确选中的边界或片段。

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/review/prepare.py --workers 8
.\.venv\Scripts\python.exe -X utf8 scripts/review/run.py --dry-run      # 不读 key、不联网、不写文件
$env:SJTU_API_KEY = "..."
.\.venv\Scripts\python.exe -X utf8 -u scripts/review/run.py --workers 5 --batch-size 10
```

两个账号可以用 `scripts/review/run_multi_key.py`（读取 `SJTU_API_KEY` / `SJTU_API_KEY_2`，每个账号 3 并发、各自独立限额；更多账号用 `--key-env`）。401/403 会停用该账号，429 只暂停该账号。不要和 `run.py` 同时写同一个输出目录。

默认输入 / 输出目录分别是 `outputs/corpus-fast-v1/` 和 `outputs/corpus-review-v1/`，新版本用 `--help` 查看路径参数。

## 产物

| 文件 | 用途 |
| --- | --- |
| `train/validation/test.txt` | 每行一个完整句子，供 SentencePiece 使用 |
| `train/validation/test.jsonl` | 与 txt 同顺序，带来源，供编码时追踪 |
| `provenance.jsonl` | 重复来源与 span 的完整追踪 |
| `documents.jsonl` / `index.sqlite` | 文档与审核索引 |
| `fragments/` `dropped_blocks/` `review_blocks.jsonl` | 隔离内容与规则审计 |
| `manifest.json` / `stats.json` | 版本指纹与计数 |
| `integrity-check.json` / `build-records/` | 完整性与构建依据 |

## 校验

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/tools/check_corpus.py         # manifest、核心代码、统计、文件大小
.\.venv\Scripts\python.exe -X utf8 scripts/tools/check_corpus.py --full  # 重新读取全部数据并计算哈希
```

合并 manifest 里的 `stage=merged_corpus_staging`、`ready_for_lm_training=false` 是构建时写入的，为了保持指纹不回写；v1 后来已审核并用于训练。`src/vimeml/data/` 和合并核心代码保持原路径和内容，否则已有指纹会失效。
