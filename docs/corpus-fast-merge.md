# 复用已完成 worker 的独立并行汇总

入口为 `src/vimeml/data/merge_parts_fast.py`。它读取已有 `preprocessed_part`，不重新清洗原始 parquet/TSV，不发送 API 请求，也不读取正在写入的旧汇总数据库。依赖仍是 `requirements-data.txt` 中的 pyarrow 和 Python 标准库。

当前全量任务可直接执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u src/vimeml/data/merge_parts_fast.py --workers 8
```

默认输入是 `outputs/corpus-parallel-v1-work/parts/`；结果写入 `outputs/corpus-fast-v1/`，中间文件写入 `outputs/corpus-fast-v1-work/`。旧 worker 和旧 corpus 保留。新入口不会停止旧进程；同时运行会竞争磁盘和 CPU。

其他机器或其他批次可显式指定：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u src/vimeml/data/merge_parts_fast.py `
  --config configs/corpus-parallel.toml `
  --parts-dir outputs/corpus-parallel-v1-work/parts `
  --output outputs/corpus-fast-v2 `
  --workers 8 --buckets 64 --cache-mb 128
```

配置、清洗代码和审批文件必须与 worker 一致；脚本验证所有分区完整、输入覆盖和产物 SHA256。文件很大时，初始校验本身也会耗时。输出和工作目录必须为空且不能与输入重叠。当前版本不续跑中间桶；失败后可换一个输出目录，继续复用原始 worker，仍无需重新清洗。

## 工作方式

1. 全局汇集文档与来源元数据。同内容文档选最小分区编号作为清洗结果所有者，所有原始来源仍保留。
2. 内存中的并查集把相同内容、相同 source ID、相同 canonical URL 的文档合为传递分组。按组分配 split，校准与批准标注涉及的组强制进入 train。
3. 多进程顺序读取 worker 里的句子，按 SHA256 前缀分到有序的 Parquet 桶。相同句子必定进入同一个桶。
4. 多进程分别去重、选来源和导出各桶。每个进程使用自己的小型 SQLite 工作库，不共享写入锁；先聚合文档来源，再做句子排序，避免把每个原始来源都展开到宽句子排序中。
5. 按桶序拼接最终 TXT/JSONL，保留原有确定性的哈希排序。输出全量 provenance，包括跨 split 移除的来源记录。

文档分组和最终结果组织仍有串行部分；句子重新分桶、去重与导出是多进程。控制台分别显示阶段、完成分区/桶数量和时间。实际耗时记录在 `merge-timing.json`。

默认 64 桶、4 个进程。当前机器可以先用 8 个进程。`--cache-mb` 是每个 SQLite 连接的缓存预算；桶进程有两个连接，因此 8 进程、128MB 时数据库缓存预算约 2GB，此外还有 Python/Arrow、排序和系统文件缓存。文档分组暂时还需要保存全局哈希与 ID/URL 映射，百万文档级别应预留数 GB 内存。它不是一个严格的内存上限。磁盘出现竞争时减少 workers；不建议机械地开到 CPU 数上限。

缓存参数的语义见 [SQLite cache_size 文档](https://www.sqlite.org/pragma.html#pragma_cache_size)；分桶文件使用 [PyArrow ParquetWriter](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html)。

## 输出与兼容性

`train/validation/test.txt`、对应 `.jsonl`、`documents.jsonl`、`provenance.jsonl`、三个 audit JSONL 和 `stats.json` 保持原有格式和语义，可继续用于审核、SentencePiece 和 token 文件生成。

**新版 `index.sqlite` 只保存 `origins`、`contents`、`annotations` 三张文档/审核索引表，不含 `units`、`winners`、`parents`、`aliases`。** 现有 `prepare_corpus_review.py` 读取的字段仍然兼容；直接针对旧 `units`/`winners` 表的自定义 SQL 需要改读句子 JSONL 或 provenance。这个变化已写入 manifest 的 `index_format`。句子数据不再重复写进一个全局大库。

工作目录保留 Parquet 桶及分桶导出文件，便于核查；成功桶的临时 SQLite 文件自动关闭并移除。**全流程仍有中间文件和完整 JSONL 来源开销**，全局索引变小不代表总磁盘只剩 TXT 文件大小。确认最终结果后，可由你自行清理工作目录。

验证结果：复用了现有 20,000 文档、232,994 条唯一句子的 worker 产物，4 进程/16 桶耗时约 8 秒；11 个 TXT/JSONL 文件与旧汇总输出 SHA256 全部相同，`stats.json` 也相同。这批样本的文档索引从约 213MB 降到约 33MB。**这不是全量耗时承诺**；全量数据还受文档映射大小、磁盘读写和 provenance 输出量影响。

回归测试另外覆盖跨分区重复文档、跨来源句子、ID/URL 传递分组、train/validation/test 之间重复句子、校准文本强制 train、标注、空桶、不同进程/桶数、缺少原始输入和拒绝覆盖已有目录。

后续审核准备时指定新 corpus：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u src/vimeml/data/prepare_corpus_review.py `
  --corpus outputs/corpus-fast-v1 --output outputs/corpus-fast-review-v1 --workers 8
```
