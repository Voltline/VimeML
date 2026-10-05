# 本机并行 corpus builder

入口为 `src/vimeml/data/build_parallel.py`，配置为 `configs/corpus-parallel.toml`。运行在当前 Python 环境，没有新增依赖。

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/build_parallel.py --workers 10
```

机器当前有 20 个逻辑 CPU；十个输入文件可分给十个独立 Python 进程，九个进程分别处理一个 FineWeb shard，另一个处理 Tatoeba。也可以使用 `--workers 4` 或 `--workers 8`。省略参数时，自动取逻辑 CPU 数减一、输入数和 10 的最小值。按输入文件分工，进程数上限是当前输入文件数。

每个进程流式处理自己的输入并写入独立 SQLite 数据库，避免共享数据库写入锁。所有 worker 成功后，脚本自动执行一次全局文档分组、精确去重、split 和导出。最后这一步仍由一个进程完成，控制台会明确切换至 `[merge]` 阶段。

已有 worker 全部完成时，可使用[独立并行汇总入口](corpus-fast-merge.md)复用产物，跳过清洗。它写入新的输出目录，原来的自动汇总入口保持不变。

默认位置：

| 内容 | 位置 |
| --- | --- |
| 最终 corpus 与 `stats.json` | `outputs/corpus-parallel-v1/` |
| 独立 worker 结果 | `outputs/corpus-parallel-v1-work/parts/` |
| 每个 worker 及汇总日志 | `outputs/corpus-parallel-v1-work/logs/` |
| worker PID 和各阶段实际耗时 | `outputs/corpus-parallel-v1-work/run-summary.json` |

总进度每十秒打印一次。运行中的数量根据 worker 最近一次进度日志更新；配置每 5,000 条刷新，因此短暂偏低。worker 完成后采用其精确数量。

使用其他输出目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/build_parallel.py --workers 10 --output outputs/corpus-parallel-v2
```

中断后重试同一个并行任务：

```powershell
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/build_parallel.py --workers 10 --resume
```

保持 worker 数、输入、代码和标注一致。已完成 worker 校验后直接复用；未完成的 worker 从头处理自己的文件。未完成的目录会保留为 `*-interrupted-*`，不会被删除。若中断发生在汇总阶段，重试只重新汇总。已完成的最终 corpus 不会重复构建。

新入口不修改或接管旧 `build.py` 的进度。旧单机输出、旧中断目录不能作为并行任务的续跑输入。两版使用不同输出目录；若同时运行，会竞争 CPU 与磁盘，由你决定是否保留旧进程。

清洗仍然使用现有本地规则和四条精确审批标注，没有 API 请求。输出还是 staging，后续的质量审核、近重复检查和正式 tokenizer 训练保持独立。
