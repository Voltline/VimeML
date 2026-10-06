# 操作流程

## 1. 准备 review 和抽检材料

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/review/prepare.py --workers 8
```

默认读取 `outputs/corpus-fast-v1/`，写入 `outputs/corpus-review-v1/`。它并行扫描、分层选样，并按原始文件重放相同规则和批准标注；不请求 API。预计包含全部 638 条 review 目标及约 700 条分层抽检，重叠目标合并，最终数量以 `stats.json` 为准。

抽检覆盖保留文本、已合并换行、疑似漏合并、排除片段、明确 drop 与特殊文本。诊断抽检使用 train；全量 review 可能包含 held-out 文档，若后续用于调规则，必须在新版本隔离相关组及重复文本。这个样本集不是全库均匀抽样。

## 2. 先查看 API 批次估算

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/review/run.py --dry-run
```

只校验材料、缓存和请求预算，不读取 key、不发送请求。超预算样本会明确列为待处理，不能当作审核完成。

## 3. 确认后才运行审核

在当前终端设置 `SJTU_API_KEY` 后：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/review/run.py --workers 5 --batch-size 10
```

默认 `deepseek-chat`，所有请求和重试共享 8 RPM / 80,000 TPM 限流，给学校额度留余量。API 只提供建议，不润色、补写或自动恢复语料。模型结果不自动写入审批台账。

结果在 `outputs/corpus-review-v1/review/`：`issues.md`、`results.json`、`pending.json`、统计、原始回复与缓存。失败后同一命令续跑；成功案例不会重复请求。鉴权失败停止后续发送，429 按服务端要求暂停。输入或提示版本变化时使用新输出目录。

核对所有 issue/uncertain，另随机核对至少 30 条 ok；同类问题集中出现时补查相关来源。规则隔离块不能仅凭模型 keep 建议恢复，修复需明确证据和新版本。`scripts/review/approve.py` 只支持明确选中的边界/片段审批；使用前查看 `--help`，不要批量自动批准。

### 多账号续跑

`scripts/review/run_multi_key.py` 使用独立账号的 key，共享剩余样本队列；每个账号默认 3 个并发、8 RPM / 80,000 TPM，重试也计入各自额度。默认从 `SJTU_API_KEY`、`SJTU_API_KEY_2` 读取，密钥不写进文件或命令行参数。更多账号用 `--key-env ENV_NAME ...` 指定环境变量名称。

切换前在旧审核窗口按 Ctrl+C，等待 Python 返回终端提示符。旧版本已启动的进程没有新的输出锁，不要同时运行两个审核进程。切换后复用相同的 `review/` 目录，已完成的案例和响应缓存仍有效。

```powershell
# 当前窗口没有第一把 key 时才执行这条
$env:SJTU_API_KEY = [System.Net.NetworkCredential]::new("", (Read-Host "第一把 SJTU key" -AsSecureString)).Password
$env:SJTU_API_KEY_2 = [System.Net.NetworkCredential]::new("", (Read-Host "第二把 SJTU key" -AsSecureString)).Password

# dry-run 不读取任何密钥，不写文件或请求 API
.\.venv\Scripts\python.exe -X utf8 scripts/review/run_multi_key.py --dry-run
.\.venv\Scripts\python.exe -X utf8 -u scripts/review/run_multi_key.py --workers-per-key 3
```

统计新增 `accounts_this_run`，记录各账号的请求数、成功数、429 和服务端报告的 token 消耗；日志只保存环境变量名称，不保存密钥。401/403 禁用本次运行中对应账号，当前批次可转交其他账号；429 只暂停对应账号。全部账号失效时保留待完成案例并退出。失败或中断后以同一命令续跑即可。两份独立额度的 dry-run 理论下限按合计额度计算，不代表服务端响应时间或实际耗时。

## 4. 验收后再进入 tokenizer

先完成必要的近重复、跨 split 泄漏检查和 corpus 冻结，再确认训练抽样量与资源。`configs/tokenizer.toml` 只是 16K unigram 模板，当前实现会读取所有 train 行；不要未经规模规划直接启动全量训练。

后续入口已整理好：`scripts/tokenizer/train.py`、`scripts/tokenizer/encode.py`。它们不会被其他整理/审核命令自动调用。旧 pilot 参数保存在 `legacy/configs/tokenizer-pilot.toml`，已有 pilot 产物原地保留。

TokenStore 的新导入路径为 `vimeml.tokenizer.store.TokenStore`；token 文件是小端 uint16，句子边界 offsets 是小端 uint64，单条序列含 BOS/text/EOS。后续 PyTorch DataLoader、Transformer 和训练调试继续逐步实现。

## 其他入口

| 入口 | 用途 |
| --- | --- |
| `scripts/corpus/build.py` | 多进程清洗后自动并行合并；使用新输出目录，避免重做当前已完成语料 |
| `scripts/corpus/preprocess.py` | 只处理 `--part INDEX/COUNT` 指定分区，可用于多台机器 |
| `scripts/corpus/merge.py` | 合并现有完成分区，指定 `--parts` 或 `--parts-dir`、`--output`；支持进程数和桶数 |
| `scripts/tools/inspect_data.py` | 查看本地原始数据 schema、数量和样本 |
| `scripts/tools/check_corpus.py` | 根据保存的完整性报告快速检查当前 corpus；`--full` 才重新读所有导出并计算 SHA256 |

所有语料构建和审核保留来源、文档分组、span、标注和版本信息。最终 SQLite 是文档/审核索引，不含完整 `units`/`winners` 表；句子位置读取 split JSONL 与 provenance。历史原始审批材料在 `outputs/history/`，不要当作当前 review 输出。
