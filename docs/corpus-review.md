# Corpus 审核工具

对应 [review 与抽检计划](corpus-review-plan.md)。这两个工具不改原始数据、corpus 或标注台账。默认准备约 700 个不同的抽检目标，另加合并结果中全部 review 目标；类别重叠或可用样本不足时会报告实际数量。

## 1. 等合并成功，再准备材料

必须等合并进程成功退出，且 corpus 的 `manifest.json` 为 `status=complete`。在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/prepare_corpus_review.py `
  --corpus outputs/corpus-fast-v1 `
  --output outputs/corpus-review-v1 `
  --workers 8
```

这一步没有 API 请求，也不需要 key。依赖现有数据环境的 pyarrow 和 Python 标准库。

准备器按 JSONL 字节范围并行扫描，保留每层固定数量的候选，不把整库载入内存。随后按来源文件并行、按 parquet row group 批量读取选中原文，重放 corpus 构建时相同的规则和已批准标注，核对文档哈希及处理结果。

保留文本每个输入文件目标 30 条；换行、片段、drop、特殊文本按计划单独抽样。每层按文档哈希选样，同一文档/层只取一个稳定目标。漏合并检查采用两阶段诊断：先从无标点且无旗标的 fragments 中取有限文档池，再验证其相邻未合并的 keep 块；它不是对全部未合并边界的均匀抽样。短缺不会用 validation/test 填补，也不自动再次扩大原文扫描。

`review_all` 可包含 held-out 文档，材料保留 `assigned_split`。若将其用于调规则或恢复正文，需要在新版本中隔离对应文档组及重复句；准备器本身不修改 split。其余抽检只使用 train 文档，保留文本来自最终 train 导出。

输出包括 `cases.json`、`prompt.txt`、`stats.json` 和完成状态的 `manifest.json`。其中 `cases.json` 保存文档/文本位置、来源引用、原文邻行、处理结果和抽样层。过长上下文有明确截断标记；不会暗中截断正文后声称审核完整。

极高重复文档的来源引用最多内嵌 256 条，并保存总数和截断标记；全部来源仍可按 doc_hash 在原 corpus 的 SQLite / provenance 中追踪。

输出目录必须不存在或为空；准备中断后保留旧目录排查，用新的 `--output` 重跑。可用 `--sample-scale 0.1` 做小规模材料检查；它缩放抽检配额，但不缩减全量 review。

## 2. 先 dry-run，看批次数量

```powershell
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/review_corpus_parallel.py `
  --input-dir outputs/corpus-review-v1 `
  --dry-run
```

dry-run 不发送网络请求、不读取 API key，也不创建审核输出。它检查材料哈希和已有缓存，显示待审核数量、动态批次分布、超预算案例及配额估算。

token 数是离线估计，不是服务端 tokenizer 的测量值。估算预留全部最大输出，收到服务端 usage 后更新共享预算。默认每请求最多 10 个案例、预估输入加输出最多 10,000 tokens；因此上下文较长时一批会少于 10 条。

如有 `oversized_cases`，不要直接当作已审核。它们不会发送，正式运行会保留为 pending。可以明确增加 `--max-batch-tokens`（不超过 TPM 配额），或另行准备更适合判断的局部上下文。不能单纯删除关键原文证据来让请求变小。

## 3. 准备好后才开始 API 审核

在当前 PowerShell 中已设置 `SJTU_API_KEY` 后执行下面命令。不要把真实 key 写进脚本、配置或提交记录。

```powershell
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/review_corpus_parallel.py `
  --input-dir outputs/corpus-review-v1 `
  --workers 3 `
  --batch-size 10
```

默认学校端点与 `deepseek-chat`，3 个线程共享滚动限流器，8 RPM / 80,000 TPM，全部重试也消耗同一个额度。RPM 同时采用间隔发送，避免瞬间挤满窗口。它只控制本进程；同一个账号的其他脚本请求仍需一起考虑。

401/403 停止后续请求；已经在途的请求可能仍会返回。429 按 Retry-After 或退避等待，并暂停所有 worker 的后续发送。网络、解析和截断失败最多额外重试两次；失败案例保持 pending，命令以非零退出码结束。Ctrl+C 阻止新请求，在途请求最长仍可能等待到超时。

每次 HTTP 回复在解析前保存，包括错误、截断和非 JSON 回复。请求材料不含鉴权头，保存内容中的 key 会替换为 `[REDACTED]`。只接受 ID 恰好覆盖、字段合法、动作与目标类型兼容的整批结果，不把缺失回复当作成功。

## 输出与续跑

默认审核输出目录为材料目录下的 `review/`：

| 文件 | 内容 |
| --- | --- |
| `results.json` | 已完成判断与本地来源、原文证据 |
| `issues.json` / `issues.md` | 模型认为有问题或拿不准的案例 |
| `pending.json` | 尚未完成的 ID |
| `failures.json` | 本次执行的失败尝试；之后成功的尝试也可能在这里保留历史 |
| `stats.json` | 完成数量、分层判断、请求次数与服务端 usage |
| `responses/` | 每次请求的原始回复，跨续跑保留 |
| `cache/` / `decisions/` | 批次与逐案例缓存 |
| `run-manifest.json` | 材料、提示、模型、端点及本次运行参数 |

失败后使用同一命令、同一输出目录续跑，已完成案例从缓存恢复；即使减少 `--batch-size`，也不会重复付费审核已有逐案例结果。若材料、提示、端点或模型发生变化，必须使用新的 `--output-dir`，脚本会拒绝混用。

模型结果全部是建议，没有自动批准、修复或恢复。当前规则已隔离/删除的块不能简单用 keep 台账恢复；这类建议需要先设计精确、可追踪的修复，再生成新 corpus。报告按层统计，不把模型 issue 比例等同于全库坏数据率。先核对问题项和随机 ok 项，再决定是否通过本轮验收。

## 本地验证

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

测试使用临时 parquet/TSV 和模拟 API transport，不请求学校接口。覆盖并行选样可复现、UTF-8 分段不漏不重、已批准标注重放、train-only 抽检、材料版本检查、并发与共享限流、缓存续跑、限流/鉴权/畸形回复和 key 脱敏。
