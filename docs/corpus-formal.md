# 正式 corpus 的第一步：全量 staging

此前 `corpus-pilot`、16K tokenizer 和 token 文件验证了链路；正式数据尚未构建。现在先处理全部本地输入，之后做代表性质量抽查、近重复处理和版本冻结，再训练正式 tokenizer。当前不实现 Transformer。

## 目录与职责

| 路径 | 用途 |
| --- | --- |
| `datasets/` | 本地原始 parquet / TSV，只读 |
| `src/vimeml/data/clean.py` | Unicode、确定的噪声规则 |
| `src/vimeml/data/segment.py` | 保守换行恢复、句子切分 |
| `src/vimeml/data/annotations.py` | 验证并应用已审批的本地决定 |
| `src/vimeml/data/approve_audit.py` | 将明确选中的审核 case 写入标注台账，无 API 调用 |
| `annotations/approved-v1.jsonl` | 可追踪的审批记录，应随代码版本保存 |
| `configs/corpus-integration.toml` | 20,000 条原文的接入检查 |
| `configs/corpus-full-staging.toml` | 全部本地输入，两个 limit 都是 0 |
| `src/vimeml/data/build.py` | 流式读取、SQLite 索引、分组、精确去重、划分和导出 |

## 清洗与标注契约

默认先执行规则，DeepSeek 的回复只提供建议。只有 `status=approved` 的本地标注进入 builder；它必须带审批者、理由、文档 NFC 哈希、精确清洗文本和原文位置。

支持三种目标：

- `block`：把明确的噪声块排除或隔离。规则已经隔离/删除的块不能用 `keep` 越过保护。
- `boundary`：确认相邻原始行合并或分开；不跨空行、被排除的行、列表边界、完整句末或长度限制。
- `fragment`：确认特定无句号表达完整后恢复；必须同时匹配文字和清洗块内的 Unicode 位置。括号异常等旗标仍阻止恢复。

初始台账只包含核对过上下文的 `F01 / F02 / F09 / F16` 四个片段。八个可信合并候选尚待核对完整前后链条；只补一条边界可能继续留下被截断的句子，不能直接全部审批。

后续手动审批示例（先读原文上下文；不要重复审批已在台账中的 ID）：

```powershell
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/approve_audit.py `
  --results outputs/corpus-recovery-audit/review/results.json `
  --output annotations/approved-v1.jsonl `
  --ids B02 --action join --reason "核对完整前后行链条后的具体理由"
```

台账 ID、目标重复或格式不合法会报错，原台账不被替换。审批工具不表示整篇文档已经通过语义质量审核。改变台账后必须输出一个新的 corpus 版本。

## 运行

在仓库根目录执行；现有环境的 pyarrow 25.0.1 可直接运行，无新增依赖。

```powershell
# 已有集成输出时无需重跑。需要新的对照时指定新的 --output。
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/build.py --config configs/corpus-integration.toml

# 集成检查通过后，读取全部本地数据。
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/build.py --config configs/corpus-full-staging.toml
```

全量配置读取本地九个 FineWeb2 shard 及全部 Tatoeba，预计 2,289,346 个原文来源记录。运行过程中不访问网络、不调用 DeepSeek。每 5,000 个原文来源记录输出进度并提交 SQLite 事务。后续分组、去重和导出阶段仍会继续运行。

输出目录必须为空或不存在；脚本不覆盖旧版，也暂不支持中断续跑。中断的目录不是完成的 corpus；保留它供排查，重新运行时用新的 `--output outputs/corpus-full-staging-v2`。不要同时修改本次使用的代码、输入和标注文件；完成前会检查这些文件是否变化。

## 导出与验收

`train / validation / test.txt` 为一行一个训练单位；同名 JSONL 增加来源、文档分组、清洗块位置和 `approved_annotation_ids`。`lm_reviewed=false` 表示没有断言整条语料已通过 LLM 质量审核。

`documents.jsonl` 保留全部来源，`provenance.jsonl` 保留精确重复句的所有出现位置及跨 split 排除记录。`index.sqlite` 支持后续抽样和去重。`fragments / dropped_blocks / review_blocks.jsonl` 保留排除与隔离记录。它们可能同时包含规则阶段和标注阶段记录，应按 `reason` 区分。

原文 NFC 哈希、来源 ID 和规范化 URL 形成传递分组。分组后按 seed 42 划分 98% / 1% / 1%，精确重复句只在一个 split 导出；一般优先 test、validation、train。已用于调规则/审批的文档组固定在 train，其中出现的相同句子也不会通过其他文档进入 validation/test。近重复检测尚未实现，不能据此声称完全无泄漏。

检查 `stats.json`：

- 全量 `document_origins` 与预期输入数量一致。
- `annotation_records=4`、`matched_annotations=4`、`unmatched_annotations=0`，`approved_fragment_recoveries=4`（台账未扩展时）。
- `keep/drop/review_blocks` 是规则阶段计数；`effective_block_actions` 是应用标注后的计数。
- 记录各 split、来源、字符数、片段和重复排除数量。token 场景估算不能替代正式 tokenizer 的实测。

完成的 `manifest.json` 保存有效配置、代码/输入哈希和阶段标识；只有成功结束才出现 `status=complete`。`ready_for_lm_training=false` 提醒这一版仍是 staging。

全量构建完成后再做：按 shard/来源、长度和保留/排除类型抽查，审视日语比例与 Web 噪声；检查近重复和 split 分组；冻结 corpus。随后只从正式 train 选取覆盖各来源的 tokenizer 训练样本，并在 validation 上衡量切分质量。pilot tokenizer 保留用于对照，不作为已冻结的正式词表。

本地回归检查：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```
