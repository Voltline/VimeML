# 当前进度与下一步

截至 2026-10-06：全量清洗、全局分组、精确句子去重、split 和导出已完成。当前本地 corpus 是 `outputs/corpus-fast-v1/`，仍为 staging，尚未完成日语质量审核、近重复检查和正式 tokenizer 训练。

| 项目 | 数量 |
| --- | ---: |
| 原始来源记录 | 2,289,346 |
| 唯一文档 | 2,289,319 |
| 最终唯一句子 | 25,713,003 |
| 最终字符 | 1,103,360,011 |
| train / validation / test 句子 | 25,185,368 / 260,337 / 267,298 |
| 待 review 的规则隔离块 | 638 |
| 已匹配批准标注 | 4 / 4 |

全量导出校验已通过：TXT 的句数/字符数/排序、三个 split 间完全重复为零；最终导出与分桶产物 SHA256/行数一致；来源覆盖、原始输入 SHA256、SQLite 完整性和校准文档归属一致。记录在本地 `outputs/corpus-fast-v1/integrity-check.json`。这些是结构及精确重复检查，不替代日语质量审核或近重复检查。

旧合并已停止，旧库、worker、分桶及重复验证目录已删除，释放约 109.19 GiB。小量 worker manifest/统计和日志保存在最终 corpus 的 `build-records/`，实际删除记录在 `cleanup-report.json`。重新合并不能直接复用已删除的 worker；最终语料无需重做。

## 下一次从这里继续

在仓库根目录，先准备约 700 条分层抽检材料，加上全部 review 目标：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u src/vimeml/data/prepare_corpus_review.py `
  --corpus outputs/corpus-fast-v1 `
  --output outputs/corpus-review-v1 `
  --workers 8
```

这一步只读本地，不需要 API key。随后先估算请求批次：

```powershell
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/review_corpus_parallel.py `
  --input-dir outputs/corpus-review-v1 --dry-run
```

确认材料后，再设置当前终端的 `SJTU_API_KEY`，按[审核工具说明](corpus-review.md)运行正式审核。工具已实现共享限流、原始回复保存、失败重试和逐案例缓存；模型结果仅作为建议，不会自动修改 corpus 或批准标注。下一次工作先完成这一轮 review 与人工核对，再决定是否需要修复。

质量验收和必要的近重复/泄漏检查后，冻结 corpus，重新训练正式 SentencePiece，再生成 token 文件。现有 `artifacts/tokenizers/ja-unigram-16k-pilot/` 与 token-data 只是此前 pilot 成果，不是全量正式版。Transformer 训练仍在后续阶段。

## Git 与另一台机器

Git 保存代码、配置、测试、文档和少量审批标注；`datasets/` 的原始文件、`outputs/`、`artifacts/`、`.venv/` 与密钥均不随提交上传。另一台机器仅 clone 不会得到这份完成的语料，需要另外复制 `datasets/` 和 `outputs/corpus-fast-v1/` 才能继续审核。

数据环境依赖为 `requirements-data.txt`，tokenizer 另用 `requirements-tokenizer.txt`。现有回归测试用临时数据与模拟 API，不请求学校服务。
