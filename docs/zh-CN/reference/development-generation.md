# 开发数据生成与审核

V1开发集采用137条冻结样本，用于组合系数选择。本文记录生成、核验与版本化方法；API凭据从环境变量读取。

## DeepSeek 生成真实转换开发样本

入口 `scripts/benchmarks/generate_development.py`；`hybrid.py prepare-dev` 只转换现有 JSON，不生成数据。

```powershell
# 只检查计划 / 缓存，不读 key、不联网、不创建文件
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/generate_development.py --dry-run
# 本地设置 SJTU_API_KEY 后，用新输出目录运行新版本
.\.venv\Scripts\python.exe -X utf8 -u scripts/benchmarks/generate_development.py --output artifacts/benchmarks/ime-dev-generation-v3
```

默认 DeepSeek `deepseek-chat`，生成 200 条、10 个主题、每批 10 条，约一半有句内左文。另起盲核验请求，只给左文和读音，不给答案、理由或本地 LM 得分。默认同模型核验，可能有共同错误；可在首次实验时用 `--judge-model` 指定另一模型，所有通过标签仍需逐条复核。

system prompt / 生成指令为日语，context 与答案须日语，仅 reason_zh 为中文短结论。输入是转换用假名，助词 は/へ/を 写作 ハ/ヘ/ヲ；不是语音串。不能靠“全汉字”确定文本是中文；字符检查也不能证明读音与语义正确。

输入结构：

```json
[
  {
    "index": "dev-0001",
    "context_text": "川を渡るために、",
    "input": "ハシヲワタル",
    "expected_output": ["橋を渡る"],
    "reason_zh": "跨河，应为桥。"
  }
]
```

左文为已提交部分，input 为本次待转换的完整读音，不重叠；答案列出合理同读音表记。不同意思的同音词不自动视作等价答案，不伪造负例候选池。最终候选由 Mac AzooKey 导出，开发集不加入训练，不按 Tiny LM 失分筛选标签。

## 并发、续跑和输出

默认每账号 4 并发、10 RPM / 80K TPM，生成、核验和重试共享额度。两个独立账号可加 `--key-envs SJTU_API_KEY,SJTU_API_KEY_2`；不得填实际 key，同 key 拒绝启动。同账号其他进程没有共享限流，避免同时占用。

成功缓存按模型、endpoint、完整 payload 和版本复用；失败重跑相同命令，不改批次编号。更改数量、seed、提示词、模型或排除源后使用新目录。原始回复、错误与请求脱敏；鉴权/限流/截断失败保存 pending，不伪报整套完成。Ctrl+C 后等待在途请求退出，再续跑。

| 输出 | 用途 |
| --- | --- |
| `draft.json` | 尚未核验的生成草稿 |
| `ime-dev-draft.json` | AI 盲核验通过，仍需进一步审核 |
| `quarantine.json` / `excluded.json` | 歧义、错误、分歧、重复 |
| `review-pack.json` / `verification.json` | 全部核验依据 |
| `cache/` / `responses/` / `failures/` / `pending.json` | 续跑与原始记录 |
| `manifest.json` / `stats.json` | 冻结参数、hash 与实际分布 |

请求 200 条不保证 200 条合格，不无限补齐失败批次。字符与 JSON 校验通过不代表语言标签通过。生成 seed 不保证外部模型确定性；复现依赖冻结请求、回复与 hash。

## 少数批次失败时离线导出

生成v2共20批，最终采用18批成功草稿，余下两批未补齐。离线导出入口：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/export_cached_development.py --source artifacts/benchmarks/ime-dev-generation-v2 --output artifacts/benchmarks/ime-dev-review-new
```

只导出成功缓存，不抢救失败回复；保存 missing-jobs 与 unverified 标记，不改原目录。当前原始 snapshot 为 `ime-dev-review-v2/`。模型辅助逐条复核后保留 137/180，43 隔离，9 条删除读音不同的替代表记；记录在 `ime-dev-reviewed-v2/`。这是 AI 复核，未经过外部母语者裁定。

新审核同样不改原 input/context，不用同义不同读音掩盖错误；不确定专名和语境可隔离。审核后保存新版本 JSON 和决定，再声明 `--labels-reviewed` 转换输入。真实候选、选 λ 与固定测试见 [组合排序](hybrid-ranking.md)。

## 可选合成诊断工具

早期的 `scripts/benchmarks/generate.py` / `evaluate.py` 保留为扩展工具，不属于当前主评测链路。默认草稿为 400 条短语二选一与 155 条受控同音词样本；生成 DeepSeek、盲核验 Qwen，后续仍需抽检。它们不提供真实 AzooKey 候选，不测开放联想自然度。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/generate.py --dry-run
# 新实验，需 SJTU_API_KEY
.\.venv\Scripts\python.exe -X utf8 -u scripts/benchmarks/generate.py --output artifacts/benchmarks/ime-synthetic-v2
.\.venv\Scripts\python.exe -X utf8 -u scripts/benchmarks/evaluate.py --help
```

默认账号 3 并发、8 RPM / 80K TPM；独立多账号与缓存方式相同。输出 benchmark.jsonl、quarantine.jsonl、manual-audit.json、manifest 和回复缓存；盲核验未完成时不报告正式准确率。评测支持 `--benchmark` 指定新冻结目录，分别报告同音词和短语辨别准确率，不混作 Vime 线上准确率。旧 `ime-synthetic-v1/` 为未完成实验缓存，继续保留少量请求依据，不纳入当前基线。
