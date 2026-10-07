# V2.0 token store 与窗口索引

2026-10-07。冻结 corpus 用 V2 tokenizer 重新编码，8 workers、batch 4096，按原句顺序合并。编码、合并与新产物指纹用时 89.39 秒，索引构建约 1.42 秒。

| 项目 | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| 句数 | 25,185,368 | 260,337 | 267,298 |
| Stored tokens，含 BOS/EOS | 581,648,025 | 5,976,419 | 6,079,517 |
| Prediction pairs | 556,462,657 | 5,716,082 | 5,812,219 |
| Context-128 windows | 25,196,843 | 260,464 | 267,421 |
| 额外长句窗口 | 11,475 | 127 | 123 |

二进制产物 **1,624,528,997 bytes（1.513 GiB）**，窗口 NPZ 共 264,360 bytes。Token、字符、句子和来源计数与冻结 corpus 一致；序列为 BOS+完整句子+EOS，长句预测目标无重叠、无遗漏，后续窗口不补 BOS。

编码及索引使用 metadata 验收，复用原 corpus 指纹与 tokenizer 全量 roundtrip 记录；合并检查 offsets、大小和计数，新二进制指纹仅计算一次。首尾及相邻长句窗口完成结构抽查。

## 产物与复现

```text
artifacts/tokenizers/ja-unigram-16k-v2/
artifacts/token-data/corpus-v2-16k/
artifacts/training-data/corpus-v2-c128/
```

```powershell
.venv\Scripts\python.exe -X utf8 scripts/tokenizer/encode.py --tokenizer artifacts/tokenizers/ja-unigram-16k-v2 --output artifacts/token-data/NEW-VERSION --workers 8 --batch-size 4096 --verification metadata
```

训练直接读取二进制，不依赖原 20 GB corpus JSONL；`rows.bin` 仅用于回溯来源。batch=256 时每轮 98,426 updates，4 轮 393,704；prefix crop 后实际训练 token 单独计数。

记录：`outputs/history/v2-20261007-session/token-data-v2-20261007.log`、`outputs/history/v2-20261007-session/window-index-v2-20261007.log`、`outputs/model-checks/v2-phase-c.json`。
