# 3,000 条 IME 输入与 AzooKey 导出

2026-10-07。2,000 development + 1,000 blind 的输入及真实 Mac 候选已收集、导入；参考标签为双词典检查的草稿，blind 尚未 LM 计分。[开发集初评](expanded-ime-evaluation.md)记录结果。

| 数据项 | 数量 |
| --- | ---: |
| Development / blind | 2,000 / 1,000，来自 corpus validation / test |
| 有 / 无左文 | 2,160 / 840 |
| FineWeb / Tatoeba | 2,914 / 86 |
| 多参考表记 | 421 |
| 唯一读音、来源组、原句、URL | 各 3,000 |
| 参考联合分词最长序列（含 BOS） | V1 38 / V2 37 tokens |

种子 20261007；转换区间保留原文连续字符，边界由 UniDic/Sudachi 共同确认。独立区间和原句内读音一致，过滤 OOV、常见高风险多读音和已发现的噪声；左文不含转换答案。来源组、原句、读音和 URL 跨 split 不重复，排除原 200/137 条 query/context，采样不使用模型得分或候选是否命中。

上下文与读音长度分层已做；语义类别均衡和正式标签审核未完成。网页原句占多数，双词典一致不保证读音和表记完全正确，近重复审计有限。Corpus test 文本已用于盲集输入准备，未做 test BPC 或 blind LM 计分。召回失败保留在完整分母，covered subset 单独报告。

## 文件与复现

- 输入：`artifacts/benchmarks/ime-expanded-v21-final/`。
- 交接：`handoff/ime-v21-3000-20261007.zip`，含 NOTICE、输入、映射及导出脚本。
- 返回候选：`artifacts/benchmarks/ime-expanded-v21-candidates-v1/`。
- 验证：`handoff/ime-v21-3000-20261007/handoff-validation.json`。
- 工具：`prepare_expanded_ime.py`、`build_v21_handoff.py`、`import_expanded_ime.py`；读音实现为 `src/vimeml/benchmarks/readings.py`。

Mac 导出入口（交接包解压目录）：

```bash
bash export_azookey.sh /path/to/AzooKeyKanaKanjiConverter
```

固定 converter `d59a28e4c7ca049aef04f29a91eae9677a7753f2`，n-best20、typo/Zenzai/stable off。返回 development/blind 各自的候选、输入映射、版本、flags、日志和 `cli_exit=0`。当前返回包已完成验收。

逐条来源 URL 与 NOTICE 随包保留；许可来源见 [FineWeb2 Japanese 数据卡](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/raw/main/README.md)与 [Tatoeba 条款](https://tatoeba.org/en/terms_of_use)。
