# 本地数据产物清点（2026-10-05）

**后续清理已执行完成。** 新版全量 corpus 通过索引完整性、输入覆盖、原始文件 SHA256、全部 TXT 的句数/字符数/排序及跨 split 精确重复检查，分桶导出与最终导出的 SHA256、行数一致。旧合并及控制进程已停止。下面列出的废弃目录共 22 个已删除，释放约 109.19 GiB。最终 corpus、原始数据、标注、审核依据和 pilot 产物保留。10 个 worker 的 manifest/统计及旧构建日志已保存到 `outputs/corpus-fast-v1/build-records/`。

完整校验记录：[integrity-check.json](../outputs/corpus-fast-v1/integrity-check.json)；实际删除名单：[cleanup-report.json](../outputs/corpus-fast-v1/cleanup-report.json)。以下是清理前的历史清点记录。

本次仅检查，未停止进程、未删除文件。根目录为 `D:\Sources\VimeML`。首次扫描约 108.10 GiB（116.07 GB）；运行中的目录仍在增长。表中大小是扫描快照，不是最终大小。

## 可以先删：约 18.24 GiB

| 目录（均在 outputs 下） | 大小 GiB | 用途 |
| --- | ---: | --- |
| corpus-full-staging-v1 | 14.118 | 已中断的旧单机全量产物；没有完成 manifest，当前无进程使用 |
| corpus-parallel-verification-v1 | 0.773 | 旧并行入口的 2 万文档验证副本 |
| corpus-parts-verification-v1 | 0.772 | 旧分区验证副本 |
| corpus-parts-verification-v2 | 0.772 | 第二套分区验证副本 |
| corpus-fast-verification-v1 | 0.416 | 新合并入口第一轮验证结果 |
| corpus-fast-verification-v2 | 0.416 | 新合并入口第二轮验证结果 |
| corpus-fast-verification-v1-work | 0.396 | 第一轮验证的分桶中间产物 |
| corpus-fast-verification-v2-work | 0.396 | 第二轮验证的分桶中间产物 |
| corpus-smoke | 0.054 | 旧 2,000 文档 smoke 结果 |
| corpus-smoke-02 | 0.055 | 旧 smoke 第二版 |
| corpus-smoke-03 | 0.055 | 旧 smoke 第三版；对应审核记录另存于 audit 目录 |
| corpus-fast-review-verification-v2 | 0.002 | 审核工具兼容性验证材料 |
| corpus-review-verification-v1 | 0.002 | 旧审核工具验证材料 |
| verification-deps | 0.005 | 早期测试临时依赖；当前 .venv 已有依赖 |

测试临时目录也可删：`parts-test-4kw6n08r`、`parts-test-hhk3e_ng`、`parts-test-m1tyn1fb`、`parts-test-mfpaz010`、`review-test-0waghswy`，合计不足 3MB。

这些旧结果均不参与当前全量合并。删 smoke 结果后，旧的恢复审核脚本默认目录将不存在；如果以后要重新跑历史 smoke 审核，需重建或显式指定其他 corpus。已有审批记录和校准材料不删除。

## 需要满足条件后再删

| 目录 | 扫描大小 GiB | 条件 |
| --- | ---: | --- |
| outputs/corpus-parallel-v1 | 24.820 | 旧 merge_parts.py 仍在运行。决定弃用旧合并并停止它后可删整个目录；不要连带删除下面的 worker 目录 |
| outputs/corpus-fast-v1-work | 34.616+ | 新并行合并正在使用，且继续增长。新版命令成功退出、manifest.json 的 status 为 complete、统计及必要检查通过后可删 |
| outputs/corpus-parallel-v1-work | 21.377 | 保存 10 个完成的 worker，是当前清洗成果。新版最终 corpus 验证完成、旧合并进程也停止后可删；删后无法直接复用这些 worker 重新合并 |

两套合并同时在运行，占用不同目录，但竞争同一块磁盘。此次清点没有替你停止其中任何一套。

新版工作目录包含分桶 Parquet、audit 副本、分桶 TXT/JSONL 导出和运行中临时 SQLite。分桶导出之后还会被拼接成最终文件，所以成功完成后工作目录可以整体释放。最终 manifest 中的 work_directory 是构建记录，审核和 tokenizer 不读取这个工作目录。

## 保留

- `datasets/`：FineWeb 约 2.261 GiB，Tatoeba 约 0.015 GiB，是唯一原始数据；后续审核也会回读。
- `outputs/corpus-fast-v1/`：当前新全量目标，扫描时 5.407 GiB，仍未完成，不能删。
- `annotations/`：审批标注与校准 ID。正式构建和版本验证需要。
- `outputs/review/`、`outputs/corpus-smoke-audit/`、`outputs/corpus-recovery-audit/`：历史请求、人工/助手审批及校准依据，现有历史配置仍引用，合计很小。
- `outputs/corpus-pilot/`（0.555 GiB）及 `artifacts/`（约 0.081 GiB）：已有 tokenizer 和 token-data 实验成果；当前配置及产物 manifest 仍指向 pilot。
- `outputs/corpus-integration-v1/`（0.576 GiB）：保留一份小规模对照语料即可，其余重复验证副本可删除。
- 代码、配置、文档、测试、`.git/`、`.venv/`。

空间主要由重复 SQLite、完整来源 JSONL 和工作目录里的导出副本占据，并非 2.4GB 原始 parquet 本身。以后保留原始数据、一份最终 corpus、小量审批材料和模型产物即可；中间副本应在最终结果验证完成后清理。
