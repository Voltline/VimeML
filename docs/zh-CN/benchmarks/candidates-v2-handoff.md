# 候选导出 v2（Mac 交接）

2026-10-21 准备，尚未在 Mac 上运行。包为 `artifacts/benchmarks/ime-candidates-v2-handoff.zip`，由 [`build_candidate_handoff_v2.py`](../../../scripts/benchmarks/build_candidate_handoff_v2.py) `package` 生成；Mac 端运行 `bash run_candidates_mac.sh <converter checkout>`，返回 `ime-candidates-v2-azookey-results.zip` 后用 `import --archive` 导入。导出不运行任何语言模型；converter、字典、typo/Zenzai/stable 设置与此前完全相同，只改变 `--config_n_best`。

| 轨道 | 题数 | n-best | 目的 |
| --- | ---: | --- | --- |
| [Standard v2 blind](standard-ime-v2-blind.md) | 729 | 20、50 | 新确认集合；覆盖率与评分一起打开 |
| [同音词挑战](homophone-ime.md) | 619 | 20、50 | 书面语语义 |
| Standard v1 development | 728 | 20、50 | n50 能找回多少池外题；n20 与原导出比对，检验确定性（输入、左文、答案与原导出逐条一致） |
| v1 development 空左文 | 728 | 20 | 候选池是否依赖左文（[分段输入](incremental-ime.md)自由运行的前提） |
| [分段输入原型](incremental-ime.md) | 2,000 | 20、50 | 段级协议原型 |

导入时逐个（轨道, n-best）核对返回的输入与映射和包内文件逐字节一致，校验 converter、字典版本、flags 和完成标记，再按标准导出格式加载；加载器从 manifest 读取 n-best（原导出默认 20，不受影响）。

n-best 50 能找回多少 Standard 池外题，先在 v1 development 上测量（可以读参考答案）；v2 blind 与同音词集合的主池仍为 n20，n50 并列报告，规则已写入 [指标 v2](metrics-v2.md)。
