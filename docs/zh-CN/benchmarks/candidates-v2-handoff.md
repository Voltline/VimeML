# 候选导出 v2（Mac 交接）

2026-10-21 准备并在 Mac 上完成导出（Swift 6.4，converter 与字典版本符合要求），9 个（轨道, n-best）全部导入，回传包 sha256 `aef93ac5…`。交付的包中 `tracks.tsv` 为 CRLF 换行，Mac 端读到 `n_best="20\r"` 时 CliTool 报错退出；运行方只把包副本中的 `tracks.tsv` 转为 LF 后重跑，导入时逐字节核对的输入文件未改动。打包脚本此后以 LF 写出该文件，导出脚本也会去掉行尾的 `\r`。包为 `artifacts/benchmarks/ime-candidates-v2-handoff.zip`，由 [`build_candidate_handoff_v2.py`](../../../scripts/benchmarks/build_candidate_handoff_v2.py) `package` 生成；Mac 端运行 `bash run_candidates_mac.sh <converter checkout>`，返回 `ime-candidates-v2-azookey-results.zip` 后用 `import --archive` 导入。导出不运行任何语言模型；converter、字典、typo/Zenzai/stable 设置与此前完全相同，只改变 `--config_n_best`。

| 轨道 | 题数 | n-best | 目的 |
| --- | ---: | --- | --- |
| [Standard v2 blind](standard-ime-v2-blind.md) | 729 | 20、50 | 新确认集合；覆盖率与评分一起打开 |
| [同音词挑战](homophone-ime.md) | 619 | 20、50 | 书面语语义 |
| Standard v1 development | 728 | 20、50 | n50 能找回多少池外题；n20 与原导出比对，检验确定性（输入、左文、答案与原导出逐条一致） |
| v1 development 空左文 | 728 | 20 | 候选池是否依赖左文（[分段输入](incremental-ime.md)自由运行的前提） |
| [分段输入原型](incremental-ime.md) | 2,000 | 20、50 | 段级协议原型 |

导入时逐个（轨道, n-best）核对返回的输入与映射和包内文件逐字节一致，校验 converter、字典版本、flags 和完成标记，再按标准导出格式加载；加载器从 manifest 读取 n-best（原导出默认 20，不受影响）。

n-best 50 能找回多少 Standard 池外题，先在 v1 development 上测量（可以读参考答案）；v2 blind 与同音词集合的主池仍为 n20，n50 并列报告，规则已写入 [指标 v2](metrics-v2.md)。

## 开发集结果（2026-10-21）

只读 v1 development 与分段原型（由 development 派生）；v2 blind 与同音词集合未读取、未计算覆盖率。脚本 [`evaluate_candidate_pools.py`](../../../scripts/benchmarks/evaluate_candidate_pools.py)，输出在 `outputs/candidates-v2-dev-20261021-r2/`（本地）。

**候选池**

- 确定性：n20 重导出与原导出 728/728 输入一致、候选列表逐条一致。
- 左文：407 道有左文的题清空左文后，候选列表 407/407 完全一致。Zenzai 关闭时 AzooKey 的池不依赖左文，分段输入自由运行只需按读音导出一次。
- n50：池内题 630 → 652（+22：WRIME 16、JMultiWOZ 6），没有题掉出池；平均池大小 28.8（不足 50 的读音很多）。725/728 题 n20 池是 n50 池的子集，但 33 题 AzooKey 首位改变，新候选会插入前 20。
- 分段原型：池内段 n20 1,913/2,000，n50 1,941/2,000。

**重排（FP32，与冻结对比相同的打分）**

n20 一列为已记录的打分；n50 为同一 checkpoint 对 n50 池重新打分。共享候选的分数在每题内只差一个常数（题内最大离散 6.7e-5，共享候选的相对顺序 728/728 不变）：分数从所有候选的公共前缀之后开始累加，n50 新候选缩短公共前缀的少数题（V2.1 5 题、V4.2 3 题）整体平移，不影响排序。

| 模型 | Exact n20 → n50 | 净变化（得/失） | 聚类符号翻转 p | 聚类 bootstrap 95% | 宽松 n20 → n50 | 净变化（得/失） | 聚类符号翻转 p | 聚类 bootstrap 95% |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| V2.1 | 505 → 499 | −6（11/17） | 0.37 | [−17, 4] | 578 → 588 | +10（14/4） | 0.051 | [1, 19] |
| V4.2 epoch 20 | 523 → 524 | +1（12/11） | 1.00 | [−8, 9] | 594 → 603 | +9（14/5） | 0.033 | [2, 17] |

错误类别迁移（n20 → n50）：

| 迁移 | V2.1 | V4.2 |
| --- | ---: | ---: |
| 池外 → 正确 | 11 | 12 |
| 池外 → 表记 / 表记（大写） | 4 | 6 |
| 池外 → 语义 | 7 | 4 |
| 正确 → 表记 / 表记（大写） | 15 | 9 |
| 正确 → 语义 | 2 | 2 |

结论：n50 找回的 22 道池外题约一半直接答对，其余多为表记差异；同时新候选里的表记变体会抢走原本答对的题。Exact Top-1 没有净收益，宽松 Top-1 约 +9～+10，但 p 值在 0.03～0.06 之间，仍属开发集观察。n50 扩大的主要是表记竞争，不是语义区分；在模型能稳定选对表记之前，扩大 n-best 不会提高 Exact 主指标。
