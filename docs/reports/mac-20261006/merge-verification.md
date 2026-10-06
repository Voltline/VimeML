# Mac 快照合并校验（2026-10-06）

来源为仓库根目录的 `VimeML-fromMac/`。合并采用逐文件比对与SHA256校验，保留Windows已有基线；不会训练模型、执行Core ML转换或连接设备。文档按当前指南、详细参考、实测报告和历史计划分层。

## 保存范围

| 检查 | 结果 |
| --- | --- |
| Mac `src/` 对比 | 全部51个文件一致，原训练、数据、评分、推理及部署包核心未改写 |
| 交接文件校验 | SHA256SUMS列出的7个文件通过 |
| ZIP内部逐文件校验 | source286、results4923、training-inputs38、Vime-client293条全部通过 |
| 逐文件保存 | 新保存4367条，复用或另存888条；89条仅为可重建缓存或OS元数据 |
| 冲突与退役产物 | 另存 `outputs/history/mac-import-20261006/`，不覆盖正式版本 |
| 原始文档与改过的导入脚本 | 另存 `outputs/maintenance/mac-merge-20261006/mac-source/` |
| 另一个Vime仓库 | ZIP独立保存在 `handoff/mac-20261006-v1/`，没有覆盖VimeML源码或修改外部仓库 |

原始Mac快照ZIP、结果、失败实验、首次内存测试失败和真机trace均保留在本地产物目录，Git只提交代码、测试、文档和iOS参考。5255条保存记录最终哈希复核通过后，已删除 `VimeML-fromMac/`；仅删除导入副本。详细逐文件记录为 `outputs/maintenance/mac-merge-20261006/merge-report.json` 和 `final-preservation-verification.json`。

## 冻结身份

```text
best.pt                  93b6139aa05031df02038efcaf26788bc9916712919f3d2a4c4f398615cef75b
tokenizer.model          d9f1ba1e456ce72dd9c06a62b12e804cf14090c30179e6078cd51d03063b2982
token manifest           c7ebf3672d46cfc75d12013c34f35fb7ac9eac613f29f3454be0cb0e991854dc
inference manifest       4284cdc5491513cc817633081677b363b14f59c0b4ffb78cdf7fcfb221f71b3c
INT8 Core ML manifest    a9d516233fa57aaadd09477e2e909760972ff49b5577919f8a441c489f5c2245
```

正式checkpoint、tokenizer、token manifest、推理包全部文件及LF核心指纹均通过校验。6个Core ML实验包按原manifest核对所有包文件、尺寸和bundle来源。原始manifest不重写；旧转换脚本哈希可能不同于后来修改的工具版本。

## 排序结果重算与文档纠正

从原 `scores.jsonl` 用冻结评分汇总代码重算，开发集和AJIMEE的metrics均与原报告完全一致，候选、上下文、答案、token目标与对应冻结输入一致。未在Windows执行Core ML预测。

| INT8相对对照 | 完整排序变化 | 首选变化 | sum分数最大绝对差 |
| --- | ---: | ---: | ---: |
| 开发集 vs conservative Core ML（Mac原报告） | 118 | 0 | 1.994759 |
| 开发集 vs Windows FP32（本次新增离线核对） | 119 | 0 | 1.985330 |
| AJIMEE vs Windows FP32（Mac原报告） | 163 | 2 | 2.692832 |

Mac文档曾把上述开发集对照统称FP32，本次已修正。开发集Top-1=122/137、Top-5=134/137，AJIMEE Top-1=124/200、Top-5=151/200，均与Windows FP32命中数相同。AJIMEE两条首选变化在可接受表记之间，不能由此声称完整排序不变。联想16/20的首选一致是对未压缩conservative包，亦已注明。

严格INT8 logits对齐失败依然保留。概率分布统计是按样例平均，不是全库token加权；模拟INT8与Core ML的聚合KL接近不能证明转换完全无误。已测palette4 g16质量差不代表所有4bit方法。宿主内存增量不能直接证明内部反量化布局，ANE瓶颈解释保留为假设。

## 内存与运行证据

从原始模拟器场景JSON重新生成汇总，与已保存summary完全一致；从真机sysmon XML重建曲线，各进程完整采样和汇总与原summary完全一致。真实扩展127个样本，physical footprint采样峰值21.25MiB，私有驻留采样峰值51.922MiB。保留未固定安装构建／模型哈希、约1秒采样和短时长的限制，不能替代长期设备验收。

宿主App性能数据在 `Vime-client.zip` 内 `Vime/Docs/LanguageModel.md` 中交叉核对；它与键盘扩展的实测范围分开呈现。

## 工具修订与回归

新增转换／压缩适配与测试放在独立脚本，不改变冻结 `src/`。合并修订包括：ANE原型改为显式main及新路径保护；分布诊断补模型／输入身份、截断和平均方式；报告拒绝覆盖已有输出；iOS打包说明标明历史集成快照；handoff按流读取大文件并更新文档路径。

Windows完整Python回归共124项：123项通过，1项Mac专用Core ML测试跳过。本地HTTP测试受沙箱localhost限制，已在沙箱外复跑完整回归通过。Core ML和Swift/iPhone运行时未在Windows重测。原创代码与文档的暂存差异空白检查通过；SentencePiece vendor中原有的生成代码与许可文本空白按原始字节保留。具体计数和最后状态见同目录 `verification.json` 及本地维护报告。
