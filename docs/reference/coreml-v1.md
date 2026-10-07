# V1 Core ML 实测（2026-10-06）

V1为7,386,624参数，以下仅为原版本的Mac、宿主与键盘扩展记录。V2.1没有完成Core ML验收。转换入口与命令见[Core ML指南](../coreml.md)，原始包与失败结果保留。

## 质量

冻结开发集137条、AJIMEE200条，纯LM重排：

| 检查 | INT8结果与范围 |
| --- | --- |
| 严格FP32 logits | **失败**，最大绝对差约3.66；没有将门槛改宽后声称通过 |
| 候选fixture | 排序相同，sum最大差约0.129；末位argmax相同 |
| PAD / causal | PAD最大差约2.81e-5、超容差0；causal差0 |
| 开发集 | Top-1 122/137、Top-5 134/137；相对Windows FP32首选变化0、完整顺序变化119条；相对conservative源包为118条 |
| AJIMEE | Top-1 124/200、Top-5 151/200；首选变化2条，均为可接受表记；完整顺序变化163条 |
| 分数差 | 相对Windows FP32，sum最大差开发集约1.985、AJIMEE约2.693；开发集相对conservative源包约1.995。不能称作逐分数一致 |
| 联想 | 20前缀中16个首选与**未压缩conservative Core ML**对照相同；仍有重复、跑题等质量问题 |

分布诊断使用AJIMEE左文＋第一个可接受答案，内容token截至127后加BOS。下表是**先逐样例对位置求均值，再对样例求均值**，不是全库按token数加权；KL和首选一致率包含最后位置，NLL不含最后位置。旧报告未完整记录截断数量与输入哈希，新脚本补充这些字段，原始JSON不改写。

| 版本 | KL(FP32‖模型) | NLL（FP32约5.567） | 各位置首选一致率 | 包体积 |
| --- | ---: | ---: | ---: | ---: |
| FP16计算v2 | 0.0003 | 5.568 | 98.7% | 见包manifest |
| INT8 block32 | 0.0059 | 5.573 | 94.4% | 8.08MB |
| PyTorch模拟INT8 | 0.0060 | — | — | 诊断对照 |
| 4bit palette g16 | 1.316 | 6.944 | 33.0% | 3.94MB |

INT8的聚合KL接近模拟结果，支持量化是主要误差来源，但不能证明转换没有额外误差；模拟scale与Core ML实际存储也非逐位等价。严格logits与概率分布衡量不同性质，分布更接近不能消除PAD、causal和排序复测要求。测试只支持当前冻结候选池上的有限接入结论。

原始报告在 `outputs/deployment/conservative-int8-b32-{alignment,dev,ajimee,phrases,timing,distribution,release-review}-v1/`。开发集原报告的comparison是对未压缩conservative源包，AJIMEE是对Windows FP32；合并时又从保存的逐候选分数重算了开发集对Windows FP32的差异，详见 [合并校验](../reports/mac-20261006/merge-verification.md)。未压缩对照为 `tiny-ja-v1-conservative-v3`，16.57MB。

## 性能与设备范围

Mac Python CPU_ONLY报告：reranking411次，p50 18.2ms、p95 21.9ms；beam联想60次，p50 53.8ms、p95 57.3ms；加载约95ms，可能命中编译缓存。

Mac客户端文档记录的iPhone16 Pro Max优化构建、CPU_ONLY、**测试宿主App进程**：加载首次80ms/再次16ms；T16/T64/T128预测0.67/1.8/3.0ms；约14候选reranking p50 10.4ms/p95 12.7ms；下一词p50 5.8ms；beam8×8 p50 44.5ms。宿主加载增量约24.2MB、宿主峰值74.8MB。这些不是键盘扩展的延迟或内存峰值，Windows未复跑其运行时。

真实键盘扩展有约152.7秒手动输入记录，操作记录确认LM排序和下一词开启。physical footprint采样峰值21.25MiB，私有驻留采样峰值51.922MiB；两者不可混用。约1秒采样会漏掉短峰值，未取得安装构建／模型哈希，末段仍有缓慢增长。见 [真实扩展报告](../reports/mac-20261006/iphone-keyboard-memory.md) 和 [模拟器报告](../reports/mac-20261006/simulator-memory.md)。加载增量不能直接证明CPU内部权重布局，也不能推定INT8一定不降低运行内存。

长期验收仍缺固定构建下的10～15分钟重复输入记录，范围包括冷启动、reranking、下一词、取消、切换宿主与终止。

## GPU与ANE实验

当前INT8 FP32计算包的Maccompute plan为119/119算子在CPU。客户端记录GPU/ALL触发 `MPSNDArrayQuantizedGatherND` 动态形状断言，CPU_AND_NE也未把当前包移至ANE。

独立实验入口为 `coreml_ane.py`（FP16计算、16/32/64/128档位及每通道INT8）、`compute_plan.py`、`ane_output_probe.py`（缩小输出的原型）。这些不是当前部署接口；输入需补到下一档并重新验证有效位置。Mac实验中ANE每次调用比CPU慢约3～4倍，缩小输出也未改善；调度与搬运是可能原因，现有数据未隔离证明瓶颈，也不代表所有iPhone或更大模型。功耗尚未测量。

FP16、4bit、ANE失败与放弃的路线均保留原包和报告。转换脚本后续有修改，旧manifest的脚本哈希仍代表当时版本，不更新为当前源码哈希。
