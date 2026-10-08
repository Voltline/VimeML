# 客户端变更后的模拟器与历史内存审计（2026-10-06）

本报告范围为模拟器压力测试与既有trace离线分析。同日另有iPhone16 Pro Max手动输入测量：LM排序和词联想开启，真实扩展footprint采样峰值21.25MiB，见[真机记录](iphone-keyboard-memory.md)。两组测量分别解释。

参考 Vime 当前工作区的 `VimeLanguageModel.swift`、`JapaneseCandidateWorker.swift`、`KeyboardSession.swift`、`KeyboardView.swift`、`KeyboardPreferences.swift`、原生测试和 `Docs/LanguageModel.md`。测量以当时客户端实现为基点，仅增加可选内存审计测试。

结论：在这组有界压力测试中没有观察到随候选评分、词联想或重复加载次数持续攀升的物理内存。模型、词典和框架存在常驻/预热开销。模拟器结果不能认证 iPhone 键盘扩展的内存预算；一次带联想场景曾发生候选等待超时，重跑通过，但原因未定位。

## 方法

优化的 Release、CPU_ONLY、arm64 iOS27.0 模拟器 `Vime Keyboard QA`，Xcode27.1。每个场景重新启动测试宿主进程，10ms后台采样 `task_vm_info.phys_footprint`，并读取进程生命周期物理峰值 `ledger_phys_footprint_peak`。单位为 MiB（1,048,576字节），不是 Activity Monitor 的全部 resident memory。记录 baseline、阶段终点、采样峰值和内核峰值。峰值包含测试宿主、UIKit、XCTest和框架，不是纯模型增量。

压力测试在模拟器执行，历史Instruments trace采用离线导出。

## 当前工作区的对照结果

| 场景 | 初始 | 80轮后（worker仍在） | 进程物理峰值 | 第20/40/60/80轮检查点的范围 |
| --- | ---: | ---: | ---: | ---: |
| 引擎排序、关闭联想 | 36.21 | 46.52 | 48.24 | 0.47 |
| LM排序、关闭联想 | 37.63 | 49.24 | 54.02 | 0.25 |
| LM排序＋下一个词联想（重跑） | 37.25 | 48.89 | 52.52 | 0.16 |

每组先运行一轮预热，再运行80轮「输入→候选发布→允许后台重排/纠错→选择完整候选→确定」，覆盖5种读音及长句；随后100次快速输入/退格/取消。最后释放测试 session/worker 并等待1秒，分别约43.19/44.00/43.66 MiB。

不同新进程的 baseline 和框架缓存不同，不能从表格推出「联想使内存下降」，也不能把两个场景相减当成精确模型增量。联想重跑的81轮中，69轮在确定后的100ms观察窗口内返回非空建议；其余轮可能无有效建议或仍未发布，未计作生成成功。

首次「LM＋联想」在第24轮候选等待超过5秒，测试失败，进程峰值62.77 MiB。失败后的诊断采集可能增加峰值，故与正常场景分别列出；不删除、不宣称首次通过。保持相同5秒标准重跑后测试通过，峰值52.52 MiB。没有定位该偶发超时的根因，真机验收需要继续关注请求延迟与取消。

## 单独模型压力测试

- 模型加载含资源SHA256校验和分词器：33.91 → 37.74 MiB；加载期间进程峰值47.71 MiB。这个模拟器增量不代表真机的权重存储精度。
- T8/T64/T128各100次推理：阶段末39.97/42.10/43.58 MiB。模型/工作区预热随长度增大，最高采样值约44.56 MiB。
- 100次候选联合评分：第20→100次检查点仅增加0.016 MiB。
- 100次 `nextWords`：第20→100次仅增加0.047 MiB。
- 20次beam生成：阶段末43.78 MiB，采样峰值44.08 MiB。
- 20次「加载→预测→释放」：释放后第一次36.58、第二十次37.25 MiB；第10→20次只增加0.047 MiB。期间进程峰值53.35 MiB。

逐候选/每个生成步骤的 autoreleasepool，以及 worker 缓存单个模型的实现，在本次工作负载下没有显示线性累积。有限次数的物理占用平台期不能排除所有泄漏；没有运行堆对象图或 Leaks 认证。

## 现有真机记录能说明什么

既有`outputs/deployment/iphone16promax-keyboard-memory-v1/keyboard-activity.trace` 中提取到146个 `VimeKeyboard` 进程采样，覆盖约150秒，物理占用24.00 → 29.95 MiB，采样峰值30.36 MiB。原始 Instruments 时间序列已存在，本轮只是离线导出。

该记录没有嵌入模型加载、设置、输入脚本和代码哈希的完整证据；另一个保存的日志trace中也未找到明确的模型加载消息。因此不能将30.36 MiB认定为「启用LM、重排、联想后的完整键盘峰值」。客户端`Docs/LanguageModel.md` 中宿主App加载增加约24 MiB、峰值74.8 MiB的记录，亦不是同口径的真实键盘扩展结果。

## 开关与缓存

当前工作区的排序模式和联想开关独立。`candidateRanking=.engine`只停止候选重排；`phraseSuggestions=true`仍可在确定后加载并调用LM。worker首次加载后持有 `cachedLanguageModel`，切开关/重置组合文本没有主动清空这个缓存。若需要「关闭LM后回收内存」的产品行为，应另行设计队列上的模型释放及再次加载策略，而不能只修改UI开关。

`sentenceContext`按字符限长，不能保证含emoji/byte fallback的输入一定少于128token；现有 `scores`/`nextWords` 的token长度检查会回退，未发现因此发生无界分配。保留这些检查。

## 文件与复现

- Vime：`Tests/UIKit/VimeLanguageModelMemoryTests.swift`，默认跳过，设置 `VIME_MEMORY_SCENARIO` 才运行。
- 场景：`direct`、`reload`、`engine`、`lm`、`lm_words`。每个场景单独进程，只选 `VimeLanguageModelMemoryTests/testProcessMemoryProfile`；采用Release。
- xcodebuild环境转发可使用 `TEST_RUNNER_VIME_MEMORY_SCENARIO=lm_words`；本轮为每个 `.xctestrun` 的测试目标写入同名环境变量并逐个运行。
- `outputs/deployment/memory-audit-v2/`：原始日志、xcresult、各场景JSON、失败与重跑JSON、历史trace导出摘要、源文件/模型指纹。
- `scripts/deployment/summarize_memory.py --directory outputs/deployment/memory-audit-v2 --output outputs/deployment/memory-audit-v2/summary-recheck.json`：生成汇总。

此前提出的真机验证清单：在 iPhone16 Pro Max 的真实键盘扩展中明确记录开关状态、LM成功加载、输入脚本、冷启动与连续操作的峰值physical footprint；单独比较引擎/LM/LM＋联想，并检查超时、宿主切换、前后台和系统终止记录。本轮不以模拟器或旧trace替代这项验收。
