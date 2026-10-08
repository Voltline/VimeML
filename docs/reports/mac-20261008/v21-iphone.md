# V2.1 iPhone 宿主与真实键盘扩展（2026-10-08）

设备为 iPhone16 Pro Max / iOS27.2（24B5099f），Xcode27.1，签名 Release，CPU_ONLY，最低iOS18。
V2.1 extend5 step40000、INT8 block32，模型与 tokenizer 的身份见 [量化报告](v21-coreml.md)。
客户端基点为main `9dc3c929c33e24cef1400b00c5639e2f9477b619`，资源接入通过独立分支实现。

## 测试宿主

最初10项功能＋1项性能测试全部通过。新增关闭式扩展审计后，增量 test 构建曾遇到
OrderedCollections 链接失败；clean test 后10项功能再次通过（另1项临时控制测试通过）。
控制测试因 XCTest 使用的共享容器与实际 App 导出路径不一致被移除，改用实际 App
启动参数控制开关。之后 App 的控制/导出构建均成功；真实扩展中的审计也成功。
初次失败、重建和成功 xcresult / 日志分别保留，没有将失败改写成首次通过。

功能覆盖6755条原生分词 fixture、4组联合评分、20条beam、PAD/causal、取消、整池回退、
排序 metadata、下一词和 session。宿主性能表见量化报告，footprint 生命周期峰值59.64MiB。
185.54ms是首次测试进程加载，OS/Core ML缓存可能温热，不是重启设备后的完全冷启动。

## 真实扩展方法与身份

Instruments Activity Monitor：北京时间 **10:08:23.641—10:11:44.938**，201.297秒。
实际扩展 `Vime.app/PlugIns/VimeKeyboard.appex/VimeKeyboard`，PID17014。
164个有效sysmon采样，间隔中位数1.036秒。首次Wi-Fi连接失败，改用USB后
重新录制成功；两份trace分别保留。导出时Apple符号分析有dylib overlap警告，
sysmon表可完整解析，录制结束原因是正常达到时间限制。

手动输入时「智能排序＋下一词联想」均开启，确定候选后可见下一词建议，未观察到异常。
操作覆盖输入、选词、确定、退格和宿主切换；测试方案列有kisha、hashi、nihongo及
ashitanokaiginisankashimasu，实际输入与操作次数未逐项保存。

扩展内opt-in审计从UTC02:08:54开始，导出191.902秒、1920个100ms样本。
实际设置为candidate_ranking=languageModel、next_words=true。
两次成功加载均记录版本 `2.1-extend5-step40000-int8-b32-v1`：第5.557秒153ms，
第136.596秒28ms。复载原因未单独标记，不将其等同于进程重启。
测量构建的宿主与扩展执行文件SHA256另存于measured-build.json；资源身份复用manifest。
取回报告时只增加App的导出入口，未改变模型、tokenizer或采样实现。

## 内存（MiB）

| 口径 | 真实键盘扩展 |
| --- | ---: |
| Instruments footprint 采样峰值（约1秒） | 28.0018 |
| 扩展内 footprint 采样峰值（100ms） | 28.2987 |
| 内核进程生命周期 footprint 峰值 | 34.7206 |
| Instruments 私有驻留采样峰值 | 73.7813 |
| Instruments 共享驻留采样峰值 | 40.2813 |
| Instruments 总驻留采样峰值 | 192.7813 |

驻留值、私有驻留、physical footprint和内核生命周期峰值分别列出；不可互相替代。
宿主Vime PID16999的Instruments footprint采样峰值24.2831MiB（20个样本），
与扩展及先前XCTest宿主59.64MiB分开。

| trace窗口（秒） | 扩展样本数 | footprint中位数 | 采样峰值 |
| --- | ---: | ---: | ---: |
| 30—60 | 28 | 18.3299 | 19.5643 |
| 60—90 | 29 | 19.6268 | 20.0018 |
| 90—120 | 29 | 19.7518 | 20.1425 |
| 120—150 | 29 | 19.9550 | 20.9550 |
| 150—180 | 29 | 20.1737 | 27.6893 |
| 180—201 | 20 | 27.2675 | 28.0018 |

末段高于中段，且记录到第二次模型加载；没有足够证据将其归因于泄漏、缓存或宿主切换。
一个扩展PID、0个recently-died标记。此有限工作负载未显示进程重启，不代表完整系统终止审计，
也不认证长期平台期。采样和周期JSON导出有测量开销；不与旧V1不同负载直接作优化对照。

## 延迟

计时为真实扩展内部单调时钟，保存各阶段最近最多256个样本。
LM阶段包含取消/失败尝试，没有将其计为全部成功请求。

| 指标 | 样本 | 平均 / p95 / 最大（ms） |
| --- | ---: | ---: |
| LM候选评分 | 127 | 32.93 / 53.92 / 58.91 |
| LM下一词 | 39 | 29.05 / 38.11 / 50.34 |
| 基础候选请求到结果 | 256 | 17.86 / 59.37 / 75.82 |
| touchDown到视觉反馈 | 256 | 1.01 / 1.86 / 4.42 |
| 结果就绪到UI发布 | 256 | 42.49 / 8.79 / 1926.09 |

UI发布有约1.93秒长尾；未保存事件关联时间戳，无法确定是否与宿主切换或调度有关。
实际输入未感到明显异常，当前长尾可接受；最大值与未定位原因保留，p95与最大延迟分别统计。

## 结论与资料

V2模型、tokenizer、分词/评分契约、取消/回退和本次实际扩展工作负载已验证。
V2.1作为当前部署版本，V1保留显式回退。严格logits量化对齐失败、扩大草稿Top-1下降0.3个百分点、
末段内存增长和UI发布长尾均保留原测量。部署代码已合入主分支；总体结论见[V2总结](../v2-summary.md)。
审计开关由实际App成功写为false；每次测量最多240秒自动停止，不记录用户输入。

原始证据：`outputs/deployment/v21-iphone-keyboard-extension-v1/` 中的成功/失败trace、toc.xml、
sysmon-process.xml、summary.json、extension-audit-17014.json、独立摘要和measured-build.json。
宿主及重建资料位于v21-iphone-host-validation-v1、v21-iphone-final-build-validation-v1和
v21-build-diagnostics-v1。报告、模型和必要编译资源随结果ZIP迁移，客户端源码由独立PR合并。
