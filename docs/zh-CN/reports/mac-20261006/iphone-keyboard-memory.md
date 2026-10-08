# iPhone 真正键盘扩展的内存记录（2026-10-06）

测量模型为冻结V1，实际输入体验与当前内存占用可接受。截图中的6.7→29.31→45.77MB为观察读数，与下表footprint口径不同；汇总JSON分别保留私有、共享和总驻留值。产物归档见[V1归档记录](handoff.md)。

使用Instruments Activity Monitor记录手动输入过程，设备为iPhone16 Pro Max / iOS27.2，项目最低iOS18。测量过程中App和模型保持不变。

运行时LM候选排序和词联想均开启，确定候选后可见下一词建议；设置状态由手动操作记录提供。本次没有成功加入 os_log/Core ML instrument，因此没有独立的模型加载日志；没有取得已安装构建和模型的哈希，不能宣称与当前源代码完全一致。

## 结果

录制时间为北京时间21:11:55.410～21:14:28.135，时长152.725秒。真正扩展 `Vime.app/PlugIns/VimeKeyboard.appex/VimeKeyboard`，PID6182，于录制第11.442秒出现，取得127个 sysmon 采样，末次为152.225秒。相邻采样中位间隔1.039秒。

| 指标 | Physical Memory Footprint（MiB） |
| --- | ---: |
| 扩展首次采样 | 11.70 |
| 扩展最低采样 | 11.49 |
| 扩展最高采样 | 21.25 |
| 扩展末次采样 | 21.22 |
| 首次到末次增加 | 9.52 |

最高采样出现在录制第139.782秒。已交叉核对 XML 原始格式值为21.25 MiB，同一行 Resident Size为167.97 MiB；这是两个不同指标，本文以 Physical Memory Footprint 为分析口径。

| 录制时间窗口（秒） | 扩展采样数 | 物理占用中位数（MiB） | 采样最高值（MiB） |
| --- | ---: | ---: | ---: |
| 0～30 | 16 | 18.12 | 19.13 |
| 30～60 | 29 | 19.06 | 19.77 |
| 60～90 | 28 | 19.71 | 20.52 |
| 90～120 | 29 | 20.22 | 20.60 |
| 120～150 | 22 | 20.36 | 21.25 |
| 150～152.225 | 3 | 21.22 | 21.22 |

初始增长包含启动和预热；30～60秒至120～150秒的窗口中位数仍增加约1.30 MiB，末段尚不能称为完全平台期。没有 `recently-died` 标记采样，记录中只有一个扩展PID，且末尾仍有扩展采样；这些记录没有显示扩展进程重启或死亡，但不能替代系统终止日志的审计。

这次操作观察到的物理占用最高采样为21.25 MiB。约1秒的采样可能漏掉短时峰值，该值不是内核生命周期最高值。一次约2.5分钟、输入内容和操作次数未逐项记录的手动测试，不能判断后段增长是缓存还是泄漏，不能认证固定的键盘扩展内存预算。下一项有价值的验证是重复相同输入的10～15分钟持续记录，并增加模型加载日志与更细粒度的峰值/分配采集。

宿主 `Vime` PID6111仅在开始约13.5秒内有7个采样，物理占用50.36～51.24 MiB；它与真正键盘扩展分开统计。旧记录的30.36 MiB与本次21.25 MiB来自不同、未固定的工作负载和构建身份，不能当作优化前后对照。

## 保存位置与复现

- `outputs/deployment/iphone16promax-mirroring-memory-v1/keyboard-manual-activity.trace`：原始 Instruments trace，保留原始录制。
- 同目录 `toc.xml`、`sysmon-process.xml`：离线导出的目录和原始统计。
- 同目录 `summary.json`：Vime、VimeKeyboard的独立汇总及完整逐点曲线。
- 同目录 `provenance.json`：录制时间、运行设置与证据限制。
- `scripts/deployment/summarize_keyboard_trace.py`：按 XML 列名和 id/ref 引用解析，重建各进程曲线。

```sh
python3 scripts/deployment/summarize_keyboard_trace.py \
  --input outputs/deployment/iphone16promax-mirroring-memory-v1/sysmon-process.xml \
  --output outputs/deployment/iphone16promax-mirroring-memory-v1/summary-recheck.json
```
