> 2026-10-06 Mac 阶段报告，经 Windows 合并校对；原文和原始证据另存于本地交接归档。这里的历史任务描述不授权启动新的训练或设备操作。

# Mac 阶段结论与双仓库交接（2026-10-06）

## 当前结论

当前 tiny-ja-v1 为7,386,624参数（约7.39M），已完成 Core ML INT8 权重量化、iOS18客户端接入、质量评测和 iPhone16 Pro Max 的实际键盘操作。用户确认 LM候选排序、下一词联想均开启，并实际看到建议。以用户认可的驻留内存表现和输入体验为依据，当前模型的内存占用可以接受，Mac 阶段曾建议探索10M～15M参数的新模型；这不是本次 Windows 合并的执行任务。

交接时提出的后续实验设想以现有 INT8 block32、FP32计算、CPU_ONLY、最低iOS18为基线，先比较10M级和15M级候选模型的质量收益与实际端侧成本。模型放大后重新做量化评测、延迟和同口径驻留内存测量；当前结果支持进入实验阶段，不是新模型已经通过验收的结论。保持当前7.39M版本用于对照和回退，不继续推进4bit。

内存口径：用户在 Instruments 中观察到约6.7→29.31→45.77MB的驻留内存读数，认可这一表现。该数列按「用户观察的图表读数」保存，不冒充全程最大值。当前原始 sysmon 导出的 `memory-real-private` 首次7.125MiB、最高51.922MiB、末次49.625MiB；这是整段记录的私有驻留统计，图中某个观测点与全程最大值不能混写。`memory-physical-footprint` 的21.25MiB最高采样是另一口径，不能用来替代用户所说的驻留内存。

整段键盘记录152.725秒，扩展PID6182、127个采样；用户看到建议是本次模型运行场景的确认依据。没有独立模型加载日志或已安装构建哈希，不将本次设备镜像自动认定为当前工作区相同版本。详细原始文件在 `outputs/deployment/iphone16promax-mirroring-memory-v1/`。

## 交付目录

`handoff/mac-20261006-v1/` 包含：

| 文件 | 内容与用途 |
| --- | --- |
| `VimeML-source.zip` | 当前Mac端代码、配置、测试、文档、示例、批准记录；Windows合并审查后提交Git |
| `VimeML-results.zip` | 所有outputs/runs，以及artifacts中部署、tokenizer、benchmark、策略、tracking；Windows本地产物归档 |
| `VimeML-training-inputs.zip` | Mac端原有的models和token-data全量备份；Windows已有相同文件可按哈希跳过传输/恢复 |
| `Vime-client.zip` | Vime当前tracked和非ignored新增源文件、模型资源、SentencePiece、测试；含基于当前HEAD的tracked差异patch |
| `MANIFEST.json` | 每个ZIP中文件的路径、大小、SHA256，以及Vime Git基点和归档排除记录 |
| `SHA256SUMS.txt` | ZIP及交接说明的SHA256，用于传输后校验 |
| `README.md` | 本文的独立副本，供两侧直接阅读 |

排除的是可重建的虚拟环境、缓存、`.DS_Store`、Git内部数据库、Xcode用户界面状态和个人断点；不是删除实验数据。失败/放弃的FP16、4bit、ANE路线和首次内存测试失败结果仍在产物包内。没有自动发给其他机器或聊天，也没有在本机提交Git。

## Windows侧：合并与提交

Mac VimeML目录没有 `.git`，无法生成相对Windows当前分支的准确diff。因此源文件ZIP是完整快照；在 `D:\Sources\VimeML` 外解压审查，与Windows已有工作区逐文件合并，保留Windows的新改动及冻结的核心代码指纹，不直接整目录覆盖。

合并重点：`requirements.txt` 平台条件依赖、Core ML转换/量化/检查/打包脚本、内存汇总脚本、`examples/ios/`、测试和更新后的README/docs。保留Windows torch/CUDA安装路线；CoreML依赖仅作用于darwin，不把Mac虚拟环境复制到Windows。

`src/vimeml/deployment/`、训练/数据核心代码与当前checkpoint、推理包之间有指纹校验，先核对再合并；不要为了通过旧模型加载而随意重写文件。新的10M～15M训练另用版本目录，训练后重新export inference bundle和FP32 reference，复制回Mac进行Core ML转换和INT8评测。

`artifacts/ outputs/ runs/` 是本地归档，不纳入源代码Git提交。Windows确认忽略规则含这些目录及 `venv/ .venv/ handoff/ __pycache__/ *.pyc .DS_Store`。对源码运行适当Python测试、审阅 `git diff --check`/`git diff` 后，只暂存确定合并的代码、配置、文档、测试、例子和批准记录，再由Windows侧提交。README/docs中的原始失败结果不能改写为通过。

产物主要索引：

- 发布模型：`artifacts/deployment/tiny-ja-v1-conservative-int8-b32-v1/`，8,077,801逻辑字节。
- 旧集成发布包：`artifacts/deployment/tiny-ja-v1-ios18-int8-release-v1/`；它是早期包，其Swift示例及「未真机验证/关闭自动生成」说明早于Claude后续更新。
- 量化评测：`outputs/deployment/conservative-int8-b32-{dev,ajimee,alignment,release-review}-v1/`。严格FP32 logits对齐失败仍保留；dev Top1=122/137、AJIMEE Top1=124/200是冻结候选池上的纯LM结果，不能直接当作客户端完整正确率。
- 内存：`outputs/deployment/memory-audit-v2/`、两组 `iphone16promax-*-memory-v1/`。模拟器、旧trace、本次真实扩展各自标明范围；首次LM＋联想测试超时与重跑成功均保留。
- 环境：`outputs/coreml-environment-v{1,2}.txt`；迁移参考：`docs/coreml.md`、`docs/artifacts.md`。

## Vime侧：优化与重新打包

当前客户端来源是 `/Users/voltline/Documents/Sources/Vime` 工作区，Git基点 `abc3e4d`（最低兼容版本iOS18）。ZIP中当前 `Shared/` 实现优先于旧发布包和 `examples/ios/integration/`：保留Claude新增的句内上下文、下一词联想、设置、取消和session接入，以及内存审计测试。

请以当前7.39M INT8模型先完成客户端优化和新的可审查Release包，不预先替换成尚未训练的10M～15M模型：

1. 检查宿主App和键盘扩展分别包含模型、tokenizer、manifest；加载哈希与资源一致。保留CPU_ONLY，最低iOS18；不要仅为提速改成GPU/ALL，已有动态shape量化路径曾出现崩溃。
2. 保留单个worker缓存模型、串行推理、逐候选/生成步骤autoreleasepool、取消旧请求和词典回退；保留句内上下文与token长度检查，不在主线程重复加载。关闭排序和关闭词联想是两个独立开关，关闭UI开关并不自动回收已缓存模型。
3. 验证候选选择/学习/脚本选择/纠错规则、快速输入与退格、取消、词联想与跨宿主切换。现有原生tokenizer/logit fixtures及必要回归应通过；内存审计测试默认opt-in，首次超时记录需要保留。
4. 更新资源包和发布文档的版本、模式及设备验证状态，避免沿用旧README的过时说明。重新打包时保留SentencePiece/第三方许可，按现有签名配置执行Release/Archive并交付构建结果和尺寸；此交接包本身不是已签名IPA或App Store发布。
5. 后续新模型保持相同tokenizer/词表/输入输出接口时，刷新资源与哈希并复跑验证；如更换tokenizer、词表、context或输出接口，必须同步分词器和客户端约束，不能只替换weight.bin。

当前文件快照和差异patch是交接材料；未执行新的客户端改写、签名、上架或Windows Git提交。


## Windows 合并后的保存位置

VimeML 源码已逐文件合并，全部51个 `src/` 文件与 Mac 快照一致；训练、评分与推理核心保持原始指纹。所有交接ZIP保存在 `handoff/mac-20261006-v1/`，新增产物进入对应本地目录，冲突或已退役的 pilot 文件另存于 `outputs/history/mac-import-20261006/`。Vime 客户端仍以 ZIP 独立保存，没有写入 VimeML 的源码目录，也没有改写另一个仓库。详见 [合并记录](merge-verification.md)。
