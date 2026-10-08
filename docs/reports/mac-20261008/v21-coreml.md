# V2.1 Core ML / INT8 block32（2026-10-08）

模型为 extend5 best / step40000，12,537,920 参数。原 checkpoint 身份为
`fa202f18d00cf5e77469bd3be5b8b264117fb73233ec9966fa3213257a470104`，V2 tokenizer 为
`cdb0c60529300619fb0d7b01370c87b3f1028f2687ff67d97c778b15bfa2d177`。
部署 checkpoint 的文件身份单独记录；评分比较复用原 best.pt 身份。

本轮新增 `src/vimeml/deployment/v2.py`、`scripts/deployment/coreml_v2.py` 和
`scripts/deployment/package_v2_ios.py`。V1 bundle、转换、压缩入口保持原样。
V2 bundle 明确声明 `tiny_gpt_v2`，通过 model_factory 分派 RMSNorm/SwiGLU 模型，
只保存一份共享 embedding/head 权重，不保存训练状态。
架构源文件与 checkpoint 签名一致；历史 `runtime_v2.py` 的 optimizer continuation
辅助函数在训练后有改动，该文件未被部署推理调用，差异保留在 bundle 的 source_audit。

## 数值与存储

Python3.11.17、torch2.7.1、coremltools9.0、numpy2.3.5、SentencePiece0.2.1。
CPU_ONLY、最低 iOS18、batch1、无 KV cache；INT32 `[1,T]` → FLOAT32 `[1,T,16384]`，
1≤T≤128，完整词表，无 softmax。压缩与未压缩均使用 FP32 计算。

| 检查 | 结果 |
| --- | --- |
| V2 inference bundle → Windows FP32 fixtures | 通过 |
| 未压缩 Core ML → Windows FP32 fixtures | 通过；最大绝对差 0.0000457764 |
| INT8 → 同一 FP32 fixtures | 严格对齐失败；最大绝对差 1.00949144 |
| Core ML FP32 / INT8 右 PAD、未来 token 修改 | 有效 16-token 前缀完全相同（最大差 0） |
| FP32 package | 50,358,014 字节 |
| INT8 block32 package | 14,330,856 字节，减少约 71.5% |

严格阈值保持 atol=3e-4、rtol=3e-4，没有为量化放宽阈值。
32 个有限学习矩阵被选中；2 个结构常量排除，causal mask 不量化，RMSNorm 参数保持 FP32。
共享 embedding/head 在原图共享同一常量；参数选择清单保存于 INT8 manifest。
包文件字节数不是编译资源大小、IPA 大小、驻留内存或 footprint。

## 同池质量

复用交接包的 FP32 逐候选分数；原候选、标签、版本、分词目标和公共前缀均检查一致。
联合 context+candidate 编码，公共 token 前缀后的 logP sum；不加 EOS、不截断，
并列保留 AzooKey 顺序，整池不适合评分时回退。没有使用 blind 或修改标签。

| 数据 | FP32 Top-1 | INT8 Top-1 | Top-1 变化 | 改对 / 改错 | 分数 sum 平均 / 最大绝对差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AJIMEE | 144/200 | 144/200 | 7 | 3 / 3 | 0.288214 / 1.820013 |
| 原 development | 122/137 | 122/137 | 1 | 0 / 0 | 0.192015 / 1.373440 |
| 扩大 development 草稿 | 1487/2000 | 1481/2000 | 30 | 8 / 14 | 0.228864 / 1.790113 |

原 development 的 1 条变化为两个可接受答案之间的变化。
AJIMEE 另有 1 条可接受答案变化；扩大 development 另有 1 条可接受答案变化和 7 条仍错误的变化。
INT8 在扩大草稿下降 0.3 个百分点。命中数相同不表示逐条预测或分数一致。
12 条固定前缀 next-token Top-1 全部相同；32-token greedy 完全相同 8/12。
这些是开发诊断，不是盲测或母语 gold 认证。

## 客户端与设备

Vime 最新 main 为基点，另建 `codex/v21-coreml-client`。新增 V2 资源名、manifest、
6755 条 native tokenizer fixtures、4 组联合评分和 20 条 beam fixture；保留 V1 全部资源，
可显式选择 `.v1` 回退。默认 V2 身份或加载失败时保留词典候选，不混用 tokenizer。
大模型和 tokenizer 使用资源 ZIP / 安装脚本迁移，不提交二进制到代码 PR。

iPhone16 Pro Max / iOS27.2，Release，CPU_ONLY：最初 10 项原生功能测试加 1 项宿主
性能测试通过。覆盖分词、联合评分、PAD/causal、整池回退、取消、排序 metadata、
20 条 beam、下一词和 session。最终构建及真实扩展证据见[设备报告](v21-iphone.md)：
真实扩展内核 footprint 峰值34.72MiB，私有驻留采样峰值73.78MiB；UI发布存在未定位长尾。

| 宿主指标 | 实测 |
| --- | ---: |
| 首次进程加载（OS/Core ML 缓存可能温热） | 185.54 ms |
| 再加载 | 19.06 ms |
| 候选评分 p50 / p95 | 21.35 / 26.46 ms |
| 候选评分超过 250ms | 0/90 |
| 下一词 p50 / p95 | 13.79 / 18.87 ms |
| beam p50 / p95 | 99.20 / 102.11 ms |
| 测试宿主进程 footprint 生命周期峰值 | 59.64 MiB |

宿主与真实键盘扩展分开，驻留内存与 footprint 分开。未将旧 V1 trace 或模拟器数值用作 V2 验收。
扩展测量为 opt-in，100ms 内存采样、最多240秒，记录模型版本、设置和有界请求计时，
不记录按键或输入文本。测量自身有开销，正常操作默认关闭。

## 复现与资料

在新工作区使用 `venv/coreml/bin/python scripts/deployment/coreml_v2.py --help`：
依次 export → convert → align → compress → align（保留失败）→ evaluate/samples。
`compress --alignment` 强制要求这个未压缩 FP32 package 的通过报告。
每一步必须使用新输出目录。三个 V2 针对性 Python 测试通过；未运行训练、完整 Python 回归或旧模型冒烟。
导出时绑定权重/tokenizer身份，后续阶段复用manifest并检查大小、格式及已加载契约。
资源变更或迁移时可显式执行verify_bundle的完整身份检查；客户端安装时验证资源身份。

产物：`artifacts/deployment/tiny-ja-v2.1-extend5-{bundle,fp32,int8-b32,client-resources}-v1/`。
报告：`outputs/deployment/v21-*/`，包含原始分数、差异、失败对齐及真机 xcresult。
iOS 编译使用 `--platform ios --deployment-target 18.0`。首次省略 platform 的编译警告产物
另存于 ios18-v1，不作为客户端资源交接；实际资源来自明确平台的 ios18-v2。

初次受限运行 Core ML 编译和设备服务失败，授权系统运行时访问后通过。
新增审计后的增量 test 构建曾出现 OrderedCollections 链接失败；失败日志与干净重建结果分别保留。
结果 ZIP 仅新增模型、报告和必要编译资源；按路径、数量和大小验证，不重复全目录 SHA256。
