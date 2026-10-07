> 历史文档：这是 Mac 实测前的 Windows 计划；当前状态与操作入口以 [Core ML 指南](../coreml.md) 和 [文档索引](../index.md) 为准。

# Core ML 手动操作指南

工具已实现；正式权重冻结，尚未执行真实导出、Core ML 转换、压缩或 iPhone 测试。各阶段使用独立手动命令，不自动串联、不训练、不调用 API。每个 `--output` 必须不存在；失败留下的目录也保留，重试用新版本名。

入口为 `scripts/deployment/coreml.py`，独立适配代码在 `src/vimeml/deployment/`，设备辅助在 `examples/ios/CoreMLProbe.swift`。冻结的模型、推理、评分和联想核心文件不变。

## 策略与接口

默认 **纯 LM contextual logP sum 排序**，AzooKey 继续按假名检索候选，数值分数不参与排序，原候选索引用于稳定并列和现有错误回退。旧 FP32 λ=2 策略保留为历史对照；此路线没有 λ。

取消 AzooKey 检索是另一项任务：当前 GPT 没有假名→表记转换训练目标和读音约束搜索，直接生成可能脱离读音，也会改变候选池与评测定义。本轮是 `Kana → AzooKey candidates → LM-only rerank`；句内联想由 LM 搜索。

| 张量/行为 | 首版 Core ML 合约 |
| --- | --- |
| `input_ids` | INT32 `[1,T]`，1≤T≤128，ID 0..16383 |
| `logits` | FLOAT32 `[1,T,16384]`；FP16 计算/未压缩权重，无 softmax |
| 位置与 mask | 从 0 开始绝对位置，显式 causal mask；右 PAD，禁止左 PAD |
| 状态 | 无 KV cache，每个生成步骤重新 forward 整段 |

SentencePiece、候选评分、生成搜索都在应用侧。`context+candidate` 联合编码，不能分别编码再拼接。比较 context 与所有候选的公共 token 前缀（包含 BOS），从 `common-1` 对下一个 token 累计**完整词表** log-softmax。输入 `seq[:-1]`，目标 `seq[1:]`，不加候选最终 EOS，不计 PAD 输出。context 最多含 BOS 在内 128 token；候选输入最多 128，允许最后目标位于第129个 token。超长/不 roundtrip/空池沿用原整条回退，不删除评测样本。

生成取最后有效位置，禁止 PAD/UNK/BOS，允许 EOS；beam 沿用原 full log-softmax 后屏蔽的规则，不重新归一化允许词表。实现复用冻结的 `score_candidates`、`rerank`、`PhraseDemo`。sum 是 token 后缀似然代理，不是精确字符串条件概率；mean、无 context 为次要诊断。

## 1. Windows：导出推理包与跨机参照

在 `D:\Sources\VimeML` 执行。先保存当前 Git 版本，Mac 必须使用同版本源码。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/deployment/coreml.py export --output artifacts/deployment/tiny-ja-v1-inference-v1
```

读取并验证三个正式输入：`best.pt`、`tokenizer.model`、token-data `manifest.json`，不改原件。输出：

```text
artifacts/deployment/tiny-ja-v1-inference-v1/
  weights.pt          # 唯一 FP32 state_dict；不重复保存 lm_head.weight
  config.json         # 模型尺寸；无训练配置
  tokenizer.model     # 原字节复制
  token-manifest.json # 原字节复制，仅追踪；应用无须加载
  manifest.json       # 输入 hash、训练签名、参数量、特殊 ID、环境、输出校验
```

加载时重建共享 embedding/head；拒绝其他缺失权重。无 optimizer、scaler、RNG 或训练状态。保留训练签名和核心原始 SHA256，另存代码 LF 规范化 hash 以兼容 checkout 换行。Mac 加载校验代码/配置/权重/tokenizer；源码变化需用新版本重新导出。FP32 理论参数约29.55MB，FP16 约14.77MB，实际包大小另测。

手动生成跨机 CPU FP32 参考：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/deployment/coreml.py reference --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --output outputs/deployment/tiny-ja-v1-reference-v1
```

覆盖 BOS、EOS、显式 UTF-8 byte pieces、短句、右 PAD、128长度、未来 token 修改的 causal 对照、联合分词与严格前缀候选。`logits.npz` 保存完整 logits；`fixtures.json` 保存逐 token 候选分数；`device-fixtures.json` 给 Swift 张量/分数对照，不能替代应用 tokenizer 验证。

复制到 Mac 同版本 Git 仓库：整个推理包、整个 reference 目录，以及 [迁移清单](../artifacts.md#core-ml-效果验证的迁移) 中的冻结开发候选、AJIMEE、FP32 缓存、联想结果。转换无需原始语料、token 二进制、训练 checkpoint 或 API key。

可选：先在 Windows 用导出包核对完整纯 LM 基线（只写新目录）：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/deployment/coreml.py evaluate --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --role dev --benchmark artifacts/benchmarks/ime-dev-v2 --baseline outputs/ime-eval/tiny-ja-v1-dev-v2-scores --output outputs/deployment/windows-bundle-dev-v1
.\.venv\Scripts\python.exe -X utf8 scripts/deployment/coreml.py evaluate --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --role ajimee --benchmark artifacts/benchmarks/ajimee-jwtd-v2-v1 --baseline outputs/ime-eval/tiny-ja-v1-ajimee --output outputs/deployment/windows-bundle-ajimee-v1
```

## 2. Mac：未压缩 FP16，先对齐

建议 Apple Silicon、macOS15+、Python3.11、独立环境。默认目标 iOS18，方便 grouped-channel/blockwise 对照；不承诺旧系统可用。统一 `requirements.txt` 使用平台条件：Windows torch2.10.0 保留，Mac torch2.7.1 + coremltools9.0。Apple 9.0 发布说明列出 PyTorch2.7 支持；这是保守转换组合，仍要数值验证。[Apple 发布说明](https://github.com/apple/coremltools/releases/tag/9.0)

在 Mac 仓库根目录执行（已有 uv）：

```bash
uv venv --python 3.11 venv/coreml
source venv/coreml/bin/activate
uv pip install --python venv/coreml/bin/python -r requirements.txt
mkdir -p outputs
uv pip freeze --python venv/coreml/bin/python > outputs/deployment/environment/coreml-environment-v1.txt
```

`venv/` 被忽略；环境记录用新版本名；整份 requirements 不要使用 PyTorch 专用下载源。先验证 Mac FP32，隔离跨机 torch/分词差异：

```bash
python scripts/deployment/coreml.py validate --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --reference outputs/deployment/tiny-ja-v1-reference-v1 --output outputs/deployment/mac-fp32-alignment-v1 --atol 0.0001 --rtol 0.001 --score-atol 0.001
```

再手动转换、对齐：

```bash
python scripts/deployment/coreml.py convert --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --target 18 --output artifacts/deployment/tiny-ja-v1-fp16-v1
python scripts/deployment/coreml.py validate --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --model artifacts/deployment/tiny-ja-v1-fp16-v1 --reference outputs/deployment/tiny-ja-v1-reference-v1 --compute-units CPU_ONLY --output outputs/deployment/fp16-alignment-v1
```

转换前核对独立显式 attention 与原 SDPA、trace 不同长度，之后 `ct.convert` 输出 ML Program + 有限 RangeDim。首版 batch=1；Mac 适配器逐行执行批量评分/beam 以保留语义，不宣称高效 batching。[PyTorch 转换](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html)、[Flexible Inputs](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)

`alignment.json` 记录 max/mean/RMSE、超容差数量、末位 argmax、PAD/causal 不变性、候选 sum 差值与排序。初始容差 `atol=.1, rtol=.01, score-atol=.2` 只是排查起点，不是质量保证。任何有效 logits 超容差、sum 超容差或固定候选排序改变都会失败，保留报告。先定位偏差，不能为了继续压缩盲目放宽；FP32 跨机应远小于 FP16 容差。

小样本通过，再验证完整开发/AJIMEE/联想：

```bash
python scripts/deployment/coreml.py evaluate --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --model artifacts/deployment/tiny-ja-v1-fp16-v1 --role dev --benchmark artifacts/benchmarks/ime-dev-v2 --baseline outputs/ime-eval/tiny-ja-v1-dev-v2-scores --output outputs/deployment/fp16-dev-v1
python scripts/deployment/coreml.py evaluate --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --model artifacts/deployment/tiny-ja-v1-fp16-v1 --role ajimee --benchmark artifacts/benchmarks/ajimee-jwtd-v2-v1 --baseline outputs/ime-eval/tiny-ja-v1-ajimee --output outputs/deployment/fp16-ajimee-v1
python scripts/deployment/coreml.py phrases --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --model artifacts/deployment/tiny-ja-v1-fp16-v1 --output outputs/deployment/fp16-phrases-v1
```

核对137/200条分母、候选池/hash，Top1/Top5/MRR/CER、改好/改坏/整条回退、分数偏差和每条排序变化。历史纯 LM 为122/137、124/200，仅作对照；数值近似不等于排名必然一致。FP16 转换质量变化不合理时先修转换。

`phrases` 沿用20前缀、beam width8、alpha .7、max_tokens8、count5，保留 raw_outputs 与过滤标记。逐条对照 `outputs/phrase-demo/tiny-ja-v1/` 的自然度、重复、语义和截断。可单独 `--mode sample` 对照原 sample 基线；相同seed也不保证浮点变化后文本一致。

再换 `--compute-units ALL`、`CPU_AND_NE`、`CPU_AND_GPU` 与新输出目录复验计划使用的执行路径；CPU 对齐不能代替 ANE/GPU 验证。

## 3. Mac：压缩与质量/体积/性能对照

FP16 对齐、完整质量复核通过后，手动启动压缩。命令要求绑定到该 FP16 包的通过报告；开发/AJIMEE/联想质量是否满足预先写下的预算另行复核。默认 grouped-channel k-means palettization，16通道一组：

```bash
python scripts/deployment/coreml.py compress --source artifacts/deployment/tiny-ja-v1-fp16-v1 --fp16-alignment outputs/deployment/fp16-alignment-v1/alignment.json --method palette --bits 4 --group-size 16 --output artifacts/deployment/tiny-ja-v1-palette4-g16-v1
```

Palettization 是4bit索引+LUT，不表示所有内核都执行INT4算术。基础 per-tensor palette 可用iOS16；grouped-channel要求iOS18/macOS15。最低系统若选iOS16，重新 `convert --target 16`、对齐后使用 `compress --group-size 0`，不能直接降低旧包声明。[Apple Palettization](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html)

独立4bit blockwise对称线性权重对照，或必要时8bit（全部从未压缩FP16开始）：

```bash
python scripts/deployment/coreml.py compress --source artifacts/deployment/tiny-ja-v1-fp16-v1 --fp16-alignment outputs/deployment/fp16-alignment-v1/alignment.json --method linear --bits 4 --block-size 32 --output artifacts/deployment/tiny-ja-v1-linear4-b32-v1
python scripts/deployment/coreml.py compress --source artifacts/deployment/tiny-ja-v1-fp16-v1 --fp16-alignment outputs/deployment/fp16-alignment-v1/alignment.json --method palette --bits 8 --group-size 16 --output artifacts/deployment/tiny-ja-v1-palette8-g16-v1
python scripts/deployment/coreml.py compress --source artifacts/deployment/tiny-ja-v1-fp16-v1 --fp16-alignment outputs/deployment/fp16-alignment-v1/alignment.json --method linear --bits 8 --block-size 32 --output artifacts/deployment/tiny-ja-v1-linear8-b32-v1
```

线性 blockwise 实验要求iOS18。不启用activation quantization、QAT、稀疏训练或微调。只压缩符合阈值的常量（>2048元素）；norm/bias等小常量保留，不整除或不支持的block常量可能跳过，需要查看日志与包。[Apple Quantization](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html)、[Optimization Workflow](https://apple.github.io/coremltools/docs-guides/source/opt-workflow.html)

每个变体都重跑数值检查、独立开发集、固定AJIMEE和20前缀，不复用FP32分数冒充新评分。例：

```bash
python scripts/deployment/coreml.py validate --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --model artifacts/deployment/tiny-ja-v1-palette4-g16-v1 --reference outputs/deployment/tiny-ja-v1-reference-v1 --output outputs/deployment/palette4-alignment-v1
python scripts/deployment/coreml.py evaluate --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --model artifacts/deployment/tiny-ja-v1-palette4-g16-v1 --role dev --benchmark artifacts/benchmarks/ime-dev-v2 --baseline outputs/deployment/fp16-dev-v1 --output outputs/deployment/palette4-dev-v1
python scripts/deployment/coreml.py evaluate --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --model artifacts/deployment/tiny-ja-v1-palette4-g16-v1 --role ajimee --benchmark artifacts/benchmarks/ajimee-jwtd-v2-v1 --baseline outputs/deployment/fp16-ajimee-v1 --output outputs/deployment/palette4-ajimee-v1
python scripts/deployment/coreml.py phrases --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --model artifacts/deployment/tiny-ja-v1-palette4-g16-v1 --output outputs/deployment/palette4-phrases-v1
```

量化可能不能通过FP16数值容差，保留失败报告再审查开发集与联想质量，不能静默改阈值。`evaluate --baseline` 还可单独对原FP32缓存比较（新目录），隔离图/压缩偏差。8bit若要试，替换模型路径与输出目录执行相同流程。

此路线没有λ；若以后恢复融合，必须用新精度/图/字典在独立开发集新建策略和缓存，固定后再跑AJIMEE，不能沿用λ=2。旧hybrid工具要求FP32身份，不能改元数据伪造兼容。AJIMEE已用于分析，固定对照不能称全新盲测，不用其挑压缩参数。

每个模型manifest记录方法、位数、目标、环境、source hash、每文件SHA256/字节数和Core ML op计数。确认真正出现`constexpr_*`，查看package内`weight.bin`及剩余常量。共享embedding/head可能在转换时形成两份存储，实际包与MIL/Xcode图需检查；PyTorch共享不保证Core ML也共享。

纯4bit理论约3.69MB，LUT/scale/未压缩常量/结构另算。比较manifest逻辑字节、`du -sk`磁盘占用、Xcode `.mlmodelc`体积和IPA增量；都不等于运行内存。使用相同工作负载粗筛Mac性能：

```bash
python scripts/deployment/coreml.py timing --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --model artifacts/deployment/tiny-ja-v1-fp16-v1 --benchmark artifacts/benchmarks/ime-dev-v2 --compute-units ALL --repeats 5 --warmup 1 --output outputs/deployment/fp16-mac-all-timing-v1
```

其他变体同参数、新目录各跑一遍。报告p50/p95/max、每条耗时、load、predict次数/时间；包含Python逐行Core ML和全词表传输/搜索成本。Mac适配器为复用冻结评分会拼接批量logits，主机内存较多；iOS逐候选及时释放。load可能命中编译缓存，这些不是iPhone冷启动与性能。

## 4. iPhone：宿主app后进入真实键盘扩展

仓库没有Vime iOS工程。`CoreMLProbe.swift`仅提供加载、tensor预测、fixture分数与物理footprint采样，不是完整键盘/tokenizer/beam实现。Windows没有Apple SDK，Swift需在Xcode编译验证。

1. 选一个`model.mlpackage`加入测试宿主app和键盘扩展target，核对Target Membership；Xcode编译为`.mlmodelc`。别把所有实验包带进发行版。保留对应manifest/tokenizer/device-fixtures。
2. 将Swift辅助加入两个target，在串行后台队列运行，UI只接收建议。先`.cpuOnly`对照fixture，再试`.all`、`.cpuAndNeuralEngine`、`.cpuAndGPU`；每个配置都验证，记录不支持的错误。
3. 接入SentencePiece C++/Swift包装，使用同一`.model`并验SHA256。日文、空context、byte fallback、标点/接缝、特殊ID、联合encode、roundtrip、长度边界与Python逐token对照。
4. 候选完整eligibility在评分前执行，任一失败整条回退。逐项预测并及时释放，完整log-softmax sum排序，原索引稳定打破并列，不用生成过滤后的概率。
5. 联想移植原beam/sample参数、EOS/句末/长度终止、去重和显示过滤；同20前缀核对自然度、prefix_changed/replacement_character/empty。无KV cache，及时释放每步数组。

后台串行队列的`do/catch`中可先调用（资源名按Xcode模型名调整）：

```swift
let url = Bundle.main.url(forResource: "model", withExtension: "mlmodelc")!
let probe = try CoreMLProbe(compiledURL: url, units: .cpuOnly)
let fixtures = Bundle.main.url(forResource: "device-fixtures", withExtension: "json")!
let check = try probe.checkFixtures(url: fixtures)
print(probe.load, check)
```

`suffixScore`是可审查、按stride索引的标量实现，优先正确性，尚非优化CPU代码；改成Accelerate/指针后重跑分数对照。footprint前后采样不能捕获瞬间峰值，用Instruments Allocations/VM Tracker/Xcode Memory Gauge补测。

| 项目 | 设备记录要求 |
| --- | --- |
| 身份 | iPhone型号、iOS/Xcode、Git commit、package/tokenizer hash、computeUnits、Release构建 |
| 包体 | package、编译`.mlmodelc`、IPA增量分别统计 |
| 冷加载 | 新扩展进程至少10次，load与首次predict分开；编译/缓存另列 |
| 热rerank | 固定137/200候选工作负载，20候选、短/长context分组；至少100次，p50/p95/max |
| 联想 | 同20前缀，目标3/5项分别测，记录实际返回数、token步数、beam参数 |
| 端到端 | 输入发生→分词/检索→推理/搜索→UI，分别标时；热模型另列 |
| 内存 | 空扩展、load后、首predict、候选池、beam、连续输入的footprint和峰值 |
| 稳定性 | 快速输入/删字、前后台、反复显示、宿主切换、长context、取消旧任务、Jetsam/崩溃 |

保留单实例，串行推理；快速输入用generation ID丢弃过时建议，在搜索步之间检查取消。全logits T128单份约8MiB，20候选不可长期同时保留约160MiB；每候选/每步`autoreleasepool`释放，UI不持有MLMultiArray。实际扩展预算由设备/系统决定，不设未经实测的统一MB上限，不承诺4bit必然更快。

先保持FP16/4bit/8bit相同接口和搜索参数，质量与延迟/峰值都满足开发阶段写下的预算后选发行版本。未来last-position输出、候选目标投影、bucket shapes、KV cache都是新图/接口，重新对齐。

## 阶段记录与停点

| 版本 | 数值/排序变化 | Dev Top1/回退 | AJIMEE Top1/回退 | 联想复核 | package/mlmodelc | iPhone load/rerank/phrase p95 | 键盘峰值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP32历史 | 参照 | 122/137 | 124/200 | 原复核 | — | — | — |
| FP16 | 待手动运行 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Palette4/Linear4 | 待手动运行 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8bit对照 | 如有需要 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

每步保存manifest、alignment、逐候选scores、metrics、原始phrase输出和timing。先Windows导出并核对，再Mac FP16；可信FP16参照建立后压缩。每个量化变体重验开发集与固定AJIMEE；理论大小或Mac predict不能替代设备验收。
