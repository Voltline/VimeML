# Core ML 部署与实测

当前客户端接入候选为 `artifacts/deployment/tiny-ja-v1-conservative-int8-b32-v1/`：INT8 block32权重、FP32计算、CPU_ONLY、最低iOS18，`.mlpackage` 共8,077,801字节（十进制8.08MB）。训练权重和tokenizer没有改变。以下步骤由使用者手动启动；已有版本不可覆盖，请给复跑结果另起目录。

## 模型接口

| 项目 | 约定 |
| --- | --- |
| 输入 `input_ids` | INT32 `[1,T]`，1≤T≤128，ID0…16383 |
| 输出 `logits` | FLOAT32 `[1,T,16384]`，完整词表、未经softmax |
| 位置与mask | 从0开始的绝对位置，显式causal mask；仅右侧PAD，ID0 |
| 状态 | batch1，无KV cache，每次对完整输入forward |

SentencePiece、联合分词评分、候选排序和生成搜索都在应用侧。当前采用纯LM contextual logP sum，AzooKey仍负责假名检索；详见 [评分定义](evaluation.md#评分定义)。量化后没有重新证明λ=2最优。

## 环境

Windows沿用当前环境。Mac使用独立Python3.11环境：

```bash
uv venv --python 3.11 venv/coreml
uv pip install --python venv/coreml/bin/python -r requirements.txt
source venv/coreml/bin/activate
```

`requirements.txt` 的Darwin分支固定torch2.7.1、coremltools9.0、numpy2.3.5、scikit-learn1.5.1；Windows训练依赖保留。Mac曾遇到NumPy2.4以上的标量转换问题，因此沿用已验证的2.3.5。原始环境记录是 `outputs/coreml-environment-v{1,2}.txt`。

## 手动流程

### 1. Windows推理包与FP32参考

以下是初次导出示例；这两个v1目录现在已经存在，重跑应改输出版本：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/deployment/coreml.py export --output artifacts/deployment/tiny-ja-v1-inference-v1
.\.venv\Scripts\python.exe -X utf8 scripts/deployment/coreml.py reference --bundle artifacts/deployment/tiny-ja-v1-inference-v1 --output outputs/deployment/tiny-ja-v1-reference-v1
```

推理包只含唯一的FP32权重、config、原始tokenizer、token manifest与校验信息，不含optimizer。核心代码指纹在加载时校验；转换适配的后续变更放在独立脚本，不改冻结的 `src/` 来绕过校验。参考包含PAD、causal、最长输入、特殊token及候选评分fixture。

### 2. Mac未压缩转换与对齐

初始 `coreml.py convert` 输出FP16计算模型。Mac的 `tiny-ja-v1-fp16-v2` 未通过严格logits门槛（最大差约0.36），保留为实验结果。后续独立适配脚本使用FP32计算，token embedding与非QKV linear以FP16存储，位置embedding、QKV和LayerNorm以FP32存储，形成未压缩conservative包。

```bash
B=artifacts/deployment/tiny-ja-v1-inference-v1
python scripts/deployment/coreml_conservative.py convert --bundle "$B" --output artifacts/deployment/tiny-ja-v1-conservative-v3
python scripts/deployment/coreml.py validate --bundle "$B" --model artifacts/deployment/tiny-ja-v1-conservative-v3 --reference outputs/deployment/tiny-ja-v1-reference-v1 --compute-units CPU_ONLY --output outputs/deployment/conservative-alignment-v3
```

默认容差为 `atol=.1, rtol=.01, score-atol=.2`。该conservative源包通过logits、候选fixture、右PAD与causal检查；FP16失败记录不改写为通过。

### 3. 压缩

```bash
python scripts/deployment/coreml_conservative.py compress \
  --source artifacts/deployment/tiny-ja-v1-conservative-v3 \
  --fp16-alignment outputs/deployment/conservative-alignment-v3/alignment.json \
  --method linear --bits 8 --block-size 32 \
  --output artifacts/deployment/tiny-ja-v1-conservative-int8-b32-v1
```

`--fp16-alignment` 是沿用的参数名，这里指精确匹配conservative源包的通过报告。脚本只选择有限值、超过2048元素的二维linear weight和已识别embedding表；不压缩causal mask、bias与LayerNorm。选择和排除记录写入manifest，避免把含负无穷的mask当作权重量化。对应回归在 `tests/test_coreml_compression.py`。

8bit为对称线性、每32个输入维度一个scale，权重仍共享；不量化激活。运行时反量化后FP32计算，不能称作INT8算术。4bit palette实验参数为 `--method palette --bits 4 --group-size 16`，独立输出目录；该已测版本质量明显下降。线性4bit等其他方案需要独立复测，不能从palette结果推定。

### 4. 质量与计时

```bash
M=artifacts/deployment/tiny-ja-v1-conservative-int8-b32-v1
B=artifacts/deployment/tiny-ja-v1-inference-v1
python scripts/deployment/coreml.py validate --bundle "$B" --model "$M" --reference outputs/deployment/tiny-ja-v1-reference-v1 --compute-units CPU_ONLY --output outputs/deployment/int8-recheck-alignment
python scripts/deployment/coreml.py evaluate --bundle "$B" --model "$M" --role dev --benchmark artifacts/benchmarks/ime-dev-v2 --baseline outputs/ime-eval/tiny-ja-v1-dev-v2-scores --output outputs/deployment/int8-recheck-dev
python scripts/deployment/coreml.py evaluate --bundle "$B" --model "$M" --role ajimee --benchmark artifacts/benchmarks/ajimee-jwtd-v2-v1 --baseline outputs/ime-eval/tiny-ja-v1-ajimee --output outputs/deployment/int8-recheck-ajimee
python scripts/deployment/coreml.py phrases --bundle "$B" --model "$M" --output outputs/deployment/int8-recheck-phrases
python scripts/deployment/coreml.py timing --bundle "$B" --model "$M" --benchmark artifacts/benchmarks/ime-dev-v2 --output outputs/deployment/int8-recheck-timing
python scripts/deployment/compare_quantization.py --bundle "$B" --model "$M" --output outputs/deployment/int8-recheck-distribution/report.json
```

`validate/evaluate/phrases/timing` 可选择compute units，默认为CPU_ONLY；其中evaluate/phrases/timing省略model则用PyTorch包。分布诊断也默认CPU_ONLY。GPU/ALL路径曾崩溃，不能直接用于当前键盘接入。

### 5. iOS编译与参考包

```bash
xcrun coremlcompiler compile "$M/model.mlpackage" outputs/deployment/ios18-compiled-int8-recheck --deployment-target 18.0
python scripts/deployment/package_ios.py --source "$M" --compiled outputs/deployment/ios18-compiled-int8-recheck/model.mlmodelc \
  --tokenizer "$B/tokenizer.model" --review outputs/deployment/conservative-int8-b32-release-review-v1/review.json \
  --output artifacts/deployment/tiny-ja-v1-ios18-int8-reference-v2
```

打包脚本沿用首次候选重排接入的review门槛，核对源模型、tokenizer与编译资源，附带 `examples/ios/` 的参考副本。它没有重新认证当前客户端或设备，也不会自动构建、安装Vime。`tiny-ja-v1-ios18-int8-release-v1` 是早期快照；最新客户端来源为本地 `handoff/mac-20261006-v1/Vime-client.zip`，含另一个仓库的Git基点与patch。不要用旧Swift副本覆盖最新客户端，见 [iOS参考说明](../examples/ios/README.md)。

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

原始报告在 `outputs/deployment/conservative-int8-b32-{alignment,dev,ajimee,phrases,timing,distribution,release-review}-v1/`。开发集原报告的comparison是对未压缩conservative源包，AJIMEE是对Windows FP32；合并时又从保存的逐候选分数重算了开发集对Windows FP32的差异，详见 [合并校验](reports/mac-20261006/merge-verification.md)。未压缩对照为 `tiny-ja-v1-conservative-v3`，16.57MB。

## 性能与设备范围

Mac Python CPU_ONLY报告：reranking411次，p50 18.2ms、p95 21.9ms；beam联想60次，p50 53.8ms、p95 57.3ms；加载约95ms，可能命中编译缓存。

Mac客户端文档记录的iPhone16 Pro Max优化构建、CPU_ONLY、**测试宿主App进程**：加载首次80ms/再次16ms；T16/T64/T128预测0.67/1.8/3.0ms；约14候选reranking p50 10.4ms/p95 12.7ms；下一词p50 5.8ms；beam8×8 p50 44.5ms。宿主加载增量约24.2MB、宿主峰值74.8MB。这些不是键盘扩展的延迟或内存峰值，Windows未复跑其运行时。

真实键盘扩展有约152.7秒手动输入记录，用户确认LM排序和下一词开启。physical footprint采样峰值21.25MiB，私有驻留采样峰值51.922MiB；两者不可混用。约1秒采样会漏掉短峰值，未取得安装构建／模型哈希，末段仍有缓慢增长。见 [真实扩展报告](reports/mac-20261006/iphone-keyboard-memory.md) 和 [模拟器报告](reports/mac-20261006/simulator-memory.md)。加载增量不能直接证明CPU内部权重布局，也不能推定INT8一定不降低运行内存。

接下来有价值的是手动进行固定构建、记录模型哈希与开关状态的10～15分钟重复输入测试，分开记录冷启动、reranking、下一词、取消、切换宿主与终止情况。本次合并不自动执行。

## GPU与ANE实验

当前INT8 FP32计算包的Maccompute plan为119/119算子在CPU。客户端记录GPU/ALL触发 `MPSNDArrayQuantizedGatherND` 动态形状断言，CPU_AND_NE也未把当前包移至ANE。

独立实验入口为 `coreml_ane.py`（FP16计算、16/32/64/128档位及每通道INT8）、`compute_plan.py`、`ane_output_probe.py`（缩小输出的原型）。这些不是当前部署接口；输入需补到下一档并重新验证有效位置。Mac实验中ANE每次调用比CPU慢约3～4倍，缩小输出也未改善；调度与搬运是可能原因，现有数据未隔离证明瓶颈，也不代表所有iPhone或更大模型。功耗尚未测量。

FP16、4bit、ANE失败与放弃的路线均保留原包和报告。转换脚本后续有修改，旧manifest的脚本哈希仍代表当时版本，不更新为当前源码哈希。
