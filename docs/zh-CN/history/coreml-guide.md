# V1 Core ML方法参考（实测前）

本页保存2026-10-06实测前的接口与工具设计。后续实际选择CPU_ONLY、FP32计算与INT8 block32，结果见[V1实测](../reference/coreml-v1.md)；当前V2入口见[Core ML指南](../coreml.md)。

## 接口与评分

| 项目 | 原设计 |
| --- | --- |
| 输入 | INT32 `input_ids [1,T]`，1≤T≤128，token ID 0..16383 |
| 输出 | FLOAT32 `logits [1,T,16384]`，FP16计算，无softmax |
| 位置与mask | Learned absolute position，显式causal mask，右PAD |
| 状态 | 无KV cache，逐生成步重新forward |
| 应用层 | SentencePiece、联合评分、稳定排序、搜索与回退 |

评分联合编码context+candidate并加BOS，从候选池公共token前缀后累计完整词表log-softmax。输入为`seq[:-1]`，目标为`seq[1:]`；不加候选EOS、不计PAD，超长或分词失败整池回退。生成屏蔽PAD/UNK/BOS并保留EOS，允许词表不重新归一化。

## 工具与产物

入口为`scripts/deployment/coreml.py`，实现位于`src/vimeml/deployment/`。

| 子命令 | 方法与输出 |
| --- | --- |
| `export` | 仅推理weights、config、tokenizer、来源manifest；共享head只存一份 |
| `reference` | FP32 logits、输入长度／特殊token、联合候选评分和设备fixture |
| `convert` | 未压缩Core ML包，显式最低系统与精度 |
| `validate` | Logits、PAD/causal与候选fixture对照，保留失败 |
| `compress` | 在通过的未压缩对照上比较palette／linear、位数与分组 |
| `evaluate` / `phrases` | 固定候选集与20个前缀的任务结果 |
| `timing` | 加载、首次／热预测、重排与搜索，注明compute units和预热 |

各阶段保存独立版本目录及manifest，包体、compiled资源和IPA增量分别统计。完整参数由CLI帮助提供：

```bash
python scripts/deployment/coreml.py --help
```

原压缩对照包括palette4 group16、linear4 block32与8bit变体。INT4存储不保证INT4执行，理论包体不代表设备收益；转换图先对齐，再独立评价量化损失。

## 客户端与测量设计

`examples/ios/CoreMLProbe.swift`提供加载、tensor预测、fixture分数与footprint探针。完整客户端位于独立Vime仓库，分词、排序和beam由应用层实现。

客户端采用单实例和串行推理；generation ID丢弃过时请求，在搜索步之间检查取消，每候选／生成步及时释放中间数组。T128完整logits单份约8MiB，20候选的全部输出长期持有会额外占用约160MiB。

原设备方案分别记录模型／构建身份、冷加载、热重排、联想、请求至UI各阶段、footprint和峰值，并覆盖快速删字、取消、宿主切换与系统终止。加载前后采样不能捕获瞬时峰值，测试宿主也不能代表键盘扩展。

实际包、失败结果和设备trace见[V1实测](../reference/coreml-v1.md)、[模拟器记录](../reports/mac-20261006/simulator-memory.md)与[真实扩展记录](../reports/mac-20261006/iphone-keyboard-memory.md)。产物路径见[归档管理](../artifacts.md)。
