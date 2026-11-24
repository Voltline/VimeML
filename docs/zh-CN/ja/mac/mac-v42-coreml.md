# V4.2 Core ML 与量化交接

这是 macOS 会话的任务说明。内容包括：把 V4.2 epoch 20（`best-lm.pt`，第 1,081,942 步）转换为 Core ML，用与已部署 V2.1 相同的配方量化，并与 V2.1 对比测量。它不授权替换已部署模型、重新训练或改动基准输入。[跨平台交换](../../artifact-exchange.md)描述了 Git 与产物之间的一般边界。

## 环境准备

```bash
unzip vimeml-v42-mac-<date>.zip -d vimeml-v42-mac
python3 vimeml-v42-mac/setup_mac.py            # default ~/Documents/Sources/VimeML-v42-coreml
cd ~/Documents/Sources/VimeML-v42-coreml
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e .
```

`setup_mac.py` 克隆 `repository.bundle`，核对记录的提交，设置 `origin`，创建 `claude/mac-v42-coreml`，并在大小和 SHA256 检查后把载荷恢复到被忽略的 `artifacts/` 和 `outputs/` 中。macOS 上的固定版本为 Python 3.11、PyTorch 2.7.1、coremltools 9.0、NumPy 2.3.5 和 scikit-learn 1.5.1。记录实际版本、macOS 构建号和硬件。

## 载荷

| 路径 | 内容 |
| --- | --- |
| `artifacts/models/tiny-ja-v42-e8k-d384-l7/deployment.pt` | V4.2 epoch 20 FP32 张量；无优化器或 RNG 状态；内部记录了源 checkpoint 的 SHA256 |
| `artifacts/tokenizers/v4-a/ja-unigram-8192/`、`artifacts/token-data/corpus-v4-8k/manifest.json` | 匹配的 8K tokenizer 和 `JapaneseLMV4` 检查的标识 manifest |
| `outputs/deployment/tiny-ja-v42-epoch20-fp32-input/` | `bos`、`short`、`long`、`right_pad` 和 `changed_future` 的 PyTorch CPU FP32 logits，以及单 token KV 逐步得到的 `long_kv_step_logits` |
| `outputs/evaluation-v4-20261015/inputs/` | 四个冻结的候选池；哈希固定在 `vimeml.posttraining.infer.TRACKS` 中 |
| `outputs/evaluation-v42-20261020/` | V4.2 FP32 CUDA 分数、指标和配对比较：任务级参照 |
| `outputs/evaluation-v4-20261015/v21/` | V2.1 在同样四个候选池上的逐例分数 |
| `outputs/training-v42-20261020/` | 下载回执和训练记录，仅用于标识 |
| `artifacts/deployment/tiny-ja-v2.1-extend5-{bundle,fp32,int8-b32}-v1/` | 已部署的 V2.1 推理包和 Core ML 包，作为同一工具测量的基线 |

不包括 blind 和 test 数据、训练语料、token 存储、优化器状态和凭据。

## 参照数字

V4.2 FP32 在冻结候选池上的 Top-1：Standard 开发集 **523/728**，Legacy 回归集 **370/500**，AJIMEE **139/200**，Historical 开发集 **123/137**。V2.1 FP32 在同样候选池上：505、374、144、122。打分使用上下文与候选的联合分词、全词表后缀 logP 求和、不加末尾 EOS、不截断、稳定并列（`vimeml.benchmarks.evaluate_ajimee.rerank`）。

## 任务

1. **Mac PyTorch 基线。** 用 `JapaneseLMV4` 在 CPU FP32 上加载 `deployment.pt`。在 atol/rtol 3e-4 内复现 fixture logits，并复现四个 Top-1 计数。报告与 CUDA 分数的任何逐例排序差异，而不只是总数。
2. **FP32 Core ML 转换。** V4 还没有转换器。按固定的 V4 约定改造 V2.1 路径（`vimeml.deployment.v2.export_bundle/convert`、`scripts/ja/deployment/coreml_v2.py`）：`int32[1,T]`，1 ≤ T ≤ 128，原始 `float32[1,T,8192]` logits，iOS 18，`compute_precision=FLOAT32`，`CPU_ONLY`。同时参照 `vimeml.deployment.kv_cache` 和 `scripts/ja/deployment/coreml_kv.py`，从 `vimeml.deployment.v4.KVStepGraphV4` 构建单 token KV 图；缓存形状为 `(7,1,6,128,64)`。V1/V2 的格式和代码路径保持不变。
3. **FP32 对齐门槛。** 无状态：所有 fixture 用例都在 3e-4 以内，并满足 `right_pad`/`changed_future` 因果不变性。KV：一次性、单 token 和分块预填充对 `long_kv_step_logits` 和完整 logits，加上与 V2.1 KV 检查等价的快照/回滚检查。然后运行四个任务赛道；Top-1 必须与 PyTorch 基线一致。
4. **INT8 量化。** 使用 V2.1 配方：`linear_symmetric` int8，`per_block` 块大小 32，`weight_threshold=2048`，FP32 计算，`CPU_ONLY`。严格 logits 对齐预期会失败，与 V2.1 一样（最大误差约 1.0）。报告误差分布，而不是放宽门槛。任务级检查是四个赛道上相对 FP32 Core ML 的 Top-1，附配对的改善和退步。报告每个退步用例的 ID。
5. **测量。** 对 V4.2 FP32 和 INT8 以及 V2.1 FP32 和 INT8，用同一工具记录包大小、编译后 `.mlmodelc` 大小、加载时间、加载后和峰值进程内存，以及无状态和 KV 的单次调用延迟（p50/p95，热和冷）。使用同一台机器和同样的负载，并说明热和电源条件。Mac 计时不能替代 iPhone 测量。
6. **可选，仅在第 1–5 步通过时。** 按 `package_v2_ios.py` 的风格准备 iOS 客户端资源，并附上供 Vime 客户端使用的 fixture JSON。在独立的 Vime 仓库中做真机运行是后续步骤。

## 规则

- 不要读取 blind 或 test 数据。不要根据这些结果重新训练、选择其他 checkpoint、改变候选池或参考答案，或调整任何打分设置。
- 每次运行写到新的输出目录。绝不覆盖现有记录或 V2.1 发布资源。
- 失败、超出容差和跳过的步骤按实际记录；门槛失败是一个结果，不是需要绕过的东西。
- 源码改动提交到 `claude/mac-v42-coreml`。在 `docs/ja/reports/mac-<date>/` 下写一份注明日期的报告，并加入索引。
- 大文件按 `package_mac_results.py` 的风格以结果 ZIP 返回，附路径、大小、环境和通过/失败检查。源码改动通过 Git 和 pull request 提交。
