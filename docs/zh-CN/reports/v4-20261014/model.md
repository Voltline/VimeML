# V4-a 模型实现与已完成的本地检查

2026-10-14。本地 V4-a 准备已完成。完整基础 LM 实验的目标平台为 RTX 4090 D，配置与运行设计见 [V4 训练设计](../../training-v4.md)。

## 固定架构

[模型代码](../../../../src/vimeml/training/model_v4.py)、[V4 注册](../../../../src/vimeml/training/model_factory_v4.py)、[推理](../../../../src/vimeml/training/infer_v4.py)。

词表8192、context128、d384、7 层、6 heads/head_dim64、SwiGLU960、Pre-Norm RMSNorm eps1e-5、无 linear bias、dropout0、learned absolute positions、causal attention、tied embeddings。配置拒绝改变这些固定值。

| 部分 | 参数量 |
| --- | ---: |
| 每层 Transformer | 1,696,512 |
| 7 层主干 | 11,875,584 |
| Token embedding / 共用输出头 | 3,145,728 |
| Position embedding | 49,152 |
| Final RMSNorm | 384 |
| **总计** | **15,070,848** |

V4 的目标是减少 embedding 分配、增加主干可学习容量。V2.1 的 token embedding 为 5,242,880，Transformer blocks 为 7,253,760；V4 分别为 3,145,728 与 11,875,584。质量收益由正式训练与后续评测检验。

固定 V4 类复用 V2 已验证算子，`model.py`、`model_v2.py`、`model_factory.py` 保持原样。共享训练 orchestration 新增 V4 身份并显式选择 train/validation；index preparation 新增可选 splits 参数，旧默认仍处理三个 split。旧合成 fixture 对齐已有 tokenizer 身份和 validation 接口，没有放宽生产校验。

## 已完成的检查

30 项相关测试通过：V4 模型 5 项、V4 trainer/resume 1 项、旧 model/trainer 9 项，以及 data/inference/V2 deployment 回归 15 项。

覆盖精确参数量、共享权重/无 bias、因果性、padding 与已移位 masked loss、context128 forward、KV 增量与整段重算、缓存溢出、A/B/A 回滚、联合分词候选评分、下一 token/Beam、本地 TorchScript 图，以及训练中断恢复。FP32 对比阈值 atol=rtol=3e-4。上述是合成/开发检查，不是模型质量评测。

K/V 各为 `[7,1,6,128,64]`，FP32 两份合计 2,752,512 bytes（2.625 MiB）。回滚恢复两个完整 buffer 与长度，功能式单 token KV 图已在 PyTorch 本地检查。

本地 Windows 环境：PyTorch2.10.0+cu128、NVIDIA RTX5060 Laptop GPU，支持 BF16。合成 AdamW 更新、模型/优化器序列化与恢复，以及真实数据 forward（无 optimizer update）均通过。

| 合成 batch × context | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| 256 × 128 | 6.820 GiB | 7.424 GiB |
| 128 × 128 | 3.457 GiB | 3.772 GiB |

以上为本地 eager 合成测量，与云 GPU 的 batch512 配置分开记录。V4 中断/恢复 fixture 与不中断运行的最终权重逐项完全一致，已提交数据游标一致。Checkpoint format 为 `vimeml_tiny_gpt_v4`。

证据保存在本地 `outputs/model-checks/`，包含 `v4-a-20261014-b128/report.json`、随机初始化/合成更新 fixture 与回归日志。测试 fixture 不作为正式训练初始化；2026-10-14 未启动正式语料训练。

## 云 GPU 训练配置

[train-v4-b.toml](../../../../configs/train-v4-b.toml)描述从随机权重开始的正式训练：BF16、batch512、accumulation1、4 epochs、fused AdamW、LR6e-4、1000-step 学习率 warmup 与 cosine decay。主干编译、数据加载与 tracking 集成复用 V2.1/V3 的云 GPU 路径。Validation 保持4096个抽样窗口，并补齐对应句子的所有窗口。目前没有 V4 在目标设备上的实际吞吐与显存结果。

训练入口调整通过3项 tracking 入口回归检查，包括无网络初始化的新进程 V4 注册。配置核对确认预算为196,940次更新。CPU Dynamo fullgraph/dynamic capture 使用 eager backend，在两种输入形状下 masked loss 与所有参数梯度均对齐，并保持权重共享和 state-dict 名称。该检查验证图捕获，不代表已验证 Linux CUDA Inductor kernel 或目标设备吞吐。
