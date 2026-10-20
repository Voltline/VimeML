# V5 规模扩展预训练（计划）

V5 把 V4.3 的训练方案原样放大到 d512/L8，检验 15M 模型的瓶颈是否在容量。数据、seed、目标函数、优化器和训练预算都与 V4.3 controlled-10ep 相同，只改变模型规模。状态：代码与配置就绪，尚未训练。

## 动机：V4.3 已饱和

V4.3 在 V4.2 数据池上加入九个 FineWeb2-Edu 分片（跨池去重，每轮网页窗口约翻倍）。在同一份原验证文本上：

| 运行 | 监督 token | LM BPC |
| --- | ---: | ---: |
| V4.2，20 轮 | 约 12B | 3.045152 |
| V4.3 controlled，10 轮 | 约 12B | 3.048984 |
| V4.3，30 轮 | 约 36B | 3.043831 |

同样约 12B token 下，两倍的新数据没有带来 BPC 改善；token 增加到三倍，BPC 也只降低 0.0013。三次运行都收敛到约 3.04，与训练长度和数据量无关。本地四组 FP32 Top-1（Standard /728、Legacy /500、AJIMEE /200、历史 dev /137）：V4.2 第 20 轮为 523/370/139/123，V4.3 第 30 轮为 519/367/143/123，V4.3 第 27 轮为 521/372/138/124，相对 V4.2 的差异都不显著。15M 模型的容量已经用尽，V4.3 不再进行 Core ML 转换和设备测量。

V4.3 的源码与配置已导入仓库（`joint_v43.py`、`v43_data.py`、`train-v43.toml`、`train-v43-10ep.toml`），与运行记录的 `code_sha256` 逐字节一致。本地评估入口为 [`evaluate_v43.py`](../../scripts/benchmarks/evaluate_v43.py)，结果在 `outputs/evaluation-v43-20261020/`。

## 架构

| 项目 | V4.2/V4.3 | V5 |
| --- | --- | --- |
| d_model / 层数 / 头数 | 384 / 7 / 6 | 512 / 8 / 8 |
| SwiGLU 宽度 | 960 | 1280 |
| 参数 | 15,070,848 | 28,385,792 |
| 每层参数 | 1,696,512 | 3,015,680 |
| KV cache 形状 | (7,1,6,128,64) | (8,1,8,128,64) |

头宽保持 64；FFN 宽度保持 d 的 2.5 倍。其余与 V4 相同：RMSNorm、SwiGLU、无 bias、tied head、learned position、context 128。tokenizer 沿用 V4 的 8K Unigram（SHA256 `296a2bdf…81b5`）。tokenizer 只决定文本如何切分，与模型宽度无关，而且 V4.3 数据池已用它编码，所以不需要重新训练或重新测试。

V5 是独立的架构身份（`tiny_gpt_v5` / `vimeml_tiny_gpt_v5`）。V4 的固定配置与参数校验保持不变，`TinyGPTV4` 加载 V5 checkpoint 会报错，反之亦然。

## 训练设置

[`train-v5.toml`](../../configs/train-v5.toml) 与 `train-v43-10ep.toml` 的区别只有 stage、architecture、模型尺寸和输出路径：

| 项目 | 设置 |
| --- | --- |
| 数据 | 冻结的 V4.3 混合 `corpus-v43-mixed-c128`（manifest `81d7e610…`），使用 30 轮规划中的前 10 轮映射 |
| Seed | 1219827974 |
| 目标 | 普通无标记 next-token loss，按每次更新的监督 token 数归一化，5% 聊天 |
| 优化 | fused AdamW，lr 6e-4 → 6e-5 cosine，1,000 warmup，wd 0.1，clip 1.0，batch 512 |
| 精度 | BF16 AMP、编译主干；每轮完整 FP32 验证（旧 LM、扩充网页、聊天） |
| 预算 | 10 轮，约 12B 监督 token |

由于每轮数据映射、裁剪和采样都与 controlled-10ep 一致，V5 与 controlled-10ep 的差异只能来自模型规模。权重初始化随形状不同而不同，这一点无法避免。

## 集群操作

```bash
VIMEML_PYTHON=/path/to/venv/bin/python python scripts/training/train_v5.py --dry-run   # 校验配置、代码、数据身份
VIMEML_PYTHON=/path/to/venv/bin/python bash scripts/training/cluster_v5.sh             # 存在 last.pt 时自动续训
```

需要的集群数据与 V4.3 相同：`corpus-v4-8k`、`corpus-v4-c128`、`corpus-v43-exp8k`、`corpus-v41-chat-8k` 及其窗口索引、排除掩码、`corpus-v43-mixed-c128` 的前 10 个映射文件。

按计算量估计（V4.3 controlled-10ep 约 3.3 小时更新时间，V5 每 token 计算约为其 1.9 倍），单张 L20C 约需 6–7 小时更新时间。V4.3 30 轮的峰值显存为 9.7 GiB，V5 预计约 15 GiB。

## 评估与部署计划

1. **选点**：以完整 LM BPC 最佳的 epoch 为主候选；其他 epoch 只做描述，不用 IME 结果挑选。
2. **本地四组 FP32 评估**：配对对照 V4.3 controlled-10ep 的同轮 checkpoint、V4.2 和 V2.1。
3. **最终裁决**：如果 V5 有替换 V2.1 的可能，先预注册计划，再在从未评分过的 1,000 题 blind 集上评一次。
4. **部署**：Core ML 部分（`vimeml.deployment.v4*`）目前写死了 V4 的形状（如 384 宽度和 `(7,1,6,128,64)` cache），需要先推广到 V5 再转换。设备端主要风险是延迟：按 V4.2 实测线性外推，iPhone 上 predict T=128 约 9 ms，无状态重排 p95 约 230 ms，接近键盘的 250 ms 软预算，因此客户端需要接入分段 KV 打分。INT8 包约 32 MB；键盘扩展的内存上限仍待实测。

## 结论边界

V5 相对 V4.3 controlled-10ep 只改变模型规模，可以回答"更大的模型在相同数据和预算下是否更好"。它不能回答更长训练或其他数据配比的效果；也不能保证 IME 指标提升，因为 V4.2 相对 V2.1 的 Top-1 翻转中有 61% 是书写形式差异，单靠规模可能无法改善。
