# V2.0 AutoDL 环境与运行

2026-10-07。训练已完成，最终结果见 [评测](evaluation.md)。

| 项目 | 记录 |
| --- | --- |
| 项目 / 大文件目录 | `/root/autodl-tmp/vimeml` |
| GPU / CPU / RAM | RTX 4090 D 24 GB / 18 cores / 60 GB |
| Python / Torch | 3.12.3 / 2.7.0+cu128 |
| NumPy / TensorBoard | 2.2.6 / 2.19.0 |
| 其他依赖 | SentencePiece .2.1、W&B .30.0、setuptools 80.9.0 |
| 数据包 | `handoff/vimeml-v2-data.tar.gz`，1,017,407,143 bytes |
| 训练预算 | BF16、batch 256、4 epochs / 393,704 updates |
| W&B | W&B本地记录，finished |

数据包含 token store、窗口索引、tokenizer 与原两套候选。50 个文件在传输后核对大小，使用 SFTP/gzip 完整性检查；源码快照记录 commit 与工作区改动。依赖见 [requirements-autodl.txt](../../../requirements-autodl.txt)。

## 运行入口

```bash
cd /root/autodl-tmp/vimeml
screen -dmS vimeml-v2 bash -c 'bash scripts/training/autodl_v2.sh > /root/autodl-tmp/vimeml-v2-training.log 2>&1'
```

启动时间为北京时间 10:29。启动脚本读取权限 600 的外部 W&B 环境文件，数据包和代码包不含凭据。运行指标保存于模型目录 `progress.json`，结束指标保存于 `summary.json`。

每 5,000 updates 固定子集验证，每轮完整 BPC/IME。子集展开所选句子的所有窗口，以准确计算字符分母。epoch 边界先保存 last，再记录评测与退化计数；评测中断可恢复。固定子集最低 NLL 选 best，连续两轮完整 BPC 比历史最佳高超过 .01 时停止。

BPC 与 epoch 调度的针对性记录：`outputs/model-checks/v2-epoch-check-2/report.json`。正式训练初期约 116k–124k 有效 tokens/s、峰值 allocated 显存 6,278 MiB；最终纯训练吞吐 121,228 tokens/s。不同计时范围的速度不可直接比较。
