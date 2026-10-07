# V2.0 deterministic prefix crop

2026-10-07。`PrefixCropWindowDataset` 只用于 V2 train。概率 30%，仅裁剪 `window_start=0` 且可保留至少 8 个预测对的窗口；起点在允许的非零 offset 中均匀选择。短窗口、长句后续窗口、validation 和 test 保持原样。

裁剪直接切 input/label token 数组，不重新分词，不补 BOS，窗口终点不变。随机决策由 seed、epoch、sample index 的 64-bit 整数混合确定。Sampler 将 `(epoch, index)` 发给常驻 worker，bucket 长度使用同一裁剪规则；checkpoint 以已提交 batch cursor 恢复。

计数分开记录训练 token、裁掉 token 和窗口覆盖，完整 epoch 满足：

```text
seen windows = epochs × windows per epoch
trained tokens + dropped tokens = epochs × prediction pairs per epoch
```

## 已有验证

| 4,099 个窗口查询 | Epoch 0 | Epoch 1 |
| --- | ---: | ---: |
| Eligible | 3,759 | 3,759 |
| 实际裁剪 | 1,141 | 1,139 |
| Eligible 裁剪率 | 30.354% | 30.301% |
| 裁掉 token | 9,644 | 9,662 |

查询覆盖首批、相邻长句与末尾窗口，不是全库估计。单样本与 bucket 长度一致；0 worker 与 2 个常驻 worker 的 batch/metadata 一致；cursor=3 恢复一致。带 dropout 的 2-epoch CPU fixture 在第 5 步中断恢复后，权重与计数完全匹配连续运行（296 trained + 14 dropped = 310）。报告：`outputs/model-checks/v2-prefix-crop/report.json`。

在线记录裁剪比例、dropped tokens、有效 tokens/update 与 samples/s。`data/prefix_crop_fraction` 分母为全部窗口，因此通常低于 eligible 样本的配置概率 30%。
