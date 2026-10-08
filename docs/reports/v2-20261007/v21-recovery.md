# V2.1中断与恢复记录

2026-10-07 18:12（北京时间），无卡恢复的旧实例无活动训练进程，`vimeml-v21` 仅剩 dead screen socket。本轮新增产物已全部迁回本地。

## 检查结果

best/last 均为 step=0、epoch=0、batch_cursor=0、total_tokens=0，optimizer state 为空。日志止于初始 validation loss **4.3895957** 与计划 196,852 updates；没有后续 update、progress、metrics 或 summary。

本地加载 last 成功，46 个模型 tensor 与冻结 V2.0 best step375000 完全相同，没有可恢复的已保存训练进度。关机前可能存在未保存的更新，现有文件无法确认或恢复。step 0 不作为新的已训练模型。

## 本地备份

`handoff/vimeml-v21-rescue-20261007.tar.gz`：**93,596,818 bytes**，18 个源文件及清单。内容包括两个 checkpoint、配置/环境/计划、训练日志、TensorBoard、W&B、本轮启动脚本与来源记录。

快照与验证位于 `outputs/autodl-v21-recovery-20261007/{rescue-manifest,verification}.json`，模型位于 `artifacts/models/tiny-ja-v2.1-e16k-d320-l6-continue/`。文件清单与大小通过，checkpoint 可加载；旧实例文件及 V1/V2.0 保持原样。SSH/W&B 凭据不在归档中，日志凭据相关行已去除。

旧W&B run的本地二进制仅7 bytes，没有可补同步的完整训练历史。

## 迁移材料

| 材料 | 本地位置 |
| --- | --- |
| Token store / index / tokenizer / 原候选 | `handoff/vimeml-v2-data.tar.gz`，1,017,407,143 bytes |
| V2.0 起始权重与结果 | `artifacts/models/tiny-ja-v2.0-e16k-d320-l6/`、`handoff/vimeml-v2-results.tar.gz` |
| 原 V2.1 来源快照 | `handoff/vimeml-v21-code.tar.gz`、`handoff/source-provenance-v21.json` |
| 当前代码 | Git 仓库，包含新增 expanded IME reader |
| 新真实候选 / 复核开发集 | `artifacts/benchmarks/ime-expanded-v21-candidates-v1/`、`ime-dev-label-reviewed-v3/` |
| 中断记录 | `handoff/vimeml-v21-rescue-20261007.tar.gz` |

实例镜像复用 Python/CUDA 环境；`/root/autodl-tmp` 数据盘需单独迁移或恢复。原实例18:54恢复4090 D后已直接启动 restart1，无需迁移；从冻结V2.0 best初始化，batch512、两轮、screen与新W&B online run。旧step0与run归档；restart1和后续extend5均已结束，当前结果见 [V2.1配置](v21-plan.md)。
