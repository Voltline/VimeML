# V1 Mac交接历史（2026-10-06）

该次Mac快照没有VimeML Git历史，Windows采用逐文件导入；此流程已经完成，不再作为新交接方式。当前Git与ZIP约定见[2026-10-08 Mac准备](../../mac-v21-preparation.md)。

V1为7,386,624参数，INT8 block32权重、FP32计算、CPU_ONLY、最低iOS18。dev122/137、AJIMEE124/200与原FP32命中数相同，严格logits对齐失败与部分排序／分数变化仍保留。完整实测见[V1 Core ML](../../reference/coreml-v1.md)。

用户认可iPhone16 Pro Max实际输入表现。图上6.7→29.31→45.77MB是用户观察值，不是全程最大值；原私有驻留采样峰值51.922MiB，footprint峰值21.25MiB。记录约152.7秒，未固定安装构建／模型身份，不能代替长期验收。见[真实键盘](iphone-keyboard-memory.md)。

| 原始材料 | 保存位置／范围 |
| --- | --- |
| VimeML源码、结果、训练输入 | `handoff/mac-20261006-v1/VimeML-{source,results,training-inputs}.zip` |
| Vime客户端 | 同目录`Vime-client.zip`；Git基点abc3e4d、差异patch及当时资源 |
| 清单与旧校验 | 同目录`MANIFEST.json`、`SHA256SUMS.txt`，作为当时证据，不重复执行 |
| 导入与冲突 | `outputs/maintenance/mac-merge-20261006/`、`outputs/history/mac-import-20261006/` |
| 发布对照 | `artifacts/deployment/tiny-ja-v1-conservative-int8-b32-v1/`；旧发布包是早期接入快照 |

原Mac源码、失败实验、首次内存测试超时和真机trace已归档；VimeML源码导入后提交Git，Vime仓库仍独立。当前客户端以Mac现有Vime Git仓库为准，历史Swift示例不覆盖其更新。详细清单见[合并记录](merge-verification.md)。
