# vimeml-zh-v2-20m 蒸馏计划

2026-11-16 起草。手机上只加载一个模型（见[发布范围](release-zh-v2-plan.md)），所以 v2-20m 必须能用自己的分数重打分。[200M 评测](reports/zh-200m-pinyin-eval-20261116/report.md)显示：

- r2-cont 用自己的分数重打分时只比 v1 单模型高 0.8（p = 0.49），拼音后训练削弱了它的 LM 能力；
- 12K 联合拼音 200M 生成和 LM 打分都好（自身重打分 58.8%，比 v1 高 6.2）。

本计划用这个 200M 做老师，在线蒸馏 20M，目标是在一个 20M 里同时得到生成能力和 LM 能力。

## 方法

- **老师**：`tiny-zh200m-12k-pinyin-pt-d1024/final.pt`（step 844,730，sha256 `90e48fe4…`），冻结，BF16，不求梯度，与学生吃同一个批次。两者共用 12K 分词器和拼音样本格式，可以直接做 token 级蒸馏。
- **损失**：在所有被监督的位置上，`(1 − α)·CE + α·T²·KL(老师_T ‖ 学生_T)`，α = 0.5，T = 1。plain 和四种拼音任务都蒸馏；`per_kind` 归一化不变。
- **数据和配方**：沿用 r2-base——v2 拼音数据（Rime 读音）、同样的混合比例（plain 160 / full 96 / mixed 128 / initials 96 / error 32）、同一个 seed `20261116`，所以数据顺序与 r2-base 完全相同。本轮只用 r1 数据，r2 留到下一轮。
- **代码**：[joint_zh_pinyin.py](../../src/vimeml/training/joint_zh_pinyin.py) 的可选 `[distillation]` 段。训练日志按任务类型记录 KL，开训前把老师自己的验证损失写进 `teacher_validation.json`。

## 两臂

| 臂 | 初始化 | 更新数 | 峰值 LR | warmup | 配置 |
| --- | --- | ---: | ---: | ---: | --- |
| `kd-base` | v1 基座 `tiny-zh13-12k-fresh-s1-d384-l9` | 90,000 | 3e-4 → 3e-5 | 2000 | [train-zh-pinyin-kd-base.toml](../../configs/train-zh-pinyin-kd-base.toml) |
| `kd-cont` | r2-cont final（step 60,000） | 30,000 | 1.5e-4 → 1.5e-5 | 1000 | [train-zh-pinyin-kd-cont.toml](../../configs/train-zh-pinyin-kd-cont.toml) |

- `kd-base` 与 r2-base 只差老师这一项，用来回答“蒸馏本身带来多少”。
- `kd-cont` 回答“老师能不能把 r2-cont 丢掉的 LM 能力补回来”，成本只有前者的三分之一。

## 判定

全部在 development 上进行，blind 和公开集 test 都不读。

1. **主指标**：公开拼音集 development，自身重打分（λ = 1.6）的 Top-1。与 v1 单模型（LM 模式）做句子簇 sign-flip 配对检验，必须显著更高。
2. **守门**：Standard 全拼重排的 Top-1 不能显著低于 v1。
3. **描述性**：联想 A–D 和私有基准，不决定去留。

两臂都通过时，主指标更高的那一臂作为 v2-20m 候选；两臂都没有通过，就在下一轮加入 r2 数据，或者调整 α、T。

## 偏离记录

（暂无）
