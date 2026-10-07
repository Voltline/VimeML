> 详细参考与 FP32 基线。当前阶段入口见 [文档索引](../index.md)，量化后的结果见 [Core ML 指南](../coreml.md)。

# 本地短语联想演示

复用冻结的 Tiny GPT，不继续训练、不调用 API：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/tools/phrase_demo.py
```

看到 Ready 后打开 `http://127.0.0.1:8765/`。模型加载一次，默认 CPU4线程FP32，仅监听本机；已启动时无需再运行。端口占用可加 `--port 8766`，Ctrl+C 停止。

## 交互和生成规则

输入单行句内日语前缀，选择 3～5 个建议和 4～16 个新 token 上限。Ctrl/⌘+Enter 或“生成建议”开始；“接上”原样追加，“复制”复制续写。日语 IME 正在组合输入时快捷键不会误触。不保存用户输入。

首版拒绝含句末 `。！？!?` 的前缀，也包括引用内标点；长前缀报错，不静默滑动截断。接上带句末标点的建议后，开始新一句时换掉前缀，保持句内上下文。

- 排序建议：批量 beam search，width8、alpha0.7；不改候选 reranking 使用的原始 logP。
- 随机探索：greedy + 8 条批量采样轨迹，temperature0.8、top_k50、top_p0.9；网页每次递增 seed。
- 去重并排除空输出、前缀改写或解码损坏，不凑足 5 项；保留真实文本，不用模板补写。
- 到达 token/context 上限标“可能未完”；重复字符组标“可能重复”。它们是诊断提示，不是可靠语法判定。一个 token 不等于一个词。

若词表 piece 携带第二句，展示到第一句句末并记录裁剪；原始轨迹留在固定 suite。HTTP 使用同源检查和推理互斥，生成耗时含分词和解码，不含模型加载，也不是 iPhone 延迟。

## 固定场景和审核

20 个前缀在 `configs/phrase-demo-prompts.json`。新的只读实验：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/tools/phrase_demo.py --suite --output outputs/phrase-demo/new-beam-run
.\.venv\Scripts\python.exe -X utf8 scripts/tools/phrase_demo.py --suite --suite-mode sample --output outputs/phrase-demo/new-sample-run
```

非空目录拒绝覆盖。samples.json 保存模型/tokenizer hash、参数、原始输出和停止原因；已有 beam / sample 分别在 `outputs/phrase-demo/tiny-ja-v1/` 与 `tiny-ja-v1-sample/`。Codex 逐条复核保存在 beam 目录的 review.md / review.json。

默认模式 20 个前缀中：10 个有明确自然建议，3 个语义或事实较弱，7 个明显不理想；不是唯一答案准确率。正式致谢、请求、简单动作和技术场景较好，因果与自由话题容易复述、跑偏或截断。采样改善部分场景，没有稳定解决语义问题。

本机热推理中位约 50ms，采样约 116ms。当前适合实验演示，未达到成熟输入法联想质量；权重按决定冻结，继续以候选排序为主要部署能力。当前部署入口见[Core ML指南](../coreml.md)。
