# Local Chinese prediction demo

A browser page for trying `vimeml-tiny-zh-v1` by hand: what prediction offers, what free continuation says, and which of several candidates scores higher. It does not train, call an API or store input, and it listens on localhost only.

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/zh/tools/phrase_demo_zh.py
```

When Ready appears, open `http://127.0.0.1:8766/`. By default it loads the FP32 inference bundle in the release directory `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/inference`, the same weights as on Hugging Face; `--bundle` selects another bundle. CPU, four threads; `--port` changes the port; Ctrl+C stops it.

**The page shows the raw model output, without the keyboard's keyword blocking.** That is one of its uses: checking whether the model itself volunteers unsuitable content. The blocklist is not in the repository, and the page does not read it.

## Three modes

- **Prediction**: the same decoding as the [prediction benchmark](../benchmarks/predict-ime-zh.md#decoding-and-simulation-details). Beam 16, candidates of 2 to 6 Han characters, the 5 highest-scoring candidates that are not prefixes of each other, scored by summed log-probability. A pinyin constraint can be added: only the first syllable typed, or the initial of every character (readings come from the local rime-ice single-character table; without that table the constraint is unavailable). Context is one utterance per line, each its own sentence in the training format; when it does not fit, the earliest utterances are dropped.
- **Free continuation**: the first line is greedy decoding, the other 5 are samples (temperature 0.8, top-k 50, top-p 0.9, a new seed for each generation); duplicates are shown once. Characters are not restricted; generation stops at the end-of-utterance token or the length limit (4 to 48 tokens).
- **Candidate scoring**: only the current input serves as context, scored as in the candidate-rerank benchmark. The page sorts by score and shows each candidate's probability share among the candidates.

"Append" adds a candidate to the current input. Times are CPU inference on the development machine (model already loaded), not iPhone keyboard speed. Examples are in `configs/zh/phrase-demo-prompts-zh.json`.
