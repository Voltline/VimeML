"""Check saved Rust research scores against the upstream PyTorch forward pass."""
import json
import sys

from chinese_ime_research import EXP, OUT, read_rows, write

sys.path.insert(0, str(EXP / "upstream/chinese-ime-lm/neural"))
import torch
from rerank_eval import load


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    results = {}
    for name in ("pinyin-ime-reranker-4M", "pinyin-ime-reranker-25M"):
        model, cfg, index = load(str(EXP / f"models/{name}/sentence-model.safetensors"), "cpu")
        maximum, candidates, choices = 0.0, 0, 0
        for mode in ("nbest-plain", "nbest-grammar"):
            saved = {r["id"]: r for r in read_rows(OUT / f"{name}-{mode}-scores.jsonl")}
            for case in read_rows(OUT / f"rime-{mode}-cases.jsonl"):
                if case["id"] not in saved:
                    continue
                row = saved[case["id"]]
                texts = [case["candidates"][i]["text"] for i in row["comparable"]]
                if not texts:
                    continue
                for context, field in ((case["context"], "scores"), ("", "no_context_scores")):
                    room = max(0, cfg.context - max(map(len, texts)) - 1)
                    ctx = [index.get(ch, 1) for ch in context]
                    prefix = [2] + (ctx[-room:] if room else [])
                    values = []
                    for text in texts:
                        tail = [index.get(ch, 1) for ch in text][:cfg.context - len(prefix)]
                        logits, _ = model(torch.tensor([prefix + tail], dtype=torch.long))
                        start = len(prefix) - 1
                        logp = logits[0, start:start + len(tail)].log_softmax(-1)
                        values.append(logp[torch.arange(len(tail)), torch.tensor(tail)].mean().item())
                    error = max(abs(a-b) for a,b in zip(values, row[field]))
                    maximum = max(maximum, error)
                    candidates += len(values)
                    choices += max(range(len(values)), key=values.__getitem__) != max(range(len(values)), key=row[field].__getitem__)
        assert maximum < 1e-3 and choices == 0, (name, maximum, choices)
        results[name] = {"candidate_scores_compared": candidates, "max_abs_error_nats_per_char": maximum,
                         "argmax_disagreements": choices, "device": "cpu", "torch": torch.__version__}
        print(name, results[name], flush=True)
    write(OUT / "rust-pytorch-parity.json", results)


if __name__ == "__main__":
    main()
