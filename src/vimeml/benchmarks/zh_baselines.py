"""Baselines for vimeml-tiny-zh-v1 on the Standard zh benchmark (docs/zh-CN/benchmarks/baselines-zh-v1-plan.md).

engine: a Rime configuration ranks on its own; the first `--n` whole-input Han candidates of the
        engine (rime_candidates.py query, the same rule that built the frozen pools) are the ranking.
lm:     an external causal LM reranks the frozen pool with the v1 scoring rule: start token + left
        context + candidate tokenized jointly, the sum of FP32 full-vocabulary log-probabilities
        from the first token where any candidate differs from the context alone, no EOS.
Both write <out>/result/{items.jsonl,summary.json} in score_standard_ime_zh.py's format, so its
`compare` pairs them with the v1 runs. blind needs a committed plan that names the benchmark.
"""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

from vimeml.benchmarks.zh13_lm_scores import VERSION, common_prefix, read_jsonl
from vimeml.training.data import file_sha

FROZEN = "outputs/benchmarks/ime-standard-zh-v1/frozen-r1"
SCORER = "scripts/benchmarks/score_standard_ime_zh.py"
RIME = "scripts/benchmarks/rime_candidates.py"
OPTIONAL = "artifacts/external-eval/site-packages"
ENGINES = ("ice", "ice-gram", "luna")
EXTERNAL = {
    "qwen2.5-0.5b": {"id": "Qwen/Qwen2.5-0.5B", "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987"},
    "qwen3-0.6b-base": {"id": "Qwen/Qwen3-0.6B-Base", "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"},
}
GRAMMAR_SHA256 = "3457b69f5fe0fc4f5d211c878bbfa2e2361b90639952c9845c8e31e9d047cde6"


def scorer_module(root):
    spec = importlib.util.spec_from_file_location("score_standard_ime_zh", root / SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate(root, split, plan):
    if split not in ("development", "blind"):
        raise ValueError(f"Unknown split {split}")
    if split == "blind":
        if not plan:
            raise ValueError("blind needs --blind-plan, a committed plan naming the benchmark")
        return scorer_module(root).check_blind_plan(plan)
    return None


def load_items(root, split, track):
    return {i["id"]: i for i in read_jsonl(root / FROZEN / "items.jsonl")
            if i["split"] == split and (track == "full" or i["keys_abbrev"] is not None)}


def engine(root, variant, split, track, out, plan=None, n=20):
    root, out = Path(root), Path(out)
    plan_record = gate(root, split, plan)
    if variant not in ENGINES:
        raise ValueError(f"Unknown engine {variant}")
    if variant == "ice-gram" and file_sha(root / "datasets/ime-standard-zh-v1/engine/grammar/wanxiang-lts-zh-hans.gram") != GRAMMAR_SHA256:
        raise ValueError("grammar model differs from the registered SHA256")
    out.mkdir(parents=True)
    items = load_items(root, split, track)
    with open(out / "inputs.jsonl", "w", encoding="utf-8", newline="\n") as stream:
        for row in read_jsonl(root / FROZEN / f"rime-{track}.jsonl"):
            if row["id"] in items:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    done = subprocess.run([sys.executable, "-I", "-X", "utf8", str(root / RIME), "query", "--variant", variant, "--n", str(n),
                           str(out / "inputs.jsonl"), str(out / "candidates.jsonl")], capture_output=True, text=True, encoding="utf-8")
    if done.returncode:
        raise SystemExit(f"rime_candidates.py failed:\n{done.stderr}")
    found = {r["id"]: r for r in read_jsonl(out / "candidates.jsonl")}
    if set(found) != set(items):
        raise SystemExit("engine output does not cover every item")
    scorer = scorer_module(root)
    overlap = {r["id"] for r in read_jsonl(root / FROZEN / "overlap" / "overlap.jsonl")}
    manifest = json.loads((root / FROZEN / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item_id in sorted(items):
        item, ranked = items[item_id], found[item_id]["candidates"]
        accepted = {item["target"], *item["alternatives"]}
        rank = next((k + 1 for k, text in enumerate(ranked) if text in accepted), None)
        rows.append({"id": item_id, "source": item["source"], "dialogue": f'{item["source"]}:{item["dialogue_id"]}',
                     "length": scorer.length_bin(item["target"]), "target": item["target"],
                     "choice": ranked[0] if ranked else None, "rank": rank, "overlap": item_id in overlap})
    summary = {"version": VERSION, "track": track, "split": split, "system": f"rime-{variant}",
               "engine": found[next(iter(found))]["engine"], "scores": None, "fallbacks": 0, "blind_plan": plan_record,
               "labels": manifest["labels"], **scorer.summarize(rows),
               "sensitivity_without_overlap": scorer.rates([r for r in rows if not r["overlap"]]),
               "overlap_items": sum(r["overlap"] for r in rows)}
    result = out / "result"
    result.mkdir()
    with open(result / "items.jsonl", "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (result / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return summary


class External:
    def __init__(self, root, key, device):
        sys.path.insert(0, str(Path(root) / OPTIONAL))
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        spec = EXTERNAL[key]
        path = snapshot_download(spec["id"], revision=spec["revision"], local_files_only=True)
        self.torch, self.device, self.spec = torch, device, spec
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
        self.start = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def score(self, context, candidates):
        torch = self.torch
        context_ids = [self.start, *self.encode(context)]
        sequences = []
        for candidate in candidates:
            ids = self.encode(context + candidate)
            if self.tokenizer.decode(ids, clean_up_tokenization_spaces=False) != context + candidate:
                raise ValueError("Context + candidate does not roundtrip with the tokenizer")
            sequences.append([self.start, *ids])
        common = common_prefix([context_ids, *sequences])
        if common < 1 or any(len(s) <= common for s in sequences):
            raise ValueError("Every candidate needs scored targets")
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), self.start, dtype=torch.long)
        attention = torch.zeros_like(ids)
        for row, s in enumerate(sequences):
            ids[row, :len(s)] = torch.tensor(s)
            attention[row, :len(s)] = 1
        rows, positions, targets = [], [], []
        for row, s in enumerate(sequences):
            for position in range(common - 1, len(s) - 1):
                rows.append(row); positions.append(position); targets.append(s[position + 1])
        with torch.inference_mode():
            hidden = self.model.model(input_ids=ids.to(self.device), attention_mask=attention.to(self.device),
                                      use_cache=False).last_hidden_state
            selected = hidden[torch.tensor(rows, device=self.device), torch.tensor(positions, device=self.device)]
            logits = self.model.lm_head(selected).float().log_softmax(-1)
            values = logits.gather(-1, torch.tensor(targets, device=self.device).unsqueeze(-1)).squeeze(-1).cpu()
        scores = [0.0] * len(sequences)
        for row, value in zip(rows, values.tolist()):
            scores[row] += value
        if not all(math.isfinite(v) for v in scores):
            raise ValueError("Non-finite candidate score")
        return scores, any(s[:len(context_ids)] != context_ids for s in sequences)


def lm(root, key, split, track, out, plan=None, device="cuda"):
    root, out = Path(root), Path(out)
    gate(root, split, plan)
    if key not in EXTERNAL:
        raise ValueError(f"Unknown model {key}")
    out.mkdir(parents=True)
    items = load_items(root, split, track)
    pools = {p["id"]: p for p in read_jsonl(root / FROZEN / f"pool-{track}.jsonl") if p["id"] in items}
    if set(pools) != set(items):
        raise SystemExit(f"pool-{track}.jsonl does not cover every {split} item")
    model = External(root, key, device)
    fallbacks = retokenized = empty = 0
    with open(out / "scores.jsonl", "w", encoding="utf-8", newline="\n") as stream:
        for item_id in sorted(items):
            texts = [c["text"] for c in pools[item_id]["candidates"]]
            scores = []
            if not texts:
                empty += 1
            else:
                try:
                    scores, changed = model.score(items[item_id]["left_context"], texts)
                    retokenized += changed
                except ValueError:
                    fallbacks += 1  # the benchmark rule: this item keeps the pool order
                    scores = None
            stream.write(json.dumps({"id": item_id, "candidates": texts, "scores": scores}, ensure_ascii=False) + "\n")
    meta = {"model": model.spec, "parameters": model.parameters, "dtype": "float32", "device": device, "version": VERSION,
            "track": track, "split": split, "items": len(items), "empty_pools": empty, "fallbacks": fallbacks,
            "boundary_retokenized_items": retokenized, "scores_sha256": file_sha(out / "scores.jsonl"),
            "scoring": "start <|endoftext|> + left context + candidate, joint tokenization, common-prefix suffix log-probability sum, FP32, no EOS"}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    command = [sys.executable, "-I", "-X", "utf8", str(root / SCORER), "score", str(root / FROZEN), str(out / "result"),
               "--track", track, "--split", split, "--scores", str(out / "scores.jsonl"), "--name", key]
    if plan:
        command += ["--blind-plan", str(plan)]
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if done.returncode:
        raise SystemExit(f"scorer failed:\n{done.stderr}")
    return json.loads((out / "result" / "summary.json").read_text(encoding="utf-8"))
