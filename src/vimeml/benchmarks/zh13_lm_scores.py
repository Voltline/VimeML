"""LM rerank scores of the Standard Chinese IME benchmark for vimeml-tiny-zh-v1 checkpoints.

Scoring is V4.2's (evaluate_ime.score_candidates): BOS + left context + candidate
is tokenized jointly; the scored region starts at the common token prefix of the
context alone and every candidate sequence, so when the last context piece merges
with a candidate's first characters, that piece is rescored for every candidate
from the first divergence; no candidate-final EOS. The score is the sum of
full-vocabulary FP32 log-probabilities over the region. Only split=='development'
items are read unless run() is given split='blind' and a verified committed plan.
"""
import json
import math
from pathlib import Path

import torch

from vimeml.training.data import file_sha
from vimeml.training.joint_zh13 import ARCHITECTURE, FORMAT, SPECIAL
from vimeml.training.model_zh13 import GPTZh13Config, TinyGPTZh13

VERSION = "ime-standard-zh-v1-r1"


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load(checkpoint, tokenizer, device="cpu"):
    import sentencepiece as spm
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != FORMAT or saved.get("architecture") != ARCHITECTURE:
        raise ValueError("Not a vimeml-tiny-zh-v1 checkpoint")
    identity = saved["tokenizer_identity"]
    if file_sha(tokenizer) != identity["tokenizer_model_sha256"] or identity["special_ids"] != SPECIAL:
        raise ValueError("Tokenizer differs from the checkpoint's tokenizer identity")
    model = TinyGPTZh13(GPTZh13Config(**saved["model_config"]))
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    meta = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha(checkpoint), "step": saved["step"],
            "arm": saved["config"].get("arm"), "tokenizer_model_sha256": identity["tokenizer_model_sha256"]}
    return model, processor, meta


def common_prefix(sequences):
    length = min(map(len, sequences))
    for i in range(length):
        if any(s[i] != sequences[0][i] for s in sequences):
            return i
    return length


@torch.inference_mode()
def score_item(model, processor, context, candidates, device="cpu"):
    if not candidates or len(set(candidates)) != len(candidates) or any(not c for c in candidates):
        raise ValueError("Candidates must be distinct nonempty strings")
    context_ids = [SPECIAL["bos"], *processor.encode(context)]
    sequences = []
    for candidate in candidates:
        content = processor.encode(context + candidate)
        if processor.decode(content) != context + candidate:
            raise ValueError("Context + candidate does not roundtrip with the tokenizer")
        sequence = [SPECIAL["bos"], *content]
        if not 1 <= len(sequence) - 1 <= model.config.context_length:
            raise ValueError("Context + candidate exceeds the model context; no silent truncation")
        sequences.append(sequence)
    common = common_prefix([context_ids, *sequences])
    if common < 1 or any(len(s) <= common for s in sequences):
        raise ValueError("Every candidate needs scored targets")
    width = max(len(s) - 1 for s in sequences)
    inputs = torch.full((len(sequences), width), SPECIAL["pad"], dtype=torch.long)
    targets = inputs.clone()
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    for row, s in enumerate(sequences):
        inputs[row, :len(s) - 1] = torch.tensor(s[:-1])
        targets[row, :len(s) - 1] = torch.tensor(s[1:])
        mask[row, common - 1:len(s) - 1] = True
    logits = model(inputs.to(device)).float()
    values = logits.log_softmax(-1).gather(-1, targets.to(device).unsqueeze(-1)).squeeze(-1).cpu()
    scores = [float(values[row][mask[row]].sum()) for row in range(len(sequences))]
    if not all(math.isfinite(v) for v in scores):
        raise ValueError("Non-finite candidate score")
    return {"scores": scores, "scored_tokens": [int(mask[row].sum()) for row in range(len(sequences))],
            "common_prefix_tokens_including_bos": common, "context_tokens_including_bos": len(context_ids),
            "boundary_retokenized": any(s[:len(context_ids)] != context_ids for s in sequences)}


def run(checkpoint, tokenizer, frozen, out, track="full", device="cpu", split="development", blind_plan=None,
        loaded=None):
    """`loaded` = (model, processor, meta) scores another forward (e.g. Core ML) with the same rules."""
    if split not in ("development", "blind"):
        raise ValueError(f"Unknown split {split}")
    if split == "blind" and not blind_plan:
        raise ValueError("blind needs a verified committed plan")
    frozen, out = Path(frozen), Path(out)
    if out.exists():
        raise ValueError(f"{out} exists; choose a new file")
    manifest = json.loads((frozen / "manifest.json").read_text("utf-8"))
    if manifest["version"] != VERSION:
        raise ValueError(f"Expected {VERSION}")
    items = {i["id"]: i for i in read_jsonl(frozen / "items.jsonl")
             if i["split"] == split and (track == "full" or i["keys_abbrev"] is not None)}
    pools = {p["id"]: p for p in read_jsonl(frozen / f"pool-{track}.jsonl") if p["id"] in items}
    if set(pools) != set(items):
        raise ValueError(f"pool-{track}.jsonl does not cover every {split} item")
    model, processor, meta = loaded or load(checkpoint, tokenizer, device)
    with torch.autocast(torch.device(device).type, enabled=False):
        torch.backends.cuda.matmul.allow_tf32 = False
        retokenized = empty = fallbacks = 0
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(out.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
            for item_id in sorted(items):
                texts = [c["text"] for c in pools[item_id]["candidates"]]
                if not texts:  # empty Rime pool: no ranking; the item stays in the denominator
                    empty += 1
                    result = {"scores": []}
                else:
                    try:
                        result = score_item(model, processor, items[item_id]["left_context"], texts, device)
                    except ValueError:
                        if split != "blind":
                            raise
                        fallbacks += 1  # the benchmark rule: this item keeps the pool order
                        result = {"scores": None, "boundary_retokenized": False}
                    retokenized += result["boundary_retokenized"]
                stream.write(json.dumps({"id": item_id, "candidates": texts, **result}, ensure_ascii=False) + "\n")
        temporary.replace(out)
    summary = {**meta, "version": VERSION, "track": track, "split": split, "items": len(items),
               "boundary_retokenized_items": retokenized, "empty_pools": empty, "fallbacks": fallbacks, "scores_sha256": file_sha(out),
               "scoring": "joint tokenization, common-prefix suffix log-probability sum, FP32, no EOS",
               **({"blind_plan": blind_plan} if blind_plan else {})}
    out.with_suffix(".meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return summary
