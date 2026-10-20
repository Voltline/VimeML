"""Modern-runtime PinyinGPT-Concat diagnostic, not a paper-score reproduction.

Uses segmented, gold-derived syllables (and their first-letter abbreviations),
the upstream segmented prompt, aligned positions, and per-syllable constraints.
This does not implement segmentation, typo correction, or nine-key input.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "artifacts/chinese-ime-research-20261017"
sys.path.insert(0, str(ROOT / "artifacts/external-eval/site-packages"))
sys.path.insert(0, str(EXP / "site-packages"))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from pypinyin import lazy_pinyin, Style
from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel, LogitsProcessor, LogitsProcessorList
from chinese_ime_research import read_rows, write, jsonl, sha, percentile


class AlignedGPT2(GPT2LMHeadModel):
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        result = super().prepare_inputs_for_generation(input_ids, **kwargs)
        n = input_ids.shape[1]
        positions = torch.arange(n, device=input_ids.device)
        positions[self.prompt_length:] -= self.prompt_length - self.target_position
        count = result["input_ids"].shape[1]
        result["position_ids"] = positions[-count:].unsqueeze(0).expand(input_ids.shape[0], -1)
        return result


class SyllableConstraint(LogitsProcessor):
    def __init__(self, prompt_length, allowed):
        self.prompt_length, self.allowed = prompt_length, allowed

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_length
        allowed = self.allowed[step]
        result = torch.full_like(scores, -float("inf"))
        result[:, allowed] = scores[:, allowed]
        return result


@torch.inference_mode()
def run(limit, output):
    torch.set_num_threads(1)
    torch.manual_seed(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False
    directory = EXP / "models/pinyingpt-concat"
    manifest = json.loads((EXP / "downloads/pinyingpt-aihi-manifest.json").read_text(encoding="utf-8-sig"))
    expected = next(s["lfs"]["sha256"] for s in manifest["siblings"] if s["rfilename"] == "pytorch_model.bin")
    assert sha(directory / "pytorch_model.bin") == expected
    extra = json.loads((directory / "additional_special_tokens.json").read_text(encoding="utf-8"))
    tokenizer = BertTokenizer.from_pretrained(directory, additional_special_tokens=extra, local_files_only=True)
    config = GPT2Config.from_pretrained(directory, local_files_only=True)
    # The published config omits the 443 added pinyin tokens present in its weights.
    config.vocab_size = len(tokenizer)
    config._attn_implementation = "eager"
    state = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model = AlignedGPT2(config)
    loaded = model.load_state_dict(state, strict=False)
    assert not loaded.missing_keys, loaded
    assert all(k.endswith(".attn.bias") or k.endswith(".attn.masked_bias") for k in loaded.unexpected_keys), loaded
    del state
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    model.generation_config.do_sample = False
    model.generation_config.eos_token_id = None
    table = json.loads((directory / "pinyin2char.json").read_text(encoding="utf-8"))
    vocab = tokenizer.get_vocab()
    allowed = {p: sorted({vocab[ch] for ch in chars if ch in vocab}) for p, chars in table.items()}
    cases = [c for c in read_rows(EXP / "cases.jsonl") if c["suite"] != "authored_robustness"]
    if limit:
        cases = cases[:limit]
    rows = []
    validations = []
    for ci, case in enumerate(cases):
        # As in upstream evaluation, pronunciation is derived from the reference text.
        full = [p[:-1] for p in lazy_pinyin(case["gold"], style=Style.TONE3, neutral_tone_with_five=True)]
        for mode, syllables in (("full", full), ("abbr", [p[0] for p in full])):
            row = {"id": case["id"], "suite": case["suite"], "mode": mode, "gold": case["gold"],
                   "syllables": syllables, "matches_raw_input": "".join(full) == case["input"].replace("'", "")}
            if any(p not in allowed or not allowed[p] or f"[{p}]" not in vocab for p in syllables):
                row["error"] = "unsupported syllable"
                rows.append(row)
                continue
            prompt = [tokenizer.cls_token_id] + tokenizer.encode(case["context"], add_special_tokens=False) + [tokenizer.sep_token_id]
            model.target_position = len(prompt)
            prompt += tokenizer.convert_tokens_to_ids([f"[{p}]" for p in syllables]) + [tokenizer.sep_token_id]
            model.prompt_length = len(prompt)
            inputs = torch.tensor([prompt], device=device)
            if not validations:
                first = model(inputs, use_cache=True)
                token = allowed[syllables[0]][0]
                appended = torch.cat([inputs, torch.tensor([[token]], device=device)], dim=1)
                positions = torch.arange(appended.shape[1], device=device).unsqueeze(0)
                positions[0, -1] = model.target_position
                fresh = model(appended, position_ids=positions).logits[:, -1]
                cached = model(torch.tensor([[token]], device=device), past_key_values=first.past_key_values,
                               position_ids=torch.tensor([[model.target_position]], device=device)).logits[:, -1]
                error = (fresh - cached).abs().max().item()
                assert error < 1e-3, error
                validations.append({"cache_vs_full_max_abs_logit_error": error})
            constraints = LogitsProcessorList([SyllableConstraint(len(prompt), [allowed[p] for p in syllables])])
            def generate():
                return model.generate(inputs, attention_mask=torch.ones_like(inputs),
                    num_beams=16, num_return_sequences=10, do_sample=False,
                    max_new_tokens=len(syllables), min_new_tokens=len(syllables),
                    bos_token_id=tokenizer.cls_token_id, eos_token_id=None,
                    pad_token_id=tokenizer.pad_token_id, logits_processor=constraints,
                    renormalize_logits=False, use_cache=True)
            if ci == 0 and mode == "full":
                generate() # Warmup outside latency observations.
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            output_ids = generate()[:, len(prompt):]
            if device == "cuda":
                torch.cuda.synchronize()
            row["ms"] = (time.perf_counter() - start) * 1000
            for sequence in output_ids.tolist():
                assert all(t in allowed[p] for t, p in zip(sequence, syllables))
            predictions = [tokenizer.decode(ids, clean_up_tokenization_spaces=False).replace(" ", "") for ids in output_ids]
            row.update(predictions=predictions, top1=predictions[0] == case["gold"], top10=case["gold"] in predictions)
            rows.append(row)
        if (ci + 1) % 10 == 0:
            print(f"{ci+1}/{len(cases)} segmented cases", flush=True)
    jsonl(output / "pinyingpt-scores.jsonl", rows)
    summary = {"model_id": manifest["id"], "revision": manifest["sha"], "sha256": expected,
        "weight_bytes": (directory / "pytorch_model.bin").stat().st_size,
        "parameters": sum(p.numel() for p in model.parameters()), "device": device,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "precision": "FP32", "position_mode": "aligned", "num_beams": 16,
        "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "validations": validations, "ignored_legacy_buffers": loaded.unexpected_keys, "groups": {}}
    for suite in sorted({r["suite"] for r in rows}):
        for mode in ("full", "abbr"):
            group = [r for r in rows if r["suite"] == suite and r["mode"] == mode]
            summary["groups"][suite + "/" + mode] = {"n": len(group), "errors": sum("error" in r for r in group),
                "top1": sum(r.get("top1", False) for r in group), "top10": sum(r.get("top10", False) for r in group),
                "raw_reading_mismatch": sum(not r["matches_raw_input"] for r in group),
                "p50_ms": percentile([r["ms"] for r in group if "ms" in r], .5),
                "p95_ms": percentile([r["ms"] for r in group if "ms" in r], .95)}
    write(output / "pinyingpt-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chinese-ime-research-20261017/pinyingpt")
    args = parser.parse_args()
    run(args.limit, args.output)
