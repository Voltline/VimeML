"""Read-only FP32 prefix continuation, without training or a KV cache."""
import argparse
import json
import time
from pathlib import Path

import sentencepiece as spm
import torch

from vimeml.training.data import file_sha, write_json
from vimeml.training.model_factory import model_from_checkpoint

ROOT = Path(__file__).resolve().parents[3]


def filtered_logits(logits, forbidden, temperature=1.0, top_k=0, top_p=1.0):
    if temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling parameters.")
    logits = logits.float().clone() / temperature
    logits[list(forbidden)] = -torch.inf
    if top_k:
        cutoff = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < cutoff] = -torch.inf
    if top_p < 1:
        sorted_values, order = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_values, dim=-1)
        # Keep the token that crosses the threshold, including the first token.
        removed = probabilities.cumsum(-1) - probabilities >= top_p
        logits[order[removed]] = -torch.inf
    return logits


class JapaneseLM:
    def __init__(self, checkpoint, tokenizer_dir, device="cpu"):
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Choose CPU or CUDA for this first inference tool.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable.")
        checkpoint, tokenizer_dir = Path(checkpoint), Path(tokenizer_dir)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") not in {"vimeml_tiny_gpt_v1", "vimeml_tiny_gpt_v2"}:
            raise ValueError("Unsupported checkpoint format.")
        token_manifest_path = ROOT / saved["config"]["token_dir"] / "manifest.json"
        if file_sha(token_manifest_path) != saved["signatures"]["tokens"]:
            raise ValueError("Token manifest does not match checkpoint.")
        token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
        expected = [digest for path, digest in token_manifest["input_sha256"].items()
                    if path.replace("\\", "/").rsplit("/", 1)[-1] == "tokenizer.model"]
        model_path = tokenizer_dir / "tokenizer.model"
        if len(expected) != 1 or file_sha(model_path) != expected[0]:
            raise ValueError("Tokenizer model does not match checkpoint vocabulary.")
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))
        self.special = {name: getattr(self.processor, f"{name}_id")() for name in ("pad", "unk", "bos", "eos")}
        if self.special != token_manifest["special_ids"]:
            raise ValueError("Special token IDs do not match training.")
        self.model = model_from_checkpoint(saved)
        if self.processor.vocab_size() != self.model.config.vocab_size:
            raise ValueError("Vocabulary size mismatch.")
        self.model.to(self.device).eval()
        self.metadata = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha(checkpoint),
                         "tokenizer_sha256": expected[0], "checkpoint_step": saved["step"],
                         "model": self.model.configuration(), "device": str(self.device), "precision": "fp32",
                         "policy": "Plain prefix continuation, separate sentence; no sliding context or KV cache."}
        self.forbidden = tuple(self.special[name] for name in ("pad", "unk", "bos"))
        del saved

    def prefix_ids(self, prompt):
        content = self.processor.encode(prompt, out_type=int)
        if self.processor.decode(content) != prompt:
            raise ValueError("Prompt does not roundtrip with this tokenizer.")
        ids = [self.special["bos"], *content]
        if len(ids) > self.model.config.context_length:
            raise ValueError("Prompt exceeds context length; shorten it rather than silently truncating.")
        return ids

    @torch.inference_mode()
    def next_logits(self, ids):
        inputs = torch.tensor([ids], device=self.device, dtype=torch.long)
        return self.model(inputs)[0, -1].float()

    def next_pieces(self, prompt, count=10):
        logits = filtered_logits(self.next_logits(self.prefix_ids(prompt)), self.forbidden)
        probabilities = logits.softmax(-1)
        values, indices = torch.topk(probabilities, min(count, probabilities.numel() - len(self.forbidden)))
        return [{"id": int(index), "piece": self.processor.id_to_piece(int(index)), "probability": float(value)}
                for index, value in zip(indices, values)]

    def generate(self, prompt, max_new_tokens=32, temperature=0.8, top_k=50, top_p=0.9, seed=42):
        if max_new_tokens < 1 or temperature < 0 or top_k < 0 or not 0 < top_p <= 1:
            raise ValueError("Invalid generation parameters.")
        prefix = self.prefix_ids(prompt)
        ids = list(prefix)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        stop = "max_new_tokens"
        for _ in range(max_new_tokens):
            if len(ids) > self.model.config.context_length:
                stop = "context_limit"
                break
            logits = self.next_logits(ids)
            if temperature == 0:
                logits = filtered_logits(logits, self.forbidden)
                token = int(logits.argmax())
            else:
                logits = filtered_logits(logits, self.forbidden, temperature, top_k, top_p)
                token = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
            ids.append(token)
            if token == self.special["eos"]:
                stop = "eos"
                break
        text = self.processor.decode(ids[1:])
        return {"prompt": prompt, "text": text, "new_token_ids": ids[len(prefix):],
                "stop_reason": stop, "temperature": temperature, "top_k": top_k, "top_p": top_p,
                "seed": seed, "contains_replacement_character": "\ufffd" in text,
                "prefix_preserved": text.startswith(prompt)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/models/tiny-ja-v1/best.pt")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/tokenizers/ja-unigram-16k-v1")
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/inference-prompts.json")
    parser.add_argument("--prompt", help="Single custom prefix, instead of the fixed suite.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/model-checks/tiny-ja-v1")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive.")
    torch.set_num_threads(args.threads)
    lm = JapaneseLM(args.checkpoint, args.tokenizer, args.device)
    cases = [{"kind": "custom", "prompt": args.prompt}] if args.prompt is not None else json.loads(args.prompts.read_text(encoding="utf-8"))
    results, paragraphs = [], []
    started = time.perf_counter()
    for index, case in enumerate(cases):
        prompt = case["prompt"]
        greedy = lm.generate(prompt, args.max_new_tokens, temperature=0, seed=args.seed)
        sampled = lm.generate(prompt, args.max_new_tokens, seed=args.seed + index)
        results.append({**case, "next_pieces": lm.next_pieces(prompt), "greedy": greedy, "sampled": sampled})
        paragraph = f"Prompt: {prompt}\nGreedy: {greedy['text']}\nSampled: {sampled['text']}\n"
        paragraphs.append(paragraph)
        print(paragraph, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "samples.json", {"metadata": lm.metadata, "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "note": "Qualitative developer prompts, not held-out metrics or an iOS latency benchmark. Top pieces are subword tokens, not phrase candidates.",
        "cases": results})
    (args.output / "samples.txt").write_text("\n".join(paragraphs), encoding="utf-8")
    print(f"Samples: {args.output / 'samples.json'}")


if __name__ == "__main__":
    main()
