"""V4 checkpoint inference and cached joint-tokenized candidate scoring."""
from pathlib import Path

import sentencepiece as spm
import torch

from vimeml.training.data import file_sha
from vimeml.training.infer import JapaneseLM, ROOT
from vimeml.training.model_factory_v4 import FORMAT
from vimeml.training.model_factory import model_from_checkpoint
from vimeml.training.evaluate_ime import common_prefix_length
from vimeml.tokenizer.v4 import read_json


class JapaneseLMV4(JapaneseLM):
    def __init__(self, checkpoint, tokenizer_dir, device="cpu"):
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != FORMAT or saved.get("architecture") != "tiny_gpt_v4":
            raise ValueError("Explicit V4 checkpoint required")
        token_manifest = ROOT / saved["config"]["token_dir"] / "manifest.json"
        if file_sha(token_manifest) != saved["signatures"]["tokens"]:
            raise ValueError("Token data identity changed")
        tokens = read_json(token_manifest)
        model_path = Path(tokenizer_dir) / "tokenizer.model"
        digest = file_sha(model_path)
        if digest != tokens["tokenizer_model_sha256"]:
            raise ValueError("V4 tokenizer identity changed")
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))
        self.special = {name: getattr(self.processor, f"{name}_id")() for name in ("pad", "unk", "bos", "eos")}
        identity = {"tokenizer_model_sha256": digest, "vocab_size": 8192, "special_ids": self.special}
        if (self.special != tokens["special_ids"] or self.processor.vocab_size() != 8192
                or saved.get("tokenizer_identity") != identity):
            raise ValueError("Checkpoint tokenizer contract mismatch")
        self.model = model_from_checkpoint(saved).to(self.device).eval()
        self.forbidden = (0, 1, 2, 4, 5, 6)
        self.metadata = {"architecture": "tiny_gpt_v4", "checkpoint_sha256": file_sha(Path(checkpoint)),
                         "checkpoint_step": saved["step"], "tokenizer_sha256": digest,
                         "model": self.model.configuration(), "precision": "fp32",
                         "random_initialization_test_only": saved.get("random_initialization_test_only", False)}

    def prefix_ids(self, prompt):
        ids = super().prefix_ids(prompt)
        if any(i in (4, 5, 6) for i in ids):
            raise ValueError("Reserved conversion markers are not ordinary V4-a text")
        return ids


@torch.inference_mode()
def score_candidates_cached(lm, context, candidates):
    """Identical full-vocabulary suffix logP sum; no final EOS/truncation."""
    if not candidates or len(set(candidates)) != len(candidates) or any(not c for c in candidates):
        raise ValueError("Distinct nonempty candidates required")
    context_ids = lm.prefix_ids(context)
    sequences = [[2, *lm.processor.encode(context + c)] for c in candidates]
    for text, sequence in zip(candidates, sequences, strict=True):
        if (lm.processor.decode(sequence[1:]) != context + text or any(i in range(7) for i in sequence[1:])
                or not 1 <= len(sequence) - 1 <= 128):
            raise ValueError("Invalid candidate or context overflow")
    common = common_prefix_length([context_ids, *sequences])
    if common < 1 or any(len(seq) <= common for seq in sequences):
        raise ValueError("Candidate suffix must contain prediction targets")
    shared = lm.model.new_cache()
    initial = lm.model.forward_cached(torch.tensor([sequences[0][:common]], device=lm.device), shared)[0, -1]
    results = []
    for candidate, sequence in zip(candidates, sequences, strict=True):
        cache, logits = shared.clone(), initial
        values = []
        suffix = sequence[common:]
        for index, token in enumerate(suffix):
            values.append(float(logits.float().log_softmax(-1)[token]))
            if index + 1 < len(suffix):
                logits = lm.model.forward_cached(torch.tensor([[token]], device=lm.device), cache)[0, -1]
        results.append({"text": candidate, "log_probability_sum": sum(values),
            "log_probability_mean": sum(values) / len(values), "scored_tokens": len(values),
            "token_ids": suffix, "token_log_probabilities": values,
            "pieces": [lm.processor.id_to_piece(i) for i in suffix],
            "boundary_retokenized": sequence[:len(context_ids)] != context_ids})
    return {"common_prefix_tokens_including_bos": common,
            "context_tokens_including_bos": len(context_ids), "candidates": results}
