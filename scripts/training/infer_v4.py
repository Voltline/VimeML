"""Read-only V4 next-piece, beam and optional candidate scoring diagnostics."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.infer_v4 import JapaneseLMV4, score_candidates_cached
from vimeml.training.evaluate_ime import beam_suggestions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/tokenizers/v4-a/ja-unigram-8192")
    parser.add_argument("--prompt", default="今日は")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    lm = JapaneseLMV4(args.checkpoint, args.tokenizer, args.device)
    result = {"metadata": lm.metadata, "next_pieces": lm.next_pieces(args.prompt),
              "beam": beam_suggestions(lm, args.prompt, max_tokens=3, width=3, count=2)}
    if args.candidate:
        result["candidate_scores"] = score_candidates_cached(lm, args.prompt, args.candidate)
    print(json.dumps(result, ensure_ascii=False, indent=2))
