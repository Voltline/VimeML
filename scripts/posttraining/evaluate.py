"""Evaluate a C1 checkpoint only on the frozen nonblind snapshots."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from vimeml.posttraining.infer import (TRACKS, read_snapshot, load_conditional_checkpoint,
                                         fp32_evaluation, evaluate_snapshot)
    from vimeml.training.data import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tracks", choices=tuple(TRACKS), nargs="+", default=["standard-development"])
    parser.add_argument("--condition", choices=("context_reading", "empty_reading", "empty_context"), default="context_reading")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new evaluation output directory")
    rows = {track: read_snapshot(track) for track in args.tracks}
    model, codec, metadata = load_conditional_checkpoint(args.checkpoint,
        ROOT / "artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model", args.device)
    report = {"checkpoint": metadata, "condition": args.condition, "tracks": {}}
    with fp32_evaluation(model):
        for track, data in rows.items():
            report["tracks"][track] = evaluate_snapshot(model, codec, data, args.output / track, args.condition)
    write_json(args.output / "summary.json", report)
    print(report)


if __name__ == "__main__":
    main()
