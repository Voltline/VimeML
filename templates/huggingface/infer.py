"""Read-only FP32 inference using the included, fingerprint-checked VimeML source."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', default='今日は雨が降っているので、')
    parser.add_argument('--candidate', action='append', help='Rerank these strings after --prompt; repeat for each candidate.')
    parser.add_argument('--max-new-tokens', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.max_new_tokens < 1:
        parser.error('threads and max-new-tokens must be positive.')
    # The wrapper has no model math or scoring changes. Original code lives in source/.
    sys.path.insert(0, str(ROOT / 'source/src'))
    import torch
    from vimeml.deployment.bundle import BundleLM
    from vimeml.training.evaluate_ime import score_candidates
    torch.set_num_threads(args.threads)
    lm = BundleLM(ROOT / 'inference')
    if args.candidate:
        result = score_candidates(lm, args.prompt, args.candidate)
        result['ranked_candidates'] = sorted(result['candidates'],
            key=lambda item: item['log_probability_sum'], reverse=True)
    else:
        result = lm.generate(args.prompt, max_new_tokens=args.max_new_tokens, temperature=0)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
