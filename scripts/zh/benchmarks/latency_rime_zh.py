"""Desktop latency of the Rime path in PyTorch: librime + rime-ice candidates, reranked by a zh model.

usage:
  python -X utf8 scripts/zh/benchmarks/latency_rime_zh.py --checkpoint CKPT --out FILE.json [--device cuda|cpu]
      [--threads N] [--every 6] [--n 16]

Per input of ime-pinyin-zh-v1 development (every Nth record, all of its inputs, in-field context):
Rime's candidates for the keys (the ice engine of rime_candidates.py, deployed for the Standard
benchmark), the first --n that are all Han and as long as the reference (the Standard benchmark's
rule), then the model's LM score of those after the context in one batch
(zh13_lm_scores.score_item), timed separately with the device synchronized. Top-1 after reranking is
recorded as a check. One warm-up pass over the first 5 inputs is not counted.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/zh/benchmarks"))

import torch  # noqa: E402

import rime_candidates as rc  # noqa: E402
from latency_zh import summary, sync  # noqa: E402
from vimeml.benchmarks import pinyin_gen_zh as pg  # noqa: E402
from vimeml.benchmarks.zh13_lm_scores import load, score_item  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--every", type=int, default=6)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--scan", type=int, default=300)
    p.add_argument("--tokenizer", default=str(ROOT / "artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model"))
    p.add_argument("--frozen", default=str(ROOT / "outputs/benchmarks/ime-pinyin-zh-v1/frozen-r1"))
    args = p.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists")
    if args.threads:
        torch.set_num_threads(args.threads)
    v = rc.VARIANTS["ice"]
    rime = rc.Rime(str(v["shared"]), str(v["work"]), str(rc.ENGINE / "librime/dist/lib/rime.dll"))
    session = rime.session()
    model, processor, meta = load(args.checkpoint, args.tokenizer, args.device)
    model.float().eval()
    records, contexts = pg.load_set(args.frozen)
    items = [(r["id"], kind, keys, contexts.get(r["id"], ""), r["text"])
             for r in records[::args.every] for kind, keys in pg.inputs(r)]

    def one(keys, context, text):
        start = time.perf_counter()
        cands = rc.full(rime.candidates(session, keys, args.scan), len(text), args.n)
        listed = time.perf_counter()
        best = None
        if len(cands) > 1:
            with torch.inference_mode():
                scores = score_item(model, processor, context, cands, args.device)["scores"]
            sync(args.device)
            best = cands[max(range(len(cands)), key=lambda i: scores[i])]
        elif cands:
            best = cands[0]
        end = time.perf_counter()
        return {"total": end - start, "rime": listed - start, "rerank": end - listed, "candidates": len(cands),
                "top1": best == text}

    for _, _, keys, context, text in items[:5]:
        one(keys, context, text)
    rows = [{"id": rid, "kind": kind, "keys": len(keys), **one(keys, context, text)}
            for rid, kind, keys, context, text in items]
    result = {"checkpoint": Path(args.checkpoint).parent.name, "checkpoint_sha256": meta["checkpoint_sha256"],
              "engine": f"librime {rime.version} + {rime.label}", "device": args.device,
              "threads": torch.get_num_threads(), "n": args.n, "every": args.every, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
              "cpu": platform.processor(),
              "summary": {kind: {**{k: summary([r for r in rows if kind in ("all", r["kind"])], k)
                                    for k in ("total", "rime", "rerank")},
                                 "top1": 100 * sum(r["top1"] for r in rows if kind in ("all", r["kind"]))
                                 / max(1, sum(kind in ("all", r["kind"]) for r in rows))}
                          for kind in ("all", "full", "mixed", "initials")},
              "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", "utf-8")
    s = result["summary"]["all"]
    print(f"rime + {result['checkpoint']} {args.device}: n {s['total']['n']}, total p50/p95 "
          f"{s['total']['p50']:.0f}/{s['total']['p95']:.0f} ms (rime {s['rime']['p50']:.0f}, rerank "
          f"{s['rerank']['p50']:.0f}), top-1 {s['top1']:.1f}")


if __name__ == "__main__":
    main()
