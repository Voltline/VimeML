"""Desktop latency of the full zh pinyin pipeline in PyTorch: Vime bucketed search plus self-rescoring.

usage:
  python -X utf8 scripts/benchmarks/latency_zh.py --checkpoint CKPT --out FILE.json [--device cuda|cpu]
      [--backend eager|graph] [--dtype fp32|fp16|bf16] [--threads N] [--every 6] [--mode task|lm] [--no-rescore]

Per input of ime-pinyin-zh-v1 development (every Nth record, all of its inputs: full, mixed,
initials; in-field context): search (beam 16, top 10) and, unless --no-rescore, the model's own LM
score of the 10 candidates after the context (pinyin_gen_zh.prior_scores), timed end to end with
the device synchronized. Model time (prefill, decode, rescoring forward) is reported separately; the
rest is the search's Python work. One warm-up pass over the first 5 inputs is not counted.

--backend graph replays the decode step from CUDA graphs (zh_torch_graph.GraphBackend; the capture
time is reported and not counted); --dtype casts the weights, which the rescoring forward uses too.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from vimeml.benchmarks import pinyin_gen_zh as pg  # noqa: E402
from vimeml.benchmarks.zh13_lm_scores import load  # noqa: E402
from vimeml.deployment import zh_search as zs  # noqa: E402
from vimeml.deployment.zh_torch_backend import ModelBatchBackend  # noqa: E402
from vimeml.deployment.zh_torch_graph import GraphBackend  # noqa: E402
from vimeml.posttraining.zh_readings import ReadingTable  # noqa: E402

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class Timed:
    """Wraps a backend; adds the synchronized wall time of every call to `seconds`."""

    def __init__(self, inner, device):
        self.inner, self.device, self.seconds = inner, device, 0.0

    def _time(self, fn, *args):
        sync(self.device)
        start = time.perf_counter()
        out = fn(*args)
        sync(self.device)
        self.seconds += time.perf_counter() - start
        return out

    def prefill(self, *args):
        return self._time(self.inner.prefill, *args)

    def decode(self, *args):
        return self._time(self.inner.decode, *args)


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summary(rows, key):
    v = [r[key] for r in rows]
    return {"n": len(v), "p50": 1000 * pct(v, 0.5), "p95": 1000 * pct(v, 0.95), "mean": 1000 * statistics.fmean(v),
            "max": 1000 * max(v)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", choices=("eager", "graph"), default="eager")
    p.add_argument("--dtype", choices=tuple(DTYPES), default="fp32")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--every", type=int, default=6, help="take every Nth development record")
    p.add_argument("--mode", choices=("task", "lm"), default="task")
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--no-rescore", action="store_true")
    p.add_argument("--tokenizer", default=str(ROOT / "artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model"))
    p.add_argument("--readings", default=str(ROOT / "outputs/zh-readings/dict/vime_pinyin.dict.yaml"))
    p.add_argument("--frozen", default=str(ROOT / "outputs/benchmarks/ime-pinyin-zh-v1/frozen-r1"))
    args = p.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists")
    if args.threads:
        torch.set_num_threads(args.threads)
    model, processor, meta = load(args.checkpoint, args.tokenizer, args.device)
    capture = None
    if args.backend == "graph":
        start = time.perf_counter()
        backend = GraphBackend(model, args.device, rows=args.beam, dtype=DTYPES[args.dtype])
        capture = time.perf_counter() - start
    else:
        backend = ModelBatchBackend(model, args.device)            # casts to FP32
        if args.dtype != "fp32":
            model.to(DTYPES[args.dtype])
    timed = Timed(backend, args.device)
    index = zs.Index(processor, ReadingTable(args.readings))
    records, contexts = pg.load_set(args.frozen)
    items = [(r["id"], kind, keys, contexts.get(r["id"], "")) for r in records[::args.every] for kind, keys in pg.inputs(r)]

    def one(keys, context):
        timed.seconds = 0.0
        sync(args.device)
        start = time.perf_counter()
        with torch.inference_mode():
            ranked, stats = zs.search(timed, processor, index, keys, context, args.mode, args.beam, 10)
            searched = time.perf_counter()
            texts = [t for t, _, _ in ranked]
            if not args.no_rescore and texts:
                pg.prior_scores(model, processor, context, texts, args.device)
            sync(args.device)
        end = time.perf_counter()
        return {"total": end - start, "search": searched - start, "rescore": end - searched, "model": timed.seconds,
                "python": searched - start - timed.seconds, "calls": stats.calls, "rows": stats.rows,
                "candidates": len(texts)}

    for _, _, keys, context in items[:5]:
        one(keys, context)
    rows = []
    for rid, kind, keys, context in items:
        rows.append({"id": rid, "kind": kind, "keys": len(keys), **one(keys, context)})
    result = {"checkpoint": Path(args.checkpoint).parent.name, "checkpoint_sha256": meta["checkpoint_sha256"],
              "device": args.device, "backend": args.backend, "capture_seconds": capture,
              "dtype": args.dtype, "threads": torch.get_num_threads(), "mode": args.mode,
              "beam": args.beam, "rescore": not args.no_rescore, "every": args.every,
              "torch": torch.__version__, "python": platform.python_version(),
              "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
              "cpu": platform.processor(),
              "summary": {kind: {k: summary([r for r in rows if kind in ("all", r["kind"])], k)
                                 for k in ("total", "search", "rescore", "model", "python")}
                          for kind in ("all", "full", "mixed", "initials")},
              "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", "utf-8")
    s = result["summary"]["all"]
    print(f"{result['checkpoint']} {args.device} {args.backend} {args.dtype}: n {s['total']['n']}, total p50/p95 "
          f"{s['total']['p50']:.0f}/{s['total']['p95']:.0f} ms (model {s['model']['p50']:.0f}, python "
          f"{s['python']['p50']:.0f}, rescore {s['rescore']['p50']:.0f})")


if __name__ == "__main__":
    main()
