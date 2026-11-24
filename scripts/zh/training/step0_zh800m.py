"""Step 0 of the zh-800m arch2 plan: where an update's time goes (src/vimeml/training/step0_zh800m.py).

usage:
  # the real data-parallel run (one process per GPU), with a profile of the last 5 updates on rank 0
  python -m torch.distributed.run --standalone --nproc_per_node 8 scripts/zh/training/step0_zh800m.py updates \
      --config configs/zh/train-zh800m-12k-c512-pinyin.toml --out outputs/step0-zh800m/ddp8 --profile 5
  # the single-process control: rank 0's share of an 8-rank run, no communication
  python -X utf8 scripts/zh/training/step0_zh800m.py updates --config configs/zh/train-zh800m-12k-c512-pinyin.toml \
      --out outputs/step0-zh800m/emulated8 --emulate-world 8 --profile 5
  # all-reduce bandwidth of the gradient size (FP32), one tensor and 25 MB buckets
  NCCL_DEBUG=INFO python -m torch.distributed.run --standalone --nproc_per_node 8 \
      scripts/zh/training/step0_zh800m.py allreduce --out outputs/step0-zh800m/allreduce
"""
import argparse
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    u = sub.add_parser("updates")
    u.add_argument("--config", type=Path, required=True)
    u.add_argument("--out", type=Path, required=True)
    u.add_argument("--updates", type=int, default=30, help="timed updates")
    u.add_argument("--warmup", type=int, default=10, help="untimed updates first (compilation)")
    u.add_argument("--profile", type=int, default=0, help="profile the last K timed updates on rank 0")
    u.add_argument("--emulate-world", type=int, default=0, help="one process running rank 0's share of N ranks")
    a = sub.add_parser("allreduce")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--gigabytes", type=float, default=3.106, help="default: 776.5M FP32 gradients")
    a.add_argument("--iterations", type=int, default=10)
    a.add_argument("--bucket-mb", type=float, default=25)
    args = parser.parse_args()
    from vimeml.training import step0_zh800m as S
    if args.command == "updates":
        config = tomllib.loads(args.config.read_text("utf-8"))
        S.run_updates(config, args.out, args.updates, args.warmup, args.profile, args.emulate_world, ROOT)
    else:
        S.run_allreduce(args.out, args.gigabytes, args.iterations, args.bucket_mb)


if __name__ == "__main__":
    main()
