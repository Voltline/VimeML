"""W&B training entry; synchronizes the trainer's TensorBoard events."""

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train-smoke.toml")
    parser.add_argument("--project", help="Defaults to tracking.project in the config, or vimeml.")
    parser.add_argument("--entity", help="W&B account/team; defaults to your logged-in account.")
    parser.add_argument("--name", help="Run display name; defaults to the output folder name.")
    parser.add_argument(
        "--offline", action="store_true", help="Local W&B logs only; no live remote view."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan; no login, sync or training."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume model checkpoint in a new grouped W&B run."
    )
    args = parser.parse_args(argv)
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    tracking_config = config.get("tracking", {})
    tracking = {
        "project": args.project or tracking_config.get("project", "vimeml"),
        "entity": args.entity or tracking_config.get("entity"),
        "name": args.name or output.name,
        "group": output.name,
        "mode": "offline" if args.offline else "online",
    }
    if args.dry_run:
        from vimeml.training.train import run

        run(config, resume=args.resume, dry_run=True)
        print(json.dumps({"wandb": tracking, "sync_tensorboard": True}, indent=2))
        return
    # Catch invalid output choices before creating a remote experiment.
    if args.resume:
        if not (output / "last.pt").exists():
            parser.error("No last.pt checkpoint to resume.")
    elif output.exists() and any(output.iterdir()):
        parser.error("Output is nonempty; use --resume or a new output/configuration.")
    elif (ROOT / config["log_dir"]).exists() and any((ROOT / config["log_dir"]).iterdir()):
        parser.error("TensorBoard directory is nonempty; use a new configuration.")
    try:
        import wandb
    except ImportError:
        parser.error("Install W&B for this environment and authenticate with wandb login.")
    tracking_dir = ROOT / "artifacts" / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    # Initialize before importing/creating SummaryWriter, so the SDK patches it.
    # Fresh run per launch avoids mixing rolled-back checkpoint steps into the
    # monotonic history of a previous cloud run. Runs share a comparison group.
    with wandb.init(
        **tracking,
        dir=str(tracking_dir),
        sync_tensorboard=True,
        save_code=False,
        force=not args.offline,
        config={
            "architecture": config.get("architecture", "tiny_gpt_v1"),
            "model": config["model"],
            "training": config["training"],
            "initialization": config.get("initialization"),
            "runtime": config.get("runtime"),
            "data_mixture": config.get("data_mixture"),
            "checkpoint_resume": args.resume,
        },
        settings=wandb.Settings(disable_git=True, init_timeout=30),
    ) as cloud:
        print(
            f"W&B run: {cloud.url}" if cloud.url else "W&B offline mode: logs are saved locally.",
            flush=True,
        )
        record = {
            "id": cloud.id,
            "url": cloud.url,
            **tracking,
            "training_config": str(args.config.resolve()),
            "checkpoint_resume": args.resume,
        }
        (tracking_dir / f"{output.name}-live.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        from vimeml.training.train import run

        summary = run(config, resume=args.resume)
        # Scalars flow from TensorBoard; checkpoints/corpus are not uploaded.
        cloud.summary.update({"training_result": summary})
    return summary


if __name__ == "__main__":
    main()
