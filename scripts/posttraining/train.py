"""Explicit V4-C1 entry; --plan never starts CUDA, W&B or training."""
import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttrain-v4-c1.toml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--offline", action="store_true", help="W&B local logs only; credentials never supplied as CLI args")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    if args.plan:
        from vimeml.posttraining.train import build_plan
        plan = build_plan(config)
        print(json.dumps({k: v for k, v in plan.items() if k not in {"code_sha256", "required_files"}}, indent=2))
        return
    output = ROOT / config["output_dir"]
    logs = ROOT / config["log_dir"]
    if args.resume and not (output / "last.pt").exists():
        parser.error("No C1 last.pt checkpoint to resume")
    if not args.resume and (output.exists() and any(output.iterdir()) or logs.exists() and any(logs.iterdir())):
        parser.error("Fresh C1 output/log directories must be empty")
    tracking = config.get("tracking", {})
    if not tracking.get("enabled", True):
        from vimeml.posttraining.train import run
        return run(config, resume=args.resume)
    import wandb
    directory = ROOT / "artifacts/tracking"
    directory.mkdir(parents=True, exist_ok=True)
    mode = "offline" if args.offline else tracking.get("mode", "online")
    # Initialize before importing SummaryWriter. New W&B launch, same group,
    # so a checkpoint rollback cannot overwrite a previous run's step history.
    with wandb.init(project=tracking["project"], entity=tracking.get("entity"), name=output.name,
                    group=output.name, mode=mode, dir=str(directory), sync_tensorboard=True,
                    save_code=False, force=mode == "online", config=config,
                    settings=wandb.Settings(disable_git=True, init_timeout=30)) as cloud:
        from vimeml.training.data import write_json
        write_json(directory / f"{output.name}-live.json", {"id": cloud.id, "url": cloud.url, "resume": args.resume})
        print(f"W&B run: {cloud.url}" if cloud.url else "W&B offline", flush=True)
        from vimeml.posttraining.train import run
        result = run(config, resume=args.resume)
        cloud.summary.update({"training_result": result})
        return result


if __name__ == "__main__":
    main()
