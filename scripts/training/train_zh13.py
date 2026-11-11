"""vimeml-tiny-zh-v1 arm pretraining entry point; W&B without any machine-wide login.

Credentials come only from the environment (WANDB_API_KEY, optional WANDB_ENTITY),
as cluster_zh13.sh sets them from a private file with isolated W&B directories.
Without a key, or when online initialization fails, the run logs offline under
artifacts/tracking/<model>/ for a later `wandb sync`; training never waits on W&B.
"""
import argparse
import os
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Verify configuration, code and data identities, then exit")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text("utf-8"))
    output, logs = ROOT / config["output_dir"], ROOT / config["log_dir"]
    if args.dry_run:
        from vimeml.training.joint_zh13 import build_plan
        plan = build_plan(config, ROOT, verify_files=True)
        s = plan["settings"]
        print({"stage": plan["stage"], "arm": plan["arm"], "parameters": plan["parameters"],
               "max_steps": s["max_steps"], "epochs": s["epochs"], "milestone_steps": s["milestone_steps"][:3] + ["..."],
               "budget": plan["budget"], "required_files": len(plan["required_files"])})
        return
    if not args.resume and ((output.exists() and any(output.iterdir())) or (logs.exists() and any(logs.iterdir()))):
        parser.error("Fresh output/log directories must be empty")
    if args.resume and not (output / "last.pt").exists():
        parser.error("No checkpoint to resume")
    from vimeml.training.joint_zh13 import run
    tracking = config.get("tracking", {})
    if not tracking.get("enabled"):
        print(run(config, args.resume), flush=True)
        return
    import wandb
    from vimeml.training.data import write_json
    directory = ROOT / "artifacts/tracking" / output.name
    directory.mkdir(parents=True, exist_ok=True)

    def start(mode):
        return wandb.init(project=tracking.get("project", "vimeml"), entity=os.environ.get("WANDB_ENTITY"),
                          group=tracking["group"], name=config["arm"] + ("-resume" if args.resume else ""),
                          job_type="pretraining", tags=["zh-v1.3", config["arm"]], mode=mode, dir=str(directory),
                          sync_tensorboard=True, save_code=False, config=config,
                          settings=wandb.Settings(disable_git=True, init_timeout=90))
    mode = tracking.get("mode", "online") if os.environ.get("WANDB_API_KEY") else "offline"
    try:
        cloud = start(mode)
    except Exception as error:
        if mode == "offline":
            raise
        print(f"W&B online initialization failed ({type(error).__name__}: {error}); logging offline", flush=True)
        mode = "offline"
        cloud = start(mode)
    with cloud:
        write_json(directory / f"live-{cloud.id}.json", {"id": cloud.id, "url": cloud.url if mode != "offline" else None,
                                                         "mode": mode, "resume": args.resume, "directory": str(directory)})
        print(f"W&B run ({mode}): {cloud.url if mode != 'offline' else directory}", flush=True)
        result = run(config, args.resume)
        cloud.summary.update({"training_result": result})


if __name__ == "__main__":
    main()
