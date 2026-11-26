"""ja-gen stage 2 entry point (vimeml.training.joint_ja_kana); single GPU or torchrun.

    python -X utf8 scripts/ja/training/train_ja_kana.py --config configs/ja/train-ja-kana-20m-pt.toml
    torchrun --standalone --nproc_per_node 8 scripts/ja/training/train_ja_kana.py --config configs/ja/train-ja-kana-200m-joint.toml
    ... --resume     continue from last.pt (same config, code, data and rank count)
    ... --dry-run    verify config, code and data identities and print the plan, then exit

W&B only on rank 0 and only from the environment (WANDB_API_KEY, optional WANDB_ENTITY); without
a key, or when online initialization fails, it logs offline under artifacts/tracking/<model>/.
"""
import argparse
import os
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text("utf-8"))
    output = ROOT / config["output_dir"]
    if args.dry_run:
        from vimeml.training.joint_ja_kana import build_plan
        plan = build_plan(config, ROOT, verify_files=True, world=int(os.environ.get("WORLD_SIZE", "1")))
        print({k: plan[k] for k in ("stage", "architecture", "parameters", "mix", "batch", "passes")}
              | {"arm": config["arm"], "max_steps": plan["settings"]["max_steps"],
                 "normalization": plan["settings"]["normalization"], "required_files": len(plan["required_files"])})
        return
    if args.resume and not (output / "last.pt").exists():
        parser.error("No checkpoint to resume")
    from vimeml.training.joint_ja_kana import run
    tracking = config.get("tracking", {})
    if not tracking.get("enabled") or os.environ.get("RANK", "0") != "0":
        print(run(config, args.resume)["status"], flush=True)
        return
    import wandb
    directory = ROOT / "artifacts/tracking" / output.name
    directory.mkdir(parents=True, exist_ok=True)

    def start(mode):
        return wandb.init(project=tracking.get("project", "vimeml"), entity=os.environ.get("WANDB_ENTITY"),
                          group=tracking["group"], name=config["arm"] + ("-resume" if args.resume else ""),
                          job_type="training", tags=["ja-kana", config["arm"]], mode=mode, dir=str(directory),
                          sync_tensorboard=True, save_code=False, config=config,
                          settings=wandb.Settings(disable_git=True, init_timeout=90))
    mode = tracking.get("mode", "online") if os.environ.get("WANDB_API_KEY") else "offline"
    try:
        cloud = start(mode)
    except Exception as error:
        if mode == "offline":
            raise
        print(f"W&B online initialization failed ({type(error).__name__}: {error}); logging offline", flush=True)
        cloud = start("offline")
    with cloud:
        result = run(config, args.resume)
        cloud.summary.update({"status": result["status"], "step": result["step"]})
        print(result["status"], flush=True)


if __name__ == "__main__":
    main()
