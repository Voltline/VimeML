"""Formal V4.1 random-initialized joint pretraining entry point."""
import argparse
from pathlib import Path
import sys
import tomllib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/train-v41.toml')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    config=tomllib.loads(args.config.read_text('utf-8'))
    output=ROOT/config['output_dir']
    logs=ROOT/config['log_dir']
    if not args.resume and ((output.exists() and any(output.iterdir())) or (logs.exists() and any(logs.iterdir()))):
        parser.error('Fresh V4.1 output/log directories must be empty')
    if args.resume and not (output/'last.pt').exists():
        parser.error('No V4.1 checkpoint to resume')
    import wandb
    directory=ROOT/'artifacts/tracking'
    directory.mkdir(parents=True,exist_ok=True)
    with wandb.init(project=config['tracking']['project'],name=output.name,group=output.name,
                    mode='online',dir=str(directory),sync_tensorboard=True,save_code=False,force=True,
                    config=config,settings=wandb.Settings(disable_git=True,init_timeout=30)) as cloud:
        from vimeml.training.data import write_json
        write_json(directory/f'{output.name}-live.json',{'id':cloud.id,'url':cloud.url,'resume':args.resume})
        print(f'W&B run: {cloud.url}',flush=True)
        from vimeml.training.joint_v41 import run
        result=run(config,args.resume)
        cloud.summary.update({'training_result':result})


if __name__=='__main__':
    main()
