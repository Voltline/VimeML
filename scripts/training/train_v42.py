"""Formal V4.2 LM+chat pretraining entry point; optional W&B tracking."""
import argparse
from pathlib import Path
import sys
import tomllib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/train-v42.toml')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true',
        help='Verify configuration, code and data identities, then exit')
    args=parser.parse_args()
    config=tomllib.loads(args.config.read_text('utf-8'))
    output=ROOT/config['output_dir']
    logs=ROOT/config['log_dir']
    if args.dry_run:
        from vimeml.training.joint_v42 import build_plan
        plan=build_plan(config,ROOT,verify_files=True)
        print({'stage':plan['stage'],'parameters':plan['parameters'],
               'max_steps':plan['settings']['max_steps'],
               'updates_per_epoch':plan['settings']['updates_per_epoch'][:3]+['...'],
               'required_files':len(plan['required_files'])})
        return
    if not args.resume and ((output.exists() and any(output.iterdir())) or (logs.exists() and any(logs.iterdir()))):
        parser.error('Fresh V4.2 output/log directories must be empty')
    if args.resume and not (output/'last.pt').exists():
        parser.error('No V4.2 checkpoint to resume')
    from vimeml.training.joint_v42 import run
    tracking=config.get('tracking',{})
    if not tracking.get('enabled'):
        result=run(config,args.resume)
        print(result,flush=True)
        return
    import wandb
    directory=ROOT/'artifacts/tracking'
    directory.mkdir(parents=True,exist_ok=True)
    with wandb.init(project=tracking.get('project','vimeml'),name=output.name,group=output.name,
                    mode=tracking.get('mode','online'),dir=str(directory),sync_tensorboard=True,
                    save_code=False,force=True,config=config,
                    settings=wandb.Settings(disable_git=True,init_timeout=30)) as cloud:
        from vimeml.training.data import write_json
        write_json(directory/f'{output.name}-live.json',{'id':cloud.id,'url':cloud.url,'resume':args.resume})
        print(f'W&B run: {cloud.url}',flush=True)
        result=run(config,args.resume)
        cloud.summary.update({'training_result':result})


if __name__=='__main__':
    main()
