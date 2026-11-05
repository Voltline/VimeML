"""V4.5 ensemble distillation: a V4.2-recipe 15M student trained on a four-seed teacher ensemble.

Each student is the exact twin of one CE-trained V4.2-recipe seed (`TWINS`): same
configuration, seed, initialization and frozen 20-epoch mixture, so the only change is
the target. The loss per supervised token is (1 - weight) * CE(label) + weight *
CE(teacher distribution), where the teacher distribution pools the four frozen epoch-20
teachers either as a normalized geometric mean ('geo', the token-level form of the
summed-logP ranking ensemble) or as an arithmetic mixture ('mix'). Temperature is 1.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time
import tomllib
import torch
import torch.nn.functional as F

from vimeml.training.data import SentenceWindowDataset,file_sha,write_json
from vimeml.training.data_mixture import mixture_dataset
from vimeml.training.model_v4 import TinyGPTV4,GPTV4Config
from vimeml.training.train import (optimizer_for,learning_rate,atomic_checkpoint,rng_state,
                                   restore_rng,resolve_precision,amp_context)
from vimeml.posttraining.streams import CommittedStream,loader_for,microbatches
from vimeml.posttraining.train import validate
from vimeml.posttraining.data import TOKENIZER_SHA

ROOT=Path(__file__).resolve().parents[3]
STAGE='v4.5'
MIXTURE_EPOCHS=20
DISTILLATION_WEIGHT=0.5
POOLS=('geo','mix')
# Frozen teachers: epoch 20 of the four V4.2-recipe seeds, fetched from the private HF
# training-output repos at fixed revisions by prepare_v45_teachers.py.
TEACHERS={
    'v42':{'repo_id':'Voltline/vimeml-v42-training-output','revision':'99d914882ae7410bbd99d08090b7d59b3becbded',
           'path':'checkpoints/best-lm.pt','step':1081942,
           'sha256':'0e83ea7417b9816b37183044468343b151f4e610da0764c31f1f780c5fa5796d'},
    's1':{'repo_id':'Voltline/vimeml-v44-training-output','revision':'a77fd281907b24919ab3e1fa0a6feabc896096fc',
          'path':'s1/checkpoints/epoch-20.pt','step':1081944,
          'sha256':'31cc81207de132b9c8c51f975ef8a550d50f35196401c69b24f70f46f4502338'},
    's2':{'repo_id':'Voltline/vimeml-v44-training-output','revision':'a77fd281907b24919ab3e1fa0a6feabc896096fc',
          'path':'s2/checkpoints/epoch-20.pt','step':1081940,
          'sha256':'dd8a33acee77981cdb3b07b602bbdd665f6db058eed505e8bcb06a54859040cb'},
    's3':{'repo_id':'Voltline/vimeml-v44-training-output','revision':'a77fd281907b24919ab3e1fa0a6feabc896096fc',
          'path':'s3/checkpoints/epoch-20.pt','step':1081937,
          'sha256':'bac6f0fe8d20ad643b25ff8291599f74b4274efa79965dbe366972dbb6b5786e'},
}
TEACHER_DIR='artifacts/teachers/v45'
# Twin configuration of each student seed; the twin is one of the teachers.
TWINS={'v42':'configs/train-v42.toml','s1':'configs/train-v44-s1.toml',
       's2':'configs/train-v44-s2.toml','s3':'configs/train-v44-s3.toml'}
ARMS={f'{pool}-{seed}':(pool,seed) for pool in POOLS for seed in TWINS}
# Fields a student may differ from its twin in; everything else must be identical.
STUDENT_KEYS=('stage','arm','output_dir','log_dir','tracking','distillation')
TASK_CONTRACT={
    'objective':'ensemble_distillation_token_normalized',
    'tasks':['old_web_lm','chat_lm'],
    'normalization':'total supervised tokens per update',
    'lm':'(1-w)*CE(label)+w*CE(pooled teacher distribution) on all valid targets',
    'distillation_weight':DISTILLATION_WEIGHT,'temperature':1.0,
    'teachers':{k:v['sha256'] for k,v in TEACHERS.items()},
    'teacher_precision':'BF16 autocast, frozen eval mode; pooled in FP32',
    'conversion':'none; V4.1 conversion pool removed',
    'ime_validation':'none_in_training; deferred post-training over saved epoch checkpoints',
    'context_length':128,'tokenizer_sha256':TOKENIZER_SHA,
}


def teacher_file(name,root=ROOT):
    return Path(root)/TEACHER_DIR/f'{name}.pt'


def twin_config(seed,root=ROOT):
    return tomllib.loads((Path(root)/TWINS[seed]).read_text('utf-8'))


def strip_student(config):
    config={k:copy.deepcopy(v) for k,v in config.items() if k not in STUDENT_KEYS}
    config['data_mixture'].pop('chat_fraction',None)
    config['training'].pop('lm_microbatch',None)
    config['training'].pop('smoke_updates',None)
    return config


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def build_plan(config,root=ROOT,verify_files=True):
    root=Path(root)
    if config.get('stage')!=STAGE or config.get('architecture')!='tiny_gpt_v4' or config.get('initialization')!={'mode':'random'}:
        raise ValueError('V4.5 students start from random initialization (their twin\'s)')
    GPTV4Config(**config['model'])
    s=dict(config['training'])
    arm=config.get('arm')
    if arm not in ARMS:
        raise ValueError('Unknown V4.5 student arm')
    pool,seed=ARMS[arm]
    if config.get('distillation')!={'pool':pool,'twin':seed,'weight':DISTILLATION_WEIGHT,
                                    'teachers':sorted(TEACHERS)}:
        raise ValueError(f'V4.5 {arm} distillation block differs from its registered arm')
    twin=twin_config(seed,root)
    if strip_student(config)!=strip_student(twin):
        raise ValueError(f'V4.5 {arm} differs from its twin {TWINS[seed]} outside the student fields')
    if not 1<=s['epochs']<=MIXTURE_EPOCHS:
        raise ValueError('V4.5 uses at most the 20-epoch mixture plan')
    for key in ('lm_batch','lm_microbatch','epochs','num_workers','cpu_threads','log_every',
                'eval_every','checkpoint_every','validation_batch','bucket_multiplier','warmup_steps'):
        if type(s[key]) is not int or s[key]<1:
            raise ValueError(f'Invalid training integer: {key}')
    if type(s.get('smoke_updates')) is not int or s['smoke_updates']<0:
        raise ValueError('Invalid smoke update budget')
    if type(s['seed']) is not int or not 0<=s['seed']<2**63:
        raise ValueError('Invalid random seed')
    for key in ('learning_rate','min_learning_rate','beta1','beta2','weight_decay','grad_clip'):
        if not math.isfinite(s[key]) or s[key]<=0:
            raise ValueError(f'Invalid training scalar: {key}')
    compiled=config['runtime']['compile_backbone']
    if (s['device']=='cuda' and s['precision']=='bf16' and compiled):
        pass
    elif (s['device']=='cpu' and s['precision']=='fp32' and not compiled and s['smoke_updates']>0):
        pass
    else:
        raise ValueError('V4.5 formal training is CUDA/BF16/compiled; only smoke runs are CPU/FP32')
    if (not 0<s['min_learning_rate']<=s['learning_rate'] or s['beta1']>=1 or s['beta2']>=1
            or not 0<=s['prefix_crop_probability']<1 or s['lm_microbatch']>s['lm_batch']):
        raise ValueError('V4.5 optimizer/microbatch contract mismatch')
    mix_dir=root/config['data_mixture']['directory']
    mix=read(mix_dir/'manifest.json')
    request=mix['request']
    # The twin's own frozen mixture (V4.2's, or the V4.4 seed arm's), pinned by its manifest SHA.
    if (mix['status']!='complete' or len(mix['epochs'])!=request['epochs']
            or request['epochs']<MIXTURE_EPOCHS
            or request['seed']!=s['seed'] or request['crop_probability']!=s['prefix_crop_probability']
            or request['crop_minimum']!=8 or request['chat_fraction']!=.05):
        raise ValueError('Frozen twin mixture differs from training configuration')
    if file_sha(mix_dir/'manifest.json')!=config['dataset_sha256']:
        raise ValueError('V4.5 twin dataset manifest changed')
    required={}
    signatures={'dataset':config['dataset_sha256']}
    for bank in request['joined_stores']:
        folder,index=root/bank['token_dir'],root/bank['index_dir']
        manifest=read(folder/'manifest.json')
        if file_sha(folder/'manifest.json')!=bank['token_manifest_sha256'] or file_sha(index/'manifest.json')!=bank['window_manifest_sha256']:
            raise ValueError('Changed LM/chat bank manifest')
        if manifest['tokenizer_model_sha256']!=TOKENIZER_SHA or manifest['vocab_size']!=8192:
            raise ValueError('LM/chat vocabulary mismatch')
        windows=read(index/'manifest.json')
        for split in ('train','validation'):
            for suffix in ('tokens.bin','offsets.bin','sources.bin'):
                required[folder/f'{split}.{suffix}']=manifest['output_sha256'][f'{split}.{suffix}']
            required[index/f'{split}.windows.npz']=windows['splits'][split]['index_sha256']
        required[folder/'manifest.json']=bank['token_manifest_sha256']
        required[index/'manifest.json']=bank['window_manifest_sha256']
    primary=request['joined_stores'][0]
    if config['token_dir']!=primary['token_dir'] or config['index_dir']!=primary['index_dir']:
        raise ValueError('Primary model tokenizer/LM bank mismatch')
    signatures.update(tokens=primary['token_manifest_sha256'],windows=primary['window_manifest_sha256'])
    required[root/request['old_keep']]=request['old_keep_sha256']
    for record in mix['epochs']:
        required[mix_dir/record['file']]=record['sha256']
    required[mix_dir/'manifest.json']=config['dataset_sha256']
    required[mix_dir/'request.json']=file_sha(mix_dir/'request.json')
    tokenizer=root/config['tokenizer_dir']/'tokenizer.model'
    if file_sha(tokenizer)!=TOKENIZER_SHA:
        raise ValueError('Tokenizer changed')
    required[tokenizer]=TOKENIZER_SHA
    for name,teacher in TEACHERS.items():
        required[teacher_file(name,root)]=teacher['sha256']
    signatures['teachers']=TASK_CONTRACT['teachers']
    code={p.relative_to(root).as_posix():file_sha(p) for p in sorted((root/'src/vimeml').rglob('*.py'))}
    code['scripts/training/train_v45.py']=file_sha(root/'scripts/training/train_v45.py')
    signatures['code']=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    if verify_files:
        for path,digest in required.items():
            if file_sha(path)!=digest:
                raise ValueError('Changed training dependency: '+str(path))
    s['updates_per_epoch']=[math.ceil(r['windows']/s['lm_batch']) for r in mix['epochs'][:s['epochs']]]
    s['windows_per_epoch']=[r['windows'] for r in mix['epochs'][:s['epochs']]]
    s['max_steps']=sum(s['updates_per_epoch'])
    return {'stage':STAGE,'parameters':15_070_848,'settings':s,'signatures':signatures,'code_sha256':code,
        'required_files':{p.relative_to(root).as_posix():h for p,h in required.items()},
        'request':request,'task_contract':{**TASK_CONTRACT,'arm':arm,'pool':pool,'twin':seed}}


def initial_state():
    return {'epoch':0,'epoch_batch':0,'step':0,'lm_cursor':{'epoch':0,'batch':0,'offset':0},
        'epoch_results':[],'best_lm':None,'training_seconds':0.,
        'totals':{'lm':{k:0 for k in ('examples','input_tokens','supervised_tokens','padded_positions')}}}


def consumed_examples(state,plan):
    """Examples consumed by committed updates; the tail batch is smaller."""
    s=plan['settings']
    e,b=state['epoch'],state['epoch_batch']
    updates=s['updates_per_epoch']
    total=sum(s['windows_per_epoch'][:e])
    if e<len(updates) and b:
        total+=s['windows_per_epoch'][e] if b==updates[e] else b*s['lm_batch']
    return total


def validate_state(state,plan):
    e,b=state['epoch'],state['epoch_batch']
    updates=plan['settings']['updates_per_epoch']
    if not 0<=e<=len(updates) or b<0 or (e==len(updates) and b) or (e<len(updates) and b>updates[e]):
        raise ValueError('Invalid committed LM epoch cursor')
    if state['step']!=sum(updates[:e])+b:
        raise ValueError('Step does not match committed epoch updates')
    if state['totals']['lm']['examples']!=consumed_examples(state,plan):
        raise ValueError('LM exposure does not match committed updates')
    if state['lm_cursor']!={'epoch':e,'batch':b,'offset':0}:
        raise ValueError('LM stream cursor mismatch')
    epochs=[r['epoch'] for r in state['epoch_results']]
    allowed=list(range(1,e+1))
    if e<len(updates) and b==updates[e] and len(epochs)==e+1:
        allowed.append(e+1)
    if epochs!=allowed:
        raise ValueError('Invalid completed/pending validation history')


def progress(state,plan,status='running',latest=None):
    updates=plan['settings']['updates_per_epoch']
    e=state['epoch']
    fraction=state['epoch_batch']/updates[e] if e<len(updates) else 0
    return {'status':status,'step':state['step'],'max_steps':plan['settings']['max_steps'],
        'completed_lm_epochs':e+fraction,'lm_cursor':state['lm_cursor'],
        'totals':state['totals'],'training_seconds':state['training_seconds'],'latest_update':latest,
        'full_validation':state['epoch_results'][-1] if state['epoch_results'] else None,
        'best_lm_bpc':state['best_lm']}


def load_teachers(root,device,compiled):
    """Frozen teacher models; built under a forked RNG so the student's streams are untouched."""
    teachers=[]
    with torch.random.fork_rng(devices=[device] if device.type=='cuda' else []):
        for name in sorted(TEACHERS):
            saved=torch.load(teacher_file(name,root),map_location='cpu',weights_only=True)
            if (saved.get('format')!='vimeml_tiny_gpt_v4' or saved['step']!=TEACHERS[name]['step']
                    or saved.get('model_config')!=GPTV4Config().__dict__):
                raise ValueError(f'Teacher {name} is not the registered V4.2-recipe epoch-20 checkpoint')
            teacher=TinyGPTV4(GPTV4Config(**saved['model_config']))
            teacher.load_state_dict(saved['model'],strict=True)
            del saved
            teacher=teacher.to(device).eval().requires_grad_(False)
            if compiled:
                from vimeml.training.runtime_v2 import compile_backbone
                compile_backbone(teacher)
            teachers.append(teacher)
    return teachers


def pooled_teacher(teachers,ids,valid,pool,device,precision):
    """FP32 log-probabilities of the pooled teacher distribution at valid positions."""
    total=None
    with torch.no_grad():
        for teacher in teachers:
            with amp_context(device,precision):
                logits=teacher(ids)[valid]
            logp=logits.float().log_softmax(-1)
            if total is None:
                total=logp
            elif pool=='geo':
                total=total+logp
            else:
                total=torch.logaddexp(total,logp)
        if pool=='geo':
            return (total/len(teachers)).log_softmax(-1)
        return total-math.log(len(teachers))


def update_lm(model,optimizer,batches,s,device,precision,teachers,pool):
    """One atomic distillation update; sums are normalized by the update's supervised tokens."""
    denominator=sum(int((b['labels']!=-100).sum()) for b in batches)
    if denominator<=0:
        raise ValueError('The LM task needs supervised targets')
    optimizer.zero_grad(set_to_none=True)
    inputs=sum(int(b['lengths'].sum()) for b in batches)
    examples=sum(len(b['lengths']) for b in batches)
    positions=0
    sums={'loss':[],'ce':[],'soft':[],'kl':[],'teacher_ce':[]}
    w=DISTILLATION_WEIGHT
    for batch in microbatches(batches,s['lm_microbatch']):
        ids=batch['input_ids'].to(device,non_blocking=True)
        labels=batch['labels'].to(device,non_blocking=True)
        valid=labels!=-100
        target=labels[valid]
        teacher=pooled_teacher(teachers,ids,valid,pool,device,precision)
        with amp_context(device,precision):
            logits=model(ids)[valid]
        logp=logits.float().log_softmax(-1)
        ce=F.nll_loss(logp,target,reduction='sum')
        probability=teacher.exp()
        soft=-(probability*logp).sum()
        loss_sum=(1-w)*ce+w*soft
        (loss_sum/denominator).backward()
        with torch.no_grad():
            sums['loss'].append(loss_sum.detach())
            sums['ce'].append(ce.detach())
            sums['soft'].append(soft.detach())
            sums['kl'].append(soft.detach()+(probability*teacher).sum())
            sums['teacher_ce'].append(F.nll_loss(teacher,target,reduction='sum'))
        positions+=ids.numel()
    values={k:float(torch.stack(v).sum())/denominator for k,v in sums.items()}
    if not all(math.isfinite(v) for v in values.values()):
        raise FloatingPointError('Non-finite distillation training loss')
    grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),s['grad_clip'],error_if_nonfinite=True))
    optimizer.step()
    return {**values,'supervised_tokens':denominator,'input_tokens':inputs,
        'examples':examples,'padded_positions':positions,'grad_norm':grad_norm}


def checkpoint_payload(model,optimizer,state,config,plan,precision,initialization):
    return {'format':'vimeml_tiny_gpt_v4','architecture':'tiny_gpt_v4','model':model.state_dict(),
        'model_config':model.configuration(),'optimizer':optimizer.state_dict(),
        'config':config,'signatures':plan['signatures'],'precision':precision,
        'step':state['step'],'epoch':state['epoch'],'training_state':copy.deepcopy(state),
        'tokenizer_identity':{'tokenizer_model_sha256':TOKENIZER_SHA,'vocab_size':8192,
                              'special_ids':{'pad':0,'unk':1,'bos':2,'eos':3}},
        'posttraining':{'stage':STAGE,'task_contract':plan['task_contract']},
        'initialization':initialization,'rng':rng_state(),
        'scheduler':{'step':state['step'],'max_steps':plan['settings']['max_steps'],
                     'warmup_steps':plan['settings']['warmup_steps'],
                     'name':plan['settings'].get('learning_rate_schedule','cosine'),
                     'stable_steps':plan['settings'].get('stable_steps',0)}}


def restore_checkpoint(saved,model,optimizer,config,plan,precision):
    signatures=plan['signatures']
    if (saved.get('format')!='vimeml_tiny_gpt_v4' or saved.get('architecture')!='tiny_gpt_v4'
            or saved.get('model_config')!=GPTV4Config().__dict__
            or saved.get('posttraining',{}).get('stage')!=STAGE
            or saved['posttraining'].get('task_contract')!=plan['task_contract']):
        raise ValueError('Explicit compatible V4.5 checkpoint required')
    expected={'tokenizer_model_sha256':TOKENIZER_SHA,'vocab_size':8192,
              'special_ids':{'pad':0,'unk':1,'bos':2,'eos':3}}
    if saved.get('tokenizer_identity')!=expected:
        raise ValueError('V4.5 checkpoint tokenizer identity mismatch')
    if saved['config']!=config or saved['signatures']!=signatures or saved['precision']!=precision:
        raise ValueError('V4.5 resume requires identical config, code, data and precision')
    if saved['step']!=saved['training_state']['step'] or saved['epoch']!=saved['training_state']['epoch']:
        raise ValueError('Checkpoint header and committed state disagree')
    model.load_state_dict(saved['model'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    return copy.deepcopy(saved['training_state']),saved['rng'],saved['initialization']


def run(config,resume=False,root=ROOT):
    from torch.utils.tensorboard import SummaryWriter
    from vimeml.posttraining.infer import fp32_evaluation
    root=Path(root)
    output,log_dir=root/config['output_dir'],root/config['log_dir']
    if not resume and ((output.exists() and any(output.iterdir())) or (log_dir.exists() and any(log_dir.iterdir()))):
        raise ValueError('Fresh V4.5 requires empty output and log directories')
    print('Verifying frozen V4.5 arm mixture, tokenizer and per-epoch maps...',flush=True)
    plan=build_plan(config,root,verify_files=True)
    s=plan['settings']
    torch.set_num_threads(s['cpu_threads'])
    random.seed(s['seed'])
    torch.manual_seed(s['seed'])
    device=torch.device(s['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('A GPU instance is required for formal V4.5 training')
    precision=resolve_precision(device,s['precision'])
    torch.set_float32_matmul_precision('high')
    model=TinyGPTV4(GPTV4Config(**config['model'])).to(device)
    optimizer=optimizer_for(model,s,device)
    state=initial_state()
    initialization={'mode':'random','seed':s['seed'],'source_checkpoint':None,'optimizer':'fresh_AdamW'}
    restored=None
    if resume:
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        state,restored,initialization=restore_checkpoint(saved,model,optimizer,config,plan,precision)
        validate_state(state,plan)
        del saved
        if state['epoch']==s['epochs']:
            print('V4.5 arm is already complete',flush=True)
            return progress(state,plan,'complete')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'config.json',config)
    write_json(output/'plan.json',plan)
    if config['runtime']['compile_backbone']:
        from vimeml.training.runtime_v2 import compile_backbone
        compilation=compile_backbone(model)
    else:
        compilation=None
    environment={'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':str(device),
        'precision':precision,'parameters':model.parameter_count(),
        'initialization':initialization,'compilation':compilation,'validation_precision':'primitive_FP32',
        'smoke_updates':s['smoke_updates']}
    if device.type=='cuda':
        environment['gpu']=torch.cuda.get_device_name(device)
    teachers=load_teachers(root,device,config['runtime']['compile_backbone'])
    environment['teachers']={name:{**TEACHERS[name],'file':str(teacher_file(name,root))} for name in sorted(TEACHERS)}
    write_json(output/'environment.json',environment)
    lm_train=mixture_dataset(config,root)
    lm_val=SentenceWindowDataset(root/config['token_dir'],root/config['index_dir'],'validation')
    chat_bank=plan['request']['joined_stores'][1]
    chat_val=SentenceWindowDataset(root/chat_bank['token_dir'],root/chat_bank['index_dir'],'validation')
    options={'workers':s['num_workers'],'multiplier':s['bucket_multiplier'],'pin_memory':device.type=='cuda'}
    lm_loader=loader_for(lm_train,s['lm_batch'],seed=s['seed'],**options)
    def val_loader(data):
        return loader_for(data,s['validation_batch'],workers=0,seed=s['seed'],shuffle=False,
                          pin_memory=device.type=='cuda')
    old_validation,chat_validation=val_loader(lm_val),val_loader(chat_val)
    subset=sorted(random.Random(s['seed']+1000019).sample(range(len(lm_val)),min(4096,len(lm_val))))
    subset_loader=loader_for(lm_val,s['validation_batch'],workers=0,seed=s['seed'],indices=subset,
                             shuffle=False,pin_memory=device.type=='cuda')
    writer=SummaryWriter(str(log_dir),purge_step=state['step']+1 if resume else None,flush_secs=5)
    metrics=(output/'metrics.jsonl').open('a',encoding='utf-8',buffering=1)
    requested=False
    handlers={}
    def request_stop(signum,frame):
        nonlocal requested
        if requested:
            raise KeyboardInterrupt('Second interrupt; keep last committed checkpoint')
        requested=True
        print('Saving after current committed update/validation',flush=True)
    for sig in (signal.SIGINT,signal.SIGTERM):
        handlers[sig]=signal.getsignal(sig)
        signal.signal(sig,request_stop)
    def save(name='last'):
        payload=checkpoint_payload(model,optimizer,state,config,plan,precision,initialization)
        payload['training_regime']='random_initialized_ensemble_distillation'
        atomic_checkpoint(output/f'{name}.pt',payload)
    if restored is not None:
        restore_rng(restored)
    write_json(output/'progress.json',progress(state,plan,'running' if resume else 'initializing'))
    try:
        if not resume:
            save()
        while state['epoch']<s['epochs'] and not requested:
            e=state['epoch']
            lm_stream=CommittedStream(lm_loader,state['lm_cursor'])
            updates=s['updates_per_epoch'][e]
            model.train()
            while state['epoch_batch']<updates and not requested:
                began=time.perf_counter()
                lm_batch=lm_stream.next_batch()
                rate=learning_rate(state['step']+1,s)
                for group in optimizer.param_groups:
                    group['lr']=rate
                result=update_lm(model,optimizer,[lm_batch],s,device,precision,teachers,plan['task_contract']['pool'])
                if device.type=='cuda':
                    torch.cuda.synchronize(device)
                state['step']+=1
                state['epoch_batch']+=1
                state['lm_cursor']=lm_stream.state_dict()
                seconds=time.perf_counter()-began
                state['training_seconds']+=seconds
                for field in state['totals']['lm']:
                    state['totals']['lm'][field]+=result[field]
                if state['step']%s['log_every']==0:
                    latest={'step':state['step'],'learning_rate':rate,'seconds':seconds,**result}
                    metrics.write(json.dumps(latest)+'\n')
                    for key in ('loss','ce','soft','kl','teacher_ce'):
                        writer.add_scalar('lm/train_'+key,result[key],state['step'])
                    writer.add_scalar('lm/supervised_tokens',state['totals']['lm']['supervised_tokens'],state['step'])
                    writer.add_scalar('lm/input_tokens',state['totals']['lm']['input_tokens'],state['step'])
                    writer.add_scalar('optimizer/learning_rate',rate,state['step'])
                    writer.add_scalar('optimizer/grad_norm',result['grad_norm'],state['step'])
                    write_json(output/'progress.json',progress(state,plan,latest=latest))
                    print(f"step={state['step']}/{s['max_steps']} epoch={e+1} loss={result['loss']:.4f} ce={result['ce']:.4f} kl={result['kl']:.4f} seconds={seconds:.3f}",flush=True)
                if state['step']%s['checkpoint_every']==0:
                    save()
                if state['step']%s['eval_every']==0 and state['epoch_batch']<updates:
                    with fp32_evaluation(model):
                        subset_result=validate(model,subset_loader,device)
                    writer.add_scalar('lm/fixed_subset_loss',subset_result['loss'],state['step'])
                    write_json(output/'subset-validation.json',{'step':state['step'],**subset_result})
                if s['smoke_updates'] and state['step']>=s['smoke_updates']:
                    break
            if s['smoke_updates'] and state['step']>=s['smoke_updates']:
                break
            if requested:
                break
            if not any(r['epoch']==e+1 for r in state['epoch_results']):
                save()
                write_json(output/'progress.json',progress(state,plan,'validating'))
                print(f'[epoch {e+1}] full old LM and chat validation',flush=True)
                before=rng_state()
                with fp32_evaluation(model):
                    old_stats=read(root/config['token_dir']/'manifest.json')['splits']['validation']
                    chat_stats=read(root/chat_bank['token_dir']/'manifest.json')['splits']['validation']
                    lv=validate(model,old_validation,device,old_stats['prediction_pairs'],old_stats['characters'])
                    chat=validate(model,chat_validation,device,chat_stats['prediction_pairs'],chat_stats['characters'])
                restore_rng(before)
                report={'epoch':e+1,'step':state['step'],'lm':lv,'chat':chat}
                roles=[]
                if state['best_lm'] is None or lv['bpc']<state['best_lm']:
                    state['best_lm']=lv['bpc'];roles.append('best-lm')
                state['epoch_results'].append(report)
                write_json(output/'full-validation.json',report)
                for tag,value in (('lm/full_validation_bpc',lv['bpc']),('chat/full_validation_bpc',chat['bpc'])):
                    writer.add_scalar(tag,value,state['step'])
                for role in roles:
                    save(role)
                save(f'epoch-{e+1}')
                print(f"[epoch {e+1}] LM BPC={lv['bpc']:.6f} chat BPC={chat['bpc']:.6f}",flush=True)
            state['epoch']+=1
            state['epoch_batch']=0
            state['lm_cursor']={'epoch':state['epoch'],'batch':0,'offset':0}
            validate_state(state,plan)
            save()
        if s['smoke_updates'] and state['step']>=s['smoke_updates']:
            save()
            terminal='smoke_complete'
        else:
            terminal='complete' if state['epoch']==s['epochs'] else 'interrupted'
        save()
        result=progress(state,plan,terminal)
        result['checkpoints']={n:str(output/f'{n}.pt') for n in ('best-lm','last') if (output/f'{n}.pt').exists()}
        result['epoch_checkpoints']=sorted(p.name for p in output.glob('epoch-*.pt'))
        if device.type=='cuda':
            result['peak_allocated_MiB']=torch.cuda.max_memory_allocated(device)/2**20
            result['peak_reserved_MiB']=torch.cuda.max_memory_reserved(device)/2**20
        write_json(output/'summary.json',result)
        write_json(output/'progress.json',result)
        return result
    except BaseException as error:
        write_json(output/'failure.json',{'status':'failed','active':True,'error_type':type(error).__name__,
            'error':str(error),'last_committed_in_memory_step':state['step'],'safe_resume_checkpoint':str(output/'last.pt')})
        raise
    finally:
        writer.close();metrics.close()
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        lm_train.close()
        for data in (lm_val,chat_val):
            data.close()
