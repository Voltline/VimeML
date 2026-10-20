"""V4.2 random-initialized LM/chat pretraining; V4.1 minus the conversion pool."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time
import torch

from vimeml.training.data import SentenceWindowDataset,file_sha,write_json
from vimeml.training.data_mixture import mixture_dataset
from vimeml.training.model_v4 import TinyGPTV4,GPTV4Config
from vimeml.training.train import (optimizer_for,learning_rate,atomic_checkpoint,rng_state,
                                   restore_rng,resolve_precision,amp_context)
from vimeml.posttraining.streams import CommittedStream,loader_for,microbatches
from vimeml.posttraining.train import validate
from vimeml.posttraining.data import TOKENIZER_SHA

ROOT=Path(__file__).resolve().parents[3]
STAGE='v4.2'
TASK_CONTRACT={
    'objective':'single_task_lm_token_normalized',
    'tasks':['old_web_lm','chat_lm'],
    'normalization':'total supervised tokens per update',
    'lm':'ordinary_unmarked_LM_all_valid_targets',
    'conversion':'none; V4.1 conversion pool removed',
    'ime_validation':'none_in_training; deferred post-training over saved epoch checkpoints',
    'context_length':128,'tokenizer_sha256':TOKENIZER_SHA,
    'chat_fraction':0.05,
}


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def build_plan(config,root=ROOT,verify_files=True):
    root=Path(root)
    if config.get('stage')!=STAGE or config.get('architecture')!='tiny_gpt_v4' or config.get('initialization')!={'mode':'random'}:
        raise ValueError('V4.2 requires explicit random initialization; no source checkpoint')
    GPTV4Config(**config['model'])
    s=dict(config['training'])
    if s['epochs']!=20:
        raise ValueError('Use the frozen 20-epoch V4.2 budget')
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
        raise ValueError('V4.2 formal training is CUDA/BF16/compiled; only smoke runs are CPU/FP32')
    if (not 0<s['min_learning_rate']<=s['learning_rate'] or s['beta1']>=1 or s['beta2']>=1
            or not 0<=s['prefix_crop_probability']<1 or s['lm_microbatch']>s['lm_batch']):
        raise ValueError('V4.2 optimizer/microbatch contract mismatch')
    mix_dir=root/config['data_mixture']['directory']
    mix=read(mix_dir/'manifest.json')
    request=mix['request']
    if (mix['status']!='complete' or request['epochs']!=20 or len(mix['epochs'])!=20
            or request['seed']!=s['seed'] or request['crop_probability']!=s['prefix_crop_probability']
            or request['crop_minimum']!=8 or request['chat_fraction']!=.05):
        raise ValueError('Frozen V4.2 mixture differs from training configuration')
    if file_sha(mix_dir/'manifest.json')!=config['dataset_sha256']:
        raise ValueError('V4.2 dataset manifest changed')
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
    code={p.relative_to(root).as_posix():file_sha(p) for p in sorted((root/'src/vimeml').rglob('*.py'))}
    code['scripts/training/train_v42.py']=file_sha(root/'scripts/training/train_v42.py')
    signatures['code']=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    if verify_files:
        for path,digest in required.items():
            if file_sha(path)!=digest:
                raise ValueError('Changed training dependency: '+str(path))
    s['updates_per_epoch']=[math.ceil(r['windows']/s['lm_batch']) for r in mix['epochs']]
    s['windows_per_epoch']=[r['windows'] for r in mix['epochs']]
    s['max_steps']=sum(s['updates_per_epoch'])
    return {'stage':STAGE,'parameters':15_070_848,'settings':s,'signatures':signatures,'code_sha256':code,
        'required_files':{p.relative_to(root).as_posix():h for p,h in required.items()},
        'request':request,'task_contract':TASK_CONTRACT}


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


def update_lm(model,optimizer,batches,s,device,precision):
    """One atomic logical LM update; never average unequal microbatch means."""
    denominator=sum(int((b['labels']!=-100).sum()) for b in batches)
    if denominator<=0:
        raise ValueError('The LM task needs supervised targets')
    optimizer.zero_grad(set_to_none=True)
    inputs=sum(int(b['lengths'].sum()) for b in batches)
    examples=sum(len(b['lengths']) for b in batches)
    positions=0
    pieces=[]
    for batch in microbatches(batches,s['lm_microbatch']):
        ids=batch['input_ids'].to(device,non_blocking=True)
        labels=batch['labels'].to(device,non_blocking=True)
        with amp_context(device,precision):
            result=model(ids,labels)
            loss=result['loss_sum']/denominator
        pieces.append(result['loss_sum'].detach())
        loss.backward()
        positions+=ids.numel()
    value=float(torch.stack(pieces).sum())
    if not math.isfinite(value):
        raise FloatingPointError('Non-finite LM training loss')
    grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),s['grad_clip'],error_if_nonfinite=True))
    optimizer.step()
    return {'loss':value/denominator,'supervised_tokens':denominator,'input_tokens':inputs,
        'examples':examples,'padded_positions':positions,'grad_norm':grad_norm}


def checkpoint_payload(model,optimizer,state,config,plan,precision,initialization):
    return {'format':'vimeml_tiny_gpt_v4','architecture':'tiny_gpt_v4','model':model.state_dict(),
        'model_config':model.configuration(),'optimizer':optimizer.state_dict(),
        'config':config,'signatures':plan['signatures'],'precision':precision,
        'step':state['step'],'epoch':state['epoch'],'training_state':copy.deepcopy(state),
        'tokenizer_identity':{'tokenizer_model_sha256':TOKENIZER_SHA,'vocab_size':8192,
                              'special_ids':{'pad':0,'unk':1,'bos':2,'eos':3}},
        'posttraining':{'stage':STAGE,'task_contract':TASK_CONTRACT},
        'initialization':initialization,'rng':rng_state(),
        'scheduler':{'step':state['step'],'max_steps':plan['settings']['max_steps'],
                     'warmup_steps':plan['settings']['warmup_steps'],'name':'cosine'}}


def restore_checkpoint(saved,model,optimizer,config,signatures,precision):
    if (saved.get('format')!='vimeml_tiny_gpt_v4' or saved.get('architecture')!='tiny_gpt_v4'
            or saved.get('model_config')!=GPTV4Config().__dict__
            or saved.get('posttraining',{}).get('stage')!=STAGE
            or saved['posttraining'].get('task_contract')!=TASK_CONTRACT):
        raise ValueError('Explicit compatible V4.2 checkpoint required')
    expected={'tokenizer_model_sha256':TOKENIZER_SHA,'vocab_size':8192,
              'special_ids':{'pad':0,'unk':1,'bos':2,'eos':3}}
    if saved.get('tokenizer_identity')!=expected:
        raise ValueError('V4.2 checkpoint tokenizer identity mismatch')
    if saved['config']!=config or saved['signatures']!=signatures or saved['precision']!=precision:
        raise ValueError('V4.2 resume requires identical config, code, data and precision')
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
        raise ValueError('Fresh V4.2 requires empty output and log directories')
    print('Verifying frozen V4.2 union, tokenizer and per-epoch maps...',flush=True)
    plan=build_plan(config,root,verify_files=True)
    s=plan['settings']
    torch.set_num_threads(s['cpu_threads'])
    random.seed(s['seed'])
    torch.manual_seed(s['seed'])
    device=torch.device(s['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('A GPU instance is required for formal V4.2 training')
    precision=resolve_precision(device,s['precision'])
    torch.set_float32_matmul_precision('high')
    model=TinyGPTV4(GPTV4Config(**config['model'])).to(device)
    optimizer=optimizer_for(model,s,device)
    state=initial_state()
    initialization={'mode':'random','seed':s['seed'],'source_checkpoint':None,'optimizer':'fresh_AdamW'}
    restored=None
    if resume:
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        state,restored,initialization=restore_checkpoint(saved,model,optimizer,config,plan['signatures'],precision)
        validate_state(state,plan)
        del saved
        if state['epoch']==s['epochs']:
            print('V4.2 is already complete',flush=True)
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
        payload['training_regime']='random_initialized_lm_chat_pretraining'
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
                result=update_lm(model,optimizer,[lm_batch],s,device,precision)
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
                    writer.add_scalar('lm/train_loss',result['loss'],state['step'])
                    writer.add_scalar('lm/supervised_tokens',state['totals']['lm']['supervised_tokens'],state['step'])
                    writer.add_scalar('lm/input_tokens',state['totals']['lm']['input_tokens'],state['step'])
                    writer.add_scalar('optimizer/learning_rate',rate,state['step'])
                    writer.add_scalar('optimizer/grad_norm',result['grad_norm'],state['step'])
                    write_json(output/'progress.json',progress(state,plan,latest=latest))
                    print(f"step={state['step']}/{s['max_steps']} epoch={e+1} lm={result['loss']:.4f} seconds={seconds:.3f}",flush=True)
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
