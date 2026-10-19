"""V4.1 random-initialized LM/chat/conversion pretraining, with exact epoch quotas."""
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
from vimeml.training.v41_data import IndexedConversionWindows,conversion_quota
from vimeml.training.model_v4 import TinyGPTV4,GPTV4Config
from vimeml.training.train import optimizer_for,learning_rate,atomic_checkpoint,rng_state,restore_rng,resolve_precision
from vimeml.posttraining.data import ConversionDataset,ConversionCodec,TOKENIZER_SHA
from vimeml.posttraining.streams import ConversionWindows,CommittedStream,loader_for
from vimeml.posttraining.runtime import update,restore_checkpoint,TASK_CONTRACT
from vimeml.posttraining.train import validate,checkpoint_payload
from vimeml.posttraining.infer import read_snapshot,fp32_evaluation,evaluate_snapshot

ROOT=Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def build_plan(config,root=ROOT,verify_files=True):
    root=Path(root)
    if config.get('stage')!='v4.1' or config.get('architecture')!='tiny_gpt_v4' or config.get('initialization')!={'mode':'random'}:
        raise ValueError('V4.1 requires explicit random initialization; no source checkpoint')
    GPTV4Config(**config['model'])
    s=dict(config['training'])
    if s['epochs']!=16 or s['lm_weight']!=.8 or s['conversion_weight']!=.2:
        raise ValueError('Use the frozen 16-epoch V4.1 joint objective')
    for key in ('lm_batch','conversion_batch','lm_microbatch','conversion_microbatch','epochs','num_workers',
                'cpu_threads','log_every','eval_every','checkpoint_every','validation_batch','bucket_multiplier','warmup_steps'):
        if type(s[key]) is not int or s[key]<1:
            raise ValueError(f'Invalid training integer: {key}')
    if type(s['seed']) is not int or not 0<=s['seed']<2**63:
        raise ValueError('Invalid random seed')
    for key in ('learning_rate','min_learning_rate','lm_weight','conversion_weight','beta1','beta2','weight_decay','grad_clip'):
        if not math.isfinite(s[key]) or s[key]<=0:
            raise ValueError(f'Invalid training scalar: {key}')
    if (s['device']!='cuda' or s['precision']!='bf16' or not config['runtime']['compile_backbone']
            or not 0<s['min_learning_rate']<=s['learning_rate'] or s['beta1']>=1 or s['beta2']>=1
            or not 0<=s['prefix_crop_probability']<1 or s['conversion_microbatch']<s['conversion_batch']
            or s['lm_microbatch']>s['lm_batch']):
        raise ValueError('V4.1 CUDA/precision/optimizer/microbatch contract mismatch')
    mix_dir=root/config['data_mixture']['directory']
    mix=read(mix_dir/'manifest.json')
    request=mix['request']
    if (mix['status']!='complete' or request['epochs']!=16 or len(mix['epochs'])!=16
            or request['seed']!=s['seed'] or request['crop_probability']!=s['prefix_crop_probability']
            or request['crop_minimum']!=8 or request['chat_fraction']!=.05):
        raise ValueError('Frozen mixture differs from training configuration')
    if file_sha(mix_dir/'manifest.json')!=config['dataset_sha256']:
        raise ValueError('V4.1 dataset manifest changed')
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
    conversion=root/config['conversion_dir']
    cv=read(conversion/'manifest.json')
    acceptance=read(conversion/'acceptance.json')
    signatures['conversion']=file_sha(conversion/'manifest.json')
    if signatures['conversion']!=request['conversion_manifest_sha256'] or cv['contract']['tokenizer_sha256']!=TOKENIZER_SHA or acceptance['manifest_sha256']!=signatures['conversion'] or acceptance['status']!='pass':
        raise ValueError('Conversion bank identity mismatch')
    for name,digest in cv['output_sha256'].items():
        if name.endswith('.bin'):
            required[conversion/name]=digest
    required[conversion/'manifest.json']=signatures['conversion']
    required[conversion/'acceptance.json']=file_sha(conversion/'acceptance.json')
    for name in ('old_keep','conversion_keep'):
        required[root/request[name]]=request[name+'_sha256']
    for record in mix['epochs']:
        required[mix_dir/record['file']]=record['sha256']
    required[mix_dir/'manifest.json']=config['dataset_sha256']
    required[mix_dir/'request.json']=file_sha(mix_dir/'request.json')
    tokenizer=root/config['tokenizer_dir']/'tokenizer.model'
    if file_sha(tokenizer)!=TOKENIZER_SHA:
        raise ValueError('Tokenizer changed')
    required[tokenizer]=TOKENIZER_SHA
    standard=read_snapshot('standard-development',root)
    required[root/'outputs/evaluation-v4-20261015/inputs/standard-development.json']=file_sha(root/'outputs/evaluation-v4-20261015/inputs/standard-development.json')
    code={p.relative_to(root).as_posix():file_sha(p) for p in sorted((root/'src/vimeml').rglob('*.py'))}
    code['scripts/training/train_v41.py']=file_sha(root/'scripts/training/train_v41.py')
    signatures['code']=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    if verify_files:
        for path,digest in required.items():
            if file_sha(path)!=digest:
                raise ValueError('Changed training dependency: '+str(path))
    s['updates_per_epoch']=[math.ceil(r['windows']/s['lm_batch']) for r in mix['epochs']]
    s['max_steps']=sum(s['updates_per_epoch'])
    if request['conversion_examples']<max(s['updates_per_epoch']):
        raise ValueError('Not enough conversion examples for the proportional update schedule')
    return {'stage':'v4.1','parameters':15_070_848,'settings':s,'signatures':signatures,'code_sha256':code,
        'required_files':{p.relative_to(root).as_posix():h for p,h in required.items()},
        'conversion_examples':request['conversion_examples'],'request':request,
        'standard_cases':len(standard),'task_contract':TASK_CONTRACT}


def initial_state():
    return {'epoch':0,'epoch_batch':0,'step':0,'lm_cursor':{'epoch':0,'batch':0,'offset':0},
        'conversion_cursor':{'epoch':0,'batch':0,'offset':0},'epoch_results':[],
        'best_conversion':None,'best_lm':None,'best_rerank_key':None,'best_ordinary_key':None,
        'training_seconds':0.,'totals':{n:{k:0 for k in ('examples','input_tokens','supervised_tokens','padded_positions')}
                                      for n in ('lm','conversion')}}


def validate_state(state,plan):
    e,b=state['epoch'],state['epoch_batch']
    updates=plan['settings']['updates_per_epoch']
    if not 0<=e<=len(updates) or b<0 or (e==len(updates) and b) or (e<len(updates) and b>updates[e]):
        raise ValueError('Invalid committed LM epoch cursor')
    if state['step']!=sum(updates[:e])+b:
        raise ValueError('Step does not match committed epoch updates')
    expected=e*plan['conversion_examples']+(b*plan['conversion_examples']//updates[e] if e<len(updates) else 0)
    if state['totals']['conversion']['examples']!=expected:
        raise ValueError('Conversion exposure is not exactly proportional to committed LM updates')
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
        'completed_lm_epochs':e+fraction,'completed_conversion_epochs':state['totals']['conversion']['examples']/plan['conversion_examples'],
        'lm_cursor':state['lm_cursor'],'conversion_cursor':state['conversion_cursor'],
        'totals':state['totals'],'training_seconds':state['training_seconds'],'latest_update':latest,
        'full_validation':state['epoch_results'][-1] if state['epoch_results'] else None,
        'best_conversion_loss':state['best_conversion'],'best_lm_bpc':state['best_lm'],
        'best_rerank_key':state['best_rerank_key'],'best_ordinary_key':state['best_ordinary_key']}


def combine_conversion(parts):
    """Rejoin a cursor boundary into one small conversion forward/backward."""
    if len(parts)==1:
        return parts
    lengths=torch.cat([p['lengths'] for p in parts])
    width=math.ceil(int(lengths.max())/8)*8
    pinned=parts[0]['input_ids'].is_pinned()
    ids=torch.zeros((len(lengths),width),dtype=torch.long,pin_memory=pinned)
    labels=torch.full_like(ids,-100)
    start=0
    for part in parts:
        rows=len(part['lengths'])
        columns=min(width,part['input_ids'].shape[1])
        ids[start:start+rows,:columns]=part['input_ids'][:,:columns]
        labels[start:start+rows,:columns]=part['labels'][:,:columns]
        start+=rows
    return [{'input_ids':ids,'labels':labels,'lengths':lengths}]


@torch.inference_mode()
def evaluate_ordinary(model,codec,rows,destination):
    from vimeml.training.infer_v4 import JapaneseLMV4
    from vimeml.training.evaluate_ime import score_candidates
    from vimeml.benchmarks.evaluate_ajimee import eligibility
    lm=JapaneseLMV4.__new__(JapaneseLMV4)
    lm.model,lm.processor,lm.device=model,codec.sp,next(model.parameters()).device
    lm.special={'pad':0,'unk':1,'bos':2,'eos':3}
    records=[]
    for row in rows:
        order=row['orders']['azookey']
        reason=eligibility(lm,row['left_context'],order)
        scores=None
        if reason:
            ranked=order
        else:
            scored=score_candidates(lm,row['left_context'],order)
            scores=[r['log_probability_sum'] for r in scored['candidates']]
            ranked=[order[i] for i in sorted(range(len(order)),key=lambda i:(-scores[i],i))]
        records.append({'id':row['id'],'correct':ranked[0] in row['answers'],'ranked':ranked,'scores':scores,'fallback':reason})
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'scores.jsonl').open('w',encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    result={'correct':sum(r['correct'] for r in records),'total':len(rows),'fallbacks':sum(r['fallback'] is not None for r in records),'precision':'fp32','score':'unchanged ordinary LM contextual suffix sum'}
    write_json(destination/'metrics.json',result)
    return result


def run(config,resume=False,root=ROOT):
    from torch.utils.tensorboard import SummaryWriter
    root=Path(root)
    output,log_dir=root/config['output_dir'],root/config['log_dir']
    if not resume and ((output.exists() and any(output.iterdir())) or (log_dir.exists() and any(log_dir.iterdir()))):
        raise ValueError('Fresh V4.1 requires empty output and log directories')
    print('Verifying frozen V4.1 union, tokenizer and per-epoch maps...',flush=True)
    plan=build_plan(config,root,verify_files=True)
    s=plan['settings']
    torch.set_num_threads(s['cpu_threads'])
    random.seed(s['seed'])
    torch.manual_seed(s['seed'])
    device=torch.device(s['device'])
    if not torch.cuda.is_available():
        raise ValueError('A GPU instance is required for formal V4.1 training')
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
            print('V4.1 is already complete',flush=True)
            return progress(state,plan,'complete')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'config.json',config)
    write_json(output/'plan.json',plan)
    from vimeml.training.runtime_v2 import compile_backbone
    compilation=compile_backbone(model)
    write_json(output/'environment.json',{'torch':str(torch.__version__),'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'precision':precision,'parameters':model.parameter_count(),
        'initialization':initialization,'compilation':compilation,'validation_precision':'primitive_FP32'})
    lm_train=mixture_dataset(config,root)
    conv_train=IndexedConversionWindows(ConversionDataset(root/config['conversion_dir'],'train'),root/plan['request']['conversion_keep'])
    conv_val=ConversionWindows(ConversionDataset(root/config['conversion_dir'],'validation'))
    lm_val=SentenceWindowDataset(root/config['token_dir'],root/config['index_dir'],'validation')
    chat_bank=plan['request']['joined_stores'][1]
    chat_val=SentenceWindowDataset(root/chat_bank['token_dir'],root/chat_bank['index_dir'],'validation')
    options={'workers':s['num_workers'],'multiplier':s['bucket_multiplier'],'pin_memory':True}
    lm_loader=loader_for(lm_train,s['lm_batch'],seed=s['seed'],**options)
    conv_loader=loader_for(conv_train,s['conversion_batch'],seed=s['seed']+1,**options)
    conv_stream=CommittedStream(conv_loader,state['conversion_cursor'])
    def val_loader(data):
        return loader_for(data,s['validation_batch'],workers=0,seed=s['seed'],shuffle=False,pin_memory=True)
    validation=val_loader(conv_val)
    old_validation,chat_validation=val_loader(lm_val),val_loader(chat_val)
    subset=sorted(random.Random(s['seed']+1000019).sample(range(len(conv_val)),min(4096,len(conv_val))))
    subset_loader=loader_for(conv_val,s['validation_batch'],workers=0,seed=s['seed'],indices=subset,shuffle=False,pin_memory=True)
    codec=ConversionCodec(root/config['tokenizer_dir']/'tokenizer.model')
    standard=read_snapshot('standard-development',root)
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
        payload['posttraining']['stage']='v4.1'
        payload['training_regime']='random_initialized_joint_pretraining'
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
                cursor=state['epoch_batch']
                lm_batch=lm_stream.next_batch()
                nconv=conversion_quota(cursor,updates,plan['conversion_examples'])
                conversion=combine_conversion(conv_stream.next_group(nconv))
                rate=learning_rate(state['step']+1,s)
                for group in optimizer.param_groups:
                    group['lr']=rate
                result=update(model,optimizer,conversion,[lm_batch],s,device,precision)
                torch.cuda.synchronize(device)
                state['step']+=1
                state['epoch_batch']+=1
                state['lm_cursor']=lm_stream.state_dict()
                state['conversion_cursor']=conv_stream.state_dict()
                seconds=time.perf_counter()-began
                state['training_seconds']+=seconds
                for task in ('lm','conversion'):
                    for field in state['totals'][task]:
                        state['totals'][task][field]+=result[field][task]
                if state['step']%s['log_every']==0:
                    latest={'step':state['step'],'learning_rate':rate,'seconds':seconds,**result}
                    metrics.write(json.dumps(latest)+'\n')
                    for task in ('lm','conversion'):
                        writer.add_scalar(f'{task}/train_loss',result['losses'][task],state['step'])
                        writer.add_scalar(f'{task}/supervised_tokens',state['totals'][task]['supervised_tokens'],state['step'])
                        writer.add_scalar(f'{task}/input_tokens',state['totals'][task]['input_tokens'],state['step'])
                    writer.add_scalar('optimizer/learning_rate',rate,state['step'])
                    writer.add_scalar('optimizer/grad_norm',result['grad_norm'],state['step'])
                    write_json(output/'progress.json',progress(state,plan,latest=latest))
                    print(f"step={state['step']}/{s['max_steps']} epoch={e+1} lm={result['losses']['lm']:.4f} conv={result['losses']['conversion']:.4f} seconds={seconds:.3f}",flush=True)
                if state['step']%s['checkpoint_every']==0:
                    save()
                if state['step']%s['eval_every']==0 and state['epoch_batch']<updates:
                    with fp32_evaluation(model):
                        subset_result=validate(model,subset_loader,device)
                    writer.add_scalar('conversion/fixed_subset_loss',subset_result['loss'],state['step'])
                    write_json(output/'subset-validation.json',{'step':state['step'],**subset_result})
            if requested:
                break
            if not any(r['epoch']==e+1 for r in state['epoch_results']):
                save()
                write_json(output/'progress.json',progress(state,plan,'validating'))
                destination=output/'epoch-evaluation'/f'epoch-{e+1}'
                destination.mkdir(parents=True,exist_ok=True)
                print(f'[epoch {e+1}] full old LM, chat, conversion and frozen Standard',flush=True)
                before=rng_state()
                with fp32_evaluation(model):
                    cv=validate(model,validation,device,conv_val.data.manifest['splits']['validation']['supervised_tokens_full_reading'])
                    old_stats=read(root/config['token_dir']/'manifest.json')['splits']['validation']
                    chat_stats=read(root/chat_bank['token_dir']/'manifest.json')['splits']['validation']
                    lv=validate(model,old_validation,device,old_stats['prediction_pairs'],old_stats['characters'])
                    chat=validate(model,chat_validation,device,chat_stats['prediction_pairs'],chat_stats['characters'])
                    ime=evaluate_snapshot(model,codec,standard,destination/'standard-development')
                    ordinary=evaluate_ordinary(model,codec,standard,destination/'standard-ordinary')
                restore_rng(before)
                report={'epoch':e+1,'step':state['step'],'conversion':cv,'lm':lv,'chat':chat,'ime':ime,'ordinary_ime':ordinary}
                roles=[]
                if state['best_conversion'] is None or cv['loss']<state['best_conversion']:
                    state['best_conversion']=cv['loss'];roles.append('best-conversion')
                if state['best_lm'] is None or lv['bpc']<state['best_lm']:
                    state['best_lm']=lv['bpc'];roles.append('best-lm')
                key=[ime['correct'],-cv['loss'],-state['step']]
                if state['best_rerank_key'] is None or key>state['best_rerank_key']:
                    state['best_rerank_key']=key;roles+=['best-rerank','best']
                ordinary_key=[ordinary['correct'],-lv['bpc'],-state['step']]
                if state['best_ordinary_key'] is None or ordinary_key>state['best_ordinary_key']:
                    state['best_ordinary_key']=ordinary_key;roles.append('best-ordinary')
                state['epoch_results'].append(report)
                write_json(destination/'summary.json',report)
                write_json(output/'full-validation.json',report)
                for tag,value in [('conversion/full_validation_loss',cv['loss']),('lm/full_validation_bpc',lv['bpc']),
                                  ('chat/full_validation_bpc',chat['bpc']),('ime/conditional_correct',ime['correct']),('ime/ordinary_correct',ordinary['correct'])]:
                    writer.add_scalar(tag,value,state['step'])
                for role in roles:
                    save(role)
                save(f'epoch-{e+1}')
                print(f"[epoch {e+1}] LM BPC={lv['bpc']:.6f} chat BPC={chat['bpc']:.6f} conv={cv['loss']:.6f} conditional={ime['correct']}/728 ordinary={ordinary['correct']}/728",flush=True)
            state['epoch']+=1
            state['epoch_batch']=0
            state['lm_cursor']={'epoch':state['epoch'],'batch':0,'offset':0}
            validate_state(state,plan)
            save()
        terminal='complete' if state['epoch']==s['epochs'] else 'interrupted'
        save()
        result=progress(state,plan,terminal)
        result['checkpoints']={n:str(output/f'{n}.pt') for n in ('best','best-rerank','best-ordinary','best-lm','best-conversion','last') if (output/f'{n}.pt').exists()}
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
        for data in (conv_train.data,conv_val.data,lm_val,chat_val):
            data.close()
