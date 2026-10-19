"""Prepare V2.1 LM + V3 natural chat + C1r2 conversion without test/blind reads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys
import unicodedata
import numpy as np
import sentencepiece as spm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vimeml.training.data import file_sha,write_json,prepare_indexes
from vimeml.training.v41_data import prepare_maps
from vimeml.posttraining.data import TOKENIZER_SHA

OLD=ROOT/'artifacts/token-data/corpus-v4-8k'
CHAT=ROOT/'artifacts/token-data/corpus-v3-half/chat'
DEST=ROOT/'artifacts/token-data/corpus-v41-chat-8k'
INDEX=ROOT/'artifacts/training-data/corpus-v41-chat-c128'
MIX=ROOT/'artifacts/training-data/corpus-v41-mixed-c128'
CONV=ROOT/'artifacts/posttrain-data/v4-c1r2'


def canonical(text):
    return ' '.join(unicodedata.normalize('NFKC',text).split())


def key(text):
    return hashlib.blake2b(canonical(text).encode(),digest_size=16).digest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps-only',action='store_true')
    args=parser.parse_args()
    if args.maps_only:
        prepare_maps(MIX)
        return
    for d in (DEST,INDEX,MIX):
        d.mkdir(parents=True,exist_ok=True)
    if (MIX/'request.json').exists():
        prepare_maps(MIX)
        return
    seed=json.loads((ROOT/'outputs/remote/v41-20261016/randomization.json').read_text())['seed']
    old=json.loads((OLD/'manifest.json').read_text('utf-8'))
    chat_old=json.loads((CHAT/'manifest.json').read_text('utf-8'))
    tokenizer=ROOT/'artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model'
    assert file_sha(tokenizer)==TOKENIZER_SHA
    sp=spm.SentencePieceProcessor(model_file=str(tokenizer))
    rows={s:[json.loads(line) for line in (CHAT/f'{s}.jsonl').read_text('utf-8').splitlines()] for s in ('train','validation')}
    assert all(len(rows[s])==chat_old['splits'][s]['sentences'] for s in rows)
    assert all(r['source'] in ('real-persona-chat','mrmp-chat') for s in rows for r in rows[s])
    assert not ({r['doc'] for r in rows['train']}&{r['doc'] for r in rows['validation']})
    held={key(line) for line in (ROOT/'outputs/corpus-fast-v1/validation.txt').open(encoding='utf-8')}
    held.update(key(r['text']) for r in rows['validation'])
    # Only explicit development/regression provenance; never open old V3 blind filters.
    for track in ('standard-development','legacy-regression','ajimee','historical-development'):
        for r in json.loads((ROOT/f'outputs/evaluation-v4-20261015/inputs/{track}.json').read_text('utf-8')):
            text=r.get('provenance',{}).get('original_message')
            if text:
                held.add(key(text))
    chat_keys={key(r['text']) for r in rows['train']}
    overlap=set()
    keep=np.ones(old['splits']['train']['sentences'],dtype=np.bool_)
    counts=Counter()
    print('Scanning only frozen V2.1 training text for validation/cross-pool exact overlap',flush=True)
    actual=hashlib.sha256()
    count=0
    with (ROOT/'outputs/corpus-fast-v1/train.txt').open('rb') as stream:
        for i,raw in enumerate(stream):
            actual.update(raw)
            fingerprint=key(raw.decode('utf-8').rstrip('\r\n'))
            if fingerprint in held:
                keep[i]=False
                counts['old_heldout_exact']+=1
            elif fingerprint in chat_keys:
                overlap.add(fingerprint)
            count=i+1
            if count%5000000==0:
                print(json.dumps({'old_rows':count,'removed':int((~keep[:count]).sum())}),flush=True)
    assert count==len(keep)
    assert actual.hexdigest()=='9c1b4585adaf0518dc0275e7f8086e967db26f9797fb571e1ff481aeadf2ff8f'
    np.save(MIX/'old.keep.npy',keep,allow_pickle=False)
    source_ids={'mrmp-chat':0,'real-persona-chat':1}
    stats={}
    for split in ('validation','train'):
        tokens,offsets,sources=bytearray(),[0],bytearray()
        chars=content=longest=0
        primary=Counter()
        accepted_records=[]
        texts=[]
        for row in rows[split]:
            fingerprint=key(row['text'])
            if split=='train' and fingerprint in held:
                counts['chat_heldout_exact']+=1
                continue
            if split=='train' and fingerprint in overlap:
                counts['chat_exact_duplicate_of_old']+=1
                continue
            texts.append(row['text'])
            accepted_records.append(row)
        for start in range(0,len(texts),10000):
            batch=texts[start:start+10000]
            encoded=sp.encode(batch,num_threads=8)
            decoded=sp.decode(encoded,num_threads=8)
            for row,text,ids,restored in zip(accepted_records[start:start+10000],batch,encoded,decoded,strict=True):
                if restored!=text or not ids or any(i<7 or i>=8192 for i in ids):
                    raise ValueError('Chat roundtrip, reserved marker or token ID error')
                full=[2,*ids,3]
                tokens.extend(struct.pack(f'<{len(full)}H',*full))
                offsets.append(offsets[-1]+len(full))
                sources.append(source_ids[row['source']])
                chars+=len(text)
                content+=len(ids)
                longest=max(longest,len(full))
                primary[row['source']]+=1
        (DEST/f'{split}.tokens.bin').write_bytes(tokens)
        np.asarray(offsets,dtype='<u8').tofile(DEST/f'{split}.offsets.bin')
        (DEST/f'{split}.sources.bin').write_bytes(sources)
        n=len(offsets)-1
        stats[split]={'sentences':n,'stored_tokens':offsets[-1],'content_tokens':content,'prediction_pairs':offsets[-1]-n,
            'characters':chars,'max_sequence_tokens':longest,'primary_sources':dict(primary),
            'file_bytes':{s:(DEST/f'{split}.{s}').stat().st_size for s in ('tokens.bin','offsets.bin','sources.bin')}}
    manifest={'format':'vimeml_sentence_tokens_v1','status':'complete','vocab_size':8192,'token_dtype':'uint16_le',
        'offset_dtype':'uint64_le','offset_unit':'tokens','special_ids':{'pad':0,'unk':1,'bos':2,'eos':3},
        'tokenizer_model_sha256':TOKENIZER_SHA,'source_ids':source_ids,'splits':stats,
        'available_splits':['train','validation'],'sequence_policy':'BOS + unchanged independent natural chat turn + EOS; frozen dialogue splits',
        'input_sha256':{str((CHAT/f'{s}.jsonl').relative_to(ROOT)):file_sha(CHAT/f'{s}.jsonl') for s in rows},
        'output_sha256':{p.name:file_sha(p) for p in DEST.glob('*.bin')},'test_or_blind_read':False}
    write_json(DEST/'manifest.json',manifest)
    prepare_indexes(DEST,INDEX,128,'sha256',splits=('train','validation'))
    conv=[]
    with (CONV/'train.records.jsonl').open(encoding='utf-8') as stream:
        for i,line in enumerate(stream):
            ex=json.loads(line)
            if keep[ex['row']]:
                conv.append(i)
            else:
                counts['conversion_parent_heldout_exact']+=1
    np.save(MIX/'conversion.keep.npy',np.asarray(conv,dtype='<u4'),allow_pickle=False)
    banks=[]
    for folder,index,source_map in ((OLD,ROOT/'artifacts/training-data/corpus-v4-c128',{0:0,1:1}),
                                    (DEST,INDEX,{0:2,1:3})):
        banks.append({'token_dir':folder.relative_to(ROOT).as_posix(),'index_dir':index.relative_to(ROOT).as_posix(),
            'token_manifest_sha256':file_sha(folder/'manifest.json'),'window_manifest_sha256':file_sha(index/'manifest.json'),
            'source_map':source_map})
    request={'joined_stores':banks,'primary_store':0,'token_manifest_sha256':banks[0]['token_manifest_sha256'],
        'window_manifest_sha256':banks[0]['window_manifest_sha256'],'index_dir':banks[0]['index_dir'],
        'epochs':16,'seed':seed,'crop_probability':.30,'crop_minimum':8,'chat_fraction':.05,
        'old_keep':(MIX/'old.keep.npy').relative_to(ROOT).as_posix(),'old_keep_sha256':file_sha(MIX/'old.keep.npy'),
        'conversion_dir':CONV.relative_to(ROOT).as_posix(),'conversion_manifest_sha256':file_sha(CONV/'manifest.json'),
        'conversion_keep':(MIX/'conversion.keep.npy').relative_to(ROOT).as_posix(),
        'conversion_keep_sha256':file_sha(MIX/'conversion.keep.npy'),'conversion_examples':len(conv),
        'exclusions':dict(counts),'test_or_blind_read':False,'numpy_version':np.__version__}
    # JSON stringifies integer source IDs identically on both execution hosts.
    request=json.loads(json.dumps(request))
    write_json(MIX/'request.json',request)
    prepare_maps(MIX)
    print(json.dumps({'status':'complete','seed':seed,'chat':stats,'conversion_examples':len(conv),'exclusions':dict(counts)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
