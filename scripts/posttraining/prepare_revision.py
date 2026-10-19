"""Rebuild mixed-length conversion views from the audited C1 parent inventory.

Only pinned original train/validation TXT and known nonblind exclusion snapshots
are read. Old C1 binaries remain intact. Each changed view is annotated anew.
"""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from vimeml.posttraining.data import ConversionCodec, TOKENIZER_SHA, FORMAT, shifted
from vimeml.posttraining.readings_v2 import DomainReading, normalize_literal
from vimeml.training.data import file_sha, write_json
from scripts.posttraining.prepare_conversion import dictionary_contract, EXPECTED

OLD = ROOT / 'artifacts/posttrain-data/v4-c1'
CORPUS = ROOT / 'outputs/corpus-fast-v1'
TOKENIZER = ROOT / 'artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model'


def case_key(ex):
    return hashlib.sha256(json.dumps([ex['context'], normalize_literal(ex['reading']), ex['output']],
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def init_worker():
    global READER, CODEC, SOURCES
    READER, CODEC = DomainReading(), ConversionCodec(TOKENIZER)
    SOURCES = {s:(CORPUS/f'{s}.txt').open('rb') for s in ('train', 'validation')}


def process_batch(batch):
    accepted, rejected = [], Counter()
    for ex, expanded in batch:
        if expanded:
            source = SOURCES[ex['split']]
            source.seek(ex['byte_offset'])
            text = source.readline().decode('utf-8').rstrip('\r\n')
            assert hashlib.sha256(text.encode()).hexdigest() == ex['parent_sha256']
            assert text[slice(*ex['span'])] == ex['output'] and text[:ex['span'][0]] == ex['context']
            updated, reason = READER.example_near(text, ex['span'], int(ex['parent_sha256'][16:32], 16))
            if updated is None:
                rejected[reason] += 1
                continue
            ex.update(updated)
            ex['view_origin'] = 'reannotated_extended'
        else:
            ex['view_origin'] = 'inherited_audited_short'
        encoded = CODEC.encode(ex['context'], ex['reading'], ex['output'])
        if encoded is None:
            rejected['reading_output_overflow'] += 1
            continue
        ids, first, removed = encoded
        sample = shifted(ids, first)
        assert sum(t != -100 for t in sample['labels']) == len(ids)-first
        assert CODEC.sp.decode(ids[first:-1]) == ex['output']
        ex.update(case_sha256=case_key(ex), first_output=first, context_crop_characters=removed,
                  reading_dropout_view=ex['split']=='train' and int(ex['parent_sha256'][-8:],16)%10==0,
                  review='dual_dictionary_silver_not_human_gold')
        accepted.append((ex, ids))
    return accepted, dict(rejected)


def input_batches(split):
    batch = []
    with (OLD/f'{split}.records.jsonl').open(encoding='utf-8') as source:
        for raw in source:
            ex = json.loads(raw)
            selection = int(ex['parent_sha256'][:8],16) % 100
            if split == 'train' and selection >= 35:
                continue
            expanded = selection < 25 if split == 'train' else selection < 72
            batch.append((ex, expanded))
            if len(batch) == 1000:
                yield batch
                batch = []
    if batch:
        yield batch


def pack(split, directory, pool, validation_parents, excluded_cases, excluded_sequences, benchmark_parents):
    stats, hist, rejects, review = Counter(), defaultdict(Counter), Counter(), []
    seen_cases, seen_sequences, parents = set(), set(), set()
    handles = {s:(directory/f'{split}.{s}').open('wb') for s in
               ('tokens.bin','offsets.bin','out-start.bin','reading-dropout.bin')}
    handles['offsets.bin'].write(struct.pack('<Q',0))
    records = (directory/f'{split}.records.jsonl').open('w',encoding='utf-8',newline='\n')
    pending = deque()
    began, completed = time.monotonic(), 0

    def consume(future):
        nonlocal completed
        examples, rejected = future.result()
        rejects.update(rejected)
        for ex, ids in examples:
            parent = ex['parent_sha256']
            sequence = struct.pack(f'<{len(ids)}H',*ids)
            sequence_sha = hashlib.sha256(sequence).digest()
            fingerprint = ex['case_sha256']
            if parent in benchmark_parents or (split=='train' and parent in validation_parents):
                rejects['excluded_parent'] += 1
                continue
            if fingerprint in excluded_cases or sequence_sha in excluded_sequences:
                rejects['excluded_exact_holdout_case'] += 1
                continue
            if fingerprint in seen_cases or sequence_sha in seen_sequences:
                rejects['duplicate_task'] += 1
                continue
            seen_cases.add(fingerprint)
            seen_sequences.add(sequence_sha)
            parents.add(parent)
            first = ex['first_output']
            stats['examples'] += 1
            stats['stored_tokens'] += len(ids)
            stats['supervised_tokens_full_reading'] += len(ids)-first
            stats['context_cropped'] += bool(ex['context_crop_characters'])
            stats['kana_only'] += not ex['has_kanji']
            stats['reading_dropout_view'] += ex['reading_dropout_view']
            stats[ex['view_origin']] += 1
            stats['proper_name_present'] += ex.get('proper_name_present',False)
            stats['literal_present'] += ex.get('literal_present',False)
            for key,value in [('reading_characters',len(ex['reading'])),('output_characters',len(ex['output'])),
                              ('context_characters',len(ex['context'])),('source_id',ex['source_id']),('serialized_tokens',len(ids))]:
                hist[key][value] += 1
            handles['tokens.bin'].write(sequence)
            handles['offsets.bin'].write(struct.pack('<Q',stats['stored_tokens']))
            handles['out-start.bin'].write(bytes([first]))
            handles['reading-dropout.bin'].write(bytes([ex['reading_dropout_view']]))
            records.write(json.dumps(ex,ensure_ascii=False,separators=(',',':'))+'\n')
            if len(review)<200:
                review.append(ex)
            elif int(parent[-12:],16)%stats['examples'] < 200:
                review[int(parent[-12:],16)%200]=ex
        completed += 1
        if completed % 25 == 0:
            progress = {'split':split,'batches':completed,'examples':stats['examples'],
                        'seconds':round(time.monotonic()-began,1),'rejected':dict(rejects)}
            write_json(directory/'progress.json',progress)
            print(json.dumps(progress),flush=True)

    try:
        for batch in input_batches(split):
            pending.append(pool.submit(process_batch,batch))
            if len(pending)>=24:
                consume(pending.popleft())
        while pending:
            consume(pending.popleft())
    finally:
        records.close()
        for handle in handles.values():
            handle.close()
    write_json(directory/f'{split}.review-sample.json',review)
    return {**stats,'histograms':dict(hist),'filter_counts':dict(rejects)}, seen_cases, seen_sequences, parents


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/posttrain-data/v4-c1r2')
    args=parser.parse_args()
    directory=args.output.resolve()
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():
        raise ValueError('Preserve existing frozen revision output')
    began=time.monotonic()
    old=json.loads((OLD/'manifest.json').read_text('utf-8'))
    acceptance=json.loads((OLD/'acceptance.json').read_text('utf-8'))
    assert acceptance['status']=='pass' and acceptance['manifest_sha256']==file_sha(OLD/'manifest.json')
    print('Verifying pinned train/validation source and audited parent records',flush=True)
    for split in ('train','validation'):
        assert file_sha(CORPUS/f'{split}.txt')==EXPECTED[split]
        name=f'{split}.records.jsonl'
        assert file_sha(OLD/name)==old['output_sha256'][name]
    validation_parents={hashlib.sha256(raw.decode('utf-8').rstrip('\r\n').encode()).hexdigest()
                        for raw in (CORPUS/'validation.txt').open('rb')}
    excluded, benchmark_parents, provenance=set(),set(),{}
    for track in ('standard-development','legacy-regression','ajimee','historical-development'):
        path=ROOT/f'outputs/evaluation-v4-20261015/inputs/{track}.json'
        provenance[track]=file_sha(path)
        for row in json.loads(path.read_text('utf-8')):
            for answer in row['answers']:
                excluded.add(case_key({'context':row['left_context'],'reading':row['query'],'output':answer}))
                benchmark_parents.add(hashlib.sha256((row['left_context']+answer).encode()).hexdigest())
    stats={}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker) as pool:
        stats['validation'],val_cases,val_sequences,_=pack('validation',directory,pool,
            validation_parents,excluded,set(),benchmark_parents)
        stats['train'],_,_,_=pack('train',directory,pool,validation_parents,
            excluded|val_cases,val_sequences,benchmark_parents)
    assert stats['train']['examples']>100000 and stats['validation']['examples']>1000
    assert stats['train']['reannotated_extended']>stats['train']['inherited_audited_short']
    files={p.name:file_sha(p) for p in directory.iterdir() if p.name.endswith(('.bin','.records.jsonl','.review-sample.json'))}
    code={str(p.relative_to(ROOT)):file_sha(p) for p in (Path(__file__),ROOT/'src/vimeml/posttraining/readings_v2.py')}
    manifest={'format':FORMAT,'status':'complete','contract':{'version':'c1-domain-v2',
        'tokenizer_sha256':TOKENIZER_SHA,'source_sha256':EXPECTED,'inherited_parent_manifest_sha256':file_sha(OLD/'manifest.json'),
        'inherited_acceptance_sha256':file_sha(OLD/'acceptance.json'),'dictionary':dictionary_contract(),
        'code_sha256':code,'nonblind_exclusion_snapshots':provenance,
        'selection':'parent hash modulo100: <25 reannotated extended; 25..34 audited short; validation <72 extended'},
        'splits':stats,'output_sha256':files,'test_or_blind_read':False,
        'isolation':'train excludes all original validation parents and exact frozen conversion validation cases/sequences',
        'quality':'dual dictionary silver; literals preserved with inference normalization; proper names require reading agreement',
        'elapsed_seconds':time.monotonic()-began}
    write_json(directory/'manifest.json',manifest)
    write_json(directory/'acceptance.json',{'status':'pass','manifest_sha256':file_sha(directory/'manifest.json'),
        'checks':['pinned source and inherited record hashes','every changed original offset/span/parent hash',
                  'every changed dual reading agreement','every encoded OUT/EOS mask and target roundtrip',
                  'all validation parents and exact case/sequence exclusions','within-split case/sequence deduplication'],
        'human_gold':False,'test_or_blind_read':False})
    write_json(directory/'progress.json',{'status':'complete','splits':stats,'elapsed_seconds':time.monotonic()-began})
    print(json.dumps({'status':'complete','splits':stats,'seconds':time.monotonic()-began}),flush=True)


if __name__=='__main__':
    main()
