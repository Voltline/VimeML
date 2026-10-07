"""Parallel sentence encoding with compact provenance and resumable local parts."""
import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import sys
import time
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from vimeml.tokenizer.encode import SPLITS, dump_json, file_sha, write_numbers
from vimeml.tokenizer.monitor import RunMonitor
from vimeml.tokenizer.train import hash_inputs

ROOT = Path(__file__).resolve().parents[3]
FORMAT = 'vimeml_sentence_tokens_v1'


@contextlib.contextmanager
def output_lock(output):
    with (output/'.encode.lock').open('a+b') as stream:
        stream.seek(0,2)
        if not stream.tell():
            stream.write(b'1')
            stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('Another encoder is using this output directory.') from None
        try: yield
        finally:
            stream.seek(0)
            if os.name=='nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream,fcntl.LOCK_UN)


def part_paths(job):
    prefix = Path(job['parts'])/job['name']
    return {suffix:Path(str(prefix)+'.'+suffix) for suffix in
            ('tokens.bin','offsets.bin','rows.bin','sources.bin','done.json')}


def encode_part(job):
    import sentencepiece as spm
    files = part_paths(job)
    # An interrupted part has no commit marker and is safely overwritten.
    files['done.json'].unlink(missing_ok=True)
    processor = spm.SentencePieceProcessor(model_file=job['model'])
    bos,eos,unk,pad = (getattr(processor,name+'_id')() for name in ('bos','eos','unk','pad'))
    special = {bos,eos,unk,pad}
    count = total = content = characters = longest = 0
    sources = Counter()
    verify_content = job.get('verification','sha256') == 'sha256'
    text_digest = hashlib.sha256() if verify_content else None
    with Path(job['input']).open('rb') as rows, \
            files['tokens.bin'].open('wb') as tokens, files['offsets.bin'].open('wb') as offsets, \
            files['rows.bin'].open('wb') as positions, files['sources.bin'].open('wb') as source_ids:
        if job['start']:
            rows.seek(job['start']-1)
            if rows.read(1)!=b'\n': rows.readline()
        write_numbers(offsets,'Q',[0])
        while rows.tell()<job['end']:
            batch,byte_positions = [],[]
            for _ in range(job['batch_size']):
                position = rows.tell()
                if position>=job['end']: break
                raw = rows.readline()
                if not raw: break
                row = json.loads(raw)
                text = row['text']
                if not isinstance(text,str) or not text or '\n' in text or '\r' in text:
                    raise ValueError(f"Invalid text: {job['name']} at byte {position}")
                if verify_content and hashlib.sha256(text.encode('utf-8')).hexdigest()!=row['text_hash']:
                    raise ValueError(f"Text hash mismatch: {job['name']} at byte {position}")
                if row['source'] not in job['source_ids']:
                    raise ValueError(f"Unknown source: {row['source']}")
                byte_positions.append(position)
                batch.append(row)
            if not batch: break
            texts = [row['text'] for row in batch]
            encoded = processor.encode(texts,out_type=int,num_threads=1)
            decoded = processor.decode(encoded,num_threads=1) if verify_content else [None]*len(encoded)
            batch_tokens,batch_offsets,byte_sources = [],[],bytearray()
            for row,ids,restored in zip(batch,encoded,decoded,strict=True):
                text = row['text']
                if (verify_content and restored!=text) or any(token in special for token in ids):
                    raise ValueError(f"Roundtrip or special token error: {job['name']} row {count}")
                batch_tokens.extend((bos,*ids,eos))
                total += len(ids)+2
                batch_offsets.append(total)
                count += 1
                content += len(ids)
                characters += len(text)
                longest = max(longest,len(ids)+2)
                sources[row['source']] += 1
                byte_sources.append(job['source_ids'][row['source']])
                if text_digest is not None:
                    text_digest.update((text+'\n').encode('utf-8'))
            write_numbers(tokens,'H',batch_tokens)
            write_numbers(offsets,'Q',batch_offsets)
            write_numbers(positions,'Q',byte_positions)
            source_ids.write(byte_sources)
    sizes = {suffix:path.stat().st_size for suffix,path in files.items() if suffix!='done.json'}
    if sizes!={'tokens.bin':total*2,'offsets.bin':(count+1)*8,'rows.bin':count*8,'sources.bin':count}:
        raise ValueError('Part size mismatch.')
    result = {'name':job['name'],'signature':job['signature'],'split':job['split'],
              'start':job['start'],'end':job['end'],'sentences':count,'stored_tokens':total,
              'content_tokens':content,'characters':characters,'max_sequence_tokens':longest,
              'primary_sources':dict(sources),'canonical_txt_sha256':text_digest.hexdigest() if text_digest is not None else None,
              'files':{suffix:{'bytes':path.stat().st_size,**({'sha256':file_sha(path)} if verify_content else {})}
                       for suffix,path in files.items() if suffix!='done.json'}}
    dump_json(files['done.json'],result)
    return result


def cached_part(job):
    files = part_paths(job)
    if not files['done.json'].is_file(): return None
    result = json.loads(files['done.json'].read_text(encoding='utf-8'))
    if result.get('signature')!=job['signature'] or result.get('name')!=job['name']:
        raise ValueError('Cached part belongs to a different encoding run.')
    for suffix,expected in result['files'].items():
        path = files[suffix]
        if (not path.is_file() or path.stat().st_size!=expected['bytes'] or
                (job.get('verification','sha256') == 'sha256' and file_sha(path)!=expected['sha256'])):
            raise ValueError(f'Cached part is damaged: {path}; preserve evidence and use a new output directory.')
    return result


def make_jobs(corpus,model,parts,source_ids,batch_size,chunk_bytes,signature,verification='sha256'):
    jobs = []
    for split in SPLITS:
        path = corpus/f'{split}.jsonl'
        size = path.stat().st_size
        partitions = max(1,math.ceil(size/chunk_bytes))
        for index in range(partitions):
            jobs.append({'name':f'{split}-{index:05d}','split':split,'input':str(path),
                         'start':size*index//partitions,'end':size*(index+1)//partitions,
                         'model':str(model),'parts':str(parts),'source_ids':source_ids,
                         'batch_size':batch_size,'signature':signature,'verification':verification})
    return jobs


def collect_parts(jobs,workers,monitor,total_rows):
    results,pending_jobs = {},[]
    for job in jobs:
        cached = cached_part(job)
        if cached: results[job['name']]=cached
        else: pending_jobs.append(job)
    done = sum(result['sentences'] for result in results.values())
    print(f'[encode] cached_parts={len(results)}; pending_parts={len(pending_jobs)}; sentences={done:,}/{total_rows:,}',flush=True)
    monitor.update('encoding',completed_sentences=done,total_sentences=total_rows)
    started = time.monotonic()
    initial_done = done
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(encode_part,job):job for job in pending_jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            results[job['name']]=result
            done += result['sentences']
            elapsed = time.monotonic()-started
            speed = (done-initial_done)/elapsed if elapsed else 0
            eta = (total_rows-done)/speed if speed else None
            print(f'[encode] {done:,}/{total_rows:,} ({100*done/total_rows:.1f}%); {speed:,.0f} sentences/s; ETA={eta:.0f}s' if eta is not None else '[encode] no new rows yet',flush=True)
            monitor.update('encoding',completed_sentences=done,total_sentences=total_rows,
                           encoding_sentences_per_second=speed,encoding_eta_seconds=eta)
    except BaseException:
        # Python 3.13 lacks the public terminate_workers API. Stop only our pool.
        processes = list((getattr(pool,'_processes',None) or {}).values())
        for process in processes:
            if process.is_alive(): process.terminate()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    else: pool.shutdown(wait=True)
    return results


def merge_split(corpus,output,split,jobs,results,expected,monitor,done_before,total_rows,verification='sha256'):
    counts = Counter()
    sources = Counter()
    longest = 0
    target = {suffix:output/f'{split}.{suffix}' for suffix in ('tokens.bin','offsets.bin','rows.bin','sources.bin')}
    text_context = (corpus/f'{split}.txt').open(encoding='utf-8') if verification == 'sha256' else contextlib.nullcontext(None)
    with text_context as text, \
            target['tokens.bin'].open('wb') as tokens,target['offsets.bin'].open('wb') as offsets, \
            target['rows.bin'].open('wb') as positions,target['sources.bin'].open('wb') as source_ids:
        write_numbers(offsets,'Q',[0])
        for job in jobs:
            result = results[job['name']]
            files = part_paths(job)
            # Match every source TXT row, including exact whitespace, in order.
            if text is not None:
                digest = hashlib.sha256()
                for _ in range(result['sentences']):
                    line = text.readline()
                    if not line: raise ValueError(f'{split}: TXT has fewer rows than JSONL.')
                    digest.update((line.removesuffix('\n')+'\n').encode('utf-8'))
                if digest.hexdigest()!=result['canonical_txt_sha256']:
                    raise ValueError(f"{split}: TXT/JSONL alignment mismatch in {job['name']}")
            for suffix,stream in [('tokens.bin',tokens),('rows.bin',positions),('sources.bin',source_ids)]:
                with files[suffix].open('rb') as source: shutil.copyfileobj(source,stream,1024*1024)
            with files['offsets.bin'].open('rb') as source:
                if source.read(8)!=b'\0'*8: raise ValueError('Invalid part initial offset.')
                previous = 0
                while raw:=source.read(1024*1024):
                    values = array('Q')
                    values.frombytes(raw)
                    if sys.byteorder!='little': values.byteswap()
                    adjusted = []
                    for value in values:
                        if not previous<value<=result['stored_tokens']: raise ValueError('Invalid part offset sequence.')
                        adjusted.append(value+counts['stored_tokens'])
                        previous = value
                    write_numbers(offsets,'Q',adjusted)
                if previous!=result['stored_tokens']: raise ValueError('Invalid final part offset.')
            for key in ('sentences','stored_tokens','content_tokens','characters'): counts[key]+=result[key]
            sources.update(result['primary_sources'])
            longest = max(longest,result['max_sequence_tokens'])
            done = done_before+counts['sentences']
            print(f'[merge] {done:,}/{total_rows:,} sentences',flush=True)
            monitor.update('merging',completed_sentences=done,total_sentences=total_rows)
        if text is not None and text.readline(): raise ValueError(f'{split}: TXT has more rows than JSONL.')
    for field,key in [('sentences','sentences'),('characters','characters'),
                      ('content_tokens','tokens_without_special_tokens'),('stored_tokens','tokens_with_bos_eos_per_sentence')]:
        if counts[field]!=expected[key]: raise ValueError(f'{split}: {field} differs from tokenizer stats.')
    sizes = {suffix:path.stat().st_size for suffix,path in target.items()}
    if sizes!={'tokens.bin':counts['stored_tokens']*2,'offsets.bin':(counts['sentences']+1)*8,
               'rows.bin':counts['sentences']*8,'sources.bin':counts['sentences']}:
        raise ValueError('Merged binary sizes differ from expected counts.')
    return {**dict(counts),'prediction_pairs':counts['stored_tokens']-counts['sentences'],
            'max_sequence_tokens':longest,'primary_sources':dict(sources),'file_bytes':sizes}


def recorded_hash(manifest,name):
    matches = [value for path,value in manifest['input_sha256'].items() if path.replace('\\','/').rsplit('/',1)[-1]==name]
    if len(matches)!=1: raise ValueError(f'No unique recorded input hash for {name}.')
    return matches[0]


def run(corpus,tokenizer,output,workers=8,batch_size=1024,chunk_mib=128,resume=False,dry_run=False,verification='sha256'):
    import sentencepiece as spm
    for name,value in [('workers',workers),('batch_size',batch_size),('chunk_mib',chunk_mib)]:
        if type(value) is not int or value<1: raise ValueError(f'{name} must be a positive integer.')
    if verification not in {'sha256','metadata'}:
        raise ValueError('verification must be sha256 or metadata.')
    corpus,tokenizer,output = (Path(p).resolve() for p in (corpus,tokenizer,output))
    model = tokenizer/'tokenizer.model'
    cm = json.loads((corpus/'manifest.json').read_text(encoding='utf-8'))
    tm = json.loads((tokenizer/'manifest.json').read_text(encoding='utf-8'))
    stats = json.loads((tokenizer/'stats.json').read_text(encoding='utf-8'))
    if cm.get('status')!='complete' or tm.get('status')!='complete': raise ValueError('Incomplete corpus or tokenizer.')
    if tm['sentencepiece_version']!=spm.__version__ or file_sha(model)!=tm['model_sha256']:
        raise ValueError('Tokenizer version or model hash mismatch.')
    processor = spm.SentencePieceProcessor(model_file=str(model))
    vocab_size = processor.get_piece_size()
    if not 0<vocab_size<=65536 or stats['actual_vocab_size']!=vocab_size: raise ValueError('Invalid vocabulary size.')
    special = tm['special_ids']
    if len(set(special.values()))!=4 or any(value<0 or getattr(processor,name+'_id')()!=value for name,value in special.items()):
        raise ValueError('Invalid tokenizer special IDs.')
    if any(stats['splits'][split]['unknown_tokens'] or stats['splits'][split]['roundtrip_mismatches'] for split in SPLITS):
        raise ValueError('Tokenizer checks did not pass.')
    cs = json.loads((corpus/'stats.json').read_text(encoding='utf-8'))
    source_names = sorted({source for split in SPLITS for source in cs['splits'][split]['primary_sources']})
    if len(source_names)>256: raise ValueError('uint8 source IDs support at most 256 sources.')
    source_ids = {name:i for i,name in enumerate(source_names)}
    estimated = sum(stats['splits'][split]['tokens_with_bos_eos_per_sentence']*2+
                    stats['splits'][split]['sentences']*17+8 for split in SPLITS)
    total_rows = sum(stats['splits'][split]['sentences'] for split in SPLITS)
    plan = {'corpus':str(corpus),'tokenizer':str(tokenizer),'output':str(output),'workers':workers,
            'native_threads_per_worker':1,'batch_size':batch_size,'chunk_mib':chunk_mib,
            'sentences':total_rows,'estimated_binary_bytes':estimated,'estimated_peak_binary_bytes':estimated*2,
            'source_ids':source_ids,'resume':resume,'dry_run':dry_run,'encoding_started':False,'verification':verification,
            'tensorboard_log_root':str(ROOT/'runs/tokenizer'/f'{output.name}-encode')}
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if dry_run: return plan
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not resume:
        raise FileExistsError('Nonempty output exists; use --resume for this run or choose a new output directory.')
    if not output.exists() and resume: raise FileNotFoundError('Cannot resume an absent output directory.')
    output.mkdir(parents=True,exist_ok=True)
    with output_lock(output):
        if (output/'manifest.json').is_file():
            existing = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            if existing.get('status')=='complete':
                print('Token data already has a complete manifest; no re-encoding performed.',flush=True)
                return existing['splits']
        with RunMonitor(output,Path(plan['tensorboard_log_root'])/str(time.time_ns()),True) as monitor:
            monitor.update('checking_inputs')
            paths = [corpus/f'{split}.{suffix}' for split in SPLITS for suffix in ('txt','jsonl')]
            paths += [corpus/'manifest.json',corpus/'stats.json',model,tokenizer/'manifest.json',tokenizer/'stats.json']
            paths += [Path(__file__).with_name(name) for name in ('encode_parallel.py','encode.py','monitor.py','train.py')]
            input_metadata = {str(path):{'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns} for path in paths}
            if verification == 'sha256':
                print('[inputs] Hashing corpus and tokenizer files...',flush=True)
                hashes = hash_inputs(paths)
            else:
                report = json.loads((corpus/'integrity-check.json').read_text(encoding='utf-8'))
                hashes = hash_inputs(paths[6:])
                if report.get('status')!='passed' or hashes[str(corpus/'manifest.json')]!=report['manifest_sha256']:
                    raise ValueError('Corpus metadata differs from its integrity report.')
                for path in paths[:6]:
                    recorded = report['exports'][path.name]
                    if input_metadata[str(path)]['bytes']!=recorded['bytes']:
                        raise ValueError(f'Corpus file size differs from the recorded export: {path.name}')
                    hashes[str(path)] = recorded['sha256']
                print('[inputs] Reusing recorded corpus hashes; skipping per-row hashes, repeated roundtrip and full input SHA256 scans.',flush=True)
            for name in ('train.txt','validation.txt','test.txt','manifest.json','stats.json'):
                if hashes[str(corpus/name)]!=recorded_hash(tm,name): raise ValueError(f'Corpus changed since tokenizer training: {name}')
            signature = hashlib.sha256(json.dumps({'inputs':hashes,'batch_size':batch_size,'chunk_mib':chunk_mib,
                'source_ids':source_ids,'verification':verification,'code_sha256':file_sha(Path(__file__))},sort_keys=True).encode()).hexdigest()
            request = output/'run.json'
            if resume:
                if not request.is_file() or json.loads(request.read_text(encoding='utf-8'))['signature']!=signature:
                    raise ValueError('Resume inputs or encoding parameters changed; use the original parameters or a new output directory.')
            else: dump_json(request,{'signature':signature,'plan':plan,'input_sha256':hashes,'input_metadata':input_metadata})
            parts = output/'.parts'
            parts.mkdir(exist_ok=True)
            jobs = make_jobs(corpus,model,parts,source_ids,batch_size,chunk_mib*1024*1024,signature,verification)
            results = collect_parts(jobs,workers,monitor,total_rows)
            exported = {}
            done = 0
            for split in SPLITS:
                selected = [job for job in jobs if job['split']==split]
                exported[split] = merge_split(corpus,output,split,selected,results,stats['splits'][split],monitor,done,total_rows,verification)
                if exported[split]['primary_sources']!=cs['splits'][split]['primary_sources']:
                    raise ValueError(f'{split}: primary source counts differ from corpus stats.')
                done += exported[split]['sentences']
            monitor.update('verifying_inputs')
            if verification == 'sha256':
                if hash_inputs(paths)!=hashes: raise ValueError('Inputs changed during encoding.')
            elif any(path.stat().st_size!=input_metadata[str(path)]['bytes'] or
                     path.stat().st_mtime_ns!=input_metadata[str(path)]['mtime_ns'] for path in paths):
                raise ValueError('Input file metadata changed during encoding.')
            dump_json(output/'stats.json',{'splits':exported})
            artifacts = [output/f'{split}.{suffix}' for split in SPLITS for suffix in ('tokens.bin','offsets.bin','rows.bin','sources.bin')]
            artifacts += [output/'stats.json']
            artifact_hashes = {Path(path).name:value for path,value in hash_inputs(artifacts).items()}
            # Remove only known temporary files created by this run. Keep final data.
            assert parts.resolve().parent==output.resolve()
            for job in jobs:
                for path in part_paths(job).values():
                    assert path.resolve().parent==parts.resolve()
                    path.unlink()
            parts.rmdir()
            dump_json(output/'manifest.json',{'status':'complete','format':FORMAT,'vocab_size':vocab_size,
                'token_dtype':'uint16_le','offset_dtype':'uint64_le','offset_unit':'tokens','special_ids':special,
                'sentencepiece_version':spm.__version__,'splits':exported,'source_ids':source_ids,
                'sequence_policy':'Each sequence is BOS + complete encoded sentence + EOS; no truncation or concatenation.',
                'prediction_policy':'x=s[:-1], y=s[1:]; sentence boundaries remain separate.',
                'provenance_policy':'Index equals zero-based corpus JSONL/TXT row. rows.bin stores uint64 little-endian byte positions in the source JSONL; sources.bin stores uint8 primary source IDs. Full original metadata stays in corpus JSONL.',
                'corpus_dir':str(corpus),'tokenizer_dir':str(tokenizer),'input_sha256':hashes,'output_sha256':artifact_hashes,
                'input_verification':{'mode':verification,'corpus_hashes':'reused from integrity-check.json' if verification=='metadata' else 'computed before and after',
                    'row_text_hashes_checked':verification=='sha256','roundtrip_checked_during_encoding':verification=='sha256',
                    'txt_jsonl_alignment_checked_during_merge':verification=='sha256',
                    'roundtrip_source':'tokenizer full-corpus acceptance' if verification=='metadata' else 'all encoded JSONL rows',
                    'output_hashes':'computed once after merge','input_metadata':input_metadata},
                'corpus_quality_mode':cm.get('quality_mode'),'corpus_ready_for_lm_training':tm.get('corpus_ready_for_lm_training',False),
                'purpose':'Full token data for the accepted first corpus baseline; known extraction and near-duplicate limitations remain.',
                'workers':workers,'native_threads_per_worker':1,'signature':signature,
                'script_sha256':file_sha(Path(__file__))})
            monitor.update('complete',completed_sentences=total_rows,total_sentences=total_rows)
            print(json.dumps({'splits':exported},ensure_ascii=False,indent=2))
            print(f'Token data: {output}',flush=True)
            return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,default=ROOT/'outputs/corpus-fast-v1')
    parser.add_argument('--tokenizer',type=Path,default=ROOT/'artifacts/tokenizers/ja-unigram-16k-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/token-data/corpus-v1-16k')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--chunk-mib',type=int,default=128)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--verification',choices=('sha256','metadata'),default='sha256',
                        help='metadata reuses frozen corpus fingerprints and skips repeated full checks.')
    args = parser.parse_args()
    run(args.corpus,args.tokenizer,args.output,args.workers,args.batch_size,args.chunk_mib,args.resume,args.dry_run,args.verification)


if __name__=='__main__': main()
