"""Verify parallel output against serial output, provenance and safe resume."""
import contextlib
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vimeml.tokenizer import encode as serial
from vimeml.tokenizer import encode_parallel as parallel
from vimeml.tokenizer.store import TokenStore


class EncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sentencepiece as spm
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT/'outputs')
        cls.root = Path(cls.temporary.name)
        cls.model_dir = cls.root/'tokenizer'
        cls.model_dir.mkdir()
        sentences = [f'今日は{i}番の文章です。東京で友達と日本語を勉強します。' for i in range(600)]
        train = cls.root/'training.txt'
        train.write_text('\n'.join(sentences)+'\n',encoding='utf-8')
        spm.SentencePieceTrainer.train(input=str(train),model_prefix=str(cls.model_dir/'tokenizer'),
            model_type='unigram',vocab_size=320,byte_fallback=True,character_coverage=0.9995,
            normalization_rule_name='identity',remove_extra_whitespaces=False,
            pad_id=0,unk_id=1,bos_id=2,eos_id=3,num_threads=1,minloglevel=2)
        cls.processor = spm.SentencePieceProcessor(model_file=str(cls.model_dir/'tokenizer.model'))

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def fixture(self,name):
        import sentencepiece as spm
        corpus = self.root/name
        corpus.mkdir()
        texts = ['今日は日本語を勉強します。','𠮷野家🙂 ABC 123。',' leading  spaces trailing ',
                 'tab\there','短い文。','最後の行には句読点がなくても良い']
        stats,corpus_stats = {},{}
        all_rows = {}
        for split in serial.SPLITS:
            rows = []
            for i,text in enumerate(texts):
                source = 'fineweb' if i%2 else 'tatoeba'
                rows.append({'text':text,'text_hash':hashlib.sha256(text.encode()).hexdigest(),
                    'source':source,'sources':[source],'doc_id':f'{split}-{i}','doc_hash':hashlib.sha256(str(i).encode()).hexdigest()})
            all_rows[split]=rows
            (corpus/f'{split}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8',newline='\n')
            # Test the accepted last line without newline as well.
            (corpus/f'{split}.txt').write_text('\n'.join(texts),encoding='utf-8',newline='\n')
            encoded = self.processor.encode(texts,out_type=int)
            stats[split]={'sentences':len(rows),'characters':sum(map(len,texts)),
                'tokens_without_special_tokens':sum(map(len,encoded)),
                'tokens_with_bos_eos_per_sentence':sum(map(len,encoded))+2*len(rows),
                'unknown_tokens':0,'roundtrip_mismatches':0}
            corpus_stats[split]={**stats[split],'primary_sources':dict(Counter(row['source'] for row in rows))}
        serial.dump_json(corpus/'manifest.json',{'status':'complete'})
        serial.dump_json(corpus/'stats.json',{'splits':corpus_stats})
        tokenizer = corpus/'tokenizer'
        tokenizer.mkdir()
        (tokenizer/'tokenizer.model').write_bytes((self.model_dir/'tokenizer.model').read_bytes())
        serial.dump_json(tokenizer/'stats.json',{'actual_vocab_size':320,'splits':stats})
        serial.dump_json(tokenizer/'manifest.json',{'status':'complete','model_sha256':serial.file_sha(tokenizer/'tokenizer.model'),
            'sentencepiece_version':spm.__version__,'special_ids':{'pad':0,'unk':1,'bos':2,'eos':3},
            'input_sha256':{'D:\\old-host\\'+name:serial.file_sha(corpus/name) for name in
                           ('train.txt','validation.txt','test.txt','manifest.json','stats.json')}})
        return corpus,tokenizer,stats,all_rows

    def test_every_byte_cut_preserves_sequences_and_provenance(self):
        corpus,tokenizer,stats,rows = self.fixture('byte-boundaries')
        path = corpus/'train.jsonl'
        size = path.stat().st_size
        expected_sequences = [[2,*self.processor.encode(row['text']),3] for row in rows['train']]
        expected_bytes = b''.join(struct.pack('<H',token) for seq in expected_sequences for token in seq)
        # UTF-8 boundaries, first/last byte, and positions on both sides of newlines.
        raw = path.read_bytes()
        cuts = {0,1,size-1,size,size//2}
        cuts.update(i for i,c in enumerate(raw) if c>=128 or c==10)
        parts = corpus/'parts'
        parts.mkdir()
        for cut in sorted(cuts):
            jobs = [{'name':f'cut-{cut}-{i}','split':'train','input':str(path),'start':start,'end':end,
                     'parts':str(parts),'model':str(tokenizer/'tokenizer.model'),'batch_size':2,
                     'source_ids':{'fineweb':0,'tatoeba':1},'signature':'test'}
                    for i,(start,end) in enumerate([(0,cut),(cut,size)])]
            results = [parallel.encode_part(job) for job in jobs]
            actual = b''.join(parallel.part_paths(job)['tokens.bin'].read_bytes() for job in jobs)
            self.assertEqual(actual,expected_bytes,f'cut={cut}')
            self.assertEqual(sum(item['sentences'] for item in results),len(rows['train']))
            positions = b''.join(parallel.part_paths(job)['rows.bin'].read_bytes() for job in jobs)
            with path.open('rb') as source:
                for i,(position,) in enumerate(struct.iter_unpack('<Q',positions)):
                    source.seek(position)
                    self.assertEqual(json.loads(source.readline()),rows['train'][i])
            for job in jobs:
                for file in parallel.part_paths(job).values(): file.unlink()

    def test_parallel_store_matches_serial_and_completed_resume_is_noop(self):
        corpus,tokenizer,stats,rows = self.fixture('complete')
        output = corpus/'parallel'
        with patch.object(parallel,'ROOT',corpus),contextlib.redirect_stdout(io.StringIO()):
            plan = parallel.run(corpus,tokenizer,output,workers=2,dry_run=True)
            self.assertFalse(output.exists())
            self.assertFalse((corpus/'runs').exists())
            actual = parallel.run(corpus,tokenizer,output,workers=2,batch_size=2)
            prior = (output/'progress.json').read_bytes()
            again = parallel.run(corpus,tokenizer,output,workers=2,batch_size=2,resume=True)
            self.assertEqual(again,actual)
            self.assertEqual((output/'progress.json').read_bytes(),prior)
        old = corpus/'serial'
        old.mkdir()
        for split in serial.SPLITS:
            serial.export_split(corpus,old,split,self.processor,stats[split],2)
            for suffix in ('tokens.bin','offsets.bin'):
                self.assertEqual((output/f'{split}.{suffix}').read_bytes(),(old/f'{split}.{suffix}').read_bytes())
            with TokenStore(output,split) as store:
                self.assertEqual(len(store),len(rows[split]))
                for i,row in enumerate(rows[split]): self.assertEqual(self.processor.decode(store[i][1:-1]),row['text'])
            self.assertEqual((output/f'{split}.sources.bin').read_bytes(),bytes(0 if row['source']=='fineweb' else 1 for row in rows[split]))
        self.assertFalse((output/'.parts').exists())

    def test_failure_then_resume_reuses_committed_parts(self):
        corpus,tokenizer,_,_ = self.fixture('resume')
        output = corpus/'parallel'
        with patch.object(parallel,'ROOT',corpus),contextlib.redirect_stdout(io.StringIO()):
            with patch.object(parallel,'merge_split',side_effect=RuntimeError('simulated merge failure')):
                with self.assertRaisesRegex(RuntimeError,'simulated'): parallel.run(corpus,tokenizer,output,workers=2,batch_size=2)
            markers = list((output/'.parts').glob('*.done.json'))
            self.assertEqual(len(markers),3)
            # All parts must be reused; submitting even one new task is a failure.
            with patch.object(parallel.ProcessPoolExecutor,'submit',side_effect=AssertionError('cache not reused')):
                result = parallel.run(corpus,tokenizer,output,workers=2,batch_size=2,resume=True)
            self.assertEqual(result['train']['sentences'],6)


if __name__=='__main__': unittest.main()
