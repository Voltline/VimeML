"""Exercise UTF-8 partitioning, native training, exact stats and TensorBoard."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vimeml.tokenizer import train


class TokenizerTrainingTests(unittest.TestCase):
    def test_every_byte_boundary_preserves_utf8_lines(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temporary:
            path = Path(temporary)/'text.txt'
            lines = ['日本語🙂。','ABC 123。','𠮷野家。','最後の行']
            raw = ('\r\n'.join(lines)).encode('utf-8')
            path.write_bytes(raw)
            for cut in range(len(raw)+1):
                restored = list(train.range_sentences(path,0,cut))+list(train.range_sentences(path,cut,len(raw)))
                self.assertEqual(restored,lines,f'cut={cut}')

    def test_sample_size_rejects_small_and_boolean_values(self):
        settings = {'model_type':'unigram','vocab_size':384,'num_threads':2,'max_sentence_length':8192,
                    'character_coverage':0.9995,'seed':42,'byte_fallback':True,
                    'normalization_rule_name':'identity','remove_extra_whitespaces':False}
        for value in [True,-1,100,1.5]:
            with self.assertRaises(ValueError): train.validate({**settings,'input_sentence_size':value})
        for value in [0,101,2_000_000]: train.validate({**settings,'input_sentence_size':value})

    def test_real_pipeline_and_exact_partition_statistics(self):
        import sentencepiece as spm
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temporary:
            root = Path(temporary)
            corpus = root/'corpus'
            corpus.mkdir()
            rows = [f'今日は{i}番の文章です。東京で友達と日本語を勉強します。' for i in range(600)]
            texts = {'train':rows,'validation':['日本語🙂 ABC 123。',' leading  spaces trailing '],
                     'test':['𠮷野家でコーヒー☕を飲む🙂','tab\there']}
            expected = {}
            for split,items in texts.items():
                (corpus/f'{split}.txt').write_text('\n'.join(items)+'\n',encoding='utf-8',newline='\n')
                expected[split] = {'sentences':len(items),'characters':sum(map(len,items))}
            (corpus/'manifest.json').write_text(json.dumps({'status':'complete'}),encoding='utf-8')
            (corpus/'stats.json').write_text(json.dumps({'splits':expected}),encoding='utf-8')
            config = root/'tokenizer.toml'
            config.write_text('''[paths]
corpus_dir = "corpus"
output_dir = "model"
[training]
model_type = "unigram"
vocab_size = 320
num_threads = 2
input_sentence_size = 201
max_sentence_length = 8192
character_coverage = 0.9995
byte_fallback = true
normalization_rule_name = "identity"
remove_extra_whitespaces = false
seed = 42
[measurement]
workers = 2
batch_size = 128
[monitoring]
tensorboard = true
''',encoding='utf-8')
            with patch.object(train,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                dry = train.run(config,dry_run=True)
                self.assertEqual(dry['sampled_training_sentences'],201)
                self.assertFalse((root/'model').exists())
                self.assertFalse((root/'runs').exists())
                try:
                    result = train.run(config)
                except Exception:
                    log = root/'model/trainer.log'
                    if log.exists(): print(log.read_text(encoding='utf-8')[-1800:],file=sys.stderr)
                    raise
            model = root/'model/tokenizer.model'
            processor = spm.SentencePieceProcessor(model_file=str(model))
            self.assertEqual(result['actual_vocab_size'],320)
            for split,items in texts.items():
                encoded = processor.encode(items,out_type=int)
                self.assertEqual(result['splits'][split]['tokens_without_special_tokens'],sum(map(len,encoded)))
                self.assertEqual(result['splits'][split]['roundtrip_mismatches'],0)
                self.assertEqual(result['splits'][split]['unknown_tokens'],0)
            path = corpus/'train.txt'
            size = path.stat().st_size
            jobs = [(model,path,size*i//7,size*(i+1)//7,'train',128) for i in range(7)]
            combined,_ = train.aggregate_measurements(processor,[train.measure_range(job) for job in jobs],'train')
            self.assertEqual(combined,result['splits']['train'])
            progress = json.loads((root/'model/progress.json').read_text(encoding='utf-8'))
            self.assertEqual(progress['stage'],'complete')
            events = EventAccumulator(str(root/'runs/tokenizer/model')).Reload()
            tags = events.Tags()['scalars']
            self.assertIn('tokenizer/em_objective',tags)
            self.assertIn('measurement/percent',tags)
            self.assertEqual(events.Scalars('measurement/percent')[-1].value,100)
            self.assertTrue((root/'model/trainer.log').is_file())


if __name__=='__main__': unittest.main()
