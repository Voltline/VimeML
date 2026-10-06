"""Demo validation, bounded real decoder calls and local HTTP protocol."""
import json
import sys
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.tools.phrase_demo import PhraseDemo, handler_for, present, validate_request
from vimeml.training.infer import JapaneseLM


class Processor:
    def encode(self, text, out_type=int):
        return [4 if c=='a' else 5 for c in text]
    def decode(self, ids):
        return ''.join('a' if i==4 else 'b' for i in ids if i not in (0,1,2,3))


class Model:
    config = SimpleNamespace(context_length=16)
    def __call__(self, inputs):
        return torch.tensor([100,100,100,4.,5.,4.]).expand(*inputs.shape,6)
    def parameter_count(self):
        return 12


def demo():
    lm = JapaneseLM.__new__(JapaneseLM)
    lm.device=torch.device('cpu');lm.model=Model();lm.processor=Processor()
    lm.special=dict(pad=0,unk=1,bos=2,eos=3);lm.forbidden=(0,1,2);lm.metadata={'precision':'fp32'}
    return PhraseDemo(lm,[{'id':'fixture','prompt':'a','scene':'fixture'}])


class PhraseDemoTests(unittest.TestCase):
    def test_request_limits_are_strict_including_boolean_integers(self):
        for data in ([], {}, {'prompt':'a\nb'}, {'prompt':'前の文。次の文'}, {'prompt':'a','count':True},
                     {'prompt':'a','max_tokens':999}, {'prompt':'a','mode':'external_api'}):
            with self.assertRaises(ValueError):validate_request(data)

    def test_display_keeps_exact_insert_text_and_marks_unfinished_tokens(self):
        item={'text':'、本を読ん','flags':[], 'stop_reason':'max_new_tokens','new_token_ids':[4,5]}
        displayed=present(item)
        self.assertEqual(displayed['text'],'、本を読ん')
        self.assertTrue(displayed['may_be_incomplete'])
        complete=present({**item,'text':'読む。次の文'})
        self.assertEqual(complete['text'],'読む。')
        self.assertEqual(complete['raw_text'],'読む。次の文')
        self.assertFalse(complete['may_be_incomplete'])
        for flag in ('prefix_changed','replacement_character','empty'):
            self.assertIsNone(present({**item,'flags':[flag]}))

    def test_decoders_batch_and_keep_specials_out_with_reproducible_samples(self):
        d=demo()
        for mode in ('beam','sample'):
            a=d.suggest({'prompt':'a','mode':mode,'max_tokens':4,'count':3,'seed':43})
            b=d.suggest({'prompt':'a','mode':mode,'max_tokens':4,'count':3,'seed':43})
            self.assertEqual(a['suggestions'],b['suggestions'])
            self.assertLessEqual(len(a['suggestions']),3)
            self.assertTrue(a['suggestions'])
            for candidate in a['suggestions']:
                self.assertTrue(set(candidate['new_token_ids']).isdisjoint({0,1,2}))
                self.assertLessEqual(candidate['new_tokens'],4)

    def test_http_routes_validation_origin_and_busy_response(self):
        d=demo();server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(d))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        def request(method,path,data=None,origin=None):
            conn=HTTPConnection('127.0.0.1',server.server_port,timeout=10)
            headers={'Content-Type':'application/json'}
            if origin:headers['Origin']=origin
            conn.request(method,path,body=json.dumps(data) if data is not None else None,headers=headers)
            r=conn.getresponse();status=r.status;raw=r.read();conn.close();return status,raw
        status,raw=request('GET','/');self.assertEqual(status,200);self.assertIn(b'Vime',raw)
        self.assertEqual(request('GET','/api/info')[0],200)
        self.assertEqual(request('POST','/api/suggest',{'prompt':'a'})[0],200)
        self.assertEqual(request('POST','/api/suggest',{'prompt':'a','count':999})[0],400)
        self.assertEqual(request('POST','/api/suggest',{'prompt':'a'},origin='https://elsewhere.invalid')[0],403)
        with d.lock:
            self.assertEqual(request('POST','/api/suggest',{'prompt':'a'})[0],429)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
