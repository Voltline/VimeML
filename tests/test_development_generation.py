"""Mocked API lifecycle: blind reconstruction, resumability and label isolation."""
import contextlib
import io
import json
import re
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vimeml.benchmarks.development import (GENERATION_PROMPT, VERIFICATION_PROMPT, finalize,
    export_cached_drafts, generation_payload, make_jobs, parse_generated, parse_verified, run, verification_payload)
from vimeml.benchmarks.hybrid import prepare_development
from vimeml.benchmarks.evaluate_ajimee import load_export

CONFIG = {"count":20,"batch_size":1,"verify_batch_size":5,"model":"deepseek-chat","judge_model":"qwen",
          "output_tokens":2048,"seed":43,"endpoint":"https://fixture.invalid/api"}


def body(rows, finish='stop'):
    return {"choices":[{"finish_reason":finish,"message":{"content":json.dumps(rows,ensure_ascii=False)}}],
            "usage":{"total_tokens":100}}


class Transport:
    def __init__(self):
        self.calls, self.lock, self.nonces = [], threading.Lock(), {}
        self.fail_nonce = None
        self.ambiguous = False

    def __call__(self, endpoint, payload, key, timeout):
        with self.lock:
            self.calls.append(payload)
            if payload['model']=='deepseek-chat':
                nonce = re.search(r'(?:独立批次编号|バッチID)：([a-f0-9]+)',payload['messages'][1]['content'])[1]
                if nonce == self.fail_nonce:
                    return 500, {}, 'malformed fixture reply'
                index = self.nonces.setdefault(nonce,len(self.nonces)+1)
                context = '出発前に、' if any(s in payload['messages'][1]['content'] for s in ('必须有5~50','5〜50文字の句内左文が必要')) else ''
                rows = [{'context_text':context,'input':'ニモツヲカクニンスル'+chr(0x30a1+index),
                         'expected_output':[f'荷物{index}を確認する'],'reason_zh':'测试 fixture-secret'}]
            else:
                evidence = json.loads(payload['messages'][1]['content'])
                rows = [{'id':item['id'],'status':'ambiguous' if self.ambiguous else 'clear',
                         'confidence':'high','expected_output':[f"荷物{ord(item['input'][-1])-0x30a1}を確認する"],
                         'reason_zh':'独立核验 fixture-secret'} for item in evidence]
        return 200, {}, json.dumps(body(rows),ensure_ascii=False)


class DevelopmentGenerationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix='development-test-',dir=ROOT/'outputs')
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.output = self.root/'generation'
        self.exclude = self.root/'excluded.json'
        self.excluded = [{"index":1,"context_text":"","input":"テスト","expected_output":["除外専用"]}]
        self.exclude.write_text(json.dumps(self.excluded,ensure_ascii=False),encoding='utf-8')

    def run_fixture(self, transport, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(self.output,CONFIG,self.exclude,api_keys=('fixture-secret',),transport=transport,
                       window=.0001,rpm=100,tpm=1000000,retries=0,**kwargs)

    def test_default_plan_is_200_and_balanced_with_no_dry_run_requests_or_files(self):
        config = {**CONFIG,'count':200,'batch_size':10,'verify_batch_size':10}
        def forbidden(*args):
            self.fail('Dry-run sent network request')
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(self.output,config,self.exclude,dry_run=True,transport=forbidden)
        self.assertEqual(result['target_draft_cases'],200)
        self.assertEqual(result['with_context_target'],100)
        self.assertEqual(result['without_context_target'],100)
        self.assertEqual(result['generation_requests'],20)
        self.assertEqual(result['verification_requests_upper_bound'],20)
        self.assertFalse(self.output.exists())

    def test_complete_run_redacts_responses_and_resume_uses_no_credentials_or_network(self):
        transport = Transport()
        stats = self.run_fixture(transport)
        self.assertEqual(stats['automatically_accepted_cases'],20)
        self.assertEqual(len(transport.calls),24)
        self.assertFalse(stats['manual_review_done'])
        for path in self.output.rglob('*.json'):
            self.assertNotIn('fixture-secret',path.read_text(encoding='utf-8'))
        response = json.loads(next((self.output/'responses').glob('*.json')).read_text(encoding='utf-8'))
        self.assertIn('raw_response',response)
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = run(self.output,CONFIG,self.exclude,transport=lambda *a:self.fail('Resume sent request'))
        self.assertEqual(resumed['network_requests_sent_this_run'],0)
        self.assertEqual(json.loads((self.output/'manifest.json').read_text(encoding='utf-8'))['status'],'complete')

    def test_partial_failure_retries_only_missing_fixed_generation_slot(self):
        transport = Transport()
        transport.fail_nonce = make_jobs(CONFIG)[0]['id'][:16]
        with self.assertRaisesRegex(RuntimeError,'pending'):
            self.run_fixture(transport)
        self.assertEqual(len(list((self.output/'cache').glob('*.json'))),19)
        self.assertTrue(json.loads((self.output/'pending.json').read_text(encoding='utf-8')))
        before = len(transport.calls)
        transport.fail_nonce = None
        result = self.run_fixture(transport)
        self.assertEqual(len(transport.calls)-before,5)  # 1 missing slot + 4 verification batches
        self.assertEqual(result['automatically_accepted_cases'],20)
        self.assertEqual(json.loads((self.output/'pending.json').read_text(encoding='utf-8')),[])

    def test_offline_export_keeps_partial_drafts_without_api_verification_or_source_writes(self):
        transport = Transport()
        transport.fail_nonce = make_jobs(CONFIG)[0]['id'][:16]
        with self.assertRaisesRegex(RuntimeError,'pending'):
            self.run_fixture(transport)
        originals = {str(p.relative_to(self.output)):p.read_bytes() for p in self.output.rglob('*.json')}
        before = len(transport.calls)
        snapshot = self.root/'snapshot'
        result = export_cached_drafts(self.output,snapshot,self.exclude)
        self.assertEqual(result['unique_draft_cases'],19)
        self.assertEqual(result['missing_generation_batches'],1)
        self.assertEqual(result['automatically_verified_cases'],0)
        self.assertEqual(result['network_requests_sent'],0)
        self.assertFalse(result['manual_review_done'])
        self.assertEqual(len(transport.calls),before)
        self.assertEqual(originals,{str(p.relative_to(self.output)):p.read_bytes() for p in self.output.rglob('*.json')})
        self.assertEqual(len(json.loads((snapshot/'review-pack.json').read_text(encoding='utf-8'))['unverified']),19)
        with self.assertRaisesRegex(ValueError,'not empty'):
            export_cached_drafts(self.output,snapshot,self.exclude)

    def test_verifier_never_receives_gold_rationale_category_or_candidates(self):
        case = {'context_text':'仕事の後に、','input':'カエル','expected_output':['帰る'],
                'reason_zh':'SECRET_LABEL','category':'work'}
        payload = verification_payload([case],CONFIG)
        evidence = json.loads(payload['messages'][1]['content'])
        self.assertEqual(set(evidence[0]),{'id','context_text','input'})
        self.assertNotIn('帰る',json.dumps(payload,ensure_ascii=False))
        self.assertNotIn('SECRET_LABEL',json.dumps(payload,ensure_ascii=False))
        self.assertIn('助詞は/へ/をはハ/ヘ/ヲ',GENERATION_PROMPT)
        self.assertIn('ハ/ヘ/ヲ',VERIFICATION_PROMPT)
        self.assertIn('中国語への翻訳',GENERATION_PROMPT)
        self.assertIn('80文字以内',VERIFICATION_PROMPT)

    def test_duplicate_forms_repaired_and_invalid_rows_quarantined_without_losing_batch(self):
        cases = [{'input':'ニモツヲモツ'}] * 4
        template = {'status':'clear','confidence':'high','reason_zh':'读音正确。'}
        rows = [
            {**template,'id':'0','expected_output':['荷物を持つ','荷物を持つ']},
            {**template,'id':'1','expected_output':['这是中文句子','A']},
            {**template,'id':'2','expected_output':['荷物を持つ'],'reason_zh':'不，是这样吗？'*300},
            {**template,'id':'3','expected_output':['にもつをかう']},
        ]
        parsed = parse_verified(body(rows),cases)
        self.assertEqual(parsed[0]['expected_output'],['荷物を持つ'])
        self.assertEqual(parsed[0]['repairs'],['duplicate_forms_removed'])
        self.assertEqual(parsed[0]['status'],'clear')
        for row in parsed[1:]:
            self.assertEqual((row['status'],row['confidence']),('invalid','low'))
            self.assertTrue(row['validation_errors'])
        self.assertLessEqual(len(parsed[2]['reason_zh']),1000)

    def test_all_kanji_japanese_forms_remain_valid(self):
        job = make_jobs(CONFIG)[1]  # no left context
        row = {'context_text':'','input':'シュウセイイライ','expected_output':['修正依頼'],'reason_zh':'请求修改。'}
        self.assertEqual(parse_generated(body([row]),job)[0]['expected_output'],['修正依頼'])

    def test_reuse_old_generation_keeps_provenance_and_never_reuses_verifier(self):
        transport = Transport()
        with patch('vimeml.benchmarks.development.GENERATION_PROMPT','legacy system prompt'), \
             patch('vimeml.benchmarks.development.VERIFICATION_PROMPT','legacy verification prompt'):
            self.run_fixture(transport,stage='generate')
        original_files = {str(p.relative_to(self.output)):p.read_bytes() for p in self.output.rglob('*.json')}
        new_output = self.root/'generation-v2'
        before = len(transport.calls)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(new_output,CONFIG,self.exclude,api_keys=('fixture-secret',),transport=transport,
                         window=.0001,rpm=100,tpm=1000000,retries=0,reuse_generation_from=self.output)
        self.assertEqual(len(transport.calls)-before,4)
        self.assertTrue(all(p['model']=='qwen' for p in transport.calls[before:]))
        self.assertEqual(result['reused_generation_cases'],20)
        self.assertEqual(result['automatically_accepted_cases'],20)
        manifest = json.loads((new_output/'manifest.json').read_text(encoding='utf-8'))
        self.assertNotEqual(manifest['generation_prompt_sha256'],manifest['configured_generation_prompt_sha256'])
        self.assertEqual(original_files,{str(p.relative_to(self.output)):p.read_bytes() for p in self.output.rglob('*.json')})
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = run(new_output,CONFIG,self.exclude,reuse_generation_from=self.output,
                          transport=lambda *a:self.fail('Cached import resume sent a request'))
        self.assertEqual(resumed['network_requests_sent_this_run'],0)
        cache = next((self.output/'cache').glob('*.json'))
        damaged = json.loads(cache.read_text(encoding='utf-8'))
        damaged['signature']='tampered'
        cache.write_text(json.dumps(damaged),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'signature'):
            run(self.root/'bad-import',CONFIG,self.exclude,reuse_generation_from=self.output,dry_run=True)

    def test_ambiguous_disagreement_and_low_confidence_are_quarantined(self):
        case = {'index':'new','context_text':'仕事の後に、','input':'カエル','expected_output':['帰る']}
        for decision in (
            {'status':'ambiguous','confidence':'high','expected_output':['帰る']},
            {'status':'clear','confidence':'medium','expected_output':['帰る']},
            {'status':'clear','confidence':'high','expected_output':['変える']},
        ):
            accepted, quarantined = finalize([case],[decision],self.excluded)
            self.assertFalse(accepted)
            self.assertEqual(len(quarantined),1)

    def test_blindly_verified_alternative_forms_preserved_and_overlap_rechecked(self):
        case = {'index':'new','context_text':'今日も','input':'ガンバル','expected_output':['頑張る']}
        decision = {'status':'clear','confidence':'high','expected_output':['頑張る','がんばる']}
        accepted, quarantined = finalize([case],[decision],self.excluded)
        self.assertEqual(accepted[0]['expected_output'],['頑張る','がんばる'])
        excluded = [{'index':2,'context_text':'今日も','input':'ガンバル','expected_output':['がんばる']}]
        accepted, quarantined = finalize([case],[decision],excluded)
        self.assertFalse(accepted)
        self.assertEqual(quarantined[0]['quarantine_reason'],'verified_form_overlaps_ajimee')

    def test_malformed_reading_counts_and_truncated_responses_rejected(self):
        job = make_jobs(CONFIG)[0]
        row = {'context_text':'今日の予定は','input':'ヨテイヲカクニンスル','expected_output':['予定を確認する'],'reason_zh':'测试'}
        self.assertEqual(len(parse_generated(body([row]),job)),1)
        for bad in ('予定を確認する','ヨテイをカクニンスル','ABC'):
            with self.assertRaises(ValueError):
                parse_generated(body([{**row,'input':bad}]),job)
        with self.assertRaises(ValueError):
            parse_generated(body([],finish='length'),job)
        decision = {'id':'00','status':'clear','confidence':'high','expected_output':['予定を確認する'],'reason_zh':'测试'}
        with self.assertRaisesRegex(ValueError,'ID'):
            parse_verified(body([decision]),[row])

    def test_config_change_and_frozen_file_edits_rejected(self):
        self.run_fixture(Transport())
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError,'changed'):
            run(self.output,{**CONFIG,'seed':44},self.exclude,dry_run=True)
        (self.output/'ime-dev-draft.json').write_text('[]',encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError,'Frozen'):
            run(self.output,CONFIG,self.exclude)

    def test_verified_draft_is_compatible_with_preparation_and_real_export_loader(self):
        self.run_fixture(Transport())
        prepared = self.root/'prepared'
        manifest = prepare_development(self.output/'ime-dev-draft.json',prepared,self.exclude,labels_reviewed=False)
        self.assertFalse(manifest['labels_reviewed'])
        exported = prepared/'ajimee-results'
        exported.mkdir()
        (exported/'ajimee-input.json').write_bytes((prepared/'ajimee-input.json').read_bytes())
        inputs=json.loads((prepared/'ajimee-input.json').read_text(encoding='utf-8'))
        result={'n_best':20,'items':[{'query':r['query'],'left_context':r['left_context'],'right_context':'',
             'answers':r['answer'],'outputs':[{'text':r['answer'][0],'score':-1.}],'max_rank':0} for r in inputs]}
        (exported/'azookey-candidates.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
        for name,value in [('converter-version.txt','a'*40),('dictionary-versions.txt',' '+'b'*40+' Dictionary'),('swift-version.txt','Swift')]:
            (exported/name).write_text(value,encoding='utf-8')
        self.assertEqual(len(load_export(prepared)[2]),20)


if __name__=='__main__':
    unittest.main()
