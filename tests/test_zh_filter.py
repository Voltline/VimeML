"""Stage-A rules for the Chinese web corpus: term logic, decisions, script conversion."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vimeml.corpus.zh_filter import (HAN, Thresholds, classify, counts_as_hit, load_rules,
                                     random_looking, registrable, repeated_line_fraction,
                                     traditional_ratio)

RULES="""
[categories]
gambling = ["百家乐", "赌场", "澳门赌场"]
porn = ["色情"]
political = ["台独"]
[rules]
weak = ["赌场", "色情", "台独"]
[domains]
block = ["bad.com"]
block_tier2 = ["news.tw"]
block_tlds = ["xyz"]
host_terms = ["yulecheng", "duchang"]
"""
BODY='今天天气很好，我们一起去公园散步，看到很多人在跑步和放风筝，孩子们玩得很开心。'*4


class ZhFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'terms.toml'; p.write_text(RULES,encoding='utf-8')
            cls.rules=load_rules(p)

    def test_weak_terms_need_a_second_term(self):
        weak=self.rules.weak
        self.assertFalse(counts_as_hit({'gambling':['赌场']},weak))
        self.assertTrue(counts_as_hit({'gambling':['赌场'],'porn':['色情']},weak))
        self.assertTrue(counts_as_hit({'gambling':['百家乐']},weak))

    def test_longest_term_wins(self):
        c=classify(BODY+'澳门赌场','a.com',self.rules)
        self.assertEqual((c['decision'],c['hits']),('drop',['澳门赌场']))

    def test_decisions(self):
        r=self.rules
        self.assertEqual(classify(BODY,'www.ok.cn',r)['decision'],'keep')
        self.assertEqual(classify(BODY+'赌场','ok.cn',r)['reasons'],['weak_term'])
        self.assertEqual(classify(BODY+'百家乐','ok.cn',r)['reasons'],['term_gambling'])
        self.assertEqual(classify(BODY,'www.bad.com',r)['reasons'],['domain_tier1'])
        self.assertEqual(classify(BODY,'x.news.tw',r)['decision'],'review')
        self.assertEqual(classify(BODY+'百家乐','x.news.tw',r)['decision'],'drop')
        self.assertEqual(classify(BODY,'spam.xyz',r)['reasons'],['tld'])
        self.assertEqual(classify('太短了','ok.cn',r)['reasons'],['short'])
        self.assertEqual(classify(BODY+'abc '*400,'ok.cn',r)['reasons'],['low_han'])

    def test_host_terms_and_farms(self):
        r=self.rules
        self.assertEqual(classify(BODY,'hongyunyulecheng.u07bgqr7k.com',r)['reasons'],['host_term'])
        self.assertEqual(classify(BODY,'www.duchang.gov.cn',r)['decision'],'keep')
        self.assertEqual(classify(BODY,'www.ceg8rb.gov.cn.vyrtf.cn',r)['reasons'],['fake_gov'])
        self.assertEqual(classify(BODY,'www.beijing.gov.cn',r)['decision'],'keep')
        self.assertEqual(registrable('a.b.example.com.cn'),'example.com.cn')
        self.assertEqual(registrable('x.u07bgqr7k.com'),'u07bgqr7k.com')
        self.assertTrue(random_looking('u07bgqr7k')); self.assertTrue(random_looking('kzlue2y'))
        for ok in ('pixnet','lady8844','it168','51cto','3dmgame'): self.assertFalse(random_looking(ok),ok)

    def test_traditional_is_converted_with_taiwan_phrasing(self):
        c=classify('我們用軟體在網路上看影片，記憶體不夠，印表機也壞了。'*6,'ok.tw',self.rules)
        self.assertEqual(c['script'],'hant')
        self.assertIn('软件',c['text']); self.assertIn('网络',c['text'])
        self.assertLess(traditional_ratio(HAN.findall(BODY)),Thresholds().traditional)

    def test_repeated_lines(self):
        self.assertEqual(repeated_line_fraction('甲\n乙\n丙'),0)
        self.assertAlmostEqual(repeated_line_fraction('首页\n首页\n正文内容很长很长'),4/12)

    def test_shipped_terms_load(self):
        r=load_rules(ROOT/'configs/zh-filter-terms.toml')
        self.assertTrue(r.weak<=set(r.category) and r.tier1 and r.tier2)


if __name__=='__main__':
    unittest.main()
