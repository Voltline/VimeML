"""Request validation and local HTTP protocol of the Chinese prediction demo."""
import json
import sys
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.tools.phrase_demo_zh import BUNDLE, handler_for, validate_request  # noqa: E402


class Validation(unittest.TestCase):
    def test_predict(self):
        s = validate_request({"prefix": "我们", "turns": ["", "你好"], "kind": "initials", "keys": "KDY"})
        self.assertEqual((s["mode"], s["turns"], s["keys"]), ("predict", ["你好"], "kdy"))
        self.assertEqual(validate_request({"prefix": "我们"})["kind"], "")
        for bad in ({"prefix": ""}, {"prefix": "a\nb"}, {"prefix": "我", "kind": "initials", "keys": "abcdefg"},
                    {"prefix": "我", "kind": "syllable", "keys": "c1"}, {"prefix": "我", "kind": "x"},
                    {"prefix": "我", "turns": ["一"] * 9}, {"mode": "other", "prefix": "我"}):
            with self.assertRaises(ValueError):
                validate_request(bad)

    def test_continue_and_score(self):
        self.assertEqual(validate_request({"mode": "continue", "prefix": "我"})["max_tokens"], 16)
        with self.assertRaises(ValueError):
            validate_request({"mode": "continue", "prefix": "我", "max_tokens": 49})
        s = validate_request({"mode": "score", "prefix": "我", "candidates": ["吃饭", " 吃饭 ", "", "看"]})
        self.assertEqual(s["candidates"], ["吃饭", "看"])
        for bad in ({"mode": "score", "prefix": "", "candidates": ["吃"]},
                    {"mode": "score", "prefix": "我", "candidates": []},
                    {"mode": "score", "prefix": "我", "candidates": ["吃"] * 0 + [str(i) for i in range(21)]}):
            with self.assertRaises(ValueError):
                validate_request(bad)


class Stub:
    lock = threading.Lock()

    def info(self):
        return {"ok": True}

    def suggest(self, data):
        return {"echo": validate_request(data)["prefix"]}


class Http(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Stub()))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, body=None, headers=None):
        c = HTTPConnection("127.0.0.1", self.port)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        return r.status, r.read()

    def test_protocol(self):
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("没有经过键盘端的关键词屏蔽".encode(), page)
        body = json.dumps({"prefix": "你好"}).encode()
        status, raw = self.request("POST", "/api/suggest", body, {"Content-Type": "application/json"})
        self.assertEqual((status, json.loads(raw)), (200, {"echo": "你好"}))
        status, _ = self.request("POST", "/api/suggest", body,
                                 {"Content-Type": "application/json", "Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.request("POST", "/api/suggest", json.dumps({"prefix": ""}).encode(),
                                 {"Content-Type": "application/json"})
        self.assertEqual(status, 400)


@unittest.skipUnless((BUNDLE / "manifest.json").exists(), "zh-v1 bundle not present")
class RealModel(unittest.TestCase):
    def test_modes(self):
        from vimeml.benchmarks.predict_zh import all_han
        from vimeml.deployment.zh_v1 import BundleLMZh
        from vimeml.tools.phrase_demo_zh import PhraseDemoZh
        demo = PhraseDemoZh(BundleLMZh(BUNDLE), None, [])
        rows = demo.suggest({"prefix": "我们明天一起去"})["results"]
        self.assertTrue(1 <= len(rows) <= 5 and all(all_han(r["text"]) and 2 <= len(r["text"]) <= 6 for r in rows))
        rows = demo.suggest({"mode": "continue", "prefix": "今天", "max_tokens": 4, "seed": 1})["results"]
        self.assertEqual(rows[0]["kind"], "greedy")
        rows = demo.suggest({"mode": "score", "prefix": "我们去", "candidates": ["吃饭", "看"]})["results"]
        self.assertAlmostEqual(sum(r["share"] for r in rows), 1.0)
        with self.assertRaises(ValueError):
            demo.suggest({"prefix": "我", "kind": "initials", "keys": "k"})


if __name__ == "__main__":
    unittest.main()
