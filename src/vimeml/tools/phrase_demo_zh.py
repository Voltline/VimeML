"""Local prediction demo for vimeml-tiny-zh-v1; no training, no API, no output filter.

Three modes, all on the published FP32 inference bundle:
- predict: the ime-predict-zh-v1 beam decoder (2-6 Chinese characters, top 5), optionally
  constrained by typed pinyin (first syllable or initials, rime-ice single-character readings);
- continue: free continuation, greedy plus sampled paths, any characters, stops at end of turn;
- score: log-probability of each given candidate after the prefix (the reranking scorer).

The page shows raw model output. The client-side keyword blocking of the keyboard is not
applied here, so the demo shows what the model itself would propose.
"""
import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from vimeml.benchmarks import predict_zh as P
from vimeml.benchmarks.zh13_lm_scores import score_item
from vimeml.deployment.zh_v1 import BundleLMZh

ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(__file__).with_name("phrase_demo_zh.html")
PROMPTS = ROOT / "configs/phrase-demo-prompts-zh.json"
BUNDLE = ROOT / "artifacts/releases/vimeml-tiny-zh-v1-hf-v1/inference"
READINGS = ROOT / "datasets/ime-standard-zh-v1/engine/rime-ice/cn_dicts"
MODES = {"predict", "continue", "score"}
SAMPLING = {"temperature": 0.8, "top_k": 50, "top_p": 0.9, "samples": 5}


def validate_request(data):
    if not isinstance(data, dict):
        raise ValueError("请求必须是 JSON 对象。")
    mode = data.get("mode", "predict")
    if mode not in MODES:
        raise ValueError("未知的模式。")
    prefix = data.get("prefix", "")
    turns = data.get("turns", [])
    if not isinstance(prefix, str) or len(prefix) > 300 or any(ord(c) < 32 for c in prefix):
        raise ValueError("当前输入应为单行文字，最多 300 字。")
    if (not isinstance(turns, list) or len(turns) > 8
            or any(not isinstance(t, str) or len(t) > 300 or any(ord(c) < 32 for c in t) for t in turns)):
        raise ValueError("前文最多 8 条，每条单行、最多 300 字。")
    turns = [t for t in turns if t.strip()]
    settings = {"mode": mode, "prefix": prefix, "turns": turns}
    if mode == "predict":
        kind, keys = data.get("kind", ""), data.get("keys", "")
        if kind not in ("", "syllable", "initials") or not isinstance(keys, str):
            raise ValueError("拼音约束格式错误。")
        keys = keys.strip().lower()
        if kind and not (keys.isascii() and keys.isalpha() and 1 <= len(keys) <= (6 if kind == "initials" else 8)):
            raise ValueError("拼音约束应为字母：首音节最多 8 个，声母 1 到 6 个（每字一个）。")
        if not prefix and not turns:
            raise ValueError("请填写当前输入或前文。")
        settings.update(kind=kind, keys=keys if kind else "")
    elif mode == "continue":
        tokens, seed = data.get("max_tokens", 16), data.get("seed", 42)
        if type(tokens) is not int or not 4 <= tokens <= 48:
            raise ValueError("续写长度上限应为 4 到 48 个 token。")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed 超出范围。")
        settings.update(max_tokens=tokens, seed=seed)
    else:
        candidates = data.get("candidates", [])
        if not isinstance(candidates, list) or any(not isinstance(c, str) for c in candidates):
            raise ValueError("候选格式错误。")
        candidates = list(dict.fromkeys(c.strip() for c in candidates if c.strip()))
        if not 1 <= len(candidates) <= 20 or any(len(c) > 30 or any(ord(x) < 32 for x in c) for c in candidates):
            raise ValueError("请给 1 到 20 个候选，每行一个，每个最多 30 字。")
        if not prefix:
            raise ValueError("候选打分需要当前输入。")
        settings.update(candidates=candidates)
    return settings


class PhraseDemoZh:
    def __init__(self, bundle, readings, cases):
        self.bundle, self.readings, self.cases = bundle, readings, cases
        self.model, self.processor = bundle.model, bundle.processor
        self.lm = P.VimeLM(bundle.model, bundle.processor, "cpu", bundle.special)
        self.lock = threading.Lock()

    def info(self):
        meta = self.bundle.metadata
        return {"model": {k: meta[k] for k in ("release_name", "checkpoint_sha256", "checkpoint_step",
                                                "tokenizer_sha256", "precision", "device")},
                "parameters": self.model.parameter_count(), "pinyin": self.readings is not None,
                "examples": self.cases,
                "decoder": {"beam": P.BEAM, "min_chars": P.MIN_CHARS, "max_chars": P.MAX_CHARS, "top": P.TOP},
                "sampling": SAMPLING}

    def predict(self, s):
        constraint = None
        if s["kind"]:
            if self.readings is None:
                raise ValueError("本机没有 rime-ice 读音表，拼音约束不可用。")
            constraint = P.Constraint(s["kind"], s["keys"])
        rows = P.predict(self.lm, s["turns"], s["prefix"], constraint, self.readings)
        return [{"text": t, "logp": v, "per_char": v / len(t)} for t, v in rows]

    def continue_(self, s):
        ids = self.lm.context(s["turns"], s["prefix"])
        if len(ids) + s["max_tokens"] > self.lm.context_length:
            raise ValueError("前文加续写超过模型的 128 token 上下文，请缩短前文或续写长度。")
        eos = self.bundle.special["eos"]
        generator = torch.Generator().manual_seed(s["seed"])
        out, seen = [], set()
        for path in range(1 + SAMPLING["samples"]):
            seq, logp, stop = list(ids), 0.0, "max_tokens"
            for _ in range(s["max_tokens"]):
                logits = self.model(torch.tensor([seq]))[0, -1].float()
                logprobs = logits.log_softmax(-1)
                if path == 0:
                    token = int(logprobs.argmax())
                else:
                    scaled = (logits / SAMPLING["temperature"]).log_softmax(-1)
                    values, index = scaled.topk(SAMPLING["top_k"])
                    probs = values.exp()
                    keep = probs.cumsum(0) - probs < SAMPLING["top_p"]
                    probs = probs * keep
                    token = int(index[torch.multinomial(probs / probs.sum(), 1, generator=generator)])
                logp += float(logprobs[token])
                if token == eos:
                    stop = "end_of_turn"
                    break
                seq.append(token)
            text = self.processor.decode(seq[len(ids):])
            if text in seen:
                continue
            seen.add(text)
            out.append({"text": text, "logp": logp, "kind": "greedy" if path == 0 else "sample",
                        "stop": stop, "tokens": len(seq) - len(ids)})
        return out

    def score(self, s):
        result = score_item(self.model, self.processor, s["prefix"], s["candidates"])
        rows = [{"text": c, "logp": v, "per_char": v / len(c), "tokens": n}
                for c, v, n in zip(s["candidates"], result["scores"], result["scored_tokens"])]
        rows.sort(key=lambda r: -r["logp"])
        total = max(r["logp"] for r in rows)
        weights = [math.exp(r["logp"] - total) for r in rows]
        for r, w in zip(rows, weights):
            r["share"] = w / sum(weights)
        return rows

    @torch.inference_mode()
    def suggest(self, data):
        s = validate_request(data)
        started = time.perf_counter()
        rows = {"predict": self.predict, "continue": self.continue_, "score": self.score}[s["mode"]](s)
        return {**s, "results": rows, "elapsed_ms": (time.perf_counter() - started) * 1000,
                "context_tokens_including_bos": len(self.lm.context(s["turns"], s["prefix"]))}


def handler_for(demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # never log what is typed

        def reply(self, status, data, content_type="application/json; charset=utf-8"):
            raw = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def same_origin(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            return (self.headers.get("Host") == expected
                    and self.headers.get("Origin", f"http://{expected}") == f"http://{expected}")

        def do_GET(self):
            if not self.same_origin():
                return self.reply(403, {"error": "仅接受本机同源请求。"})
            if self.path == "/":
                return self.reply(200, ASSETS.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/info":
                return self.reply(200, demo.info())
            return self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if not self.same_origin():
                return self.reply(403, {"error": "仅接受本机同源请求。"})
            if self.path != "/api/suggest":
                return self.reply(404, {"error": "Not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 32768 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    return self.reply(400, {"error": "请求格式或长度错误。"})
                data = json.loads(self.rfile.read(size).decode("utf-8"))
                if not demo.lock.acquire(blocking=False):
                    return self.reply(429, {"error": "模型正在计算，请稍后重试。"})
                try:
                    result = demo.suggest(data)
                finally:
                    demo.lock.release()
                return self.reply(200, result)
            except (ValueError, UnicodeError) as error:
                return self.reply(400, {"error": str(error)})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=BUNDLE, help="vimeml-tiny-zh-v1 inference bundle")
    parser.add_argument("--readings", type=Path, default=READINGS, help="rime-ice cn_dicts (pinyin constraint)")
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    if args.threads < 1 or not 1 <= args.port <= 65535:
        parser.error("Invalid threads or port.")
    try:
        torch.set_num_threads(args.threads)
        cases = json.loads(args.prompts.read_text(encoding="utf-8"))
        bundle = BundleLMZh(args.bundle)
        readings = P.load_readings(args.readings) if args.readings.is_dir() else None
        demo = PhraseDemoZh(bundle, readings, cases)
        demo.suggest({"mode": "predict", "prefix": cases[0]["prefix"]})  # warm up
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(demo))
        print(f"Ready: http://127.0.0.1:{args.port}/\nPinyin constraint: {'on' if readings else 'off'}. "
              "Raw model output, no keyword filter. Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever(poll_interval=.2)
        finally:
            server.server_close()
    except KeyboardInterrupt:
        return
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
