"""Local short-continuation demo and frozen qualitative suite; no training/API."""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from vimeml.training.data import file_sha, write_json
from vimeml.training.evaluate_ime import beam_suggestions, generate_samples, unique_suggestions
from vimeml.training.infer import JapaneseLM, ROOT

ASSETS = Path(__file__).with_name("phrase_demo.html")
PROMPTS = ROOT / "configs/phrase-demo-prompts.json"


def validate_request(data):
    if not isinstance(data, dict):
        raise ValueError("请求必须是JSON对象。")
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 500 or any(ord(c) < 32 for c in prompt):
        raise ValueError("请输入单行日语句内前缀（1～500字符）；不要带换行。")
    if any(c in "。！？!?" for c in prompt):
        raise ValueError("当前演示只使用句内上下文。请删除已结束的句子，填写新一句的前缀。")
    mode = data.get("mode", "beam")
    count, tokens, seed = data.get("count", 5), data.get("max_tokens", 8), data.get("seed", 42)
    if mode not in {"beam", "sample"}:
        raise ValueError("未知的生成方式。")
    if type(count) is not int or not 3 <= count <= 5:
        raise ValueError("候选数应为3～5。")
    if type(tokens) is not int or not 4 <= tokens <= 16:
        raise ValueError("新token上限应为4～16。")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed超出范围。")
    return dict(prompt=prompt, mode=mode, count=count, max_tokens=tokens, seed=seed)


def present(item):
    """Preserve the exact decoded insertion text; do not hide incomplete endings."""
    if set(item["flags"]) & {"prefix_changed", "replacement_character", "empty"}:
        return None
    text = item["text"]
    # A vocabulary piece can contain text after punctuation. Keep the first
    # sentence only, retaining its punctuation, and explicitly record the edit.
    end = next((i + 1 for i, c in enumerate(text) if c in "。！？!?\n"), len(text))
    text = text[:end]
    if not text.strip(" 、，,。！？!?\n\t"):
        return None
    flags = list(item["flags"])
    if end < len(item["text"]):
        flags.append("display_after_sentence_removed")
    incomplete = item["stop_reason"] in {"max_new_tokens", "context_limit"} and not text.endswith(tuple("。！？!?"))
    return {**item, "text": text, "raw_text": item["text"], "flags": flags,
            "may_be_incomplete": incomplete, "new_tokens": len(item["new_token_ids"])}


class PhraseDemo:
    def __init__(self, lm, cases):
        self.lm, self.cases, self.lock = lm, cases, threading.Lock()

    def info(self):
        return {"model": self.lm.metadata, "parameters": self.lm.model.parameter_count(),
                "examples": self.cases,
                "note": "本地开发机FP32推理；耗时不代表iPhone键盘。建议为短续写，可能尚未完成一个词。"}

    @torch.inference_mode()
    def suggest(self, data):
        settings = validate_request(data)
        started = time.perf_counter()
        prompt = settings["prompt"]
        context_tokens = len(self.lm.prefix_ids(prompt))
        if settings["mode"] == "beam":
            raw = beam_suggestions(self.lm, prompt, max_tokens=settings["max_tokens"],
                                   width=8, count=8, alpha=.7)
        else:
            greedy, sampled = generate_samples(self.lm, prompt, settings["max_tokens"], settings["seed"], attempts=8)
            raw = unique_suggestions([greedy, *sampled], 9)
        suggestions, seen = [], set()
        for item in raw:
            candidate = present(item)
            if candidate is None or candidate["text"] in seen:
                continue
            seen.add(candidate["text"])
            suggestions.append(candidate)
            if len(suggestions) == settings["count"]:
                break
        if self.lm.device.type == "cuda":
            torch.cuda.synchronize(self.lm.device)
        return {**settings, "context_tokens_including_bos": context_tokens,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "suggestions": suggestions, "returned_count": len(suggestions),
                "discarded_invalid_outputs": sum(present(item) is None for item in raw),
                "raw_outputs": raw}


def handler_for(demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Do not log users' Japanese text.

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
            return self.headers.get("Host") == expected and self.headers.get("Origin", f"http://{expected}") == f"http://{expected}"

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
                if not 1 <= size <= 16384 or self.headers.get("Content-Type", "").split(';')[0] != "application/json":
                    return self.reply(400, {"error": "请求格式或长度错误。"})
                data = json.loads(self.rfile.read(size).decode("utf-8"))
                if not demo.lock.acquire(blocking=False):
                    return self.reply(429, {"error": "模型正在生成，请稍后重试。"})
                try:
                    result = demo.suggest(data)
                finally:
                    demo.lock.release()
                return self.reply(200, result)
            except (ValueError, UnicodeError) as error:
                return self.reply(400, {"error": str(error)})
    return Handler


def run_suite(demo, path, output, max_tokens=8, mode="beam"):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Suite output is not empty; use a new --output.")
    cases = []
    for i, case in enumerate(demo.cases):
        result = demo.suggest({"prompt": case["prompt"], "max_tokens": max_tokens, "mode": mode, "seed": 42+i})
        cases.append({**case, **result})
        print(f"[{i+1}/{len(demo.cases)}] {case['prompt']} -> " + ' / '.join(s['text'] for s in result['suggestions']), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    report = {"format": "vimeml_phrase_demo_suite_v1", "model": demo.lm.metadata,
              "prompts_sha256": file_sha(path), "cases": cases,
              "settings": {"mode": mode, "beam_width": 8, "alpha": .7, "max_tokens": max_tokens, "count": 5,
                           "sampling": {"temperature": .8, "top_k": 50, "top_p": .9, "attempts": 8}},
              "note": "Qualitative developer prompts, not held-out accuracy. No training/API. Timings exclude model loading and warmup; not iOS timings."}
    write_json(output / "samples.json", report)
    lines = ["# 短语联想演示样本", "", "只读当前模型；短续写可能在token上限处未完成。此表不自动判定自然度。", "",
             "| 前缀 | 建议 | 耗时 |", "| --- | --- | ---: |"]
    for case in cases:
        rendered = ' / '.join(s['text'] + ('〔未完〕' if s['may_be_incomplete'] else '') for s in case['suggestions'])
        lines.append(f"| {case['prompt']} | {rendered} | {case['elapsed_ms']:.0f} ms |")
    (output / "samples.md").write_text('\n'.join(lines)+'\n', encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/models/tiny-ja-v1/best.pt")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/tokenizers/ja-unigram-16k-v1")
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--suite", action="store_true", help="Run the fixed qualitative suite and exit; no server.")
    parser.add_argument("--max-tokens", type=int, default=8, help="Suite new-token limit (web UI sets its own).")
    parser.add_argument("--suite-mode", choices=("beam", "sample"), default="beam")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phrase-demo/tiny-ja-v1")
    args = parser.parse_args(argv)
    if args.threads < 1 or not 1 <= args.port <= 65535 or not 4 <= args.max_tokens <= 16:
        parser.error("Invalid threads, port or token limit.")
    try:
        torch.set_num_threads(args.threads)
        cases = json.loads(args.prompts.read_text(encoding="utf-8"))
        if not isinstance(cases, list) or not cases or len({c['id'] for c in cases}) != len(cases):
            raise ValueError("Expected nonempty, uniquely identified prompts.")
        lm = JapaneseLM(args.checkpoint, args.tokenizer, args.device)
        for _ in range(2):
            lm.next_logits(lm.prefix_ids(cases[0]['prompt']))
        demo = PhraseDemo(lm, cases)
        if args.suite:
            run_suite(demo, args.prompts, args.output, args.max_tokens, args.suite_mode)
            print(f"Samples: {args.output.resolve() / 'samples.json'}")
            return
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(demo))
        print(f"Ready: http://127.0.0.1:{args.port}/\nModel loaded once; no training or API. Ctrl+C to stop.", flush=True)
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
