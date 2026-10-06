"""Review one shared case queue using independently limited SJTU accounts."""

import argparse
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path

from vimeml.review.run import MODEL, ROOT, QuotaLimiter, run


@contextmanager
def output_lock(output):
    """Cross-platform, process-scoped exclusive lock; no stale PID cleanup."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".review.lock").open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Review output is already in use; stop the other reviewer before resuming.") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class AccountPool:
    """Choose an available account, with separate quotas and in-flight caps."""

    def __init__(self, api_keys, labels=None, workers_per_key=3, rpm=8, tpm=80000,
                 window=60.0, dry_run=False):
        self.secrets = tuple(api_keys)
        self.size = len(self.secrets)
        self.labels = tuple(labels or [f"account_{index + 1}" for index in range(self.size)])
        if not 1 <= self.size <= 8 or len(self.labels) != self.size or len(set(self.labels)) != self.size:
            raise ValueError("Provide 1..8 accounts with unique labels.")
        if not dry_run and (any(not isinstance(key, str) or not key.strip() for key in self.secrets)
                            or len(set(self.secrets)) != self.size):
            raise ValueError("Account keys must be nonempty and distinct; duplicate keys do not add quota.")
        if not 1 <= workers_per_key <= 8 or rpm < 1 or tpm < 1:
            raise ValueError("Invalid concurrency or quota.")
        self.workers_per_key, self.rpm, self.tpm = workers_per_key, rpm, tpm
        self.dry_run = dry_run
        self.limiters = [QuotaLimiter(rpm, tpm, window) for _ in self.secrets]
        self.lock, self.cursor = threading.Lock(), 0
        self.states = [{"account": label, "enabled": True, "disabled_http_status": None,
                        "in_flight": 0, "requests_started": 0, "successful_requests": 0,
                        "http_429": 0, "requests_with_usage": 0, "total_tokens_reported": 0}
                       for label in self.labels]

    def plan(self):
        return {"count": self.size, "labels": list(self.labels),
                "workers_per_key": self.workers_per_key, "rpm_per_key": self.rpm,
                "tpm_per_key": self.tpm, "aggregate_rpm": self.rpm * self.size,
                "aggregate_tpm": self.tpm * self.size}

    def acquire(self, tokens, stop):
        if self.dry_run:
            raise RuntimeError("A dry-run pool cannot send requests.")
        if tokens > self.tpm:
            raise ValueError("Single request exceeds per-account token quota.")
        while not stop.is_set():
            waits = []
            with self.lock:
                if not any(state["enabled"] for state in self.states):
                    return None
                for offset in range(self.size):
                    index = (self.cursor + offset) % self.size
                    state = self.states[index]
                    if not state["enabled"]:
                        continue
                    if state["in_flight"] >= self.workers_per_key:
                        waits.append(0.05)
                        continue
                    ticket, wait = self.limiters[index].try_acquire(tokens)
                    if ticket is not None:
                        state["in_flight"] += 1
                        state["requests_started"] += 1
                        self.cursor = (index + 1) % self.size
                        return {"index": index, "quota_ticket": ticket}
                    waits.append(wait)
            stop.wait(min(max(min(waits, default=0.05), 0.001), 0.05))
        return None

    def key(self, ticket):
        return self.secrets[ticket["index"]]

    def label(self, ticket):
        return self.labels[ticket["index"]]

    def release(self, ticket):
        with self.lock:
            self.states[ticket["index"]]["in_flight"] -= 1

    def finish(self, ticket, usage):
        self.limiters[ticket["index"]].finish(ticket["quota_ticket"], usage)
        if isinstance(usage, dict) and type(usage.get("total_tokens")) is int and usage["total_tokens"] >= 0:
            with self.lock:
                state = self.states[ticket["index"]]
                state["requests_with_usage"] += 1
                state["total_tokens_reported"] += usage["total_tokens"]

    def success(self, ticket):
        with self.lock:
            self.states[ticket["index"]]["successful_requests"] += 1

    def pause(self, ticket, seconds):
        self.limiters[ticket["index"]].pause(seconds)
        with self.lock:
            self.states[ticket["index"]]["http_429"] += 1

    def disable(self, ticket, status):
        with self.lock:
            state = self.states[ticket["index"]]
            state["enabled"], state["disabled_http_status"] = False, status

    def snapshot(self):
        with self.lock:
            return [dict(state) for state in self.states]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/corpus-review-v1")
    parser.add_argument("--output-dir", type=Path, help="Default: INPUT/review (compatible with single-key caches)")
    parser.add_argument("--key-env", nargs="+", default=["SJTU_API_KEY", "SJTU_API_KEY_2"],
                        metavar="ENV_NAME", help="Names of distinct accounts' environment variables; never pass actual keys")
    parser.add_argument("--workers-per-key", type=int, default=3)
    parser.add_argument("--rpm-per-key", type=int, default=8)
    parser.add_argument("--tpm-per-key", type=int, default=80000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-batch-tokens", type=int, default=10000)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dry-run", action="store_true", help="Read caches and estimate independent quotas; no keys, writes or requests")
    args = parser.parse_args()
    if (not 1 <= len(args.key_env) <= 8 or len(set(args.key_env)) != len(args.key_env)
            or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in args.key_env)):
        parser.error("--key-env requires 1..8 unique environment variable names, not API keys.")
    if not 1 <= args.workers_per_key <= 8 or not 1 <= args.rpm_per_key <= 10 or not 1 <= args.tpm_per_key <= 100000:
        parser.error("workers-per-key must be 1..8; rpm-per-key 1..10; tpm-per-key 1..100000.")
    if (min(args.batch_size, args.timeout, args.max_output_tokens) < 1
            or not args.max_output_tokens < args.max_batch_tokens <= args.tpm_per_key
            or not 0 <= args.retries <= 5):
        parser.error("Invalid batch, output budget, timeout or retries.")
    keys = [None] * len(args.key_env) if args.dry_run else [os.environ.get(name, "").strip() for name in args.key_env]
    if not args.dry_run:
        missing = [name for name, key in zip(args.key_env, keys) if not key]
        if missing:
            parser.error("Set these environment variables before starting: " + ", ".join(missing))
        if len(set(keys)) != len(keys):
            parser.error("Two environment variables contain the same key; provide independent account keys.")
    pool = AccountPool(keys, labels=args.key_env, workers_per_key=args.workers_per_key,
                       rpm=args.rpm_per_key, tpm=args.tpm_per_key, dry_run=args.dry_run)
    result = run(args.input_dir, args.output_dir or args.input_dir / "review",
                 workers=len(keys) * args.workers_per_key, batch_size=args.batch_size,
                 rpm=args.rpm_per_key, tpm=args.tpm_per_key, max_batch_tokens=args.max_batch_tokens,
                 max_output_tokens=args.max_output_tokens, timeout=args.timeout, retries=args.retries,
                 dry_run=args.dry_run, model=args.model, request_pool=pool)
    if not args.dry_run and result["pending_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
