"""Sampling of the Vime 1.3 Chinese tokenizer: web and review files pooled per shard."""
import random
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.tokenizer.zh13 import sample, shard_of  # noqa: E402


def write(path, texts, row_group_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": texts}), path, row_group_size=row_group_size)


def documents(kind, shard, n):
    # Two paragraphs per document, plus an empty line that must be dropped.
    return [f"{kind}{shard}文档{i}第一段\n\n{kind}{shard}文档{i}第二段" for i in range(n)]


def make_export(root):
    train = root / "train"
    for shard in ("000_00001", "000_00002"):
        # Web: 2,000 documents in two files of one big row group each.
        write(train / f"web-{shard}-000.parquet", documents("web", shard, 1000), 1000)
        write(train / f"web-{shard}-001.parquet", documents("wex", shard, 1000), 1000)
        # Review: 200 documents cut into 20 small row groups (10 documents each).
        write(train / f"review-{shard}-000.parquet", documents("rev", shard, 200), 10)
    write(train / "chat-lccc-train.parquet", [f"聊天{i}" for i in range(50)], 50)
    return root


def config(web_lines, seed=7):
    return {"seed": seed, "web_lines": web_lines, "chat_lines": 10, "max_line_bytes": 16384}


class Zh13SampleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.export = make_export(self.root / "export")

    def tearDown(self):
        self.temporary.cleanup()

    def out(self, name):
        path = self.root / name
        path.mkdir()
        return path

    def test_shard_of_groups_web_and_review(self):
        self.assertEqual(shard_of("web-000_00009-001.parquet"), "000_00009")
        self.assertEqual(shard_of("review-000_00009-000.parquet"), "000_00009")

    def test_review_keeps_its_natural_share(self):
        shares = []
        for seed in range(20):
            _, info = sample(config(600, seed), self.export, self.out(f"s{seed}"))
            self.assertEqual((info["web_lines"], info["chat_lines"]), (600, 10))
            picked = sum(sum(v.values()) for v in info["by_shard"].values())
            shares.append(info["review_lines"] / picked)
        # Review is 200 of 2,200 documents per shard (9%); one quota per file would give it a third.
        mean = sum(shares) / len(shares)
        self.assertTrue(0.06 < mean < 0.13, mean)

    def test_quota_filled_from_several_row_groups(self):
        # 2,401 lines per shard needs more than one 1,000-document web row group.
        path, info = sample(config(4800), self.export, self.out("big"))
        self.assertEqual(info["web_lines"], 4800)
        lines = path.read_text("utf-8").splitlines()
        self.assertEqual(len(lines), 4810)
        self.assertTrue(all(lines))
        self.assertEqual(len(set(lines)), len(lines))

    def test_same_seed_same_sample(self):
        digests = []
        for name, global_seed in (("a", 0), ("b", 1)):
            random.seed(global_seed)  # the sampler must not depend on the global RNG
            digests.append(sample(config(600), self.export, self.out(name))[1]["sha256"])
        self.assertEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
