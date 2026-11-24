"""zh-800m: model shape, encoder selections and the three-store mixture on a small fake export."""
import hashlib
from pathlib import Path
import random
import shutil
import sys
import tempfile
import tomllib
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TOKENIZERS = ROOT / "artifacts/tokenizers/zh-v13"
CHARS = "我们今天天气很好学习中文输入法模型训练数据网页聊天明天去哪里吃饭你好谢谢再见朋友工作时间问题"


def text(rng, paragraphs):
    return "\n".join("".join(rng.choice(CHARS) for _ in range(rng.randint(6, 60))) + "。"
                     for _ in range(paragraphs))


def write_export(root, prefix, documents, chat, rng):
    import pyarrow as pa
    import pyarrow.parquet as pq
    for split, count in (("train", documents), ("validation", 6)):
        (root / split).mkdir(parents=True, exist_ok=True)
        ids = [f"{prefix}-{split}-{i}" for i in range(count)]
        texts = [text(rng, rng.randint(1, 5)) for _ in ids]
        table = pa.table({"id": ids, "host": ["h"] * count, "script": ["Hans"] * count,
                          "han": [len(t) for t in texts], "text": texts})
        pq.write_table(table, root / split / "web-00000.parquet", row_group_size=64)
        if chat:
            lines = [text(rng, 1) for _ in range(chat if split == "train" else 20)]
            pq.write_table(pa.table({"text": lines, "dialogue": list(range(len(lines))), "turn": [0] * len(lines),
                                     "script": ["Hans"] * len(lines)}), root / split / f"chat-lccc-{split}.parquet")


class Zh800mModelTest(unittest.TestCase):
    def test_registered_shape_and_config(self):
        from vimeml.training.model_zh800m import LAYER_PARAMETERS, SHAPES, GPTZh800mConfig
        layers, total = SHAPES[(12288, 512)]
        self.assertEqual(total, 12288 * 1536 + 512 * 1536 + layers * LAYER_PARAMETERS + 1536)
        config = tomllib.loads((ROOT / "configs/zh/train-zh800m-12k-c512.toml").read_text("utf-8"))
        GPTZh800mConfig(**config["model"])
        with self.assertRaises(ValueError):
            GPTZh800mConfig(**{**config["model"], "n_layers": 26})
        from vimeml.training.zh800m_data import ARMS
        self.assertEqual(config["training"]["seed"], ARMS[config["arm"]][3])


@unittest.skipUnless((TOKENIZERS / "manifest.json").exists(), "needs the local 12K tokenizer")
class Zh800mDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vimeml.tokenizer.encode_zh13 import encode, measure
        cls.tmp = Path(tempfile.mkdtemp(prefix="zh800m-"))
        rng = random.Random(7)
        r1, r2 = cls.tmp / "export-r1", cls.tmp / "export-r2"
        write_export(r1, "a", 400, 3000, rng)
        write_export(r2, "b", 300, 0, rng)
        identity = {"frozen": False}
        data = cls.tmp / "data"
        total = 0
        import pyarrow.parquet as pq
        from vimeml.tokenizer.encode_zh13 import clean_paragraphs
        for t in pq.read_table(r1 / "train/web-00000.parquet")["text"].to_pylist():
            total += sum(map(len, clean_paragraphs(t)[0]))
        config = {"tokenizer_dir": str(TOKENIZERS), "workers": 2, "context_length": 512,
                  "pool_label": "zh-v13-pool", "memo_label": "zh-v13-memo",
                  "p1_target_characters": total // 2, "memo_documents": 3}
        pools = data / "pools"
        measure(config, r1, pools, identity)
        cls.pools = pools / "pools.json"

        def out(name, chat=None):
            o = {"web": data / "token-data" / name, "web_index": data / "training-data" / f"{name}-c512"}
            if chat:
                o.update(chat=data / "token-data" / chat, chat_index=data / "training-data" / f"{chat}-c512")
            return o
        encode(config, r1, 12288, out("zh13-web-12k", "zh13-chat-12k"), cls.pools, identity)
        encode(config, r1, 12288, out("zh800m-web-r1rest-12k"), cls.pools, identity, selection="rest")
        encode(config, r2, 12288, out("zh800m-web-r2-12k"), None, identity, selection="all")
        cls.data = data

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def manifest(self, name):
        from vimeml.training.zh800m_data import read
        return read(self.data / "token-data" / name / "manifest.json")

    def test_p1_and_rest_partition_export_r1(self):
        p1, rest, r2 = (self.manifest(n) for n in ("zh13-web-12k", "zh800m-web-r1rest-12k", "zh800m-web-r2-12k"))
        self.assertEqual(p1["splits"]["train"]["documents"] + rest["splits"]["train"]["documents"], 400)
        self.assertGreater(rest["splits"]["train"]["documents"], 0)
        self.assertEqual(r2["splits"]["train"]["documents"], 300)
        self.assertEqual((rest["selection"], r2["selection"]), ("rest", "all"))
        self.assertNotIn("selection", p1)
        self.assertIsNone(r2["pools_sha256"])
        self.assertNotIn("train.tiers.bin", rest["output_sha256"])
        self.assertIn("tiers", p1["splits"]["train"])

    def test_mixture_covers_every_web_window_once_and_rebuilds(self):
        from vimeml.training import zh800m_data as zd
        from vimeml.training.data import file_sha
        data_root = self.data.as_posix()
        request = zd.request_for("12k-c512", ROOT, data_root)
        target = self.tmp / "mix"
        target.mkdir()
        zd.write_lf(target / "request.json", request)
        first = zd.prepare_maps(target, ROOT)
        digest = file_sha(target / "manifest.json")
        mapping = np.load(target / "epoch-0.npy")
        record = first["epochs"][0]
        web = mapping[mapping < record["web_windows"]]
        self.assertEqual(len(np.unique(web)), record["web_windows"])
        self.assertEqual(len(web), record["web_windows"])
        self.assertAlmostEqual(record["chat_fraction"], zd.CHAT_FRACTION, delta=0.005)
        self.assertEqual(sum(record["web_tokens_by_store"].values()), record["web_tokens"])
        self.assertGreater(first["probe"]["windows"], 0)
        self.assertTrue(np.all(np.load(target / "probe-windows.npy") < record["web_windows"]))
        # A second build must reproduce the frozen maps.
        (target / "epoch-0.npy").unlink()
        zd.prepare_maps(target, ROOT)
        self.assertEqual(file_sha(target / "manifest.json"), digest)
        # The trainer accepts the mixture and reads windows from every store.
        from vimeml.training.joint_zh200m import build_plan
        config = tomllib.loads((ROOT / "configs/zh/train-zh800m-12k-c512.toml").read_text("utf-8"))
        config["data_mixture"]["directory"] = target.as_posix()
        config["dataset_sha256"] = digest
        config["runtime"]["compile_backbone"] = False
        config["training"].update(device="cpu", precision="fp32", smoke_updates=20)
        plan = build_plan(config, ROOT, verify_files=True)
        self.assertEqual(plan["stage"], "zh-800m")
        self.assertEqual(plan["parameters"], 784_157_184)
        self.assertEqual(plan["task_contract"]["windows_per_update"], 2048)
        self.assertEqual(plan["settings"]["max_steps"], -(-record["windows"] // 2048))
        dataset = zd.training_dataset(target, ROOT)
        try:
            sources = {int(dataset[i]["source_id"]) for i in range(len(dataset))}
            self.assertEqual(sources, {0, 2})
        finally:
            dataset.close()
        bad = dict(config, arm="12k-c512")
        bad["training"] = dict(config["training"], lm_batch=512)
        with self.assertRaises(ValueError):
            build_plan(bad, ROOT, verify_files=False)


if __name__ == "__main__":
    unittest.main()
