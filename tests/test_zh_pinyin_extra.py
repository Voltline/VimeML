"""[[data.extra]] in joint_zh_pinyin: extra directories join each kind's batch after the primary one."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training import joint_zh_pinyin as JP  # noqa: E402


def write_store(directory, samples):
    directory.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        lengths = np.array([len(ids) for ids, _ in samples], dtype="<u8")
        offsets = np.zeros(len(samples) + 1, dtype="<u8")
        np.cumsum(lengths, out=offsets[1:])
        np.array([t for ids, _ in samples for t in ids], dtype="<u2").tofile(directory / f"{split}.tokens.bin")
        offsets.tofile(directory / f"{split}.offsets.bin")
        np.array([f for _, f in samples], dtype="u1").tofile(directory / f"{split}.first.bin")


def make_data(directory, base, tokenizer="tok", kinds=JP.BASE_STORES):
    """Plain windows start with BOS (2); pinyin samples are [2,4,ctx,5,keys,6,out,3] with first at out."""
    stores = {}
    for kind in kinds:
        k = JP.STORES.index(kind)
        samples = []
        for i in range(7 + k):
            body = [base + 10 * k + i] * (12 + i)
            if kind == "plain":
                samples.append(([2] + body + [3], 1))
            else:
                samples.append(([2, 4, 9, 5, 7, 6] + body + [3], 6))
        write_store(directory / kind, samples)
        stores[kind] = {"examples": len(samples), "files": {}}
    manifest = {"format": JP.DATA_FORMAT, "status": "complete", "tokenizer_sha256": tokenizer,
                "stores": {"train": stores, "validation": stores}}
    (directory / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    return hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()


def ids(batch):
    return batch["input_ids"].tolist()


class ExtraTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.r1, self.r2 = self.root / "r1", self.root / "r2"
        self.r1_sha, self.r2_sha = make_data(self.r1, 100), make_data(self.r2, 500)
        self.mix = {"plain": 3, "full": 2, "mixed": 2, "initials": 1, "error": 1}
        self.extra_mix = {"plain": 2, "full": 1, "mixed": 0, "initials": 1, "error": 0}

    def tearDown(self):
        self.tmp.cleanup()

    def test_primary_part_unchanged_and_extra_follows(self):
        alone = JP.Schedule(self.r1, self.mix, 7, 0.3)
        both = JP.Schedule(self.r1, self.mix, 7, 0.3, [(self.r2, self.extra_mix)])
        for step in range(12):
            a, b = alone[step], both[step]
            self.assertEqual(set(a), set(b))
            for kind in JP.BASE_STORES:
                n, extra = self.mix[kind], self.extra_mix[kind]
                rows_a, rows_b = ids(a[kind]), ids(b[kind])
                self.assertEqual(len(rows_b), n + extra)
                # Padding width can differ, so compare unpadded rows.
                strip = lambda rows: [[t for t in r if t != 0] for r in rows]  # noqa: E731
                self.assertEqual(strip(rows_b[:n]), strip(rows_a))
                for row in strip(rows_b[n:]):
                    self.assertTrue(any(t >= 500 for t in row), kind)

    def test_extra_stream_covers_store_each_pass(self):
        both = JP.Schedule(self.r1, self.mix, 7, 0.0, [(self.r2, self.extra_mix)])
        n = len(both.stores[1, "full"])
        seen = [i for step in range(n) for i in both.indices("full", step, 1)]
        self.assertEqual(sorted(seen), list(range(n)))
        self.assertNotEqual(both.permutation("full", 0, 1).tolist(), both.permutation("full", 0, 0)[:n].tolist())
        self.assertAlmostEqual(both.epochs(n)["full@1"], 1.0)

    def test_pickle_round_trip(self):
        import pickle
        both = JP.Schedule(self.r1, self.mix, 7, 0.3, [(self.r2, self.extra_mix)])
        again = pickle.loads(pickle.dumps(both))
        for kind in JP.BASE_STORES:
            self.assertEqual(ids(again[5][kind]), ids(both[5][kind]))

    def plan(self, extra, distillation=None, mix=None):
        return JP.build_plan(self.config(extra, distillation, mix), self.root, verify_files=False)

    def config(self, extra, distillation=None, mix=None):
        config = {
            "stage": JP.STAGE, "architecture": JP.ARCHITECTURE,
            "model": {"vocab_size": 12288, "context_length": 128, "d_model": 384, "n_heads": 6, "n_layers": 9,
                      "d_ff": 960, "dropout": 0.0, "norm": "rmsnorm", "norm_eps": 1e-5, "activation": "swiglu",
                      "bias": False, "tie_embeddings": True, "position_embedding": "learned"},
            "runtime": {"compile_backbone": True},
            "initialization": {"mode": "checkpoint", "checkpoint": "x.pt", "checkpoint_sha256": "0"},
            "data": {"directory": "r1", "manifest_sha256": self.r1_sha, "mix": mix or self.mix, **extra},
            "training": {"device": "cuda", "precision": "bf16", "seed": 7, "normalization": "per_kind",
                         "cpu_threads": 1, "num_workers": 1, "max_steps": 100, "learning_rate": 1e-4,
                         "min_learning_rate": 1e-5, "warmup_steps": 0, "log_every": 1, "eval_every": 1,
                         "checkpoint_every": 1, "keep_every": 1, "validation_batch": 1,
                         "validation_plain_examples": 1},
        }
        if distillation:
            config["distillation"] = distillation
        return config

    def test_plan_totals(self):
        plain = self.plan({})
        self.assertNotIn("sources", plain)
        self.assertEqual(plain["mix"], self.mix)
        extra = {"extra": [{"directory": "r2", "manifest_sha256": self.r2_sha, "mix": self.extra_mix}]}
        plan = self.plan(extra)
        self.assertEqual(plan["mix"], {k: self.mix[k] + self.extra_mix[k] for k in JP.BASE_STORES})
        self.assertEqual(plan["batch"], sum(self.mix.values()) + sum(self.extra_mix.values()))
        self.assertEqual(plan["sources"][1]["mix"], self.extra_mix)
        self.assertAlmostEqual(plan["passes"]["full@1"], 100 * 1 / 8)
        self.assertNotIn("mixed@1", plan["passes"])
        self.assertEqual(plan["signatures"]["extra_data"], [self.r2_sha])

    def test_plan_rejects(self):
        bad = [
            {"extra": [{"directory": "r1", "manifest_sha256": self.r1_sha, "mix": self.extra_mix}]},
            {"extra": [{"directory": "r2", "manifest_sha256": "0" * 64, "mix": self.extra_mix}]},
            {"extra": [{"directory": "r2", "manifest_sha256": self.r2_sha, "mix": {"bogus": 4}}]},
            {"extra": [{"directory": "r2", "manifest_sha256": self.r2_sha, "mix": {"typo": 4}}]},     # r2 has no typo
        ]
        for extra in bad:
            with self.assertRaises(ValueError):
                self.plan(extra)
        other = self.root / "other"
        sha = make_data(other, 900, tokenizer="different")
        with self.assertRaises(ValueError):
            self.plan({"extra": [{"directory": "other", "manifest_sha256": sha, "mix": self.extra_mix}]})


class TypoExtraTest(ExtraTest):
    """A typo-only extra directory (data-zh-pinyin-typo.toml): trains `typo` and validates it."""

    def setUp(self):
        super().setUp()
        self.typo = self.root / "typo"
        self.typo_sha = make_data(self.typo, 700, kinds=("typo",))

    def typo_extra(self, n):
        return {"extra": [{"directory": "typo", "manifest_sha256": self.typo_sha, "mix": {"typo": n}}]}

    def test_typo_joins_last_and_leaves_other_kinds_unchanged(self):
        alone = JP.Schedule(self.r1, self.mix, 7, 0.3)
        both = JP.Schedule(self.r1, self.mix, 7, 0.3, [(self.typo, {"typo": 3})])
        for step in range(12):
            a, b = alone[step], both[step]
            self.assertEqual(list(b), list(a) + ["typo"])
            for kind in a:
                self.assertEqual(ids(a[kind]), ids(b[kind]))
            self.assertEqual(len(ids(b["typo"])), 3)
            self.assertTrue(all(any(t >= 700 for t in row) for row in ids(b["typo"])))

    def test_plan_with_typo(self):
        plan = self.plan(self.typo_extra(4))
        self.assertEqual(plan["mix"], {**self.mix, "typo": 4})
        self.assertEqual(plan["batch"], sum(self.mix.values()) + 4)
        self.assertEqual(plan["validation_sources"]["typo"], "typo")
        self.assertEqual(plan["validation_sources"]["full"], "r1")
        self.assertIn("typo@1", plan["passes"])
        sets = JP.validation_sets(self.r1, plan, 7, self.root)
        self.assertEqual(list(sets), list(JP.BASE_STORES) + ["typo"])
        old = JP.validation_sets(self.r1, {k: v for k, v in plan.items() if k != "validation_sources"}, 7, self.root)
        self.assertEqual(list(old), list(JP.BASE_STORES))
        for kind in old:
            self.assertEqual(old[kind][1], sets[kind][1])
        for store, _ in list(sets.values()) + list(old.values()):
            del store.ids

    def test_validation_only_extra(self):
        plan = self.plan(self.typo_extra(0))
        self.assertEqual(plan["mix"], {**self.mix, "typo": 0})
        self.assertEqual(plan["batch"], sum(self.mix.values()))
        self.assertEqual(plan["validation_sources"]["typo"], "typo")
        self.assertNotIn("typo@1", plan["passes"])

    def test_typo_rejects(self):
        with self.assertRaises(ValueError):           # the primary has no typo store
            self.plan({}, mix={**self.mix, "typo": 2})
        with self.assertRaises(ValueError):
            self.plan(self.typo_extra(4), {"teacher": "t.pt", "teacher_sha256": "0", "alpha": 0.5,
                                           "temperature": 1.0, "ce_only_kinds": ["bogus"]})
        plan = self.plan(self.typo_extra(4), {"teacher": "t.pt", "teacher_sha256": "0", "alpha": 0.5,
                                              "temperature": 1.0, "ce_only_kinds": ["typo"]})
        self.assertEqual(plan["distillation"]["ce_only_kinds"], ["typo"])


class TinyLM(__import__("torch").nn.Module):
    """model(ids) -> logits; model(ids, labels) -> {"loss_sum"} like the 12K student."""

    def __init__(self, seed):
        import torch
        super().__init__()
        torch.manual_seed(seed)
        self.embed = torch.nn.Embedding(1000, 8)
        self.head = torch.nn.Linear(8, 1000)

    def forward(self, ids, labels=None):
        import torch
        logits = self.head(self.embed(ids))
        if labels is None:
            return logits
        valid = labels != -100
        return {"loss_sum": torch.nn.functional.cross_entropy(logits[valid], labels[valid], reduction="sum")}


class CeOnlyExtraTest(ExtraTest):
    """[[data.extra]] ce_only = true: the c10-hard arm trains its extra rows on cross-entropy alone."""

    distill = {"teacher": "t.pt", "teacher_sha256": "0", "alpha": 0.5, "temperature": 1.0}

    def test_batches_unchanged_without_ce_only(self):
        both = JP.Schedule(self.r1, self.mix, 7, 0.3, [(self.r2, self.extra_mix)])
        flagged = JP.Schedule(self.r1, self.mix, 7, 0.3, [(self.r2, self.extra_mix)], [1])
        for step in range(6):
            a, b = both[step], flagged[step]
            for kind in a:
                self.assertNotIn("hard", a[kind])
                self.assertEqual(ids(a[kind]), ids(b[kind]))
                self.assertEqual(a[kind]["labels"].tolist(), b[kind]["labels"].tolist())
                n = self.mix[kind]
                self.assertEqual(b[kind]["hard"].tolist(), [False] * n + [True] * self.extra_mix[kind])

    def test_pickle_keeps_ce_only(self):
        import pickle
        flagged = JP.Schedule(self.r1, self.mix, 7, 0.3, [(self.r2, self.extra_mix)], [1])
        again = pickle.loads(pickle.dumps(flagged))
        self.assertEqual(again[3]["full"]["hard"].tolist(), flagged[3]["full"]["hard"].tolist())

    def extra(self, flag):
        return {"extra": [{"directory": "r2", "manifest_sha256": self.r2_sha, "mix": self.extra_mix,
                           "ce_only": flag}]}

    def test_plan(self):
        self.assertEqual(self.plan(self.extra(True), self.distill)["ce_only_sources"], [1])
        self.assertNotIn("ce_only_sources", self.plan(self.extra(False), self.distill))
        with self.assertRaises(ValueError):
            self.plan(self.extra(True))
        with self.assertRaises(ValueError):
            self.plan(self.extra("yes"), self.distill)

    def test_update_is_ce_on_hard_rows_plus_kd_on_the_rest(self):
        import torch
        schedule = JP.Schedule(self.r1, self.mix, 7, 0.0, [(self.r2, self.extra_mix)], [1])
        batches = schedule[2]
        mix = {k: self.mix[k] + self.extra_mix[k] for k in JP.BASE_STORES}
        d = dict(self.distill, ce_only_kinds=[])
        teacher = TinyLM(1)
        cpu = torch.device("cpu")
        for normalization in ("tokens", "per_kind"):
            plan = {"settings": {"normalization": normalization, "grad_clip": 1e9}, "distillation": d,
                    "mix": mix, "batch": sum(mix.values())}
            model = TinyLM(0)
            report = JP.update(model, torch.optim.SGD(model.parameters(), lr=0.0), batches, plan, cpu, "fp32",
                               teacher)
            got = [p.grad.clone() for p in model.parameters()]
            ref = TinyLM(0)
            tokens = {k: int((b["labels"] != -100).sum()) for k, b in batches.items()}
            total = 0
            for kind, b in batches.items():
                hard, labels = b["hard"], b["labels"]
                logits = ref(b["input_ids"])
                with torch.no_grad():
                    tl = teacher(b["input_ids"])
                soft, _, _ = JP.distilled_loss(logits[~hard], tl[~hard], labels[~hard], 0.5, 1.0)
                h = labels[hard]
                ce = torch.nn.functional.cross_entropy(logits[hard][h != -100], h[h != -100], reduction="sum")
                scale = (1 / sum(tokens.values()) if normalization == "tokens"
                         else mix[kind] / plan["batch"] / tokens[kind])
                total = total + (soft + ce) * scale
                self.assertEqual(int(hard.sum()), self.extra_mix[kind])
            total.backward()
            for g, p in zip(got, ref.parameters()):
                self.assertTrue(torch.allclose(g, p.grad, atol=1e-6, rtol=1e-5))
            self.assertIn("kl", report["kinds"]["full"])

if __name__ == "__main__":
    unittest.main()
