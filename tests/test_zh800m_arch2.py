"""zh-800m arch2: registered shape, GQA, RoPE KV path, packed forward equivalence, search backend."""
import math
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training import model_zh800m_arch2 as A  # noqa: E402

SMALL = A.Arch2Config(vocab_size=97, context_length=32, d_model=48, n_heads=6, n_kv_heads=2, n_layers=3, d_ff=80,
                      pack_length=64)


def small(seed=0):
    torch.manual_seed(seed)
    return A.Arch2Decoder(SMALL).float()


def reference_logits(model, ids):
    """Plain re-implementation: per-head repeated K/V, explicit softmax, RoPE positions 0..T-1."""
    c = model.config
    time = ids.shape[1]
    positions = torch.arange(time)
    hidden = model.token_embedding(ids)
    causal = torch.ones(time, time, dtype=torch.bool).tril()
    for block in model.blocks:
        a = block.attention
        x = block.attention_norm(hidden)
        q = a.query_norm(a.query(x).view(1, time, c.n_heads, c.head_dim)).transpose(1, 2)
        k = a.key_norm(a.key(x).view(1, time, c.n_kv_heads, c.head_dim)).transpose(1, 2)
        v = a.value(x).view(1, time, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = A.rope(q, positions, c.rope_theta), A.rope(k, positions, c.rope_theta)
        group = c.n_heads // c.n_kv_heads
        k = torch.stack([k[:, h // group] for h in range(c.n_heads)], 1)
        v = torch.stack([v[:, h // group] for h in range(c.n_heads)], 1)
        scores = (q @ k.transpose(-2, -1) / math.sqrt(c.head_dim)).masked_fill(~causal, -torch.inf)
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, time, c.d_model)
        hidden = hidden + a.projection(out)
        hidden = hidden + block.ffn(block.ffn_norm(hidden))
    return model.lm_head(model.final_norm(hidden))


class Shape(unittest.TestCase):
    def test_registered_parameters(self):
        with torch.device("meta"):
            model = A.TinyGPTZh800mArch2()
        self.assertEqual(model.parameter_count(), A.PARAMETERS)
        self.assertEqual(A.PARAMETERS, 12288 * 1792 + 23 * A.LAYER_PARAMETERS + 1792)
        with self.assertRaises(ValueError):
            A.GPTZh800mArch2Config(n_kv_heads=8)
        with self.assertRaises(ValueError):
            A.TinyGPTZh800mArch2(A.Arch2Config())

    def test_rope_is_relative(self):
        torch.manual_seed(1)
        q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)
        dot = lambda i, j: (A.rope(q, torch.tensor([i]), 10000.0) * A.rope(k, torch.tensor([j]), 10000.0)).sum()
        self.assertAlmostEqual(float(dot(5, 2)), float(dot(13, 10)), places=4)


class Forward(unittest.TestCase):
    def test_gqa_equals_repeated_heads(self):
        model = small().eval()
        ids = torch.randint(0, 97, (1, 20))
        torch.testing.assert_close(model(ids), reference_logits(model, ids), rtol=1e-4, atol=1e-5)

    def test_kv_cache_matches_forward(self):
        model = small().eval()
        ids = torch.randint(0, 97, (2, 17))
        full = model(ids)
        cache, pieces = model.new_cache(2), []
        for a, b in ((0, 5), (5, 6), (6, 17)):
            pieces.append(model.forward_cached(ids[:, a:b], cache))
        torch.testing.assert_close(torch.cat(pieces, 1), full, rtol=1e-4, atol=1e-5)

    def test_packed_loss_and_gradients_equal_separate_rows(self):
        torch.manual_seed(3)
        sequences = []
        for size in (31, 7, 20, 3, 12, 25, 9):
            ids = torch.randint(1, 97, (size,))
            labels = torch.randint(0, 97, (size,))
            labels[: size // 3] = -100
            sequences.append((ids, labels))
        separate = small()
        total = sum(separate(i[None], l[None])["loss_sum"] for i, l in sequences)
        total.backward()
        packed_model = small()
        ids, labels, segments, placement = A.pack(sequences, 64, 32)
        self.assertLess(len(ids), len(sequences))
        out = packed_model(ids, labels, segments=segments)
        torch.testing.assert_close(out["loss_sum"], total.detach(), rtol=1e-5, atol=1e-4)
        out["loss_sum"].backward()
        for a, b in zip(separate.parameters(), packed_model.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)
        logits = packed_model(ids, segments=segments)
        for n, (row, offset) in enumerate(placement):
            alone = packed_model(sequences[n][0][None])[0]
            torch.testing.assert_close(logits[row, offset:offset + len(alone)], alone, rtol=1e-4, atol=1e-5)

    def test_bf16_autocast(self):
        """QK-norm's FP32 weight must not leave query/key FP32 next to a BF16 value under autocast."""
        model = small()
        ids = torch.randint(4, 97, (2, 12))
        segments = torch.tensor([[0] * 5 + [1] * 7, [0] * 12])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            query, key, value = model.layer_qkv(model.blocks[0], model.token_embedding(ids), torch.arange(12))
            self.assertEqual({query.dtype, key.dtype, value.dtype}, {torch.bfloat16})
            out = model(ids, ids.clone(), token_weights=torch.ones(2, 12), segments=segments)
        out["loss_sum"].backward()
        self.assertTrue(math.isfinite(float(out["loss_sum"])))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertEqual(model(ids).shape, (2, 12, 97))

    def test_pack_rules(self):
        self.assertEqual(A.segment_positions(torch.tensor([[0, 0, 0, 1, 1, -1, -1]])).tolist(), [[0, 1, 2, 0, 1, 0, 1]])
        with self.assertRaises(ValueError):
            A.pack([(torch.ones(33, dtype=torch.long), torch.ones(33, dtype=torch.long))], 64, 32)

    @unittest.skipUnless(torch.cuda.is_available(), "flex_attention path runs on CUDA")
    def test_flex_mask_equals_dense_mask(self):
        model = small().cuda()
        ids, _, segments, _ = A.pack([(torch.randint(1, 97, (n,)), torch.zeros(n, dtype=torch.long))
                                      for n in (30, 5, 17, 9)], 64, 32)
        flex = model(ids.cuda(), segments=segments.cuda())
        dense = model.cpu()(ids, segments=segments)
        torch.testing.assert_close(flex.cpu(), dense, rtol=1e-3, atol=1e-4)


class Backend(unittest.TestCase):
    def test_search_backend_matches_forward(self):
        from vimeml.deployment.zh_torch_backend import ModelBatchBackend
        model = small().eval()
        backend = ModelBatchBackend(model)
        sequences = [[2, 4, 5], [2, 4, 6, 7, 8]]
        states = []
        for s in sequences:
            logits, state = backend.prefill(s)
            self.assertEqual(state[0].shape[1], SMALL.n_kv_heads)
            self.assertLess((logits - model(torch.tensor([s]))[0, -1]).abs().max().item(), 1e-4)
            states.append(state)
        for step in ([10, 20], [30, 40]):
            logits, states = backend.decode(step, states)
            sequences = [s + [t] for s, t in zip(sequences, step)]
            for r, s in enumerate(sequences):
                self.assertLess((logits[r] - model(torch.tensor([s]))[0, -1]).abs().max().item(), 1e-4)

    def test_desktop_dispatch(self):
        from vimeml.training.zh_desktop import desktop_family
        config, model = desktop_family({"format": "vimeml_tiny_gpt_zh800m_arch2", "architecture": "tiny_gpt_zh800m_arch2"})
        self.assertIs(model, A.TinyGPTZh800mArch2)


def collated(rows, width, seed):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(2, width + 1, (rows,), generator=g)
    ids = torch.randint(1, 97, (rows, width), generator=g)
    labels = torch.randint(0, 97, (rows, width), generator=g)
    for r, n in enumerate(lengths.tolist()):
        ids[r, n:] = 0
        labels[r, n:] = -100
        labels[r, : n // 4] = -100
    return {"input_ids": ids, "labels": labels, "lengths": lengths}


class Trainer(unittest.TestCase):
    def test_packed_update_equals_the_padded_update(self):
        from vimeml.training.joint_zh200m import FAMILIES, update_joint
        batches = {"lm": collated(24, 32, 1), "full": collated(5, 20, 2), "error": collated(3, 12, 3)}
        weights = {"lm": 24 / 32, "full": 5 / 32, "error": 3 / 32}
        s = {"lm_microbatch": 4, "grad_clip": 1e9}
        results = []
        for packed in (False, True):
            model = small()
            model.packed_training = packed
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            out = update_joint(model, optimizer, {k: dict(v) for k, v in batches.items()}, weights, s,
                               torch.device("cpu"), "fp32")
            results.append((out, [p.grad.clone() for p in model.parameters()]))
        (a, ga), (b, gb) = results
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-6)
        self.assertAlmostEqual(a["loss"], b["loss"], places=5)
        for kind in ("full", "error"):
            self.assertAlmostEqual(a["pinyin"][kind]["loss"], b["pinyin"][kind]["loss"], places=5)
        self.assertLess(b["padded_positions"], a["padded_positions"] * 2)
        family = FAMILIES["zh-800m-arch2"]
        self.assertEqual((family.allowed, family.shapes[(12288, 512)]), ((8192, 16384), (23, A.PARAMETERS)))

    def test_one_row_passes_get_an_empty_row(self):
        from vimeml.training.joint_zh200m import segment_microbatches
        work = segment_microbatches([(0, collated(3, 30, 4))], 64, 32, 1)
        for ids, labels, segments, kind in work:
            self.assertEqual(len(ids), 2)
            self.assertTrue(bool((labels[1] == -100).all()) and bool((segments[1] == -1).all()))


class Configs(unittest.TestCase):
    def test_k_scaling_against_arch1(self):
        import tomllib
        base = tomllib.loads((ROOT / "configs/zh/train-zh800m-12k-c512-pinyin.toml").read_text("utf-8"))
        for k in (4,):  # the k = 8 config was never run and was removed on 2026-11-24
            c = tomllib.loads((ROOT / f"configs/zh/train-zh800m-arch2-k{k}.toml").read_text("utf-8"))
            A.GPTZh800mArch2Config(**c["model"])
            t, b = c["training"], base["training"]
            self.assertEqual(t["lm_batch"], b["lm_batch"] * k)
            self.assertEqual(c["pinyin"]["mix"], {q: v * k for q, v in base["pinyin"]["mix"].items()})
            # k = 4 keeps arch1's peak LR (precheck control round, 2026-11-20).
            self.assertAlmostEqual(t["learning_rate"], b["learning_rate"], delta=5e-6)
            self.assertAlmostEqual(t["min_learning_rate"] * 10, t["learning_rate"])
            self.assertEqual((t["warmup_steps"] * k, t["bucket_multiplier"] * k, t["checkpoint_every"] * k),
                             (b["warmup_steps"], b["bucket_multiplier"], b["checkpoint_every"]))
            same = ("seed", "lm_microbatch", "weight_decay", "beta1", "beta2", "grad_clip", "prefix_crop_probability",
                    "precision", "validation_batch")
            self.assertEqual({q: t[q] for q in same}, {q: b[q] for q in same})
            for key in ("arm", "dataset_sha256", "data_mixture", "initialization"):
                self.assertEqual(c[key], base[key])
            self.assertEqual(c["pinyin"]["manifest_sha256"], base["pinyin"]["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
