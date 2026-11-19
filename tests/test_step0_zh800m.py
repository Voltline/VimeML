import unittest

import torch

from vimeml.training import step0_zh800m as S
from vimeml.training.distributed import Context


def event(cat, name, ts, dur):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur}


class Step0Test(unittest.TestCase):
    def test_union(self):
        self.assertEqual(S.union([]), 0)
        self.assertEqual(S.union([(0, 2), (1, 3), (5, 6), (5.5, 5.8)]), 4)

    def test_breakdown(self):
        # one update of 100 us: compute 10-50 and 60-70, NCCL 40-80 (10 us under compute), sync 85-95
        trace = {"traceEvents": [
            event("user_annotation", "update", 0, 100),
            event("gpu_user_annotation", "update", 0, 100),
            event("kernel", "ampere_bf16_gemm", 10, 40),
            event("kernel", "elementwise", 60, 10),
            event("kernel", "ncclDevKernel_AllReduce_Sum_f32_RING_LL", 40, 40),
            event("cuda_runtime", "cudaStreamSynchronize", 85, 10),
            event("cuda_runtime", "cudaLaunchKernel", 1, 1),
            event("cuda_runtime", "cudaDeviceSynchronize", 95, 5),       # the timing barrier: not counted
            event("kernel", "outside", 200, 10),
        ]}
        out = S.breakdown(trace)
        self.assertEqual(out["updates"], 1)
        row = out["per_update"][0]
        self.assertAlmostEqual(row["wall"], 100e-6)
        self.assertAlmostEqual(row["compute"], 50e-6)
        self.assertAlmostEqual(row["communication"], 40e-6)
        self.assertAlmostEqual(row["exposed_communication"], 20e-6)    # 50-60 and 70-80
        self.assertAlmostEqual(row["gpu_idle"], 30e-6)                 # 0-10 and 80-100
        self.assertAlmostEqual(row["host_sync_calls"], 10e-6)
        self.assertEqual(row["sync_call_count"], 1)
        self.assertAlmostEqual(row["compute"] + row["exposed_communication"] + row["gpu_idle"], row["wall"])
        self.assertAlmostEqual(out["share_of_wall"]["compute"], 0.5)

    def test_emulated_rank_shares_like_rank_0(self):
        lengths = torch.tensor([5, 3, 8, 2, 7, 4, 6, 1, 3])
        ids = torch.arange(9 * 8).reshape(9, 8)
        batch = {"input_ids": ids, "labels": ids.clone(), "lengths": lengths}
        emulated, real = S.EmulatedRank(4), Context(4, 0, 0)
        self.assertTrue(emulated.active)
        self.assertEqual(emulated.world, 4)
        a, b = emulated.share(batch), real.share(batch)
        for key in batch:
            self.assertTrue(torch.equal(a[key], b[key]))
        self.assertEqual(emulated.sum([1.0, 2.0], "cpu"), [1.0, 2.0])
        model = torch.nn.Linear(2, 2)
        self.assertIs(emulated.wrap(model, torch.device("cpu")), model)
        with emulated.no_sync(model, False):
            pass


if __name__ == "__main__":
    unittest.main()
