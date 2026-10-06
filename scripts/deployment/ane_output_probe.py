"""Manual reduced-output ANE experiment; not the deployment interface.

Usage: python scripts/deployment/ane_output_probe.py NEW_OUTPUT.mlpackage
Conversion, timing and accuracy checks start only when this command is run.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New scratch .mlpackage path")
    args = parser.parse_args()
    out = args.output
    if out.exists():
        parser.error("Output exists; choose a new experiment path.")
    if out.suffix != ".mlpackage":
        parser.error("Output must end with .mlpackage.")
    if sys.platform != "darwin":
        parser.error("Run this experiment manually on Mac.")
    import numpy as np, torch, coremltools as ct
    from torch import nn
    from vimeml.deployment.bundle import BundleLM
    from vimeml.deployment.graph import ConversionGraph
    lm = BundleLM(ROOT/"artifacts/deployment/tiny-ja-v1-inference-v1")
    K = 16
    class Head(nn.Module):
        def __init__(s, model): super().__init__(); s.g = ConversionGraph(model)
        def forward(s, input_ids, target_ids):
            logp = torch.log_softmax(s.g(input_ids), dim=-1)
            target = torch.gather(logp, 2, target_ids.long().unsqueeze(-1)).squeeze(-1)
            top, idx = torch.topk(logp, K, dim=-1)
            return target, top, idx.int()
    h = Head(lm.model).eval()
    ex = (torch.zeros(1,16,dtype=torch.int32), torch.zeros(1,16,dtype=torch.int32))
    tr = torch.jit.trace(h, ex)
    shapes = ct.EnumeratedShapes(shapes=[(1,L) for L in (16,32,64,128)], default=(1,16))
    m = ct.convert(tr, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16,
        inputs=[ct.TensorType(name="input_ids", shape=shapes, dtype=np.int32), ct.TensorType(name="target_ids", shape=shapes, dtype=np.int32)],
        outputs=[ct.TensorType(name="target_logprobs", dtype=np.float32), ct.TensorType(name="top_logprobs", dtype=np.float32), ct.TensorType(name="top_ids", dtype=np.int32)])
    m.save(str(out))
    for units in ("CPU_ONLY","CPU_AND_NE"):
        mm = ct.models.MLModel(str(out), compute_units=getattr(ct.ComputeUnit, units)); row=[units]
        for L in (16,64,128):
            x={"input_ids":np.full((1,L),5,np.int32),"target_ids":np.full((1,L),6,np.int32)}; mm.predict(x); ts=[]
            for _ in range(20): t=time.perf_counter(); mm.predict(x); ts.append((time.perf_counter()-t)*1000)
            row.append(f"T{L}={np.median(ts):.2f}ms")
        print(" ".join(row))
    # accuracy vs FP32 on AJIMEE text
    import json
    items=json.load(open(ROOT/"artifacts/benchmarks/ajimee-jwtd-v2-v1/evaluation_items.json"))
    mm = ct.models.MLModel(str(out), compute_units=ct.ComputeUnit.CPU_AND_NE)
    diffs=[]; topagree=[]
    for it in items[:100]:
        ids=[2]+lm.processor.encode((it.get("context_text") or "")+it["expected_output"][0])[:127]
        n=len(ids)-1; L=next(b for b in (16,32,64,128) if b>=n)
        inp=np.zeros((1,L),np.int32); inp[0,:n]=ids[:-1]; tgt=np.zeros((1,L),np.int32); tgt[0,:n]=ids[1:]
        o=mm.predict({"input_ids":inp,"target_ids":tgt})
        with torch.no_grad(): ref=lm.model(torch.tensor([ids[:-1]])).float().log_softmax(-1)[0]
        r=ref.gather(1,torch.tensor(ids[1:])[:,None]).squeeze(1).numpy()
        diffs.append(abs(o["target_logprobs"][0,:n]-r).max())
        topagree.append(np.mean(o["top_ids"][0,:n,0]==ref.argmax(-1).numpy()))
    print(f"ANE target logprob max|diff| mean={np.mean(diffs):.4f} max={np.max(diffs):.4f}; top1 agree={np.mean(topagree):.4f}")


if __name__ == "__main__":
    main()
