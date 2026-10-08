# V2.0 training environment — 2026-10-07

The completed run uses RTX 4090 D 24 GB, 18 CPU cores, 60 GB RAM, Python 3.12.3, PyTorch 2.7.0+cu128, NumPy 2.2.6, TensorBoard 2.19.0, SentencePiece 0.2.1, and W&B 0.30.0. Pins reside in [requirements-autodl.txt](../../../requirements-autodl.txt).

Large assets reside under `/root/autodl-tmp/vimeml`. The 1,017,407,143-byte data archive contains token store/index/tokenizer and two original candidate sets. Transport checks use size and archive integrity; credentials remain in a separate root-readable environment file.

```bash
cd /root/autodl-tmp/vimeml
screen -dmS vimeml-v2 bash -c 'bash scripts/training/autodl_v2.sh > /root/autodl-tmp/vimeml-v2-training.log 2>&1'
```

The historical run begins at 10:29 Beijing time and completes four epochs/393,704 updates, BF16 batch 256. Fixed-subset validation runs every 5,000 updates; complete BPC/IME evaluations run each epoch. Subset selection includes every window of selected sentences so the character denominator remains coherent.

Early formal updates measure approximately 116k–124k effective tokens/s; final training throughput is 121,228 tokens/s. Peak early allocated memory is approximately 6,278 MiB. Tracking finishes normally. Original progress/summary files retain run state; [evaluation](evaluation.md) records final quality.
