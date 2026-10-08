"""Freeze virtual epochs: all web windows once, chat repeated to cropped-token quota."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from vimeml.training.data import SentenceWindowDataset, file_sha, write_json
from vimeml.training.data_v2 import PrefixCropWindowDataset
from vimeml.tokenizer.encode_parallel import output_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/training-data/corpus-v3-chat05-c128")
    parser.add_argument("--chat-fraction", type=float, default=.05)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--noise-filter", type=Path)
    args = parser.parse_args()
    if not 0 < args.chat_fraction < .5 or args.epochs < 1:
        parser.error("Positive epochs and chat fraction in (0, .5) required")
    token_dir = ROOT / "artifacts/token-data/corpus-v3-half/mixed"
    index_dir = ROOT / "artifacts/token-data/corpus-v3-half/indexes/mixed"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    request = {"token_manifest_sha256": file_sha(token_dir / "manifest.json"),
               "window_manifest_sha256": file_sha(index_dir / "manifest.json"),
               "index_dir": index_dir.relative_to(ROOT).as_posix(), "epochs": args.epochs,
               "chat_fraction": args.chat_fraction, "seed": 42, "crop_probability": .30, "crop_minimum": 8}
    noise = None
    if args.noise_filter:
        noise = json.loads((args.noise_filter / "manifest.json").read_text(encoding="utf-8"))
        if noise["status"] != "complete" or noise["token_manifest_sha256"] != request["token_manifest_sha256"]:
            raise ValueError("Noise filter belongs to different token data")
        request["noise_filter_manifest_sha256"] = file_sha(args.noise_filter / "manifest.json")
    with output_lock(output):
        if (output / "manifest.json").exists():
            cached = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            cached["request"]["index_dir"] = cached["request"]["index_dir"].replace("\\", "/")
            if cached["request"] != request or cached["status"] != "complete":
                raise ValueError("Changed mixture request; use a new output directory")
            if any((output / r["file"]).stat().st_size != r["bytes"] for r in cached["epochs"]):
                raise ValueError("Incomplete cached mixture")
            print("Mixture already complete; reused", flush=True)
            return
        dataset = PrefixCropWindowDataset(token_dir, index_dir, seed=42, probability=.30)
        if len(dataset) >= 2 ** 32:
            raise ValueError("Mixture mapping exceeds uint32")
        ids = np.arange(len(dataset), dtype=np.int64)
        position = np.searchsorted(dataset.end, ids, side="right")
        previous = np.zeros(len(ids), dtype=np.int64)
        present = position > 0
        previous[present] = dataset.end[position[present] - 1] - dataset.sentences[position[present] - 1] - 1
        sentences = ids - previous
        candidates = np.flatnonzero(position < len(dataset.end))
        long = candidates[ids[candidates] >= dataset.first[position[candidates]]]
        sentences[long] = dataset.sentences[position[long]]
        tm = json.loads((token_dir / "manifest.json").read_text(encoding="utf-8"))
        chat_sources = [value for name, value in tm["source_ids"].items() if name in ("real-persona-chat", "mrmp-chat")]
        sources = np.memmap(token_dir / "train.sources.bin", dtype="u1", mode="r")
        is_chat = np.isin(sources[sentences], chat_sources)
        sources._mmap.close()
        excluded = np.zeros(len(ids), dtype=np.bool_)
        if noise:
            path = args.noise_filter / noise["file"]
            if path.stat().st_size != noise["bytes"]:
                raise ValueError("Noise filter file size mismatch")
            with_filter = np.load(path, mmap_mode="r", allow_pickle=False)
            if with_filter.shape != (tm["splits"]["train"]["sentences"],) or with_filter.dtype != np.bool_:
                raise ValueError("Invalid noise filter shape/dtype")
            excluded = with_filter[sentences].copy()
            with_filter._mmap.close()
        chat, web = ids[is_chat], ids[~is_chat & ~excluded]
        if not len(chat) or not len(web):
            raise ValueError("Both chat and web windows required")
        del sentences, previous, candidates, long, present, position
        raw = np.empty(len(ids), dtype=np.int64)
        for start in range(0, len(ids), 262144):
            raw[start:start + 262144] = SentenceWindowDataset.window_lengths(dataset, ids[start:start + 262144])
        records = []
        try:
            for epoch in range(args.epochs):
                dataset.set_epoch(epoch)
                lengths = np.empty(len(ids), dtype=np.int64)
                for start in range(0, len(ids), 262144):
                    lengths[start:start + 262144] = dataset.window_lengths(ids[start:start + 262144])
                web_tokens = int(lengths[web].sum())
                target = round(web_tokens * args.chat_fraction / (1 - args.chat_fraction))
                rng = np.random.default_rng(42 + epoch)
                chosen, used = [], 0
                while used < target:
                    order = rng.permutation(chat)
                    cumulative = np.cumsum(lengths[order], dtype=np.int64)
                    take = int(np.searchsorted(cumulative, target - used, side="right"))
                    if take:
                        chosen.append(order[:take])
                        used += int(cumulative[take - 1])
                    if take < len(order):
                        break
                if not chosen:
                    raise ValueError("Chat quota is smaller than a usable window")
                chosen = np.concatenate(chosen)
                mapping = np.concatenate((web, chosen)).astype("<u4")
                rng.shuffle(mapping)
                path = output / f"epoch-{epoch}.npy"
                temporary = path.with_suffix(".tmp.npy")
                np.save(temporary, mapping, allow_pickle=False)
                temporary.replace(path)
                record = {"epoch": epoch, "file": path.name, "bytes": path.stat().st_size,
                          "windows": len(mapping), "web_windows": len(web), "chat_windows": len(chosen),
                          "web_tokens": web_tokens, "chat_tokens": used, "effective_tokens": web_tokens + used,
                          "uncropped_tokens": int(raw[web].sum()) + int(raw[chosen].sum()),
                          "chat_fraction": used / (web_tokens + used),
                          "chat_effective_passes": used / int(lengths[chat].sum()),
                          "chat_independent_tokens": int(lengths[chat].sum())}
                records.append(record)
                print(json.dumps(record, indent=2), flush=True)
                write_json(output / "progress.json", {"status": "building", "completed_epochs": epoch + 1})
            write_json(output / "manifest.json", {"status": "complete", "format": "vimeml_token_mixture_v1",
                "request": request, "epochs": records, "policy": "Web windows exactly once per epoch; training chat only, shuffled repeated cycles; quota after deterministic prefix crop; whole-window rounding; no token copies.",
                "verification": "Small manifest identity, mapping sizes/shapes; no full corpus or mapping SHA256"})
            write_json(output / "progress.json", {"status": "complete"})
        finally:
            dataset.close()


if __name__ == "__main__":
    main()
