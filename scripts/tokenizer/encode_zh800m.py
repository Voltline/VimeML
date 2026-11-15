"""Token data for the zh-800m model (docs/zh-CN/training-zh-800m-plan.md), 12K tokenizer, c512.

The P1 web store and the chat store are the vimeml-tiny-zh-v1 ones (zh13-web-12k,
zh13-chat-12k) and are not encoded again; they only get c512 window indexes. Two new
web-only stores complete the web data:

  encode --part r1rest   export-r1 train documents outside P1 (same pools), no tiers
                         -> token-data/zh800m-web-r1rest-12k (+ c512 index)
  encode --part r2       every export-r2 train document (no pools), no tiers
                         -> token-data/zh800m-web-r2-12k (+ c512 index)
  index                  c512 window indexes for zh13-web-12k and zh13-chat-12k

The token store format is the vimeml_sentence_tokens_v1 one of encode_zh13.py.
Formal runs need the frozen exports and hash every input against SHA256SUMS.

usage:
  python -I -X utf8 scripts/tokenizer/encode_zh800m.py encode --part r1rest
  python -I -X utf8 scripts/tokenizer/encode_zh800m.py encode --part r2
  python -I -X utf8 scripts/tokenizer/encode_zh800m.py index
"""
import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.tokenizer.encode_zh13 import encode, web_files  # noqa: E402
from vimeml.tokenizer.zh13 import check_files, export_identity  # noqa: E402

VOCAB = 12288
PARTS = {"r1rest": ("export_dir", "rest", "zh800m-web-r1rest-12k"),
         "r2": ("export_r2_dir", "all", "zh800m-web-r2-12k")}
EXISTING = ("zh13-web-12k", "zh13-chat-12k")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("encode", "index"))
    parser.add_argument("--part", choices=sorted(PARTS))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data-zh800m.toml")
    parser.add_argument("--export", type=Path, help="override the part's export directory")
    parser.add_argument("--root", type=Path, help="output base directory (default: the repo artifacts tree); "
                        "write to fast node-local storage and copy back when small-write latency dominates")
    parser.add_argument("--resume", action="store_true",
                        help="encode: continue an interrupted run in the same output directory")
    parser.add_argument("--workers", type=int, help="override the config's worker count")
    parser.add_argument("--no-verify-inputs", action="store_true", help="check names in SHA256SUMS but skip hashing")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text("utf-8"))
    if args.workers:
        config["workers"] = args.workers
    config["tokenizer_dir"] = str(ROOT / config["tokenizer_dir"])
    base = Path(args.root).resolve() if args.root else ROOT / "artifacts"
    context = config["context_length"]
    if args.command == "index":
        from vimeml.training.data import prepare_indexes
        for name in EXISTING:
            result = prepare_indexes(base / "token-data" / name, base / "training-data" / f"{name}-c{context}",
                                     context, splits=("train", "validation"))
            print(json.dumps({name: {s: result["splits"][s]["windows"] for s in result["splits"]}}), flush=True)
        return
    if args.part is None:
        parser.error("encode needs --part")
    export_key, selection, name = PARTS[args.part]
    export = args.export or ROOT / config[export_key]
    identity = export_identity(export)
    public = {k: v for k, v in identity.items() if k != "sums"}
    files = web_files(export) + sorted((export / "validation").glob("web-*.parquet")) + sorted(
        (export / "validation").glob("review-*.parquet"))
    check_files(export, files, identity, not args.no_verify_inputs)
    pools = ROOT / config["pools_dir"] / "pools.json" if selection == "rest" else None
    outputs = {"web": base / "token-data" / name, "web_index": base / "training-data" / f"{name}-c{context}"}
    result = encode(config, export, VOCAB, outputs, pools, public, resume=args.resume, selection=selection)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
