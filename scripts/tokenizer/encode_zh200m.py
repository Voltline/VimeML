"""Encode the zh-200m token stores for the 16K/24K/32K vocabularies, with c512 window indexes.

encode --vocab 16384|24576|32768 writes artifacts/token-data/zh200m-{web,chat}-{16k,24k,32k}
and their c512 window indexes (configs/data-zh200m.toml). index --vocab N --context 128
builds window indexes at another context length over an existing store, for the
24k-c128 arm. The pools are the frozen vimeml-tiny-zh-v1 ones and are not re-measured.

Formal runs need the frozen export and hash every input against SHA256SUMS.

usage:
  python -I -X utf8 scripts/tokenizer/encode_zh200m.py encode --vocab 24576
  python -I -X utf8 scripts/tokenizer/encode_zh200m.py index --vocab 24576 --context 128
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

TAGS = {16384: "16k", 24576: "24k", 32768: "32k"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data-zh200m.toml")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--no-verify-inputs", action="store_true", help="check names in SHA256SUMS but skip hashing")
    parser.add_argument("command", choices=("encode", "index"))
    parser.add_argument("--vocab", type=int, choices=sorted(TAGS))
    parser.add_argument("--context", type=int, help="window context for the index command")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text("utf-8"))
    config["pools_dir"] = str(ROOT / config["pools_dir"])
    config["tokenizer_dir"] = str(ROOT / config["tokenizer_dir"])
    export = args.export or ROOT / config["export_dir"]
    identity = export_identity(export)
    public = {k: v for k, v in identity.items() if k != "sums"}
    if args.command == "encode":
        if args.vocab is None:
            parser.error("encode needs --vocab")
        files = (web_files(export) + sorted((export / "validation").glob("*.parquet"))
                 + [export / "train/chat-lccc-train.parquet"])
        check_files(export, files, identity, not args.no_verify_inputs)
        tag = TAGS[args.vocab]
        base = ROOT / "artifacts"
        outputs = {"web": base / f"token-data/zh200m-web-{tag}", "chat": base / f"token-data/zh200m-chat-{tag}",
                   "web_index": base / f"training-data/zh200m-web-{tag}-c{config['context_length']}",
                   "chat_index": base / f"training-data/zh200m-chat-{tag}-c{config['context_length']}"}
        result = encode(config, export, args.vocab, outputs, Path(config["pools_dir"]) / "pools.json", public)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if args.vocab is None or args.context is None:
        parser.error("index needs --vocab and --context")
    from vimeml.training.data import prepare_indexes
    tag = TAGS[args.vocab]
    for name in ("web", "chat"):
        token_dir = ROOT / "artifacts" / f"token-data/zh200m-{name}-{tag}"
        index_dir = ROOT / "artifacts" / f"training-data/zh200m-{name}-{tag}-c{args.context}"
        result = prepare_indexes(token_dir, index_dir, args.context, splits=("train", "validation"))
        print(json.dumps({name: {s: result["splits"][s]["windows"] for s in result["splits"]}}), flush=True)


if __name__ == "__main__":
    main()
