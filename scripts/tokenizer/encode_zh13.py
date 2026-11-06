"""Measure the nested Vime 1.3 pools, then encode token stores per vocabulary.

measure writes pools.json (thresholds q, q//5, q//20 and the memorization probe);
it does not depend on the tokenizer. encode --vocab 8192|12288 writes
artifacts/token-data/zh13-{web,chat}-{8k,12k} and their c128 window indexes.

Formal runs need the frozen export and hash every input against SHA256SUMS.
--unfrozen is for local smoke tests: it must write outside the formal
directories and may cut each file to --max-row-groups.

usage:
  python -I -X utf8 scripts/tokenizer/encode_zh13.py measure
  python -I -X utf8 scripts/tokenizer/encode_zh13.py encode --vocab 12288
  python -I -X utf8 scripts/tokenizer/encode_zh13.py --unfrozen --root OUT \\
      --tokenizers DIR --target 20000000 --max-row-groups 2 measure|encode --vocab N
"""
import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.tokenizer.encode_zh13 import encode, measure, web_files  # noqa: E402
from vimeml.tokenizer.zh13 import check_files, export_identity  # noqa: E402

TAGS = {8192: "8k", 12288: "12k"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data-zh13.toml")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--unfrozen", action="store_true", help="smoke test on an unfrozen export")
    parser.add_argument("--root", type=Path, help="smoke output root (required with --unfrozen)")
    parser.add_argument("--tokenizers", type=Path, help="tokenizer directory (smoke)")
    parser.add_argument("--target", type=int, help="P1 target characters (smoke)")
    parser.add_argument("--memo-documents", type=int, help="memorization probe size (smoke)")
    parser.add_argument("--max-row-groups", type=int, default=0, help="per-file row-group limit (smoke)")
    parser.add_argument("--no-verify-inputs", action="store_true", help="check names in SHA256SUMS but skip hashing")
    parser.add_argument("command", choices=("measure", "encode"))
    parser.add_argument("--vocab", type=int, choices=sorted(TAGS))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text("utf-8"))
    smoke = (args.tokenizers, args.target, args.memo_documents, args.root, args.max_row_groups or None)
    if args.unfrozen:
        if args.root is None:
            parser.error("--unfrozen needs --root")
        config["pools_dir"] = str(args.root / "pools")
        config["tokenizer_dir"] = str(args.tokenizers or ROOT / config["tokenizer_dir"])
        config["p1_target_characters"] = args.target or config["p1_target_characters"]
        config["memo_documents"] = args.memo_documents or config["memo_documents"]
    elif any(x is not None for x in smoke):
        parser.error("--root/--tokenizers/--target/--memo-documents/--max-row-groups need --unfrozen")
    else:
        config["pools_dir"] = str(ROOT / config["pools_dir"])
        config["tokenizer_dir"] = str(ROOT / config["tokenizer_dir"])
    export = args.export or ROOT / config["export_dir"]
    identity = export_identity(export, frozen=not args.unfrozen)
    public = {k: v for k, v in identity.items() if k != "sums"}
    pools = Path(config["pools_dir"])
    if args.command == "measure":
        check_files(export, web_files(export), identity, not args.no_verify_inputs)
        result = measure(config, export, pools, public, args.max_row_groups)
        print(json.dumps({k: result[k] for k in ("thresholds", "memo_threshold", "pools", "tiers", "all_train", "seconds")},
                         ensure_ascii=False), flush=True)
        return
    if args.vocab is None:
        parser.error("encode needs --vocab")
    files = (web_files(export) + sorted((export / "validation").glob("*.parquet"))
             + [export / "train/chat-lccc-train.parquet"])
    check_files(export, files, identity, not args.no_verify_inputs)
    tag = TAGS[args.vocab]
    base = args.root if args.unfrozen else ROOT / "artifacts"
    outputs = {"web": base / f"token-data/zh13-web-{tag}", "chat": base / f"token-data/zh13-chat-{tag}",
               "web_index": base / f"training-data/zh13-web-{tag}-c{config['context_length']}",
               "chat_index": base / f"training-data/zh13-chat-{tag}-c{config['context_length']}"}
    result = encode(config, export, args.vocab, outputs, pools / "pools.json", public, args.max_row_groups)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
