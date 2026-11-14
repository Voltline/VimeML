"""Build the pinyin-conditioned stores for the zh three-task model (stage 1).

Formal runs read the frozen export and write configs/data-zh-pinyin.toml's output_dir.
--smoke cuts every file to --max-row-groups and must write elsewhere (--output).

usage:
  python -I -X utf8 scripts/posttraining/build_zh_pinyin.py
  python -I -X utf8 scripts/posttraining/build_zh_pinyin.py --smoke --max-row-groups 1 --output OUT
"""
import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.posttraining.zh_pinyin_data import build  # noqa: E402
from vimeml.tokenizer.zh13 import check_files, export_identity  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data-zh-pinyin.toml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-row-groups", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--no-verify-inputs", action="store_true", help="check names in SHA256SUMS but skip hashing")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text("utf-8"))
    formal = ROOT / config["output_dir"]
    if args.smoke:
        if args.output is None or args.output.resolve() == formal.resolve():
            parser.error("--smoke needs an --output outside output_dir")
    elif args.max_row_groups or args.output:
        parser.error("--max-row-groups/--output need --smoke")
    export = ROOT / config["export_dir"]
    identity = export_identity(export)
    files = sorted((export / "train").glob("*.parquet")) + sorted((export / "validation").glob("*.parquet"))
    if not args.smoke:
        check_files(export, files, identity, not args.no_verify_inputs)
    public = {k: v for k, v in identity.items() if k != "sums"}
    config["tokenizer"] = str(ROOT / config["tokenizer"])
    config["pools_path"] = str(ROOT / config["pools_dir"] / "pools.json")
    if args.workers:
        config["workers"] = args.workers
    manifest = build(config, export, args.output or formal, public, args.max_row_groups)
    print(json.dumps({split: {s: v["examples"] for s, v in stores.items()}
                      for split, stores in manifest["stores"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
