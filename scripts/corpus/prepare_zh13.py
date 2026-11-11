"""Prepare one vimeml-tiny-zh-v1 arm's frozen mixture (docs/zh-CN/training-zh-v1-plan.md).

Writes artifacts/training-data/zh13-<arm>-c128/{request.json, epoch-*.npy,
probe-windows.npy, manifest.json}, LF newlines. A second run rebuilds and must
match byte for byte; a configured dataset_sha256 must equal the manifest SHA.

usage:
  python -X utf8 scripts/corpus/prepare_zh13.py --arm 12k-fresh-s1
  python -X utf8 scripts/corpus/prepare_zh13.py --arm 12k-ep20 --data-root SMOKE_DIR   # smoke, no config check
"""
import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.data import file_sha  # noqa: E402
from vimeml.training.zh13_data import ARMS, mixture_directory, prepare_maps, read, request_for, write_lf  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--data-root", help="smoke: absolute directory holding token-data/ and training-data/")
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve().as_posix() if args.data_root else "artifacts"
    target = ROOT / mixture_directory(args.arm, data_root)
    request = request_for(args.arm, ROOT, data_root)
    if (target / "request.json").exists():
        if read(target / "request.json") != request:
            raise ValueError(f"{target} holds a different request")
    else:
        target.mkdir(parents=True, exist_ok=True)
        write_lf(target / "request.json", request)
    result = prepare_maps(target, ROOT)
    report = {"arm": args.arm, "directory": mixture_directory(args.arm, data_root),
              "dataset_sha256": file_sha(target / "manifest.json"), "epochs": len(result["epochs"]),
              "windows_epoch1": result["epochs"][0]["windows"],
              "web_characters_total": result["web_characters_total"],
              "chat_passes": round(result["chat_passes"], 3), "probe_windows": result["probe"]["windows"],
              "effective_tokens": sum(r["effective_tokens"] for r in result["epochs"])}
    if not args.data_root:
        config = tomllib.loads((ROOT / f"configs/train-zh13-{args.arm}.toml").read_text("utf-8"))
        if config["data_mixture"]["directory"] != report["directory"]:
            raise ValueError("Config points at another mixture directory")
        if config["dataset_sha256"] and config["dataset_sha256"] != report["dataset_sha256"]:
            raise ValueError(f"{args.arm} mixture SHA {report['dataset_sha256']} differs from the pinned "
                             f"{config['dataset_sha256']}; do not train, report this mismatch")
        report["matches_pinned_sha"] = bool(config["dataset_sha256"])
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
