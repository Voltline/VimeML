"""Freeze a small code archive and explicit, checksummed AutoDL payload list.

No SSH, uploads, credential reads, model execution or training. Existing server
files with matching SHA256 can be reused when a later upload is authorized.
"""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from vimeml.posttraining.train import build_plan
    from vimeml.posttraining.infer import TRACKS, read_snapshot
    from vimeml.training.data import file_sha, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/handoff-v4-c1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttrain-v4-c1.toml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    plan = build_plan(config, verify_files=True)
    code = sorted((ROOT / "src/vimeml").rglob("*.py"))
    code += sorted((ROOT / "scripts/posttraining").glob("*.py"))
    code += [ROOT / "scripts/posttraining/autodl_v4_c1.sh", config_path]
    code += [ROOT / name for name in ("requirements.txt", "requirements-autodl.txt", "pyproject.toml", "README.md", "LICENSE")]
    code += [ROOT / name for name in ("docs/zh-CN/posttraining-v4.md", "docs/zh-CN/data-v4-c1.md",
                                    "docs/posttraining-v4.md", "docs/data-v4-c1.md")]
    data = [ROOT / p for p in plan["required_files"]]
    data += [ROOT / config["conversion_dir"] / name for name in ("manifest.json", "acceptance.json")]
    data += [ROOT / config["token_dir"] / "manifest.json", ROOT / config["index_dir"] / "manifest.json",
             ROOT / config["tokenizer_dir"] / "tokenizer.model", ROOT / config["initialization"]["checkpoint"]]
    for track in TRACKS:
        read_snapshot(track)
        data.append(ROOT / f"outputs/evaluation-v4-20261015/inputs/{track}.json")
    args.output.mkdir(parents=True, exist_ok=True)
    code = sorted(set(code))
    data = sorted(set(data))
    archive = args.output / "v4-c1-code.tar.gz"
    def clean(info):
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ""
        info.mode = 0o755 if info.name.endswith(".sh") else 0o644
        return info
    with tarfile.open(archive, "w:gz") as bundle:
        for path in code:
            bundle.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False, filter=clean)
    payload = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": file_sha(p)} for p in data]
    result = {"format": "vimeml_v4_c1_handoff_v1", "code_archive": {"name": archive.name, "bytes": archive.stat().st_size,
              "sha256": file_sha(archive)}, "code_files": {p.relative_to(ROOT).as_posix(): file_sha(p) for p in code},
              "payload_files": payload, "payload_bytes": sum(x["bytes"] for x in payload),
              "new_conversion_binary_bytes": sum(x["bytes"] for x in payload if x["path"].startswith(config["conversion_dir"]) and x["path"].endswith(".bin")),
              "source_step": 584000, "max_steps": plan["settings"]["max_steps"], "signatures": plan["signatures"],
              "purpose": "Frozen upload inventory; reuse identical remote files. No raw corpus, annotation dictionaries, credentials or blind data.",
              "training_started": False, "remote_operations": False}
    write_json(args.output / "manifest.json", result)
    write_json(args.output / "training-plan.json", plan)
    print(json.dumps({"output": str(args.output), "archive_bytes": result["code_archive"]["bytes"],
                      "payload_files": len(payload), "payload_GiB": result["payload_bytes"] / 2**30,
                      "new_conversion_binary_MiB": result["new_conversion_binary_bytes"] / 2**20,
                      "max_steps": result["max_steps"]}, indent=2))


if __name__ == "__main__":
    main()
