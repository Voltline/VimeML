"""Transfer vimeml-tiny-zh-v1 Mac packages/records with path/count/size validation, not repeated hashes."""
import argparse
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = [f"tiny-zh-v1-{name}" for name in ("bundle-v1", "fp32-v1", "int8-b32-v1", "kvchunk-fp32-v1",
                                            "kvchunk-int8-b32-v1", "ios-resources-v1")]
SURVEY = [f"tiny-zh-v1-survey-{name}" for name in ("int8-b32-embfp16-v2", "int4-b32-v1", "palette6-gc32-v1",
                                                   "palette4-gc32-v1")]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def package(status, output, include_survey=True, client=None):
    if output.exists():
        raise ValueError("Results ZIP already exists; use a new version.")
    directories = [ROOT / "artifacts/deployment" / name for name in [*MODELS, *(SURVEY if include_survey else [])]]
    directories += sorted(p for p in (ROOT / "outputs/deployment").glob("zh-v1-*") if p.is_dir())
    files = []
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"Missing result directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink in transfer: {path}")
            parts = path.relative_to(directory).parts
            if path.is_file() and "compiled" not in parts and not any(p.endswith(".xcresult") for p in parts):
                files.append(path)
    paths = {p.relative_to(ROOT).as_posix(): p.stat().st_size for p in files}
    if len(paths) != len(files):
        raise ValueError("Duplicate ZIP entries.")
    repositories = {"VimeML": {"head_commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                               "origin_main": git("rev-parse", "origin/main")}}
    if client:
        repositories["Vime"] = {"head_commit": git("-C", str(client), "rev-parse", "HEAD"),
                                "branch": git("-C", str(client), "branch", "--show-current")}
    record = {"format": "vimeml_zh_v1_mac_results_v1", "date": "2026-11-11", "repositories": repositories,
        "environment": json.loads((ROOT / "artifacts/deployment" / MODELS[1] / "manifest.json").read_text())["bundle"]["environment"],
        "acceptance": json.loads(status.read_text()), "files": paths,
        "survey_only": SURVEY if include_survey else [],
        "restore_policy": "Restore original relative paths to ignored artifact/output directories. Save conflicting names as a new version; source changes travel through Git on main.",
        "verification": "Model identities reused from manifests; ZIP path/count/logical-size check; no repeated full-tree SHA256."}
    content = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("handoff.json", content)
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        actual = {item.filename: item.file_size for item in archive.infolist()}
        if actual != {"handoff.json": len(content), **paths} or archive.testzip() is not None:
            raise ValueError("ZIP inventory verification failed.")
    print(json.dumps({"zip": str(output.resolve()), "zip_bytes": output.stat().st_size,
                      "files": len(paths) + 1, "logical_bytes": sum(paths.values()) + len(content)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True, help="JSON acceptance summary")
    parser.add_argument("--no-survey", action="store_true", help="Leave survey-only packages out")
    parser.add_argument("--client", type=Path, help="Vime checkout with the opt-in zh-v1 test host")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.status, args.output, not args.no_survey, args.client)


if __name__ == "__main__":
    main()
