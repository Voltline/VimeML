"""Transfer V4.2 Mac models/reports with path/count/size validation, not repeated hashes."""
import argparse
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = [f"tiny-ja-v4.2-epoch20-{name}-v1" for name in ("bundle", "fp32", "int8-b32", "kv-fp32", "kv-int8-b32")]
MEASUREMENT_ONLY = [f"tiny-ja-v2.1-extend5-kv-{name}-mac-v42-measure-v1" for name in ("fp32", "int8-b32")]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def package(status, resources, pr, output):
    if output.exists():
        raise ValueError("Results ZIP already exists; use a new version.")
    directories = [ROOT / "artifacts/deployment" / name for name in [*MODELS, *MEASUREMENT_ONLY]]
    if resources:
        directories.append(Path(resources).resolve())
    directories += sorted(p for p in (ROOT / "outputs/deployment").glob("v42-*") if p.is_dir())
    directories += [ROOT / "outputs/deployment/v21-kv-mac-v42-measure-gate-v1", ROOT / "outputs/deployment/logs"]
    files = []
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"Missing result directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink in transfer: {path}")
            # Compiled measurement copies are reproducible from the packages.
            if path.is_file() and "compiled" not in path.relative_to(directory).parts:
                files.append(path)
    paths = {p.relative_to(ROOT).as_posix(): p.stat().st_size for p in files}
    if len(paths) != len(files):
        raise ValueError("Duplicate ZIP entries.")
    acceptance = json.loads(status.read_text())
    record = {"format": "vimeml_v42_mac_results_v1", "date": "2026-10-20",
        "repositories": {"VimeML": {"base_commit": git("merge-base", "HEAD", "origin/main"),
            "head_commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "pr": pr}},
        "environment": json.loads((ROOT / "artifacts/deployment" / MODELS[1] / "manifest.json").read_text())["environment"],
        "acceptance": acceptance, "files": paths,
        "measurement_only": MEASUREMENT_ONLY,
        "restore_policy": "Restore original relative paths to ignored artifact/output directories. Save conflicting names as a new version; merge source via Git PRs.",
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
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--pr", default="not opened")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.status, args.resources, args.pr, args.output)


if __name__ == "__main__":
    main()
