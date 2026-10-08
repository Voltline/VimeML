"""Transfer new V2 models/reports with path/count/size validation, not repeated hashes."""
import argparse
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def package(client, ml_pr, client_pr, status, output):
    if output.exists():
        raise ValueError("Results ZIP already exists; use a new version.")
    models = [ROOT / "artifacts/deployment" / f"tiny-ja-v2.1-extend5-{name}-v1"
              for name in ("bundle", "fp32", "int8-b32", "client-resources")]
    reports = sorted((ROOT / "outputs/deployment").glob("v21-*"))
    files = []
    for directory in [*models, *reports]:
        if not directory.is_dir():
            raise ValueError(f"Missing result directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink in transfer: {path}")
            if path.is_file():
                files.append(path)
    paths = {p.relative_to(ROOT).as_posix(): p.stat().st_size for p in files}
    if len(paths) != len(files):
        raise ValueError("Duplicate ZIP entries.")
    record = {"format": "vimeml_v21_mac_results_v1", "date": "2026-10-08",
        "repositories": {"VimeML": {"base_commit": git(ROOT, "merge-base", "HEAD", "origin/main"),
            "head_commit": git(ROOT, "rev-parse", "HEAD"), "branch": git(ROOT, "branch", "--show-current"), "pr": ml_pr},
            "Vime": {"base_commit": git(client, "merge-base", "HEAD", "origin/main"),
            "head_commit": git(client, "rev-parse", "HEAD"), "branch": git(client, "branch", "--show-current"), "pr": client_pr}},
        "environment": json.loads((models[0] / "manifest.json").read_text())["environment"],
        "acceptance": json.loads(status.read_text()), "files": paths,
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
        if actual != {"handoff.json": len(content), **paths}:
            raise ValueError("ZIP inventory verification failed.")
    print(json.dumps({"zip": str(output.resolve()), "zip_bytes": output.stat().st_size,
                      "files": len(paths) + 1, "logical_bytes": sum(paths.values()) + len(content)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--ml-pr", required=True)
    parser.add_argument("--client-pr", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.client, args.ml_pr, args.client_pr, args.status, args.output)


if __name__ == "__main__":
    main()
