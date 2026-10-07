"""Create an isolated Mac Git checkout and install the handoff payload."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=Path.home() / "Documents/Sources/VimeML-v21-coreml",
    )
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    manifest = json.loads(
        (package / "handoff-manifest.json").read_text(encoding="utf-8")
    )
    destination = args.destination.expanduser().resolve()
    if destination.exists():
        parser.error("Destination exists; choose a new checkout directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            manifest["source_branch"],
            str(package / "repository.bundle"),
            str(destination),
        ],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=destination, text=True
    ).strip()
    if head != manifest["source_commit"]:
        raise ValueError("Git checkout does not match the recorded source commit.")
    subprocess.run(
        ["git", "remote", "set-url", "origin", manifest["remote"]],
        cwd=destination,
        check=True,
    )
    subprocess.run(
        ["git", "switch", "-c", manifest["mac_branch"]], cwd=destination, check=True
    )
    for entry in manifest["payload_files"]:
        relative = PurePosixPath(entry["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] not in {"artifacts", "outputs"}
        ):
            raise ValueError("Unexpected payload path.")
        source = package / "payload" / relative
        target = destination / relative
        if not source.is_file() or source.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing/incomplete payload file: {relative}")
        if target.exists():
            raise ValueError(f"Payload would overwrite a checkout file: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    record = destination / "outputs/deployment/mac-v21-handoff.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Ready: {destination}")
    print("Preparation: docs/mac-v21-preparation.md")
    print(
        "No dependencies installed, conversion, quantization or device operation started."
    )


if __name__ == "__main__":
    main()
