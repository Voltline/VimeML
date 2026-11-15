"""Package the pushed main branch with the vimeml-tiny-zh-v1 release bundle, its INT8 stateless and chunk-KV
packages, the reading dictionary and the Vime generation fixtures, for the batched-KV Core ML work on a Mac.

Brief: docs/reference/mac-zh-v1-kvbatch.md. No benchmark items are included (the generation fixtures come
from the public ime-pinyin-zh-v1 development split and a few hand-written keys).
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RELEASE = "artifacts/releases/vimeml-tiny-zh-v1-hf-v1"
TREES = (f"{RELEASE}/inference", f"{RELEASE}/coreml/ios18-int8-block32", f"{RELEASE}/coreml/ios18-int8-block32-kvchunk")
FILES = (f"{RELEASE}/RELEASE.json", f"{RELEASE}/SHA256SUMS.txt", "outputs/zh-readings/dict/vime_pinyin.dict.yaml")
FIXTURES = "outputs/vime-fixtures"
CHECKPOINT_SHA256 = "44c423a1e7cb0e32161afc94d6167cabb439a3d25fd9de6d1fbb150a927542d4"
TOKENIZER_SHA256 = "68bd37c45bcb018b32b2eef5b241561ab5b6dc1b77c1c3aadcc1b41c4bf47e5a"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", nargs="+", default=["zh-generation-v1-lm-r2"],
                        help="generation fixture directories under outputs/vime-fixtures")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output exists; choose a new handoff version.")
    if git("status", "--porcelain", "--untracked-files=no"):
        parser.error("Commit tracked source changes before creating the portable Git bundle.")
    if git("branch", "--show-current") != "main" or git("rev-parse", "HEAD") != git("rev-parse", "origin/main"):
        parser.error("Package the pushed main branch.")
    from vimeml.deployment import zh_v1
    from vimeml.training.data import file_sha

    bundle = zh_v1.verify_bundle(ROOT / RELEASE / "inference")
    if bundle["checkpoint_sha256"] != CHECKPOINT_SHA256 or bundle["files"]["tokenizer.model"]["sha256"] != TOKENIZER_SHA256:
        raise ValueError("Release bundle is not the confirmed vimeml-tiny-zh-v1.")
    zh_v1.verify_package(ROOT / TREES[1])
    zh_v1.verify_package(ROOT / TREES[2], zh_v1.CHUNK_FORMAT)
    fixtures = []
    for name in args.fixtures:
        meta = json.loads((ROOT / FIXTURES / name / "meta.json").read_text(encoding="utf-8"))
        if meta["tokenizer_sha256"] != TOKENIZER_SHA256 or "test never read" not in meta["source"]:
            raise ValueError(f"{name}: unexpected fixture identity or source")
        fixtures.append(f"{FIXTURES}/{name}")

    maintenance = ROOT / "outputs/maintenance"
    maintenance.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mac-zh-kvbatch-package-", dir=maintenance) as temporary:
        stage = Path(temporary)
        payload = stage / "payload"
        for tree in (*TREES, *fixtures):
            shutil.copytree(ROOT / tree, payload / tree)
        for relative in FILES:
            (payload / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, payload / relative)
        for path in payload.rglob("*"):
            if path.is_file() and (path.is_symlink() or path.name.startswith(".env") or path.suffix == ".pyc"):
                raise ValueError(f"Unexpected input: {path}")
        subprocess.run(["git", "bundle", "create", str(stage / "repository.bundle"), "main"], cwd=ROOT, check=True)
        subprocess.run(["git", "bundle", "verify", str(stage / "repository.bundle")], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        manifest = {
            "format": "vimeml_windows_to_mac_zh_kvbatch",
            "source_commit": git("rev-parse", "HEAD"), "source_branch": "main", "mac_branch": "main",
            "checkout_name": "VimeML-zh-kvbatch", "handoff_record": "outputs/deployment/mac-zh-kvbatch-handoff.json",
            "remote": git("remote", "get-url", "origin"),
            "model": {"release_name": "vimeml-tiny-zh-v1", "checkpoint_sha256": CHECKPOINT_SHA256,
                      "tokenizer_sha256": TOKENIZER_SHA256, "bundle": f"{RELEASE}/inference",
                      "stateless_int8": TREES[1], "chunk_int8": TREES[2]},
            "fixtures": fixtures,
            "payload_files": [
                {"path": p.relative_to(payload).as_posix(), "bytes": p.stat().st_size, "sha256": file_sha(p)}
                for p in sorted(payload.rglob("*")) if p.is_file()],
            "verification": "Bundle and package manifests; checkpoint and tokenizer SHA256; fixture identity; "
                            "Git bundle verify; per-file SHA256; ZIP CRC.",
        }
        write_json(stage / "handoff-manifest.json", manifest)
        shutil.copy2(ROOT / "scripts/deployment/setup_mac.py", stage / "setup_mac.py")
        shutil.copy2(ROOT / "docs/reference/mac-zh-v1-kvbatch.md", stage / "README.md")
        output.parent.mkdir(parents=True, exist_ok=True)
        expected = {}
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    name = path.relative_to(stage).as_posix()
                    archive.write(path, name)
                    expected[name] = path.stat().st_size
        with zipfile.ZipFile(output) as archive:
            if {p.filename: p.file_size for p in archive.infolist()} != expected or archive.testzip() is not None:
                raise ValueError("ZIP directory or CRC differs from packaged files.")
        summary = {"zip": output.name, "bytes": output.stat().st_size, "sha256": file_sha(output),
                   "file_count": len(expected), "source_commit": manifest["source_commit"]}
        write_json(output.with_suffix(".json"), summary)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
