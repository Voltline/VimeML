"""Download pinned WRIME/JMultiWOZ sources for local IME benchmark preparation."""
import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "VimeML-benchmark-preparation"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/ime-standard-ja-v1")
    args = parser.parse_args()
    manifest_path = args.output / "download-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["sources"].values():
            for item in entry["files"]:
                if (args.output / item["local_path"].replace("\\", "/")).stat().st_size != item["bytes"]:
                    raise ValueError("Source size changed; preserve the snapshot and inspect it.")
        print("Reusing pinned downloads.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, repo in (("wrime", "ids-cv/wrime"), ("jmultiwoz", "nu-dialogue/jmultiwoz")):
        revision = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/master"))["sha"]
        if name == "wrime":
            paths = ["wrime-ver2.tsv", "README.md", "README.en.md", "LICENSE"]
        else:
            inventory = json.loads(fetch(f"https://api.github.com/repos/{repo}/contents/dataset?ref={revision}"))
            paths = [item["path"] for item in inventory if item["name"].lower().endswith(".zip")]
            if len(paths) != 1:
                raise ValueError("Expected one JMultiWOZ data archive.")
            paths += ["dataset/README.md", "README.md", "LICENSE"]
        files = []
        for path in paths:
            local = args.output / name / path
            local.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
            content = fetch(url)
            local.write_bytes(content)
            files.append({"source_path": path, "local_path": local.relative_to(args.output).as_posix(),
                          "bytes": len(content), "url": url})
            print(f"Downloaded {name}/{path}: {len(content):,} bytes", flush=True)
            if path.lower().endswith(".zip"):
                with zipfile.ZipFile(local) as package:
                    for leaf in ("dialogues.json", "split_list.json"):
                        members = [p for p in package.namelist() if p.rsplit("/", 1)[-1] == leaf]
                        if len(members) != 1:
                            raise ValueError(f"Expected one {leaf} in the archive.")
                        (args.output / name / leaf).write_bytes(package.read(members[0]))
        sources[name] = {"repository": f"https://github.com/{repo}", "revision": revision,
                         "license": "CC-BY-NC-ND-4.0" if name == "wrime" else "CC-BY-SA-4.0",
                         "files": files}
    manifest = {"status": "complete", "purpose": "Evaluation-only; never a training input",
                "sources": sources, "verification": "Pinned revision, file size, named JSON members; no full-file SHA256"}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
