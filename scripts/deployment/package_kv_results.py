"""Transfer successful KV models and all KV reports; reuse model identities."""
import argparse
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fp32", "int8", "resources", "client", "status", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--ml-pr", required=True); p.add_argument("--client-pr", required=True)
    a = p.parse_args()
    if a.output.exists(): p.error("Output exists; use a new version.")
    files = []
    for directory in [a.fp32,a.int8,a.resources,*sorted((ROOT/"outputs/deployment").glob("v21-kv-*"))]:
        if not directory.is_dir(): raise ValueError(f"Missing directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink(): raise ValueError(f"Unexpected symlink: {path}")
            if path.is_file(): files.append(path.resolve())
    inventory = {p.relative_to(ROOT).as_posix():p.stat().st_size for p in files}
    if len(files) != len(inventory): raise ValueError("Duplicate ZIP paths.")
    repositories = {}
    for name,repo,pr in (("VimeML",ROOT,a.ml_pr),("Vime",a.client,a.client_pr)):
        if git(repo,"status","--porcelain"): raise ValueError(f"Dirty source repository: {repo}")
        repositories[name] = {"base_commit":git(repo,"merge-base","HEAD","origin/main"),
            "head_commit":git(repo,"rev-parse","HEAD"),"branch":git(repo,"branch","--show-current"),"pr":pr}
    record = {"format":"vimeml_v21_kv_mac_results_v1","date":"2026-10-09",
        "repositories":repositories,"acceptance":json.loads(a.status.read_text()),
        "environment":json.loads((a.int8/"manifest.json").read_text())["environment"],"files":inventory,
        "baseline_payload":"Client main d96d94e tracks the original V2 model/tokenizer. This archive adds KV only; re-exporting still requires the bound inference bundle from the original vimeml-v21-mac-results.zip.",
        "restore_policy":"Restore original relative ignored paths. Preserve conflicts under a new version; merge source through Git PRs.",
        "verification":"Existing manifest identities; ZIP paths/count/logical sizes verified, no repeated tree hashes."}
    data=(json.dumps(record,ensure_ascii=False,indent=2)+"\n").encode()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,"x",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr("handoff.json",data)
        for path in files:z.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(a.output) as z:
        if {item.filename:item.file_size for item in z.infolist()} != {"handoff.json":len(data),**inventory}:
            raise ValueError("ZIP inventory differs.")
    print(json.dumps({"zip":str(a.output.resolve()),"zip_bytes":a.output.stat().st_size,"files":len(files)+1},indent=2))


if __name__ == "__main__": main()
