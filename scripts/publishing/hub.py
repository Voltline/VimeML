"""Manual Hub upload/download. Network work starts only with an explicit subcommand."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_hf import verifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    upload = sub.add_parser('upload', help='Manually upload only the verified inventory into the existing model repo main branch.')
    upload.add_argument('--release', type=Path, required=True)
    download = sub.add_parser('download', help='Download a fixed Hub commit into a fresh directory, then checksum-verify.')
    download.add_argument('--revision', required=True, help='Full 40-character Hub commit from upload output')
    download.add_argument('--output', type=Path, required=True)
    for command in (upload, download):
        command.add_argument('--repo-id', default='Voltline/vimeml-tiny-ja-v1')
    args = parser.parse_args()
    check = verifier()
    if args.command == 'upload':
        manifest, files = check.verify(args.release)
        if manifest['repo_id'] != args.repo_id:
            parser.error('Release and target repository identities differ.')
        from huggingface_hub import CommitOperationAdd, HfApi
        api = HfApi()
        parent = api.repo_info(args.repo_id, repo_type='model', revision='main').sha
        result = api.create_commit(repo_id=args.repo_id, repo_type='model', revision='main', parent_commit=parent,
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(args.release / name)) for name in files],
            commit_message='Release frozen VimeML tiny-ja-v1 FP32 and Core ML INT8 (GPL-2.0)')
        print(json.dumps({'repo_id': args.repo_id, 'hub_commit': result.oid,
                          'commit_url': result.commit_url, 'uploaded_files': len(files)}, indent=2))
    else:
        if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
            parser.error('Use a full immutable 40-character Hub commit, not main.')
        if args.output.exists():
            parser.error('Output exists; use a fresh download directory.')
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.repo_id, repo_type='model', revision=args.revision, local_dir=str(args.output))
        manifest, files = check.verify(args.output)
        if manifest['repo_id'] != args.repo_id:
            raise ValueError('Downloaded release belongs to a different repository.')
        print(json.dumps({'repo_id': args.repo_id, 'hub_commit': args.revision,
                          'verified_files': len(files), 'source_commit': manifest['source_commit']}, indent=2))


if __name__ == '__main__':
    main()
